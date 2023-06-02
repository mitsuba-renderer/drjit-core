#include <nanothread/nanothread.h>
#include "state.h"
#include "llvm.h"
#include "log.h"
#include "thread_state.h"
#include "llvm_memmgr.h"

NB_TLS ThreadState* thread_state_llvm = nullptr;

/// Current top-level task in the task queue
Task *jitc_task = nullptr;

/// Helper function: enqueue parallel CPU task (synchronous or asynchronous)
template <typename Func>
void jitc_submit_cpu(KernelType type, Func &&func, uint32_t width,
                     uint32_t size = 1, bool release_prev = true,
                     bool always_async = false) {

    struct Payload { Func f; };
    Payload payload{ std::forward<Func>(func) };

    static_assert(std::is_trivially_copyable_v<Payload> &&
                  std::is_trivially_destructible_v<Payload>, "Internal error!");

    Task *new_task = task_submit_dep(
        nullptr, &jitc_task, 1, size,
        [](uint32_t index, void *payload) { ((Payload *) payload)->f(index); },
        &payload, sizeof(Payload), nullptr, (int) always_async);

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        task_wait(new_task);

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        KernelHistoryEntry entry = {};
        entry.backend = JitBackend::LLVM;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        task_retain(new_task);
        entry.task = new_task;
        state.kernel_history.append(entry);
    }

    if (release_prev)
        task_release(jitc_task);

    jitc_task = new_task;
}

class LLVMThreadState : public ThreadState {
public:
    // ================== ThreadState interface ==================

    /// Allocate private or shared memory accessible on host+device
    void *malloc(size_t size, bool shared) override;

    /// Associate the CPU memory address associated with an allocation
    void *host_ptr(void *ptr) override;

    /**
     * Compile the kernel source (buf, buf_size) and with entry point 'name'
     * and store the result in 'kernel'.
     *
     * The function returns `true` if compilation could be avoided by means of
     * a secondary caching mechanism (e.g., OptiX)
     */
    bool compile(const char *buf, size_t buf_size, const char *name,
                 Kernel &kernel) override;

    /// Enqueue a memory copy operation
    void enqueue_memcpy(void *dst, const void *src, size_t size) override;

    /// Enqueue a host callback function
    void enqueue_callback(void (*fn)(void *), void *payload) override;

    /// Wait for queued computation to finish
    void sync() override;
};

ThreadState *jitc_llvm_thread_state_new() {
    if (!jitc_llvm_target_cpu) {
        #if defined(_WIN32)
            const char *llvm_fname = "LLVM-C.dll";
        #elif defined(__linux__)
            const char *llvm_fname  = "libLLVM.so";
        #else
            const char *llvm_fname  = "libLLVM.dylib";
        #endif

        jitc_raise(
            "jit_cuda_thread_state_new(): the LLVM backend has not been "
            "initialized.\nThere could be two reasons for this:\n\n 1. The "
            "LLVM driver library (\"%s\") could not be found or does not "
            "have\n    the right version. LLVM 8 or newer is required. You can "
            "manually\n    specify the LLVM library path using the "
            "DRJIT_LIBLLVM_PATH\n    environment variable.\n\n 2. The "
            "application code did not perform the backend initialization.\n    "
            "Call `jit_init(1 << (int) JitBackend::LLVM)` in this case.",
            llvm_fname);
    }

    return new LLVMThreadState();
}

void *LLVMThreadState::malloc(size_t, bool) {
    jitc_fail("LLVMThreadState::malloc(): should never be called!");
}

void *LLVMThreadState::host_ptr(void *ptr) {
    return ptr;
}

bool LLVMThreadState::compile(const char *buf, size_t buf_size,
                              const char *name, Kernel &kernel) {
    jitc_llvm_memmgr_prepare(buf_size);

    LLVMMemoryBufferRef llvm_buf =
        LLVMCreateMemoryBufferWithMemoryRange(buf, buf_size, name, 0);
    if (unlikely(!llvm_buf))
        jitc_fail("jit_run_compile(): could not create memory buffer!");

    // 'buf' is consumed by this function.
    LLVMModuleRef llvm_module = nullptr;
    char *error = nullptr;
    LLVMParseIRInContext(jitc_llvm_context, llvm_buf, &llvm_module, &error);
    if (unlikely(error))
        jitc_fail("jit_llvm_compile(): parsing failed. Please see the LLVM "
                  "IR and error message below:\n\n%s\n\n%s", buffer.get(), error);

#if !defined(NDEBUG)
    bool status = LLVMVerifyModule(llvm_module, LLVMReturnStatusAction, &error);
    if (unlikely(status))
        jitc_fail("jit_llvm_compile(): module could not be verified! Please "
                  "see the LLVM IR and error message below:\n\n%s\n\n%s",
                  buffer.get(), error);
#endif

    LLVMRunPassManager(jitc_llvm_pass_manager, llvm_module);

    std::vector<uint8_t *> reloc(
        callable_count_unique ? (callable_count_unique + 2) : 1);

    if (jitc_llvm_use_orcv2)
        jitc_llvm_orcv2_compile(llvm_module, reloc);
    else
        jitc_llvm_mcjit_compile(llvm_module, reloc);

    if (jitc_llvm_memmgr_got)
        jitc_fail(
            "jit_llvm_compile(): a global offset table was generated by LLVM, "
            "which typically means that a compiler intrinsic was not supported "
            "by the target architecture. DrJit cannot handle this case "
            "and will terminate the application now. For reference, the "
            "following kernel code was responsible for this problem:\n\n%s",
            buffer.get());

#if !defined(_WIN32)
    void *ptr = mmap(nullptr, jitc_llvm_memmgr_offset, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED)
        jitc_fail("jit_llvm_compile(): could not mmap() memory: %s",
                  strerror(errno));
#else
    void *ptr = VirtualAlloc(nullptr, jitc_llvm_memmgr_offset,
                             MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!ptr)
        jitc_fail("jit_llvm_compile(): could not VirtualAlloc() memory: %u", GetLastError());
#endif
    memcpy(ptr, jitc_llvm_memmgr_data, jitc_llvm_memmgr_offset);

    kernel.data = ptr;
    kernel.size = (uint32_t) jitc_llvm_memmgr_offset;
    kernel.llvm.n_reloc = (uint32_t) reloc.size();
    kernel.llvm.reloc = (void **) malloc_check(sizeof(void *) * reloc.size());

    // Relocate function pointers
    for (size_t i = 0; i < reloc.size(); ++i)
        kernel.llvm.reloc[i] = (uint8_t *) ptr + (reloc[i] - jitc_llvm_memmgr_data);

    // Write address of @callables
    if (kernel.llvm.n_reloc > 1)
        *((void **) kernel.llvm.reloc[1]) = kernel.llvm.reloc + 1;

#if defined(DRJIT_ENABLE_ITTNOTIFY)
    kernel.llvm.itt = __itt_string_handle_create(name);
#endif

#if !defined(_WIN32)
    if (mprotect(ptr, jitc_llvm_memmgr_offset, PROT_READ | PROT_EXEC) == -1)
        jitc_fail("jit_llvm_compile(): mprotect() failed: %s", strerror(errno));
#else
    DWORD unused;
    if (VirtualProtect(ptr, jitc_llvm_memmgr_offset, PAGE_EXECUTE_READ, &unused) == 0)
        jitc_fail("jit_llvm_compile(): VirtualProtect() failed: %u", GetLastError());
#endif

    if (std::max(state.log_level_stderr, state.log_level_callback) >= LogLevel::Debug)
        jitc_llvm_disasm(kernel);
}

void LLVMThreadState::sync() {
    task_wait_and_release(jitc_task);
    jitc_task = nullptr;
}

void LLVMThreadState::enqueue_memcpy(void *dst, const void *src, size_t size) {
     jitc_submit_cpu(
         KernelType::Other,
         [dst, src, size](uint32_t) {
             memcpy(dst, src, size);
         },
         (uint32_t) size);
}
