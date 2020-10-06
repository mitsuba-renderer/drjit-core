/*
    src/llvm_api.cpp -- Low-level interface to LLVM driver API

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "llvm_api.h"
#include "internal.h"
#include "log.h"
#include "profiler.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <sys/mman.h>
#endif

#include <thread>
#include <lz4.h>
#include "../kernels/kernels.h"

#if !defined(ENOKI_JIT_DYNAMIC_LLVM)
#  include <llvm-c/Core.h>
#  include <llvm-c/ExecutionEngine.h>
#  include <llvm-c/Disassembler.h>
#  include <llvm-c/IRReader.h>
#  include <llvm-c/Transforms/IPO.h>
#else

/// LLVM API
using LLVMBool = int;
using LLVMDisasmContextRef = void *;
using LLVMExecutionEngineRef = void *;
using LLVMModuleRef = void *;
using LLVMMemoryBufferRef = void *;
using LLVMContextRef = void *;
using LLVMPassManagerRef = void *;
using LLVMMCJITMemoryManagerRef = void *;
using LLVMTargetMachineRef = void *;
using LLVMCodeModel = int;

using LLVMMemoryManagerAllocateCodeSectionCallback =
    uint8_t *(*) (void *, uintptr_t, unsigned, unsigned, const char *);

using LLVMMemoryManagerAllocateDataSectionCallback =
    uint8_t *(*) (void *, uintptr_t, unsigned, unsigned, const char *,
                  LLVMBool);

using LLVMMemoryManagerFinalizeMemoryCallback = LLVMBool (*)(void *, char **);

using LLVMMemoryManagerDestroyCallback = void (*)(void *Opaque);

struct LLVMMCJITCompilerOptions {
  unsigned OptLevel;
  LLVMCodeModel CodeModel;
  LLVMBool NoFramePointerElim;
  LLVMBool EnableFastISel;
  void *MCJMM;
};

static void (*LLVMLinkInMCJIT)() = nullptr;
static void (*LLVMInitializeX86AsmPrinter)() = nullptr;
static void (*LLVMInitializeX86Disassembler)() = nullptr;
static void (*LLVMInitializeX86Target)() = nullptr;
static void (*LLVMInitializeX86TargetInfo)() = nullptr;
static void (*LLVMInitializeX86TargetMC)() = nullptr;
static char *(*LLVMCreateMessage)(char *) = nullptr;
static void (*LLVMDisposeMessage)(char *) = nullptr;
static char *(*LLVMGetDefaultTargetTriple)() = nullptr;
static char *(*LLVMGetHostCPUName)() = nullptr;
static char *(*LLVMGetHostCPUFeatures)() = nullptr;
static LLVMContextRef (*LLVMGetGlobalContext)() = nullptr;
static LLVMDisasmContextRef (*LLVMCreateDisasm)(const char *, void *, int,
                                                void *, void *) = nullptr;
static void (*LLVMDisasmDispose)(LLVMDisasmContextRef) = nullptr;
static int (*LLVMSetDisasmOptions)(LLVMDisasmContextRef, uint64_t) = nullptr;
static LLVMModuleRef (*LLVMModuleCreateWithName)(const char *) = nullptr;
static LLVMBool (*LLVMCreateMCJITCompilerForModule)(LLVMExecutionEngineRef *,
                                                    LLVMModuleRef,
                                                    LLVMMCJITCompilerOptions *,
                                                    size_t, char **) = nullptr;
static LLVMMCJITMemoryManagerRef (*LLVMCreateSimpleMCJITMemoryManager)(
    void *, LLVMMemoryManagerAllocateCodeSectionCallback,
    LLVMMemoryManagerAllocateDataSectionCallback,
    LLVMMemoryManagerFinalizeMemoryCallback,
    LLVMMemoryManagerDestroyCallback) = nullptr;
static void (*LLVMDisposeExecutionEngine)(LLVMExecutionEngineRef) = nullptr;
static void (*LLVMAddModule)(LLVMExecutionEngineRef, LLVMModuleRef) = nullptr;
static void (*LLVMDisposeModule)(LLVMModuleRef) = nullptr;
static LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool) = nullptr;
static LLVMBool (*LLVMParseIRInContext)(LLVMContextRef, LLVMMemoryBufferRef,
                                        LLVMModuleRef *, char **) = nullptr;
static char *(*LLVMPrintModuleToString)(LLVMModuleRef) = nullptr;
static uint64_t (*LLVMGetFunctionAddress)(LLVMExecutionEngineRef, const char *);
static LLVMBool (*LLVMRemoveModule)(LLVMExecutionEngineRef, LLVMModuleRef,
                                    LLVMModuleRef *, char **) = nullptr;
static size_t (*LLVMDisasmInstruction)(LLVMDisasmContextRef, uint8_t *,
                                       uint64_t, uint64_t, char *,
                                       size_t) = nullptr;
static LLVMPassManagerRef (*LLVMCreatePassManager)() = nullptr;
static void (*LLVMRunPassManager)(LLVMPassManagerRef, LLVMModuleRef) = nullptr;
static void (*LLVMDisposePassManager)(LLVMPassManagerRef) = nullptr;
static void (*LLVMAddAlwaysInlinerPass)(LLVMPassManagerRef) = nullptr;
static LLVMTargetMachineRef (*LLVMGetExecutionEngineTargetMachine)(
    LLVMExecutionEngineRef) = nullptr;

#define LLVMDisassembler_Option_PrintImmHex       2
#define LLVMDisassembler_Option_AsmPrinterVariant 4

static void *jit_llvm_handle                    = nullptr;
#endif

/// Enoki API
static LLVMDisasmContextRef jit_llvm_disasm_ctx = nullptr;
static LLVMExecutionEngineRef jit_llvm_engine   = nullptr;
static LLVMContextRef jit_llvm_context          = nullptr;
static LLVMPassManagerRef jit_llvm_pass_manager = nullptr;

char    *jit_llvm_triple          = nullptr;
char    *jit_llvm_target_cpu      = nullptr;
char    *jit_llvm_target_features = nullptr;
uint32_t jit_llvm_vector_width    = 0;
size_t   jit_llvm_kernel_id       = 0;
uint32_t jit_llvm_version_major   = 0;
uint32_t jit_llvm_version_minor   = 0;
uint32_t jit_llvm_version_patch   = 0;
uint32_t jit_llvm_thread_count    = 0;

static bool     jit_llvm_init_attempted = false;
static bool     jit_llvm_init_success   = false;

static uint8_t *jit_llvm_mem           = nullptr;
static size_t   jit_llvm_mem_size      = 0;
static size_t   jit_llvm_mem_offset    = 0;
static bool     jit_llvm_got           = false;

extern "C" {

static uint8_t *jit_llvm_mem_allocate(void * /* opaque */, uintptr_t size,
                                      unsigned align, unsigned /* id */,
                                      const char *name) {
    if (align == 0)
        align = 16;

    jit_trace("jit_llvm_mem_allocate(section=%s, size=%zu, align=%u);", name,
              size, (uint32_t) align);

    /* It's bad news if LLVM decides to create a global offset table entry.
       This usually means that a compiler intrinsic didn't resolve to a machine
       instruction, and a function call to an external library was generated
       along with a relocation, which we don't support. */
    if (strncmp(name, ".got", 4) == 0)
        jit_llvm_got = true;

    size_t offset_align = (jit_llvm_mem_offset + (align - 1)) / align * align;

    // Zero-fill including padding region
    memset(jit_llvm_mem + jit_llvm_mem_offset, 0,
           offset_align - jit_llvm_mem_offset);

    jit_llvm_mem_offset = offset_align + size;

    if (jit_llvm_mem_offset > jit_llvm_mem_size)
        return nullptr;

    return jit_llvm_mem + offset_align;
}

static uint8_t *jit_llvm_mem_allocate_data(void *opaque, uintptr_t size,
                                           unsigned align, unsigned id,
                                           const char *name,
                                           LLVMBool /* read_only */) {
    return jit_llvm_mem_allocate(opaque, size, align, id, name);
}

static LLVMBool jit_llvm_mem_finalize(void * /* opaque */, char ** /* err */) {
    return 0;
}

static void jit_llvm_mem_destroy(void * /* opaque */) { }

} /* extern "C" */ ;

/// Dump assembly representation
void jit_llvm_disasm(const Kernel &kernel) {
    if (std::max(state.log_level_stderr, state.log_level_callback) <
        LogLevel::Trace)
        return;

    uint8_t *func_base = (uint8_t *) kernel.llvm.func,
            *ptr = func_base;
    char ins_buf[256];
    bool last_nop = false;
    jit_trace("jit_llvm_disasm(): =====================");
    do {
        size_t offset      = ptr - (uint8_t *) kernel.data,
               func_offset = ptr - func_base;
        if (offset >= kernel.size)
            break;
        size_t size =
            LLVMDisasmInstruction(jit_llvm_disasm_ctx, ptr, kernel.size - offset,
                                  (uintptr_t) ptr, ins_buf, sizeof(ins_buf));
        if (size == 0)
            break;
        char *start = ins_buf;
        while (*start == ' ' || *start == '\t')
            ++start;
        if (strcmp(start, "nop") == 0) {
            if (!last_nop)
                jit_trace("jit_llvm_disasm(): ...");
            last_nop = true;
            ptr += size;
            continue;
        }
        last_nop = false;
        jit_trace("jit_llvm_disasm(): 0x%08x   %s", (uint32_t) func_offset, start);
        if (strncmp(start, "ret", 3) == 0)
            break;
        ptr += size;
    } while (true);
}

static ProfilerRegion profiler_region_llvm_compile("jit_llvm_compile");

void jit_llvm_compile(const char *buffer, size_t buffer_size, Kernel &kernel,
                      bool include_supplement) {
    ProfilerPhase phase(profiler_region_llvm_compile);
    char *temp = nullptr;
    if (include_supplement) {
        jit_lz4_init();

        /* Use one of two different variants of the supplemental
           kernels depending on the version of the LLVM IR. */
        bool legacy = jit_llvm_version_major < 9;
        int size_uncompressed = legacy ? llvm_kernels_7_size_uncompressed
                                       : llvm_kernels_9_size_uncompressed;
        int size_compressed   = legacy ? llvm_kernels_7_size_compressed
                                       : llvm_kernels_9_size_compressed;
        const char *kernels   = legacy ? llvm_kernels_7 : llvm_kernels_9;

        size_t temp_size = buffer_size + jit_lz4_dict_size + size_uncompressed + 1;

        // Decompress supplemental kernel IR content
        temp = (char *) malloc_check(temp_size);
        memcpy(temp, jit_lz4_dict, jit_lz4_dict_size);
        char *buffer_new = temp + jit_lz4_dict_size;

        if (LZ4_decompress_safe_usingDict(
                kernels, buffer_new, size_compressed, size_uncompressed, temp,
                jit_lz4_dict_size) != size_uncompressed)
            jit_fail("jit_cuda_init(): decompression of supplemental kernel "
                     "fragments failed!");

        memcpy(buffer_new + size_uncompressed, buffer, buffer_size);
        buffer_new[size_uncompressed + buffer_size] = '\0';
        buffer_size += size_uncompressed;
        buffer = buffer_new;
    }

    if (jit_llvm_mem_size <= buffer_size) {
        // Central assumption: LLVM text IR is much larger than the resulting generated code.
#if !defined(_WIN32)
        free(jit_llvm_mem);
        if (posix_memalign((void **) &jit_llvm_mem, 64, buffer_size))
            jit_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", buffer_size);
#else
        _aligned_free(jit_llvm_mem);
        jit_llvm_mem = (uint8_t *) _aligned_malloc(buffer_size, 64);
        if (!jit_llvm_mem)
            jit_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", buffer_size);
#endif
        jit_llvm_mem_size = buffer_size;
    }
    jit_llvm_mem_offset = 0;

    // Temporarily change the kernel name
    char kernel_name_old[23]{}, kernel_name_new[30]{};
    snprintf(kernel_name_new, 23, "enoki_%016llx", (long long) jit_llvm_kernel_id);

    char *offset = (char *) buffer;
    do {
        offset = (char *) strstr(offset, "enoki_");
        if (offset == nullptr)
            break;
        memcpy(kernel_name_old, offset, 22);
        memcpy(offset, kernel_name_new, 22);
        offset += 22;
    } while (true);

    LLVMMemoryBufferRef buf = LLVMCreateMemoryBufferWithMemoryRange(
        buffer, buffer_size, kernel_name_new, 0);
    if (unlikely(!buf))
        jit_fail("jit_run_compile(): could not create memory buffer!");

    // 'buf' is consumed by this function.
    LLVMModuleRef llvm_module = nullptr;
    char *error = nullptr;
    LLVMParseIRInContext(jit_llvm_context, buf, &llvm_module, &error);
    if (unlikely(error))
        jit_fail("jit_llvm_compile(): could not parse IR: %s.\n", error);

    if (false) {
        char *llvm_ir = LLVMPrintModuleToString(llvm_module);
        jit_trace("jit_llvm_compile(): Parsed LLVM IR:\n%s", llvm_ir);
        LLVMDisposeMessage(llvm_ir);
    }

    // Inline supplemental code fragments into currently compiled kernel
    if (include_supplement)
        LLVMRunPassManager(jit_llvm_pass_manager, llvm_module);

    LLVMAddModule(jit_llvm_engine, llvm_module);

    uint8_t *func =
        (uint8_t *) LLVMGetFunctionAddress(jit_llvm_engine, kernel_name_new);
    if (unlikely(!func))
        jit_fail("jit_llvm_compile(): internal error: could not fetch function "
                 "address of kernel \"%s\"!\n", kernel_name_new);
    else if (unlikely(func < jit_llvm_mem))
        jit_fail("jit_llvm_compile(): internal error: invalid address: "
                 "%p < %p!\n", func, jit_llvm_mem);

    if (jit_llvm_got)
        jit_fail(
            "jit_llvm_compile(): a global offset table was generated by LLVM, "
            "which typically means that a compiler intrinsic was not supported "
            "by the target architecture. Enoki cannot handle this case "
            "and will terminate the application now. For reference, the "
            "following kernel code was responsible for this problem:\n\n%s",
            buffer);

    uint32_t func_offset        = (uint32_t) (func        - jit_llvm_mem);

#if !defined(_WIN32)
    void *ptr_result =
        mmap(nullptr, jit_llvm_mem_offset, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr_result == MAP_FAILED)
        jit_fail("jit_llvm_compile(): could not mmap() memory: %s",
                 strerror(errno));
    memcpy(ptr_result, jit_llvm_mem, jit_llvm_mem_offset);

    if (mprotect(ptr_result, jit_llvm_mem_offset, PROT_READ | PROT_EXEC) == -1)
        jit_fail("jit_llvm_compile(): mprotect() failed: %s", strerror(errno));
#else
    void* ptr_result = VirtualAlloc(nullptr, jit_llvm_mem_offset, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!ptr_result)
        jit_fail("jit_llvm_compile(): could not VirtualAlloc() memory: %u", GetLastError());
    memcpy(ptr_result, jit_llvm_mem, jit_llvm_mem_offset);
    DWORD unused;
    if (VirtualProtect(ptr_result, jit_llvm_mem_offset, PAGE_EXECUTE_READ, &unused) == 0)
        jit_fail("jit_llvm_compile(): VirtualProtect() failed: %u", GetLastError());
#endif

    LLVMRemoveModule(jit_llvm_engine, llvm_module, &llvm_module, &error);
    if (unlikely(error))
        jit_fail("jit_llvm_compile(): could remove module: %s.\n", error);
    LLVMDisposeModule(llvm_module);

    // Change the kernel name back
    offset = (char *) buffer;
    do {
        offset = (char *) strstr(offset, "enoki_");
        if (offset == nullptr)
            break;
        memcpy(offset, kernel_name_old, 22);
        offset += 22;
    } while (true);

    kernel.data = ptr_result;
    kernel.size = (uint32_t) jit_llvm_mem_offset;
    kernel.llvm.func        = (LLVMKernelFunction) ((uint8_t *) ptr_result + func_offset);
#if defined(ENOKI_ENABLE_ITTNOTIFY)
    kernel.llvm.itt = __itt_string_handle_create(kernel_name_old);
#endif
    jit_llvm_kernel_id++;
    free(temp);
}

void jit_llvm_set_target(const char *target_cpu,
                         const char *target_features,
                         uint32_t vector_width) {
    if (!jit_llvm_init_success)
        return;

    if (jit_llvm_target_cpu)
        LLVMDisposeMessage(jit_llvm_target_cpu);

    if (jit_llvm_target_features) {
        LLVMDisposeMessage(jit_llvm_target_features);
        jit_llvm_target_features = nullptr;
    }

    jit_llvm_vector_width = vector_width;
    jit_llvm_target_cpu = LLVMCreateMessage((char *) target_cpu);
    if (target_features)
        jit_llvm_target_features = LLVMCreateMessage((char *) target_features);
}

/// Convenience function for intrinsic function selection
int jit_llvm_if_at_least(uint32_t vector_width, const char *feature) {
    return jit_llvm_vector_width >= vector_width &&
           jit_llvm_target_features != nullptr &&
           strstr(jit_llvm_target_features, feature) != nullptr;
}

bool jit_llvm_init() {
    if (jit_llvm_init_attempted)
        return jit_llvm_init_success;
    jit_llvm_init_attempted = true;

#if defined(ENOKI_JIT_DYNAMIC_LLVM)
    jit_llvm_handle = nullptr;
#  if defined(_WIN32)
#    define dlsym(ptr, name) GetProcAddress((HMODULE) ptr, name)
    const char *llvm_fname = "LLVM-C.dll",
               *llvm_glob  = nullptr;
#  elif defined(__linux__)
    const char *llvm_fname  = "libLLVM.so",
               *llvm_glob   = "/usr/lib/x86_64-linux-gnu/libLLVM*.so.*";
#  else
    const char *llvm_fname  = "libLLVM.dylib",
               *llvm_glob   = "/usr/local/Cellar/llvm/*/lib/libLLVM.dylib";
#  endif

#  if !defined(_WIN32)
    // Don't dlopen libLLVM.so if it was loaded by another library
    if (dlsym(RTLD_NEXT, "LLVMLinkInMCJIT"))
        jit_llvm_handle = RTLD_NEXT;
#  endif

    if (!jit_llvm_handle) {
        jit_llvm_handle = jit_find_library(llvm_fname, llvm_glob, "ENOKI_LIBLLVM_PATH");

        if (!jit_llvm_handle) {
            jit_log(Warn, "jit_llvm_init(): %s could not be loaded -- "
                          "disabling LLVM backend! Set the 'ENOKI_LIBLLVM_PATH' "
                          "environment variable to specify its path.", llvm_fname);
            return false;
        }
    }

    #define LOAD(name)                                                         \
        symbol = #name;                                                        \
        name = decltype(name)(dlsym(jit_llvm_handle, symbol));                 \
        if (!name)                                                             \
            break;                                                             \
        symbol = nullptr

    const char *symbol = nullptr;
    do {
        LOAD(LLVMLinkInMCJIT);
        LOAD(LLVMInitializeX86Target);
        LOAD(LLVMInitializeX86TargetInfo);
        LOAD(LLVMInitializeX86TargetMC);
        LOAD(LLVMInitializeX86AsmPrinter);
        LOAD(LLVMInitializeX86Disassembler);
        LOAD(LLVMGetGlobalContext);
        LOAD(LLVMGetDefaultTargetTriple);
        LOAD(LLVMGetHostCPUName);
        LOAD(LLVMGetHostCPUFeatures);
        LOAD(LLVMCreateMessage);
        LOAD(LLVMDisposeMessage);
        LOAD(LLVMCreateDisasm);
        LOAD(LLVMDisasmDispose);
        LOAD(LLVMSetDisasmOptions);
        LOAD(LLVMModuleCreateWithName);
        LOAD(LLVMCreateMCJITCompilerForModule);
        LOAD(LLVMCreateSimpleMCJITMemoryManager);
        LOAD(LLVMDisposeExecutionEngine);
        LOAD(LLVMAddModule);
        LOAD(LLVMDisposeModule);
        LOAD(LLVMCreateMemoryBufferWithMemoryRange);
        LOAD(LLVMParseIRInContext);
        LOAD(LLVMPrintModuleToString);
        LOAD(LLVMGetFunctionAddress);
        LOAD(LLVMRemoveModule);
        LOAD(LLVMDisasmInstruction);
        LOAD(LLVMCreatePassManager);
        LOAD(LLVMRunPassManager);
        LOAD(LLVMDisposePassManager);
        LOAD(LLVMAddAlwaysInlinerPass);
        LOAD(LLVMGetExecutionEngineTargetMachine);
    } while (false);

    if (symbol) {
        jit_log(Warn,
                "jit_llvm_init(): could not find symbol \"%s\" -- disabling "
                "LLVM backend!", symbol);
        return false;
    }
#endif

#if defined(ENOKI_JIT_DYNAMIC_LLVM)
    void *dlsym_src = jit_llvm_handle;
#else
    void *dlsym_src = RTLD_NEXT;
#endif

    auto get_version_string = (const char * (*) ()) dlsym(
        dlsym_src, "_ZN4llvm16LTOCodeGenerator16getVersionStringEv");

    if (get_version_string) {
        const char* version_string = get_version_string();
        if (sscanf(version_string, "LLVM version %u.%u.%u", &jit_llvm_version_major,
                   &jit_llvm_version_minor, &jit_llvm_version_patch) != 3) {
            jit_log(Warn,
                    "jit_llvm_init(): could not parse LLVM version string \"%s\".",
                    version_string);
            return false;
        }

        if (jit_llvm_version_major < 7) {
            jit_log(Warn,
                    "jit_llvm_init(): LLVM version 7 or later must be used. (found "
                    "%s). You may want to define the 'ENOKI_LIBLLVM_PATH' "
                    "environment variable to specify the path to "
                    "libLLVM.so/dylib/dll of a particular LLVM version.",
                    version_string);
        }
    } else {
        jit_llvm_version_major = 10;
        jit_llvm_version_minor = 0;
        jit_llvm_version_patch = 0;
    }

    LLVMLinkInMCJIT();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86Disassembler();

    jit_llvm_context = LLVMGetGlobalContext();
    if (!jit_llvm_context) {
        jit_log(Warn, "jit_llvm_init(): could not obtain context!");
        return false;
    }

    jit_llvm_triple = LLVMGetDefaultTargetTriple();
    jit_llvm_target_cpu = LLVMGetHostCPUName();
    jit_llvm_target_features = LLVMGetHostCPUFeatures();

    jit_llvm_disasm_ctx =
        LLVMCreateDisasm(jit_llvm_triple, nullptr, 0, nullptr, nullptr);

    if (!jit_llvm_disasm_ctx) {
        jit_log(Warn, "jit_llvm_init(): could not create a disassembler!");
        LLVMDisposeMessage(jit_llvm_triple);
        LLVMDisposeMessage(jit_llvm_target_cpu);
        LLVMDisposeMessage(jit_llvm_target_features);
        return false;
    }

    if (LLVMSetDisasmOptions(jit_llvm_disasm_ctx,
                             LLVMDisassembler_Option_PrintImmHex |
                             LLVMDisassembler_Option_AsmPrinterVariant) == 0) {
        jit_log(Warn, "jit_llvm_init(): could not configure disassembler!");
        LLVMDisasmDispose(jit_llvm_disasm_ctx);
        LLVMDisposeMessage(jit_llvm_triple);
        LLVMDisposeMessage(jit_llvm_target_cpu);
        LLVMDisposeMessage(jit_llvm_target_features);
        return false;
    }

    LLVMMCJITCompilerOptions options;
    options.OptLevel = 3;
    options.CodeModel =
        (LLVMCodeModel)(jit_llvm_version_major == 7 ? 2 : 3); /* Small */
    options.NoFramePointerElim = false;
    options.EnableFastISel = false;
    options.MCJMM = LLVMCreateSimpleMCJITMemoryManager(
        nullptr,
        jit_llvm_mem_allocate,
        jit_llvm_mem_allocate_data,
        jit_llvm_mem_finalize,
        jit_llvm_mem_destroy);

    LLVMModuleRef enoki_module = LLVMModuleCreateWithName("enoki");
    char *error = nullptr;
    if (LLVMCreateMCJITCompilerForModule(&jit_llvm_engine, enoki_module,
                                         &options, sizeof(options), &error)) {
        jit_log(Warn, "jit_llvm_init(): could not create MCJIT: %s", error);
        LLVMDisposeModule(enoki_module);
        LLVMDisasmDispose(jit_llvm_disasm_ctx);
        LLVMDisposeMessage(jit_llvm_triple);
        LLVMDisposeMessage(jit_llvm_target_cpu);
        LLVMDisposeMessage(jit_llvm_target_features);
        return false;
    }

    /**
       The following is horrible, but it works and was without alternative.

       LLVM MCJIT uses a 'static' relocation model by default. But that's not
       good for us: we want a 'pic' relocation model that allows for *relocatable
       code*. This is important because generated kernels can be reloaded from
       the cache, at which point they may end up anywhere in memory.

       The LLVM C API is quite nice, but it's missing a function to adjust this
       crucial aspect. While we could in principle use C++ API, this totally
       infeasible without a hard dependency on the LLVM headers and being being
       permanently tied to a specific version.

       The code below simply hot-patches the TargetMachine* instance, which
       contains the following 3 fields (around byte offset 568 on my machine)

          Reloc::Model RM = Reloc::Static;
          CodeModel::Model CMModel = CodeModel::Small;
          CodeGenOpt::Level OptLevel = CodeGenOpt::Default;

       Which are all 32 bit integers. This interface has been stable since
       ancient times (LLVM 4), and the good thing is that we know the values of
       these fields and can therefore use them as a 12-byte key to find the
       precise byte offset and them overwrite the 'RM' field.
    */

    uint32_t *patch_loc =
        (uint32_t *) LLVMGetExecutionEngineTargetMachine(jit_llvm_engine) + 142 - 16;

    int key[3] = { 0, jit_llvm_version_major == 7 ? 0 : 1, 3 };
    bool found = false;
    for (int i = 0; i < 30; ++i) {
        if (memcmp(patch_loc, key, sizeof(uint32_t) * 3) == 0) {
            found = true;
            break;
        }
        patch_loc += 1;
    }

    if (!found) {
        jit_log(Warn, "jit_llvm_init(): could not hot-patch TargetMachine relocation model!");
        LLVMDisposeModule(enoki_module);
        LLVMDisasmDispose(jit_llvm_disasm_ctx);
        LLVMDisposeMessage(jit_llvm_triple);
        LLVMDisposeMessage(jit_llvm_target_cpu);
        LLVMDisposeMessage(jit_llvm_target_features);
        return false;
    }
    patch_loc[0] = 1;

    jit_llvm_pass_manager = LLVMCreatePassManager();
    LLVMAddAlwaysInlinerPass(jit_llvm_pass_manager);

    jit_llvm_vector_width = 1;

    if (strstr(jit_llvm_target_features, "+sse4.2"))
        jit_llvm_vector_width = 4;
    if (strstr(jit_llvm_target_features, "+avx"))
        jit_llvm_vector_width = 8;
    if (strstr(jit_llvm_target_features, "+avx512f"))
        jit_llvm_vector_width = 16;

#if defined(ENOKI_JIT_ENABLE_TBB)
    jit_llvm_thread_count =
        std::max(1u, (uint32_t) std::thread::hardware_concurrency());
#else
    jit_llvm_thread_count = 1;
#endif

    jit_log(Info,
            "jit_llvm_init(): found LLVM %u.%u.%u, target=%s, cpu=%s, vector width=%u, threads=%u.",
            jit_llvm_version_major, jit_llvm_version_minor, jit_llvm_version_patch, jit_llvm_triple,
            jit_llvm_target_cpu, jit_llvm_vector_width, jit_llvm_thread_count);

    jit_llvm_init_success = jit_llvm_vector_width > 1;

    if (!jit_llvm_init_success) {
        jit_log(Warn, "jit_llvm_init(): no suitable vector ISA found, shutting "
                      "down LLVM backend..");
        jit_llvm_shutdown();
    }

    return jit_llvm_init_success;
}

void jit_llvm_shutdown() {
    if (!jit_llvm_init_success)
        return;

    jit_log(Info, "jit_llvm_shutdown()");

    LLVMDisasmDispose(jit_llvm_disasm_ctx);
    LLVMDisposeExecutionEngine(jit_llvm_engine);
    LLVMDisposeMessage(jit_llvm_triple);
    LLVMDisposeMessage(jit_llvm_target_cpu);
    LLVMDisposeMessage(jit_llvm_target_features);
    LLVMDisposePassManager(jit_llvm_pass_manager);

    jit_llvm_engine = nullptr;
    jit_llvm_disasm_ctx = nullptr;
    jit_llvm_context = nullptr;
    jit_llvm_pass_manager = nullptr;
    jit_llvm_target_cpu = nullptr;
    jit_llvm_target_features = nullptr;
    jit_llvm_vector_width = 0;

#if !defined(_WIN32)
    free(jit_llvm_mem);
#else
    _aligned_free(jit_llvm_mem);
#endif

    jit_llvm_mem        = nullptr;
    jit_llvm_mem_size   = 0;
    jit_llvm_mem_offset = 0;
    jit_llvm_kernel_id  = 0;
    jit_llvm_got        = false;

#if defined(ENOKI_JIT_DYNAMIC_LLVM)
    #define Z(x) x = nullptr

    Z(LLVMLinkInMCJIT); Z(LLVMInitializeX86Target);
    Z(LLVMInitializeX86TargetInfo); Z(LLVMInitializeX86TargetMC);
    Z(LLVMInitializeX86AsmPrinter); Z(LLVMInitializeX86Disassembler);
    Z(LLVMGetGlobalContext); Z(LLVMGetDefaultTargetTriple);
    Z(LLVMGetHostCPUName); Z(LLVMGetHostCPUFeatures); Z(LLVMCreateMessage);
    Z(LLVMDisposeMessage); Z(LLVMCreateDisasm); Z(LLVMDisasmDispose);
    Z(LLVMSetDisasmOptions); Z(LLVMModuleCreateWithName);
    Z(LLVMCreateMCJITCompilerForModule); Z(LLVMCreateSimpleMCJITMemoryManager);
    Z(LLVMDisposeExecutionEngine); Z(LLVMAddModule); Z(LLVMDisposeModule);
    Z(LLVMCreateMemoryBufferWithMemoryRange); Z(LLVMParseIRInContext);
    Z(LLVMPrintModuleToString); Z(LLVMGetFunctionAddress); Z(LLVMRemoveModule);
    Z(LLVMDisasmInstruction); Z(LLVMCreatePassManager); Z(LLVMRunPassManager);
    Z(LLVMDisposePassManager); Z(LLVMAddAlwaysInlinerPass);
    Z(LLVMGetExecutionEngineTargetMachine);

#  if !defined(_WIN32)
    if (jit_llvm_handle != RTLD_NEXT)
        dlclose(jit_llvm_handle);
#  else
    FreeLibrary((HMODULE) jit_llvm_handle);
#  endif

    jit_llvm_handle = nullptr;
#endif

    jit_llvm_init_success = false;
    jit_llvm_init_attempted = false;
}
