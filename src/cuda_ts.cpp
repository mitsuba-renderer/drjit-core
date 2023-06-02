#include "cuda.h"
#include "cuda_api.h"

NB_TLS ThreadState* thread_state_cuda = nullptr;

void jitc_submit_gpu(KernelType type, CUfunction kernel, uint32_t block_count,
                     uint32_t thread_count, uint32_t shared_mem_bytes,
                     CUstream stream, void **args, void **extra,
                     uint32_t width) {

    KernelHistoryEntry entry = {};

    uint32_t flags = jit_flags();

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        cuda_check(cuEventCreate((CUevent *) &entry.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &entry.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) entry.event_start, stream));
    }

    cuda_check(cuLaunchKernel(kernel, block_count, 1, 1, thread_count, 1, 1,
                              shared_mem_bytes, stream, args, extra));

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        cuda_check(cuStreamSynchronize(stream));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        entry.backend = JitBackend::CUDA;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        cuda_check(cuEventRecord((CUevent) entry.event_end, stream));

        state.kernel_history.append(entry);
    }
}


class CUDAThreadState : public ThreadState {
public:
    // ================== ThreadState interface ==================

    /// Allocate private or shared memory accessible on host+device
    void *malloc(size_t size, bool shared) override;

    /// Associate the CPU memory address associated with an allocation
    void *host_ptr(void *ptr) override;

    /**
     * Return an opaque set of flags characterizing the kernel being compiled
     * based on backend compilation settings. This isolates the kernel from
     * other variants of the same code compiled with *different* settings.
     */
    uint64_t kernel_flags() const override;

    /**
     * Compile the kernel source (buf, buf_size) and with entry point 'name'
     * and store the result in 'kernel'.
     *
     * The function returns `true` if compilation could be avoided by means of
     * a secondary caching mechanism (e.g., OptiX)
     */
    bool compile(const char *buf, size_t buf_size, const char *name,
                 Kernel &kernel);

    /// Enqueue a memory copy operation
    void enqueue_memcpy(void *dst, const void *src, size_t size) override;

    /// Enqueue a host callback function
    void enqueue_callback(void (*fn)(void *), void *payload) override;

    /// Wait for queued computation to finish
    void sync() override;

private:
    // ================== Private members ==================

    /// Redundant copy of the device context
    CUcontext context;

    /// Associated CUDA stream handle
    CUstream stream;

    /// A CUDA event for synchronization purposes
    CUevent event;

    /// Device compute capability (major * 10 + minor)
    uint32_t compute_capability;

    /// Targeted PTX version (major * 10 + minor)
    uint32_t ptx_version;

    // Support for stream-ordered memory allocations (async alloc/free)
    bool memory_pool;

#if defined(DRJIT_ENABLE_OPTIX)
    /// OptiX pipeline associated with the next kernel launch
    OptixPipelineData *optix_pipeline = nullptr;

    OptixShaderBindingTable *optix_sbt = nullptr;
#endif
};

ThreadState *jitc_cuda_thread_state_new() {
    if (jitc_cuda_cuinit_result == CUDA_ERROR_NOT_INITIALIZED) {
        #if defined(_WIN32)
            const char *cuda_fname = "nvcuda.dll";
        #elif defined(__linux__)
            const char *cuda_fname  = "libcuda.so";
        #else
            const char *cuda_fname  = "libcuda.dylib";
        #endif

        jitc_raise(
            "jit_cuda_thread_state_new(): the CUDA backend has not been "
            "initialized.\nThere could be two reasons for this:\n\n 1. The "
            "CUDA driver library (\"%s\") could not be found. You can "
            "manually\n    specify its path using the DRJIT_LIBCUDA_PATH "
            "environment variable.\n\n 2. The application code did not perform "
            "the backend initialization.\n    Call `jit_init(1 << (int) "
            "JitBackend::CUDA)` in this case.", cuda_fname);
    } else if (jitc_cuda_cuinit_result != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(jitc_cuda_cuinit_result, &msg);
        jitc_raise(
            "jit_cuda_thread_state_new(): the CUDA backend is not available "
            "because\n`cuInit()` failed. There are two common causes of this "
            "specific failure:\n\n1. your computer simply does not contain a "
            "graphics card that supports CUDA.\n\n2. your CUDA kernel module "
            "and CUDA library are out of sync. Try to run the\n   `nvida-smi` "
            "command-line utility to test this hypothesis. If this also\n   "
            "fails with an error message, reboot your computer. If the error "
            "persists\n   after a reboot, you will need to reinstall your "
            "graphics driver.\n\n   The specific error message produced by "
            "cuInit was\n   \"%s\"", msg);
    } else if (state.devices.empty()) {
        jitc_raise("jit_init_thread_state(): the CUDA backend could not find "
                   "compatible CUDA devices on on your machine.");
    }

    Device &device = state.devices[0];
    CUDAThreadState *ts = new CUDAThreadState();
    ts->device = 0;
    ts->context = device.context;
    ts->compute_capability = device.compute_capability;
    ts->ptx_version = device.ptx_version;
    ts->memory_pool = device.memory_pool;
    ts->stream = device.stream;
    ts->event = device.event;
    return ts;
}

void jitc_cuda_set_device(int device_id) {
    CUDAThreadState *ts = (CUDAThreadState *) thread_state(JitBackend::CUDA);
    if (ts->device == device_id)
        return;

    if ((size_t) device_id >= state.devices.size())
        jitc_raise("jit_cuda_set_device(%i): must be in the range 0..%i!",
                  device_id, (int) state.devices.size() - 1);

    jitc_log(Info, "jit_cuda_set_device(%i)", device_id);

    Device &device = state.devices[device_id];

    /* Associate with new context */ {
        ts->context = device.context;
        ts->device = device_id;
        ts->compute_capability = device.compute_capability;
        ts->ptx_version = device.ptx_version;
        ts->memory_pool = device.memory_pool;
        ts->stream = device.stream;
        ts->event = device.event;
    }
}

void *CUDAThreadState::malloc(size_t size, bool shared) {
    scoped_set_context guard(context);
    CUresult ret;

    if (shared)
        ret = cuMemAllocHost(&ptr, size);
    else if (memory_pool)
        ret = cuMemAllocAsync(&ptr, size, stream);
    else
        ret = cuMemAlloc(&ptr, size);

    if (ret)
        ptr = nullptr;
}

void *CUDAThreadState::host_ptr(void *ptr) {
    return ptr;
}

uint64_t CUDAThreadState::kernel_flags() const {
#if defined(DRJIT_ENABLE_OPTIX)
    if (uses_rt) {
        const OptixPipelineCompileOptions &pco = optix_pipeline->compile_options;
        return ((uint64_t) pco.numAttributeValues << 0) +     // 4 bit
               ((uint64_t) pco.numPayloadValues << 4) +       // 4 bit
               ((uint64_t) pco.usesMotionBlur << 8) +         // 1 bit
               ((uint64_t) pco.traversableGraphFlags << 9) +  // 16 bit
               ((uint64_t) pco.usesPrimitiveTypeFlags << 25); // 32 bit
    }
#endif
    return 0;
}

bool CUDAThreadState::compile(const char *buf, size_t buf_size,
                              const char *name, Kernel &kernel) {
    const uintptr_t log_size = 16384;
    char error_log[log_size], info_log[log_size];

    CUjit_option cujit_arg[] = {
        CU_JIT_OPTIMIZATION_LEVEL,
        CU_JIT_LOG_VERBOSE,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_GENERATE_LINE_INFO,
        CU_JIT_GENERATE_DEBUG_INFO
    };

    void *cujit_argv[] = {
        (void *) 4,
        (void *) 1,
        (void *) info_log,
        (void *) log_size,
        (void *) error_log,
        (void *) log_size,
        (void *) 0,
        (void *) 0
    };

    CUlinkState link_state;
    cuda_check(cuLinkCreate(sizeof(cujit_argv) / sizeof(void *), cujit_arg,
                            cujit_argv, &link_state));

    int rt = cuLinkAddData(link_state, CU_JIT_INPUT_PTX, (void *) buf,
                           buf_size, nullptr, 0, nullptr, nullptr);
    if (rt != CUDA_SUCCESS)
        jitc_fail("jit_cuda_compile(): compilation failed. Please see the PTX "
                  "assembly listing and error message below:\n\n%s\n\n%s",
                  buf, error_log);

    void *link_output = nullptr;
    size_t link_output_size = 0;
    cuda_check(cuLinkComplete(link_state, &link_output, &link_output_size));
    if (rt != CUDA_SUCCESS)
        jitc_fail("jit_cuda_compile(): compilation failed. Please see the PTX "
                  "assembly listing and error message below:\n\n%s\n\n%s",
                  buf, error_log);

    jitc_trace("Detailed linker output:\n%s", info_log);

    kernel.data = malloc_check(link_output_size);
    kernel.size = (uint32_t) link_output_size;
    memcpy(kernel.data, link_output, link_output_size);

    // Destroy the linker invocation
    cuda_check(cuLinkDestroy(link_state));

    return false;
}

void CUDAThreadState::load(Kernel &kernel, void *name) {
    scoped_set_context guard(context);

    CUresult ret = (CUresult) 0;
    /* Unlock while synchronizing */ {
        unlock_guard guard(state.lock);
        ret = cuModuleLoadData(&kernel.cuda.mod, kernel.data);
    }
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        jitc_flush_malloc_cache(true);
        /* Unlock while synchronizing */ {
            unlock_guard guard(state.lock);
            ret = cuModuleLoadData(&kernel.cuda.mod, kernel.data);
        }
    }
    cuda_check(ret);

    // Locate the kernel entry point
    cuda_check(cuModuleGetFunction(&kernel.cuda.func, kernel.cuda.mod, name));

    // Determine a suitable thread count to maximize occupancy
    int unused, block_size;
    cuda_check(cuOccupancyMaxPotentialBlockSize(
        &unused, &block_size, kernel.cuda.func, nullptr, 0, 0));
    kernel.cuda.block_size = (uint32_t) block_size;

    // DrJit doesn't use shared memory at all, prefer to have more L1 cache.
    cuda_check(cuFuncSetAttribute(
        kernel.cuda.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
    cuda_check(cuFuncSetAttribute(
        kernel.cuda.func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        CU_SHAREDMEM_CARVEOUT_MAX_L1));

    free(kernel.data);
    kernel.data = nullptr;
}

void CUDAThreadState::enqueue_memcpy(void *dst, const void *src, size_t size) {
     scoped_set_context guard(context);
     cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, size, stream));
}

void CUDAThreadState::enqueue_callback(void (*fn)(void *), void *payload) {
    scoped_set_context guard(context);
    cuda_check(cuLaunchHostFunc(stream, fn, payload));
}

void CUDAThreadState::sync() {
    scoped_set_context guard(context);
    cuda_check(cuStreamSynchronize(stream));
}

