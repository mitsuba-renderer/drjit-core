#include "cuda_ts.h"
#include "util.h"
#include "var.h"
#include "log.h"
#include "optix.h"
#include "eval.h"
#include "util.h"
#include "optix_api.h"

static uint8_t *kernel_params_global = nullptr;

static void submit_gpu(KernelType type, KernelRecordingMode recording_mode,
                       CUfunction kernel, uint32_t block_count_x,
                       uint32_t thread_count, uint32_t shared_mem_bytes,
                       CUstream stream, void **args, void **extra,
                       uint32_t width, uint32_t block_count_y = 1) {

    KernelHistoryEntry entry = {};

    uint32_t flags = jit_flags();

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        cuda_check(cuEventCreate((CUevent *) &entry.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &entry.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) entry.event_start, stream));
    }

    cuda_check(cuLaunchKernel(kernel, block_count_x, block_count_y, 1,
                              thread_count, 1, 1, shared_mem_bytes,
                              stream, args, extra));

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        cuda_check(cuStreamSynchronize(stream));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        entry.backend = JitBackend::CUDA;
        entry.type = type;
        entry.recording_mode = recording_mode;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        cuda_check(cuEventRecord((CUevent) entry.event_end, stream));

        state.kernel_history.append(entry);
    }
}

Task *CUDAThreadState::launch(
    Kernel kernel, KernelKey * /*key*/, XXH128_hash_t /*hash*/, uint32_t size,
    std::vector<void *> *kernel_params,
    const std::vector<uint32_t> * /*kernel_param_ids*/,
    KernelHistoryEntry *kernel_history_entry, uint32_t /*operation_count*/) {
    if (kernel_history_entry) {
        auto &e = *kernel_history_entry;
        cuda_check(cuEventCreate((CUevent *) &e.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &e.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) e.event_start, this->stream));
    }

    uint32_t kernel_param_count = (uint32_t) kernel_params->size();

    // Pass parameters through global memory if too large or using OptiX
    if (uses_optix || kernel_param_count > jitc_cuda_arg_limit) {
        size_t param_size = kernel_param_count * sizeof(void *);
        uint8_t *tmp =
            (uint8_t *) jitc_malloc(AllocType::HostPinned, param_size);
        kernel_params_global =
            (uint8_t *) jitc_malloc(AllocType::Device, param_size);
        memcpy(tmp, kernel_params->data(), param_size);
        jitc_memcpy_async(backend, kernel_params_global, tmp, param_size);
        jitc_free(tmp);
        kernel_params->clear();
        kernel_params->push_back(kernel_params_global);
    }

#if defined(DRJIT_ENABLE_OPTIX)
    if (unlikely(uses_optix))
        jitc_optix_launch(this, kernel, size, kernel_params_global,
                          kernel_param_count);
#else
    (void) kernel_param_count;
    (void) kernel_params_global;
#endif

    if (!uses_optix) {
        size_t buffer_size = kernel_params->size() * sizeof(void *);

        void *config[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER,
            kernel_params->data(),
            CU_LAUNCH_PARAM_BUFFER_SIZE,
            &buffer_size,
            CU_LAUNCH_PARAM_END
        };

        uint32_t block_count, thread_count;
        const Device &device_ = state.devices[this->device];
        device_.get_launch_config(&block_count, &thread_count, size,
                                 (uint32_t) kernel.cuda.block_size);

        cuda_check(cuLaunchKernel(kernel.cuda.func, block_count, 1, 1,
                                  thread_count, 1, 1, 0, stream,
                                  nullptr, config));
        jitc_trace("jit_run(): launching %u thread%s in %u block%s ..",
                   thread_count, thread_count == 1 ? "" : "s", block_count,
                   block_count == 1 ? "" : "s");
    }

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        cuda_check(cuStreamSynchronize(stream));


    // Cleanup global kernel parameters
    jitc_free(kernel_params_global);
    kernel_params_global = nullptr;

    if (kernel_history_entry){
        cuda_check(cuEventRecord((CUevent) kernel_history_entry->event_end,
                                 this->stream));
        state.kernel_history.append(*kernel_history_entry);
    }

    return nullptr;
}

/// Fill a device memory region with constants of a given type
void CUDAThreadState::memset_async(void *ptr, uint32_t size_, uint32_t isize,
                                   const void *src) {

    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
              (uintptr_t) ptr, isize, size_);

    if (size_ == 0)
        return;

    size_t size = size_;

    // Try to convert into ordinary memset if possible
    uint64_t zero = 0;
    if (memcmp(src, &zero, isize) == 0) {
        size *= isize;
        isize = 1;
    }

    // CUDA Specific
    scoped_set_context guard(context);
    switch (isize) {
        case 1:
            cuda_check(cuMemsetD8Async((CUdeviceptr) ptr,
                                       ((uint8_t *) src)[0], size,
                                       stream));
            break;

        case 2:
            cuda_check(cuMemsetD16Async((CUdeviceptr) ptr,
                                        ((uint16_t *) src)[0], size,
                                        stream));
            break;

        case 4:
            cuda_check(cuMemsetD32Async((CUdeviceptr) ptr,
                                        ((uint32_t *) src)[0], size,
                                        stream));
            break;

        case 8: {
                const Device &dev = state.devices[device];
                uint32_t block_count, thread_count;
                dev.get_launch_config(&block_count, &thread_count, size_);
                void *args[] = { &ptr, &size_, (void *) src };
                CUfunction kernel = jitc_cuda_fill_64[dev.id];
                submit_gpu(KernelType::Other, this->recording_mode, kernel,
                           block_count, thread_count, 0, stream, args, nullptr,
                           size_);
            }
            break;
    }
}

static VarType make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

void CUDAThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                   uint32_t block_size, const void *in,
                                   void *out) {
    // To understand the logic below, refer to the documentation of the
    // associated kernel in 'resources/block_reduce.cuh'

    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise(
            "jit_block_prefix_reduce(): invalid block size (size=%u, block_size=%u)!",
            size, block_size);
    }

    uint32_t tsize = type_size[(int) vt];
    if (block_size == 1) {
        memcpy_async(out, in, size * tsize);
        return;
    }

    VarType vts = vt;
    // Signed sum/product/and/or reductions can use the unsigned kernel.
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And) {
        vt = make_int_type_unsigned(vt);

        // Float16 add/mul reductions use higher-precision intermediate Float32
        // calculation steps, which requires extra shared memory.
        if (vt == VarType::Float16)
            vts = VarType::Float32;
    }

    // How many blocks are there to reduce?
    // Note: 'size' is not necessary divisible by 'block_size'.
    uint32_t block_count = ceil_div(size, block_size);

    // We have precompiled kernels for reductions from 2..1024 values.
    // Round up to the next power of two and select one of them.
    uint32_t chunk_size = round_pow2(block_size);

    // Launch configuration
    uint32_t thread_count, grid_dim_x, grid_dim_y, chunk_count;
    uint32_t chunks_per_block, chunks_per_thread_block;
    bool x_is_block_id = true;

    // Potentially load multiple values at once
    uint32_t vector_width = 1;

    if (chunk_size < 1024) {
        // Small reduction with 1 chunk/output block and a 1D launch grid. A
        // single thread block potentially processes multiple chunks. Be careful
        // changing the numbers below, the kernel expects this configuration.

        // Minimum number of threads needed to finish the work
        uint32_t min_threads = ceil_div(block_count * chunk_size, 32) * 32;

        // In general, schedule >= 4 warps/block to improve occupancy. It's
        // okay to use fewer warps if the input data size is tiny.
        thread_count = std::min(std::max(chunk_size, 128u), min_threads);

        chunk_count = block_count;
        chunks_per_block = 1;
        chunks_per_thread_block = thread_count / chunk_size;

        grid_dim_x = ceil_div(block_count, chunks_per_thread_block);
        grid_dim_y = 1;
    } else  {
        // Big reduction with 1 chunk == 1 CUDA thread block == 1024 threads.
        // There are potentially multiple chunks per output block.

        // Can we use the vectorized reduction?
        if ((block_size * tsize) % 16 == 0 && (size * tsize) % 16 == 0 &&
            ((uintptr_t) in) % 16 == 0)
            vector_width = 16 / tsize;

        chunk_size = 1024;
        thread_count = chunk_size / vector_width;

        chunks_per_block = ceil_div(block_size, chunk_size);
        chunks_per_thread_block = 1;

        grid_dim_x = block_count;
        grid_dim_y = chunks_per_block;
        chunk_count = block_count * chunks_per_block;

        // CUDA cannot launch grids with a Y block count > 65K..
        if (grid_dim_y > grid_dim_x) {
            std::swap(grid_dim_x, grid_dim_y);
            x_is_block_id = false;
        }
    }

    // The first warp reduction reduces data per chunk to this many entries
    uint32_t after_stage_1 = chunk_size / (std::min(chunk_size, 32u) * vector_width);

    // Compute shared memory requirement
    uint32_t smem_per_chunk = (after_stage_1 == 1 ? 0 : after_stage_1) * type_size[(int) vts],
             smem_bytes = smem_per_chunk * chunks_per_thread_block;

    jitc_log(Debug,
             "jit_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u, "
             "chunk_size=%u, chunks_per_block=%u, vector_width=%u): launching "
             "a %u x %u grid with %u threads and %u bytes of shared memory per "
             "thread block.",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, block_count, chunk_size,
             chunks_per_block, vector_width, grid_dim_x, grid_dim_y,
             thread_count, smem_bytes);

    const Device &dev = state.devices[device];

    CUfunction func = nullptr;
    if (vector_width != 1) {
        func = jitc_cuda_block_reduce_vec[(int) op][(int) vt][dev.id];
    } else {
        int kernel_id = log2i_ceil(chunk_size) - 1;
        func = jitc_cuda_block_reduce[(int) op][(int) vt][kernel_id][dev.id];
    }

    if (!func)
        jitc_raise("jit_block_reduce(): no existing kernel for type=%s, op=%s, vector_width=%u!",
                  type_name[(int) vt], red_name[(int) op], vector_width);

    struct {
        const void *in;
        void *out;
        uint32_t size;
        uint32_t block_size;
        uint32_t chunks_per_block;
        uint32_t chunk_count;
        uint8_t x_is_block_id;
    } params;

    params.in = in;
    params.size = size / vector_width;
    params.block_size = block_size / vector_width;
    params.chunks_per_block = chunks_per_block;
    params.chunk_count = chunk_count;
    params.x_is_block_id = x_is_block_id;

    if (chunks_per_block == 1)
        params.out = out;
    else
        params.out = jitc_malloc(AllocType::Device, chunk_count * tsize);

    {
        scoped_set_context guard(context);
        void *args[] = { &params };
        submit_gpu(KernelType::Reduce, this->recording_mode, func, grid_dim_x,
                   thread_count, smem_bytes, stream, args, nullptr, size,
                   grid_dim_y);
    }

    if (chunks_per_block > 1) {
        // Recurse
        block_reduce(vt, op, chunk_count, chunks_per_block, params.out, out);
        jitc_free(params.out);
    }
}

void CUDAThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                 const void *ptr_2, uint32_t size, void *out) {
    const Device &dev = state.devices[device];

    CUfunction red_dot = jitc_cuda_reduce_dot[(int) vt][dev.id];

    if (!red_dot)
        jitc_raise("jit_reduce_dot(): no existing kernel for type=%s!",
                   type_name[(int) vt]);

    uint32_t thread_count = 1024,
             tsize = type_size[(int) vt],
             shared_size = thread_count * tsize,
             block_count = (size + thread_count * 2 - 1) / (thread_count * 2);

    block_count = std::min(dev.sm_count * 4, block_count);

    jitc_log(Debug, "jit_reduce_dot(" DRJIT_PTR ", " DRJIT_PTR
             ", type=%s, size=%u, smem=%u, blocks=%u)",
             (uintptr_t) ptr_1, (uintptr_t) ptr_2, type_name[(int) vt],
             size, shared_size, block_count);

    scoped_set_context guard(context);
    if (block_count == 1) {
        void *args[] = { &ptr_1, &ptr_2, &size, &out };

        submit_gpu(KernelType::Reduce, this->recording_mode, red_dot, 1,
                   thread_count, shared_size, stream, args, nullptr, size);
    } else {
        // Reduce using multiple blocks
        void *temp = jitc_malloc(AllocType::Device,
                                 block_count * (size_t) tsize);

        // First reduction
        void *args_1[] = { &ptr_1, &ptr_2, &size, &temp };

        submit_gpu(KernelType::Reduce, this->recording_mode, red_dot,
                   block_count, thread_count, shared_size, stream, args_1,
                   nullptr, size);

        block_reduce(vt, ReduceOp::Add, block_count, block_count, temp, out);

        jitc_free(temp);
    }
}

void CUDAThreadState::block_prefix_reduce(VarType vt, ReduceOp op,
                                          uint32_t size, uint32_t block_size,
                                          bool exclusive, bool reverse,
                                          const void *in, void *out) {
    // To understand the logic below, refer to the documentation of two
    // related kernels in the files 'resources/block_reduce.cuh'
    // and 'resources/block_prefix_reduce.cuh'

    uint32_t tsize = type_size[(int) vt];
    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise(
            "jit_block_prefix_reduce(): invalid block size (size=%u, block_size=%u)!",
            size, block_size);
    } else if (block_size == 1) {
        uint64_t z = 0;
        if (exclusive)
            memset_async(out, size, tsize, &z);
        else if (in != out)
            memcpy_async(out, in, size * tsize);
        return;
    }

    VarType vts = vt;
    // Signed sum/product/and/or reductions can use the unsigned kernel.
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And) {
        vt = make_int_type_unsigned(vt);

        // Float16 add/mul reductions use higher-precision intermediate Float32
        // calculation steps, which requires extra shared memory.
        if (vt == VarType::Float16)
            vts = VarType::Float32;
    }

    // How many blocks are there to reduce?
    // Note: 'size' is not necessary divisible by 'block_size'.
    uint32_t block_count = ceil_div(size, block_size);

    // We have precompiled kernels for reductions from 2..1024 values.
    // Round up to the next power of two and select one of them.
    uint32_t chunk_size = round_pow2(block_size);

    // Launch configuration
    uint32_t thread_count, grid_dim_x, grid_dim_y, chunk_count;
    uint32_t chunks_per_block, chunks_per_thread_block;
    bool x_is_block_id = true;

    if (chunk_size < 1024) {
        // Small reduction with 1 chunk/output block and a 1D launch grid. A
        // single thread block potentially processes multiple chunks. Be careful
        // changing the numbers below, the kernel expects this configuration.

        // Minimum number of threads needed to finish the work
        uint32_t min_threads = ceil_div(block_count * chunk_size, 32) * 32;

        // In general, schedule >= 4 warps/block to improve occupancy. It's
        // okay to use fewer warps if the input data size is tiny.
        thread_count = std::min(std::max(chunk_size, 128u), min_threads);

        chunk_count = block_count;
        chunks_per_block = 1;
        chunks_per_thread_block = thread_count / chunk_size;

        grid_dim_x = ceil_div(block_count, chunks_per_thread_block);
        grid_dim_y = 1;
    } else  {
        // Big reduction with 1 chunk == 1 CUDA thread block == 1024 threads.
        // There are potentially multiple chunks per output block.

        chunk_size = thread_count = 1024;

        chunks_per_block = ceil_div(block_size, chunk_size);
        chunks_per_thread_block = 1;

        grid_dim_x = block_count;
        grid_dim_y = chunks_per_block;
        chunk_count = block_count * chunks_per_block;

        // CUDA cannot launch grids with a Y block count > 65K..
        if (grid_dim_y > grid_dim_x) {
            std::swap(grid_dim_x, grid_dim_y);
            x_is_block_id = false;
        }
    }

    // Shared memory requirement of this kernel
    uint32_t smem_bytes = thread_count * type_size[(int) vts];

    jitc_log(Debug,
             "jit_block_prefix_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, exclusive=%i, "
             "reverse=%i, block_count=%u, chunk_size=%u, chunks_per_block=%u): "
             "launching a %u x %u grid with %u threads and %u bytes of shared "
             "memory per thread block.",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, exclusive, reverse,
             block_count, chunk_size, chunks_per_block, grid_dim_x, grid_dim_y,
             thread_count, smem_bytes);

    const Device &dev = state.devices[device];

    CUfunction func = nullptr;
    int kernel_id = log2i_ceil(chunk_size) - 1;
    func = jitc_cuda_block_prefix_reduce[(int) op][(int) vt][kernel_id][dev.id];

    if (!func)
        jitc_raise("jit_block_prefix_reduce(): no existing kernel for type=%s, op=%s!",
                  type_name[(int) vt], red_name[(int) op]);

    struct {
        const void *in;
        void *scratch;
        void *out;
        uint32_t size;
        uint32_t block_size;
        uint32_t chunks_per_block;
        bool x_is_block_id;
        bool exclusive;
        bool reverse;
    } params;

    params.in = in;
    params.out = out;
    params.size = size;
    params.block_size = block_size;
    params.chunks_per_block = chunks_per_block;
    params.x_is_block_id = x_is_block_id;
    params.exclusive = exclusive;
    params.reverse = reverse;

    if (chunks_per_block > 1) {
        uint32_t scratch_size = chunk_count * 2,
                 vsize = type_size[(int) vts];
        params.scratch = jitc_malloc(AllocType::Device, scratch_size * vsize);
        uint64_t z = 0;
        memset_async(params.scratch, scratch_size, vsize, &z);
    } else {
        params.scratch = nullptr;
    }

    {
        scoped_set_context guard(context);
        void *args[] = { &params };
        submit_gpu(KernelType::Reduce, this->recording_mode, func, grid_dim_x,
                   thread_count, smem_bytes, stream, args, nullptr, size,
                   grid_dim_y);
    }

    if (chunks_per_block > 1)
        jitc_free(params.scratch);
}

uint32_t CUDAThreadState::compress(const uint8_t *in, uint32_t size,
                                   uint32_t *out) {
    if (size == 0)
        return 0;

    const Device &dev = state.devices[device];
    scoped_set_context guard(context);

    uint32_t *count_out = (uint32_t *) jitc_malloc(
        AllocType::HostPinned, sizeof(uint32_t));

    if (size <= 4096) {
        // Kernel for small arrays
        uint32_t items_per_thread = 4,
                 thread_count     = round_pow2((size + items_per_thread - 1)
                                                / items_per_thread),
                 shared_size      = thread_count * 2 * sizeof(uint32_t),
                 trailer          = thread_count * items_per_thread - size;

        jitc_log(Debug,
                "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                ", size=%u, type=small, threads=%u, shared=%u)",
                (uintptr_t) in, (uintptr_t) out, size, thread_count,
                shared_size);

        if (trailer > 0)
            cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                       stream));

        void *args[] = { &in, &out, &size, &count_out };
        submit_gpu(KernelType::Other, this->recording_mode,
                   jitc_cuda_compress_small[dev.id], 1, thread_count,
                   shared_size, stream, args, nullptr, size);
    } else {
        // Kernel for large arrays
        uint32_t items_per_thread = 16,
                 thread_count     = 128,
                 items_per_block  = items_per_thread * thread_count,
                 block_count      = (size + items_per_block - 1) / items_per_block,
                 shared_size      = items_per_block * sizeof(uint32_t),
                 scratch_items    = block_count + 32,
                 trailer          = items_per_block * block_count - size;

        jitc_log(Debug,
                "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                "scratch=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_count,
                thread_count, shared_size, scratch_items * 4);

        uint64_t *scratch = (uint64_t *) jitc_malloc(
            AllocType::Device, scratch_items * sizeof(uint64_t));

        // Initialize scratch space and padding
        uint32_t block_count_init, thread_count_init;
        dev.get_launch_config(&block_count_init, &thread_count_init,
                                 scratch_items);

        void *args[] = { &scratch, &scratch_items };
        submit_gpu(KernelType::Other, this->recording_mode,
                   jitc_cuda_compress_large_init[dev.id], block_count_init,
                   thread_count_init, 0, stream, args, nullptr, scratch_items);

        if (trailer > 0)
            cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                       stream));

        scratch += 32; // move beyond padding area
        void *args_2[] = { &in, &out, &scratch, &count_out };
        submit_gpu(KernelType::Other, this->recording_mode,
                   jitc_cuda_compress_large[dev.id], block_count, thread_count,
                   shared_size, stream, args_2, nullptr, scratch_items);
        scratch -= 32;

        jitc_free(scratch);
    }
    jitc_sync_thread();
    uint32_t count_out_v = *count_out;
    jitc_free(count_out);
    return count_out_v;
}

static void cuda_transpose(ThreadState *ts, const uint32_t *in, uint32_t *out,
                           uint32_t rows, uint32_t cols) {
    const Device &device = state.devices[ts->device];

    uint16_t blocks_x = (uint16_t) ((cols + 15u) / 16u),
             blocks_y = (uint16_t) ((rows + 15u) / 16u);

    scoped_set_context guard(ts->context);
    jitc_log(Debug,
            "jit_transpose(" DRJIT_PTR " -> " DRJIT_PTR
            ", rows=%u, cols=%u, blocks=%ux%u)",
            (uintptr_t) in, (uintptr_t) out, rows, cols, blocks_x, blocks_y);

    void *args[] = { &in, &out, &rows, &cols };

    cuda_check(cuLaunchKernel(
        jitc_cuda_transpose[device.id], blocks_x, blocks_y, 1, 16, 16, 1,
        16 * 17 * sizeof(uint32_t), ts->stream, args, nullptr));
}

uint32_t CUDAThreadState::mkperm(const uint32_t *ptr, uint32_t size,
                                 uint32_t bucket_count, uint32_t *perm,
                                 uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jitc_fail("jit_mkperm(): bucket_count cannot be zero!");

    scoped_set_context guard(context);
    const Device &dev = state.devices[device];

    // Don't use more than 1 block/SM due to shared memory requirement
    const uint32_t warp_size = 32;
    uint32_t block_count, thread_count;
    dev.get_launch_config(&block_count, &thread_count, size, 1024, 1);

    // Always launch full warps (the kernel impl. assumes that this is the case)
    uint32_t warp_count = (thread_count + warp_size - 1) / warp_size;
    thread_count = warp_count * warp_size;

    uint32_t bucket_size_1   = bucket_count * sizeof(uint32_t),
             bucket_size_all = bucket_size_1 * block_count;

    /* If there is a sufficient amount of shared memory, atomically accumulate into a
       shared memory buffer. Otherwise, use global memory, which is much slower. */
    uint32_t shared_size = 0;
    const char *variant = nullptr;
    CUfunction phase_1 = nullptr, phase_4 = nullptr;
    bool initialize_buckets = false;

    if (bucket_size_1 * warp_count <= dev.shared_memory_bytes) {
        /* "Tiny" variant, which uses shared memory atomics to produce a stable
           permutation. Handles up to 512 buckets with 64KiB of shared memory. */

        phase_1 = jitc_cuda_mkperm_phase_1_tiny[dev.id];
        phase_4 = jitc_cuda_mkperm_phase_4_tiny[dev.id];
        shared_size = bucket_size_1 * warp_count;
        bucket_size_all *= warp_count;
        variant = "tiny";
    } else if (bucket_size_1 <= dev.shared_memory_bytes) {
        /* "Small" variant, which uses shared memory atomics and handles up to
           16K buckets with 64KiB of shared memory. The permutation can be
           somewhat unstable due to scheduling variations when performing atomic
           operations (although some effort is made to keep it stable within
           each group of 32 elements by performing an intra-warp reduction.) */

        phase_1 = jitc_cuda_mkperm_phase_1_small[dev.id];
        phase_4 = jitc_cuda_mkperm_phase_4_small[dev.id];
        shared_size = bucket_size_1;
        variant = "small";
    } else {
        /* "Large" variant, which uses global memory atomics and handles
           arbitrarily many elements (though this is somewhat slower than the
           previous two shared memory variants). The permutation can be somewhat
           unstable due to scheduling variations when performing atomic
           operations (although some effort is made to keep it stable within
           each group of 32 elements by performing an intra-warp reduction.)
           Buckets must be zero-initialized explicitly. */

        phase_1 = jitc_cuda_mkperm_phase_1_large[dev.id];
        phase_4 = jitc_cuda_mkperm_phase_4_large[dev.id];
        variant = "large";
        initialize_buckets = true;
    }

    bool needs_transpose = bucket_size_1 != bucket_size_all;
    uint32_t *buckets_1, *buckets_2, *counter = nullptr;
    buckets_1 = buckets_2 =
        (uint32_t *) jitc_malloc(AllocType::Device, bucket_size_all);

    // Scratch space for matrix transpose operation
    if (needs_transpose)
        buckets_2 = (uint32_t *) jitc_malloc(AllocType::Device, bucket_size_all);

    if (offsets) {
        counter = (uint32_t *) jitc_malloc(AllocType::Device, sizeof(uint32_t)),
        cuda_check(cuMemsetD8Async((CUdeviceptr) counter, 0, sizeof(uint32_t),
                                   stream));
    }

    if (initialize_buckets)
        cuda_check(cuMemsetD8Async((CUdeviceptr) buckets_1, 0,
                                   bucket_size_all, stream));

    /* Determine the amount of work to be done per block, and ensure that it is
       divisible by the warp size (the kernel implementation assumes this.) */
    uint32_t size_per_block = (size + block_count - 1) / block_count;
    size_per_block = (size_per_block + warp_size - 1) / warp_size * warp_size;

    jitc_log(Debug,
            "jit_mkperm(" DRJIT_PTR
            ", size=%u, bucket_count=%u, block_count=%u, thread_count=%u, "
            "size_per_block=%u, variant=%s, shared_size=%u)",
            (uintptr_t) ptr, size, bucket_count, block_count, thread_count,
            size_per_block, variant, shared_size);

    // Phase 1: Count the number of occurrences per block
    void *args_1[] = { &ptr, &buckets_1, &size, &size_per_block,
                       &bucket_count };

    submit_gpu(KernelType::CallReduce, this->recording_mode, phase_1,
               block_count, thread_count, shared_size, stream, args_1, nullptr,
               size);

    // Phase 2: exclusive prefix sum over transposed buckets
    if (needs_transpose)
        cuda_transpose(this, buckets_1, buckets_2,
                       bucket_size_all / bucket_size_1, bucket_count);

    uint32_t psum_count = bucket_size_all / sizeof(uint32_t);
    block_prefix_reduce(VarType::UInt32, ReduceOp::Add, psum_count, psum_count,
                        true, false, buckets_2, buckets_2);

    if (needs_transpose)
        cuda_transpose(this, buckets_2, buckets_1, bucket_count,
                       bucket_size_all / bucket_size_1);

    // Phase 3: collect non-empty buckets (optional)
    if (likely(offsets)) {
        uint32_t block_count_3, thread_count_3;
        dev.get_launch_config(&block_count_3, &thread_count_3,
                                 bucket_count * block_count);

        // Round up to a multiple of the thread count
        uint32_t bucket_count_rounded =
            (bucket_count + thread_count_3 - 1) / thread_count_3 * thread_count_3;

        void *args_3[] = { &buckets_1, &bucket_count, &bucket_count_rounded,
                           &size,      &counter,      &offsets };

        submit_gpu(KernelType::CallReduce, this->recording_mode,
                   jitc_cuda_mkperm_phase_3[dev.id], block_count_3,
                   thread_count_3, sizeof(uint32_t) * thread_count_3, stream,
                   args_3, nullptr, size);

        cuda_check(cuMemcpyAsync((CUdeviceptr) (offsets + 4 * size_t(bucket_count)),
                                 (CUdeviceptr) counter, sizeof(uint32_t),
                                 stream));

        cuda_check(cuEventRecord(this->event, stream));
    }

    // Phase 4: write out permutation based on bucket counts
    void *args_4[] = { &ptr, &buckets_1, &perm, &size, &size_per_block,
                       &bucket_count };

    submit_gpu(KernelType::CallReduce, this->recording_mode, phase_4,
               block_count, thread_count, shared_size, stream, args_4, nullptr,
               size);

    if (likely(offsets)) {
        unlock_guard guard_2(state.lock);
        cuda_check(cuEventSynchronize(this->event));
    }

    jitc_free(buckets_1);
    if (needs_transpose)
        jitc_free(buckets_2);
    jitc_free(counter);

    return offsets ? offsets[4 * bucket_count] : 0u;
}

void CUDAThreadState::memcpy(void *dst, const void *src, size_t size) {
    scoped_set_context guard_2(context);
    cuda_check(cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, size));
}

void CUDAThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    scoped_set_context guard(context);
    cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, size,
                             stream));
}

void CUDAThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    VarType type;
    switch (size) {
        case 1: type = VarType::UInt8; break;
        case 2: type = VarType::UInt16; break;
        case 4: type = VarType::UInt32; break;
        case 8: type = VarType::UInt64; break;
        default:
            jitc_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");
    }

    scoped_set_context guard(context);
    const Device &dev = state.devices[device];
    CUfunction func = jitc_cuda_poke[(int) type][dev.id];
    void *args[] = { &dst, (void *) src };
    submit_gpu(KernelType::Other, this->recording_mode, func, 1, 1, 0, stream,
               args, nullptr, 1);
}

void CUDAThreadState::aggregate(void *dst_, AggregationEntry *agg,
                                uint32_t size) {
    scoped_set_context guard(context);
    const Device &dev = state.devices[device];
    CUfunction func = jitc_cuda_aggregate[dev.id];
    void *args[] = { &dst_, &agg, &size };

    uint32_t block_count, thread_count;
    dev.get_launch_config(&block_count, &thread_count, size);

    jitc_log(InfoSym,
             "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
             ", size=%u, blocks=%u, threads=%u)",
             (uintptr_t) agg, (uintptr_t) dst_, size, block_count,
             thread_count);

    submit_gpu(KernelType::Other, this->recording_mode, func, block_count,
               thread_count, 0, stream, args, nullptr, 1);

    jitc_free(agg);
}

void CUDAThreadState::enqueue_host_func(void (*callback)(void *),
                                        void *payload) {

    scoped_set_context guard(context);
    cuda_check(cuLaunchHostFunc(stream, callback, payload));
}

void CUDAThreadState::coop_vec_pack(uint32_t count, const void *in_,
                                    const MatrixDescr *in_d, void *out_,
                                    const MatrixDescr *out_d) {
#if defined(DRJIT_ENABLE_OPTIX)
    scoped_set_context guard(context);
    OptixDeviceContext ctx = jitc_optix_context();
    const uint8_t *in = (const uint8_t *) in_;
    uint8_t *out = (uint8_t *) out_;

    std::vector<OptixCoopVecMatrixDescription> in_o, out_o;
    in_o.reserve(count);
    out_o.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        const MatrixDescr &id = in_d[i],
                          &od = out_d[i];

        uint32_t tsize = type_size[(int) id.dtype];

        if (id.cols == 1) {
            cuda_check(cuMemcpyAsync((CUdeviceptr) (out + od.offset * tsize),
                                     (CUdeviceptr) (in + id.offset * tsize),
                                     id.size * tsize,
                                     stream));
        } else {
            uint32_t type_id = jitc_optix_coop_vec_type_id(id.dtype);

            OptixCoopVecMatrixDescription io;
            io.N = id.rows;
            io.K = id.cols;
            io.offsetInBytes = id.offset * tsize;
            io.elementType = type_id;
            io.layout = jitc_optix_coop_vec_layout_id(id.layout);
            io.rowColumnStrideInBytes = id.stride * tsize;
            io.sizeInBytes = id.size * tsize;
            in_o.push_back(io);

            OptixCoopVecMatrixDescription oo;
            oo.N = od.rows;
            oo.K = od.cols;
            oo.offsetInBytes = od.offset * tsize;
            oo.elementType = type_id;
            oo.layout = jitc_optix_coop_vec_layout_id(od.layout);
            oo.rowColumnStrideInBytes = od.stride * tsize;
            oo.sizeInBytes = od.size * tsize;
            out_o.push_back(oo);
        }
    }

    OptixNetworkDescription in_net, out_net;
    in_net.layers = in_o.data();
    in_net.numLayers = (unsigned int) in_o.size();
    out_net.layers = out_o.data();
    out_net.numLayers = (unsigned int) out_o.size();

    if (!optixCoopVecMatrixConvert)
        jitc_raise("jit_coop_vec_pack(): Cooperative vectors are not "
                   "supported by your NVIDIA GPU driver. Please install "
                   "driver version 570 or newer.");

    if (in_net.numLayers)
        jitc_optix_check(optixCoopVecMatrixConvert(
            ctx, stream, 1, &in_net, (CUdeviceptr) in_, 0, &out_net,
            (CUdeviceptr) out_, 0));
#else
    (void) count; (void) in_; (void) in_d; (void) out_; (void) out_d;
    jitc_raise("CUDAThreadState::coop_vec_pack(): requires OptiX support!");
#endif
}
