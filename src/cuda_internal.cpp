#include "cuda_internal.h"
#include "var.h"
#include "log.h"

const static char *reduction_name[(int) ReduceOp::Count] = { "none", "sum", "mul",
                                                      "min", "max", "and", "or" };

static void jitc_submit_gpu(KernelType type, CUfunction kernel, uint32_t block_count,
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

/// Fill a device memory region with constants of a given type
void CUDAThreadState::jitc_memset_async(void *ptr, uint32_t size_,
                                        uint32_t isize, const void *src){
    
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("CUDAThreadState::jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jitc_trace("CUDAThreadState::jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
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
    scoped_set_context guard(this->context);
    switch (isize) {
        case 1:
            cuda_check(cuMemsetD8Async((CUdeviceptr) ptr,
                                       ((uint8_t *) src)[0], size,
                                       this->stream));
            break;

        case 2:
            cuda_check(cuMemsetD16Async((CUdeviceptr) ptr,
                                        ((uint16_t *) src)[0], size,
                                        this->stream));
            break;

        case 4:
            cuda_check(cuMemsetD32Async((CUdeviceptr) ptr,
                                        ((uint32_t *) src)[0], size,
                                        this->stream));
            break;

        case 8: {
                const Device &device = state.devices[this->device];
                uint32_t block_count, thread_count;
                device.get_launch_config(&block_count, &thread_count, size_);
                void *args[] = { &ptr, &size_, (void *) src };
                CUfunction kernel = jitc_cuda_fill_64[device.id];
                jitc_submit_gpu(KernelType::Other, kernel, block_count,
                                thread_count, 0, this->stream, args, nullptr,
                                size_);
            }
            break;
    }
}

void CUDAThreadState::jitc_reduce(VarType type, ReduceOp op, const void *ptr,
                                  uint32_t size, void *out) {
    
    jitc_log(Debug, "jit_reduce(" DRJIT_PTR ", type=%s, op=%s, size=%u)",
            (uintptr_t) ptr, type_name[(int) type],
            reduction_name[(int) op], size);

    uint32_t tsize = type_size[(int) type];
    
    // CUDA specific
    scoped_set_context guard(this->context);
    const Device &device = state.devices[this->device];
    CUfunction func = jitc_cuda_reductions[(int) op][(int) type][device.id];
    if (!func)
        jitc_raise("jit_reduce(): no existing kernel for type=%s, op=%s!",
                  type_name[(int) type], reduction_name[(int) op]);

    uint32_t thread_count = 1024,
             shared_size = thread_count * tsize,
             block_count;

    device.get_launch_config(&block_count, nullptr, size, thread_count);

    if (size <= 1024) {
        // This is a small array, do everything in just one reduction.
        void *args[] = { &ptr, &size, &out };

        jitc_submit_gpu(KernelType::Reduce, func, 1, thread_count,
                        shared_size, this->stream, args, nullptr, size);
    } else {
        void *temp = jitc_malloc(AllocType::Device, size_t(block_count) * tsize);

        // First reduction
        void *args_1[] = { &ptr, &size, &temp };

        jitc_submit_gpu(KernelType::Reduce, func, block_count, thread_count,
                        shared_size, this->stream, args_1, nullptr, size);

        // Second reduction
        void *args_2[] = { &temp, &block_count, &out };

        jitc_submit_gpu(KernelType::Reduce, func, 1, thread_count,
                        shared_size, this->stream, args_2, nullptr, size);

        jitc_free(temp);
    }
}

bool CUDAThreadState::jitc_all(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_all(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = true;
        this->jitc_memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    // CUDA specific
    bool result;
    
    uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
    this->jitc_reduce(VarType::UInt32, ReduceOp::And, values, reduced_size, out);
    jitc_sync_thread();
    result = (out[0] & out[1] & out[2] & out[3]) != 0;
    jitc_free(out);

    return result;
}

bool CUDAThreadState::jitc_any(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */
    
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_any(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = false;
        this->jitc_memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    // CUDA specific
    bool result;
    
    uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
    this->jitc_reduce(VarType::UInt32, ReduceOp::Or, values,
                      reduced_size, out);
    jitc_sync_thread();
    result = (out[0] | out[1] | out[2] | out[3]) != 0;
    jitc_free(out);

    return result;
    
}

void CUDAThreadState::jitc_prefix_sum(VarType vt, bool exclusive,
                                      const void *in, uint32_t size,
                                      void *out) {
    if (size == 0)
        return;
    if (vt == VarType::Int32)
        vt = VarType::UInt32;

    const uint32_t isize = type_size[(int) vt];

    // CUDA specific
    const Device &device = state.devices[this->device];
    scoped_set_context guard(this->context);

    if (size == 1) {
        if (exclusive) {
            cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, isize, this->stream));
        } else {
            if (in != out)
                cuda_check(cuMemcpyAsync((CUdeviceptr) out,
                                         (CUdeviceptr) in, isize,
                                         this->stream));
        }
    } else if ((isize == 4 && size <= 4096) || (isize == 8 && size < 2048)) {
        // Kernel for small arrays
        uint32_t items_per_thread = isize == 8 ? 2 : 4,
                 thread_count     = round_pow2((size + items_per_thread - 1)
                                                / items_per_thread),
                 shared_size      = thread_count * 2 * isize;

        jitc_log(Debug,
                 "jit_prefix_sum(" DRJIT_PTR " -> " DRJIT_PTR
                 ", type=%s, exclusive=%i, size=%u, type=small, threads=%u, shared=%u)",
                 (uintptr_t) in, (uintptr_t) out, type_name[(int) vt], exclusive, size,
                 thread_count, shared_size);

        CUfunction kernel =
            (exclusive ? jitc_cuda_prefix_sum_exc_small
                       : jitc_cuda_prefix_sum_inc_small)[(int) vt][device.id];

        if (!kernel)
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);

        void *args[] = { &in, &out, &size };
        jitc_submit_gpu(
            KernelType::Other, kernel, 1,
            thread_count, shared_size, this->stream, args, nullptr, size);
    } else {
        // Kernel for large arrays
        uint32_t items_per_thread = isize == 8 ? 8 : 16,
                 thread_count     = 128,
                 items_per_block  = items_per_thread * thread_count,
                 block_count      = (size + items_per_block - 1) / items_per_block,
                 shared_size      = items_per_block * isize,
                 scratch_items    = block_count + 32;

        jitc_log(Debug,
                 "jit_prefix_sum(" DRJIT_PTR " -> " DRJIT_PTR
                 ", type=%s, exclusive=%i, size=%u, type=large, blocks=%u, threads=%u, "
                 "shared=%u, scratch=%zu)",
                 (uintptr_t) in, (uintptr_t) out, type_name[(int) vt], exclusive, size,
                 block_count, thread_count, shared_size, scratch_items * sizeof(uint64_t));

        CUfunction kernel =
            (exclusive ? jitc_cuda_prefix_sum_exc_large
                       : jitc_cuda_prefix_sum_inc_large)[(int) vt][device.id];

        if (!kernel)
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);

        uint64_t *scratch = (uint64_t *) jitc_malloc(
            AllocType::Device, scratch_items * sizeof(uint64_t));

        /// Initialize scratch space and padding
        uint32_t block_count_init, thread_count_init;
        device.get_launch_config(&block_count_init, &thread_count_init,
                                 scratch_items);

        void *args[] = { &scratch, &scratch_items };
        jitc_submit_gpu(KernelType::Other,
                        jitc_cuda_prefix_sum_large_init[device.id],
                        block_count_init, thread_count_init, 0, this->stream,
                        args, nullptr, scratch_items);

        scratch += 32; // move beyond padding area
        void *args_2[] = { &in, &out, &size, &scratch };
        jitc_submit_gpu(KernelType::Other, kernel, block_count,
                        thread_count, shared_size, this->stream, args_2,
                        nullptr, scratch_items);
        scratch -= 32;

        jitc_free(scratch);
    }
}

uint32_t CUDAThreadState::jitc_compress(const uint8_t *in, uint32_t size,
                                        uint32_t *out) {
    if (size == 0)
        return 0;

    // CUDA specific
    const Device &device = state.devices[this->device];
    scoped_set_context guard(this->context);

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
                                       this->stream));

        void *args[] = { &in, &out, &size, &count_out };
        jitc_submit_gpu(
            KernelType::Other, jitc_cuda_compress_small[device.id], 1,
            thread_count, shared_size, this->stream, args, nullptr, size);
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
        device.get_launch_config(&block_count_init, &thread_count_init,
                                 scratch_items);

        void *args[] = { &scratch, &scratch_items };
        jitc_submit_gpu(KernelType::Other,
                        jitc_cuda_prefix_sum_large_init[device.id],
                        block_count_init, thread_count_init, 0, this->stream,
                        args, nullptr, scratch_items);

        if (trailer > 0)
            cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                       this->stream));

        scratch += 32; // move beyond padding area
        void *args_2[] = { &in, &out, &scratch, &count_out };
        jitc_submit_gpu(KernelType::Other,
                        jitc_cuda_compress_large[device.id], block_count,
                        thread_count, shared_size, this->stream, args_2,
                        nullptr, scratch_items);
        scratch -= 32;

        jitc_free(scratch);
    }
    jitc_sync_thread();
    uint32_t count_out_v = *count_out;
    jitc_free(count_out);
    return count_out_v;
}

void CUDAThreadState::jitc_memcpy(void *dst, const void *src, size_t size) {
    scoped_set_context guard_2(this->context);
    cuda_check(cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, size));
}
