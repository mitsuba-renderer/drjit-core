#include "internal.h"
#include "util.h"
#include "var.h"
#include "log.h"

const char *reduction_name[(int) ReductionType::Count] = { "add", "mul", "min",
                                                           "max", "and", "or" };

/// Fill a device memory region with constants of a given type
void jit_fill(VarType type, void *ptr, size_t size, const void *src) {
    jit_trace("jit_fill(" ENOKI_PTR ", type=%s, size=%zu)", (uintptr_t) ptr,
              var_type_name[(int) type], size);

    if (size == 0)
        return;

    Stream *stream = active_stream;
    if (stream) {
        switch (var_type_size[(int) type]) {
            case 1:
                cuda_check(cuMemsetD8Async((CUdeviceptr) ptr, ((uint8_t *) src)[0], size, stream->handle));
                break;

            case 2:
                cuda_check(cuMemsetD16Async((CUdeviceptr) ptr, ((uint16_t *) src)[0], size, stream->handle));
                break;

            case 4:
                cuda_check(cuMemsetD32Async((CUdeviceptr) ptr, ((uint32_t *) src)[0], size, stream->handle));
                break;

            case 8: {
                    const Device &device = state.devices[stream->device];
                    int num_threads, num_blocks;
                    if (size < 1024) {
                        num_threads = size;
                        num_blocks = 1;
                    } else {
                        num_threads = 1024;
                        num_blocks = std::min(
                            (uint32_t) size / num_threads,
                            (uint32_t) device.num_sm);
                    }
                    void *args[] = { &ptr, &size, (void *) src };
                    CUfunction kernel = jit_cuda_fill_64[device.id];
                    cuda_check(cuLaunchKernel(kernel, num_blocks, 1, 1,
                                              num_threads, 1, 1, 0,
                                              stream->handle, args, nullptr));
                }
                break;

            default:
                jit_raise("jit_fill(): unknown type!");
        }
    } else {
        unlock_guard guard(state.mutex);
        switch (var_type_size[(int) type]) {
            case 1: {
                    uint8_t value = ((uint8_t *) src)[0],
                            *p    = (uint8_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 2: {
                    uint16_t value = ((uint16_t *) src)[0],
                             *p    = (uint16_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 4: {
                    uint32_t value = ((uint32_t *) src)[0],
                             *p    = (uint32_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 8: {
                    uint64_t value = ((uint64_t *) src)[0],
                             *p    = (uint64_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            default:
                jit_raise("jit_fill(): unknown type!");
        }
    }
}

void jit_reduce(VarType type, ReductionType rtype, const void *ptr, size_t size,
                void *out) {
    jit_log(Debug, "jit_reduce(" ENOKI_PTR ", type=%s, rtype=%s, size=%zu)",
            (uintptr_t) ptr, var_type_name[(int) type],
            reduction_name[(int) rtype], size);

    size_t type_size = var_type_size[(int) type];
    Stream *stream = active_stream;

    if (stream) {
        const Device &device = state.devices[stream->device];
        CUfunction func = jit_cuda_reductions[(int) rtype][(int) type][device.id];
        if (!func)
            jit_raise("jit_reduce(): no existing kernel for type=%s, rtype=%s!",
                      var_type_name[(int) type], reduction_name[(int) rtype]);

        uint32_t num_blocks = device.num_sm * 4,
                 num_threads = 1024,
                 shared_size = num_threads * type_size;

        if (size > 1024) {
            // First reduction
            void *args_1[] = { &ptr, &size, &out };
            cuda_check(cuLaunchKernel(func, num_blocks, 1, 1, num_threads, 1, 1,
                                      shared_size, stream->handle, args_1,
                                      nullptr));

            // Second reduction
            size = num_blocks;
            void *args_2[] = { &out, &size, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, num_threads, 1, 1,
                                      shared_size, stream->handle, args_2,
                                      nullptr));
        } else {
            /// This is a small array, do everything in just one reduction.
            void *args[] = { &ptr, &size, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, num_threads, 1, 1,
                                      shared_size, stream->handle, args,
                                      nullptr));
        }
    }
}

/// Exclusive prefix sum
void jit_scan(const uint32_t *in, uint32_t *out, uint32_t size) {
    Stream *stream = active_stream;

    if (stream) {
        const Device &device = state.devices[stream->device];

        /// Exclusive prefix scan processes 4K elements / block, 4 per thread
        uint32_t block_count      = (size + 4096 - 1) / 4096,
                 thread_count     = std::min(round_pow2((size + 3u) / 4u), 1024u),
                 shared_size      = thread_count * 2 * sizeof(uint32_t);

        jit_log(Debug,
                "jit_scan(" ENOKI_PTR " -> " ENOKI_PTR
                ", size=%u, block_count=%u, thread_count=%u, shared_size=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_count,
                thread_count, shared_size);

        if (size == 0) {
            return;
        } else if (size == 1) {
            cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, sizeof(uint32_t),
                                       stream->handle));
        } else if (size <= 4096) {
            void *args[] = { &in, &out, &size };
            cuda_check(cuLaunchKernel(jit_cuda_scan_small[device.id], 1, 1, 1,
                                      thread_count, 1, 1, shared_size,
                                      stream->handle, args, nullptr));
        } else {
            uint32_t *block_sums = (uint32_t *) jit_malloc(
                AllocType::Device, block_count * sizeof(uint32_t));

            void *args[] = { &in, &out, &block_sums };
            cuda_check(cuLaunchKernel(jit_cuda_scan_large[device.id], block_count, 1, 1,
                                      thread_count, 1, 1, shared_size,
                                      stream->handle, args, nullptr));

            jit_scan(block_sums, block_sums, block_count);

            void *args_2[] = { &out, &block_sums };
            cuda_check(cuLaunchKernel(jit_cuda_scan_offset[device.id], block_count, 1, 1,
                                      thread_count, 1, 1, 0, stream->handle,
                                      args_2, nullptr));
            jit_free(block_sums);
        }
    } else {
        unlock_guard guard(state.mutex);
        uint32_t accum = 0;
        for (uint32_t i = 0; i < size; ++i) {
            uint32_t value = in[i];
            out[i] = accum;
            accum += value;
        }
    }
}

void jit_transpose(uint32_t *data, uint32_t rows, uint32_t cols) {
    Stream *stream = active_stream;

    if (stream) {
        const Device &device = state.devices[stream->device];
        void *args[] = { &data, &rows, &cols };
        cuda_check(cuLaunchKernel(jit_cuda_transpose[device.id],
                                  (cols + 31) / 32, (rows + 31) / 32, 1,
                                  32, 32, 1, 0, stream->handle,
                                  args, nullptr));
    } else {
        unlock_guard guard(state.mutex);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                uint32_t idx0 = c + r * cols,
                         idx1 = r + c * rows,
                         val0 = data[idx0],
                         val1 = data[idx1];

                data[idx0] = val1;
                data[idx1] = val0;
            }
        }
    }
}

void jit_mkperm(const uint32_t *values, uint32_t size, uint32_t bucket_count,
                uint32_t **perm_out, uint32_t **offsets_out) {
    Stream *stream = active_stream;
    const Device &device = state.devices[stream->device];

    uint32_t block_count      = device.num_sm,
             thread_count     = 1024,
             bucket_size_1    = sizeof(uint32_t) * bucket_count,
             bucket_size_all  = bucket_size_1 * block_count;

    /// Determine the amount of work to be done per SM
    uint32_t size_per_block = (size + block_count - 1) / block_count;
    size_per_block = (size_per_block + thread_count - 1) / thread_count * thread_count;

    /* If there is a sufficient amount of shared memory, atomically accumulate into a
       shared memory buffer. Otherwise, use global memory, which is much slower. */
    uint32_t shared_memory, shared_memory_small = sizeof(uint32_t) * bucket_count;
    CUfunction phase_1, phase_2;

    if (shared_memory_small <= (uint32_t) device.shared_memory_bytes) {
        phase_1 = jit_cuda_mkperm_phase_1_shared[device.id];
        phase_2 = jit_cuda_mkperm_phase_2_shared[device.id];
        shared_memory = shared_memory_small;
    } else {
        phase_1 = jit_cuda_mkperm_phase_1_global[device.id];
        phase_2 = jit_cuda_mkperm_phase_2_global[device.id];
        shared_memory = 0;
    }

    uint32_t *buckets = (uint32_t *) jit_malloc(AllocType::Device, bucket_size_all),
             *offsets = (uint32_t *) jit_malloc(AllocType::HostPinned, bucket_size_1 + sizeof(uint32_t)),
             *perm    = (uint32_t *) jit_malloc(AllocType::Device, size);

    offsets[bucket_count] = size;

    // Phase 1: Count the number of occurrences per block
    void *args_1[] = { &values, &buckets, &size, &size_per_block,
                       &bucket_count };

    cuda_check(cuLaunchKernel(phase_1, block_count, 1, 1, thread_count, 1, 1,
                              shared_memory, stream->handle, args_1, nullptr));

    // Phase 1.5: exclusive prefix sum over transposed buckets
    jit_transpose(buckets, bucket_count, block_count);
    jit_scan(buckets, buckets, bucket_count * block_count);
    cuda_check(cuMemcpyAsync((CUdeviceptr) offsets, (CUdeviceptr) buckets,
                             bucket_size_1, active_stream->handle));
    cuda_check(cuEventRecord(stream->event, stream->handle));
    jit_transpose(buckets, block_count, bucket_count);

    // Phase 2: write out permutation based on bucket counts
    void *args_2[] = { &values, &buckets, &perm, &size, &size_per_block,
                       &bucket_count };
    cuda_check(cuLaunchKernel(phase_2, block_count, 1, 1, thread_count, 1, 1,
                              shared_memory, stream->handle, args_2, nullptr));

    /* Unlock while synchronizing */ {
        unlock_guard guard(state.mutex);
        cuda_check(cuEventSynchronize(stream->event));
    }

    jit_free(buckets);

    *perm_out    = perm;
    *offsets_out = offsets;
}
