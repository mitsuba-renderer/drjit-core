#include "internal.h"
#include "util.h"
#include "var.h"
#include "log.h"

const char *reduction_name[(int) ReductionType::Count] = { "add", "mul", "min",
                                                           "max", "and", "or" };

/// Fill a device memory region with constants of a given type
void jit_fill(VarType type, void *ptr, uint32_t size, const void *src) {
    jit_trace("jit_fill(" ENOKI_PTR ", type=%s, size=%u)", (uintptr_t) ptr,
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
                    uint32_t block_count, thread_count;
                    device.get_launch_config(&block_count, &thread_count, size);
                    void *args[] = { &ptr, &size, (void *) src };
                    CUfunction kernel = jit_cuda_fill_64[device.id];
                    cuda_check(cuLaunchKernel(kernel, block_count, 1, 1,
                                              thread_count, 1, 1, 0,
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
                    for (uint32_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 2: {
                    uint16_t value = ((uint16_t *) src)[0],
                             *p    = (uint16_t *) ptr;
                    for (uint32_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 4: {
                    uint32_t value = ((uint32_t *) src)[0],
                             *p    = (uint32_t *) ptr;
                    for (uint32_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 8: {
                    uint64_t value = ((uint64_t *) src)[0],
                             *p    = (uint64_t *) ptr;
                    for (uint32_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            default:
                jit_raise("jit_fill(): unknown type!");
        }
    }
}

void jit_reduce(VarType type, ReductionType rtype, const void *ptr, uint32_t size,
                void *out) {
    jit_log(Debug, "jit_reduce(" ENOKI_PTR ", type=%s, rtype=%s, size=%u)",
            (uintptr_t) ptr, var_type_name[(int) type],
            reduction_name[(int) rtype], size);

    uint32_t type_size = var_type_size[(int) type];
    Stream *stream = active_stream;

    if (stream) {
        const Device &device = state.devices[stream->device];
        CUfunction func = jit_cuda_reductions[(int) rtype][(int) type][device.id];
        if (!func)
            jit_raise("jit_reduce(): no existing kernel for type=%s, rtype=%s!",
                      var_type_name[(int) type], reduction_name[(int) rtype]);

        uint32_t thread_count = 1024,
                 shared_size = thread_count * type_size,
                 block_count;

        device.get_launch_config(&block_count, nullptr, size, thread_count);

        if (size <= 1024) {
            /// This is a small array, do everything in just one reduction.
            void *args[] = { &ptr, &size, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, thread_count, 1, 1,
                                      shared_size, stream->handle, args,
                                      nullptr));
        } else {
            void *temp = jit_malloc(AllocType::Device, block_count * type_size);

            // First reduction
            void *args_1[] = { &ptr, &size, &temp };
            cuda_check(cuLaunchKernel(func, block_count, 1, 1, thread_count, 1, 1,
                                      shared_size, stream->handle, args_1,
                                      nullptr));

            // Second reduction
            void *args_2[] = { &temp, &block_count, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, thread_count, 1, 1,
                                      shared_size, stream->handle, args_2,
                                      nullptr));

            jit_free(temp);
        }
    }
}

/// 'All' reduction for boolean arrays
bool jit_all(bool *values, uint32_t size) {
    Stream *stream = active_stream;
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jit_log(Debug, "jit_all(" ENOKI_PTR ", size=%u)", (uintptr_t) values, size);

    bool result;
    if (stream) {
        if (trailing)
            cuda_check(cuMemsetD8Async(values + size, 0x01, trailing, stream->handle));
        uint8_t *out = (uint8_t *) jit_malloc(AllocType::HostPinned, 4);
        jit_reduce(VarType::UInt32, ReductionType::And, values, reduced_size, out);
        cuda_check(cuStreamSynchronize(stream->handle));
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
        jit_free(out);
    } else {
        if (trailing)
            memset(values + size, 0x01, trailing);
        uint8_t out[4];
        jit_reduce(VarType::UInt32, ReductionType::And, values, reduced_size, out);
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
    }

    return result;
}

/// 'Any' reduction for boolean arrays
bool jit_any(bool *values, uint32_t size) {
    Stream *stream = active_stream;
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jit_log(Debug, "jit_any(" ENOKI_PTR ", size=%u)", (uintptr_t) values, size);

    bool result;
    if (stream) {
        if (trailing)
            cuda_check(cuMemsetD8Async(values + size, 0x00, trailing, stream->handle));
        uint8_t *out = (uint8_t *) jit_malloc(AllocType::HostPinned, 4);
        jit_reduce(VarType::UInt32, ReductionType::Or, values, reduced_size, out);
        cuda_check(cuStreamSynchronize(stream->handle));
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
        jit_free(out);
    } else {
        if (trailing)
            memset(values + size, 0x00, trailing);
        uint8_t out[4];
        jit_reduce(VarType::UInt32, ReductionType::Or, values, reduced_size, out);
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
    }

    return result;
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

        jit_log(Debug, "jit_scan(" ENOKI_PTR " -> " ENOKI_PTR ", size=%u)",
                (uintptr_t) in, (uintptr_t) out, size);

        if (size == 0) {
            return;
        } else if (size == 1) {
            cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, sizeof(uint32_t),
                                       stream->handle));
        } else if (size <= 4096) {
            void *args[] = { &in, &out, &size };
            cuda_check(cuLaunchKernel(jit_cuda_scan_small_u32[device.id], 1, 1, 1,
                                      thread_count, 1, 1, shared_size,
                                      stream->handle, args, nullptr));
        } else {
            uint32_t *block_sums = (uint32_t *) jit_malloc(
                AllocType::Device, block_count * sizeof(uint32_t));

            void *args[] = { &in, &out, &block_sums };
            cuda_check(cuLaunchKernel(jit_cuda_scan_large_u32[device.id], block_count, 1, 1,
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

void jit_mkperm(const uint32_t *ptr, uint32_t size, uint32_t bucket_count,
                uint32_t *perm, uint32_t *offsets) {
    Stream *stream = active_stream;
    const Device &device = state.devices[stream->device];

    uint32_t block_count, thread_count;
    device.get_launch_config(&block_count, &thread_count, size, 1024, 1);

    uint32_t bucket_size_1    = sizeof(uint32_t) * bucket_count,
             bucket_size_all  = bucket_size_1 * block_count;

    /// Determine the amount of work to be done per SM
    uint32_t size_per_block = (size + block_count - 1) / block_count;
    size_per_block = (size_per_block + thread_count - 1) / thread_count * thread_count;

    /* If there is a sufficient amount of shared memory, atomically accumulate into a
       shared memory buffer. Otherwise, use global memory, which is much slower. */
    uint32_t shared_size, shared_size_small = sizeof(uint32_t) * bucket_count;
    CUfunction phase_1, phase_4;

    if (shared_size_small <= (uint32_t) device.shared_memory_bytes) {
        phase_1 = jit_cuda_mkperm_phase_1_shared[device.id];
        phase_4 = jit_cuda_mkperm_phase_4_shared[device.id];
        shared_size = shared_size_small;

    } else {
        phase_1 = jit_cuda_mkperm_phase_1_global[device.id];
        phase_4 = jit_cuda_mkperm_phase_4_global[device.id];
        shared_size = 0;
    }

    jit_trace("jit_mkperm(" ENOKI_PTR
              ", size=%u, block_count=%u, thread_count=%u, size_per_block_%u, shared_size=%u)",
              (uintptr_t) ptr, size, block_count, thread_count, size_per_block, shared_size);

    uint32_t *buckets =
        (uint32_t *) jit_malloc(AllocType::Device, bucket_size_all);

    // Phase 1: Count the number of occurrences per block
    void *args_1[] = { &ptr, &buckets, &size, &size_per_block,
                       &bucket_count };

    cuda_check(cuLaunchKernel(phase_1, block_count, 1, 1, thread_count, 1, 1,
                              shared_size, stream->handle, args_1, nullptr));

    // Phase 2: exclusive prefix sum over transposed buckets
    jit_transpose(buckets, bucket_count, block_count);
    jit_scan(buckets, buckets, bucket_count * block_count);
    jit_transpose(buckets, block_count, bucket_count);

    // Phase 3: collect non-empty buckets (optional)
    if (likely(offsets)) {
        uint32_t *counter = (uint32_t *) jit_malloc(AllocType::Device, sizeof(uint32_t));
        cuda_check(cuMemsetD8Async(counter, 0, sizeof(uint32_t), stream->handle));

        void *args_3[] = { &buckets, &bucket_count, &size, &counter, &offsets };

        cuda_check(cuLaunchKernel(jit_cuda_mkperm_phase_3, block_count, 1, 1,
                                  thread_count, 1, 1, 0, stream->handle, args_3,
                                  nullptr));

        cuda_check(cuMemcpyAsync((CUdeviceptr) offsets, (CUdeviceptr) counter,
                                 sizeof(uint32_t), stream->handle));

        cuda_check(cuEventRecord(stream->event, stream->handle));
        jit_free(counter);
    }

    // Phase 4: write out permutation based on bucket counts
    void *args_4[] = { &ptr, &buckets, &perm, &size, &size_per_block,
                       &bucket_count };
    cuda_check(cuLaunchKernel(phase_4, block_count, 1, 1, thread_count, 1, 1,
                              shared_size, stream->handle, args_4, nullptr));

    if (likely(offsets)) {
        unlock_guard guard(state.mutex);
        cuda_check(cuEventSynchronize(stream->event));
    }

    jit_free(buckets);
}
