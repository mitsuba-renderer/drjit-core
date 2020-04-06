#include <enoki/traits.h>
#include "internal.h"
#include "util.h"
#include "var.h"
#include "registry.h"
#include "log.h"

const char *reduction_name[(int) ReductionType::Count] = { "add", "mul", "min",
                                                           "max", "and", "or" };

/// Fill a device memory region with constants of a given type
void jit_fill(VarType type, void *ptr, uint32_t size, const void *src) {
    Stream *stream = active_stream;

    if (unlikely(!stream))
        jit_raise("jit_fill(): you must invoke jit_device_set() to "
                  "choose a target device before calling this function.");

    jit_trace("jit_fill(" ENOKI_PTR ", type=%s, size=%u)", (uintptr_t) ptr,
              var_type_name[(int) type], size);

    if (size == 0)
        return;

    if (stream->cuda) {
        switch (var_type_size[(int) type]) {
            case 1:
                cuda_check(cuMemsetD8Async((CUdeviceptr) ptr,
                                           ((uint8_t *) src)[0], size,
                                           stream->handle));
                break;

            case 2:
                cuda_check(cuMemsetD16Async((CUdeviceptr) ptr,
                                            ((uint16_t *) src)[0], size,
                                            stream->handle));
                break;

            case 4:
                cuda_check(cuMemsetD32Async((CUdeviceptr) ptr,
                                            ((uint32_t *) src)[0], size,
                                            stream->handle));
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

/// Perform a synchronous copy operation
void jit_memcpy(void *dst, const void *src, size_t size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_memcpy(): you must invoke jit_device_set() to choose a "
                  "target device before calling this function.");

    // Temporarily release the lock while copying
    unlock_guard guard(state.mutex);
    if  (stream->cuda) {
        cuda_check(cuStreamSynchronize(stream->handle));
        cuda_check(cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, size));
    } else {
        memcpy(dst, src, size);
    }
}

/// Perform an assynchronous copy operation
void jit_memcpy_async(void *dst, const void *src, size_t size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_memcpy_async(): you must invoke jit_device_set() to "
                  "choose a target device before calling this function.");

    if  (stream->cuda)
        cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, size,
                                 stream->handle));
    else
        memcpy(dst, src, size);
}

template <typename Value> struct reduction_add {
    Value init() { return (Value) 0; }
    Value operator()(Value a, Value b) const {
        return a + b;
    }
};

template <typename Value> struct reduction_mul {
    Value init() { return (Value) 1; }
    Value operator()(Value a, Value b) const {
        return a * b;
    }
};

template <typename Value> struct reduction_max {
    Value init() {
        return std::is_integral<Value>::value
                   ?  std::numeric_limits<Value>::min()
                   : -std::numeric_limits<Value>::infinity();
    }
    Value operator()(Value a, Value b) const {
        return std::max(a, b);
    }
};

template <typename Value> struct reduction_min {
    Value init() {
        return std::is_integral<Value>::value
                   ? std::numeric_limits<Value>::max()
                   : std::numeric_limits<Value>::infinity();
    }
    Value operator()(Value a, Value b) const {
        return std::min(a, b);
    }
};

template <typename Value> struct reduction_or {
    Value init() { return (Value) 0; }
    Value operator()(Value a, Value b) const {
        return a | b;
    }
};

template <typename Value> struct reduction_and {
    Value init() { return (Value) -1; }
    Value operator()(Value a, Value b) const {
        return a & b;
    }
};

template <typename Reduction, typename Value>
void jit_reduce_cpu(const Value *ptr, uint32_t size, Value *out) {
    unlock_guard guard(state.mutex);
    Reduction reduction;
    Value value = reduction.init();
    for (uint32_t i = 0; i < size; ++i)
        value = reduction(value, ptr[i]);
    *out = value;
}

template <typename Value>
void jit_reduce_cpu(ReductionType rtype, const Value *ptr, uint32_t size,
                    Value *out) {
    using UInt = uint_with_size_t<Value>;
    switch (rtype) {
        case ReductionType::Add:
            jit_reduce_cpu<reduction_add<Value>>(ptr, size, out);
            break;
        case ReductionType::Mul:
            jit_reduce_cpu<reduction_mul<Value>>(ptr, size, out);
            break;
        case ReductionType::Min:
            jit_reduce_cpu<reduction_min<Value>>(ptr, size, out);
            break;
        case ReductionType::Max:
            jit_reduce_cpu<reduction_max<Value>>(ptr, size, out);
            break;
        case ReductionType::And:
            jit_reduce_cpu<reduction_and<UInt>>((const UInt *) ptr, size, (UInt *) out);
            break;
        case ReductionType::Or:
            jit_reduce_cpu<reduction_or<UInt>>((const UInt *) ptr, size, (UInt *) out);
            break;

        default: jit_raise("jit_reduce(): unsupported reduction type!");
    }
}

void jit_reduce(VarType type, ReductionType rtype, const void *ptr, uint32_t size,
                void *out) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_reduce(): you must invoke jit_device_set() to "
                  "choose a target device before calling this function.");

    jit_log(Debug, "jit_reduce(" ENOKI_PTR ", type=%s, rtype=%s, size=%u)",
            (uintptr_t) ptr, var_type_name[(int) type],
            reduction_name[(int) rtype], size);

    uint32_t type_size = var_type_size[(int) type];

    if (stream->cuda) {
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
    } else {
        switch (type) {
            case VarType::Int32:
                jit_reduce_cpu<int32_t>(rtype, (const int32_t *) ptr, size, (int32_t *) out);
                break;

            case VarType::UInt32:
                jit_reduce_cpu<uint32_t>(rtype, (const uint32_t *) ptr, size, (uint32_t *) out);
                break;

            case VarType::Int64:
                jit_reduce_cpu<int64_t>(rtype, (const int64_t *) ptr, size, (int64_t *) out);
                break;

            case VarType::UInt64:
                jit_reduce_cpu<uint64_t>(rtype, (const uint64_t *) ptr, size, (uint64_t *) out);
                break;

            case VarType::Float32:
                jit_reduce_cpu<float>(rtype, (const float *) ptr, size, (float *) out);
                break;

            case VarType::Float64:
                jit_reduce_cpu<double>(rtype, (const double *) ptr, size, (double *) out);
                break;

            default: jit_raise("jit_reduce(): unsupported data type!");
        }
    }
}

/// 'All' reduction for boolean arrays
uint8_t jit_all(uint8_t *values, uint32_t size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_all(): you must invoke jit_device_set() to "
                  "choose a target device before calling this function.");

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jit_log(Debug, "jit_all(" ENOKI_PTR ", size=%u)", (uintptr_t) values, size);

    uint8_t result;
    if (stream->cuda) {
        if (trailing)
            cuda_check(cuMemsetD8Async((CUdeviceptr)(values + size), 0x01,
                                       trailing, stream->handle));
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
uint8_t jit_any(uint8_t *values, uint32_t size) {
    Stream *stream = active_stream;
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jit_log(Debug, "jit_any(" ENOKI_PTR ", size=%u)", (uintptr_t) values, size);

    uint8_t result;
    if (stream) {
        if (trailing)
            cuda_check(cuMemsetD8Async((CUdeviceptr)(values + size), 0x00,
                                       trailing, stream->handle));
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
    if (unlikely(!stream))
        jit_raise("jit_scan(): you must invoke jit_device_set() to "
                  "choose a target device before calling this function.");

    if (stream->cuda) {
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

void jit_transpose(const uint32_t *in, uint32_t *out, uint32_t rows, uint32_t cols) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_transpose(): you must invoke jit_device_set() to "
                  "choose a target device before calling this function.");

    if (stream->cuda) {
        const Device &device = state.devices[stream->device];

        uint16_t blocks_x = (cols + 15u) / 16u,
                 blocks_y = (rows + 15u) / 16u;

        jit_log(Debug,
                "jit_transpose(" ENOKI_PTR " -> " ENOKI_PTR
                ", rows=%u, cols=%u, blocks=%ux%u)",
                (uintptr_t) in, (uintptr_t) out, rows, cols, blocks_x, blocks_y);

        void *args[] = { &in, &out, &rows, &cols };
        cuda_check(cuLaunchKernel(
            jit_cuda_transpose[device.id], blocks_x, blocks_y, 1, 16, 16, 1,
            16 * 17 * sizeof(uint32_t), stream->handle, args, nullptr));
    } else {
        jit_log(Debug,
                "jit_transpose(" ENOKI_PTR " -> " ENOKI_PTR
                ", rows=%u, cols=%u)",
                (uintptr_t) in, (uintptr_t) out, rows, cols);

        unlock_guard guard(state.mutex);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < cols; ++c)
                out[r + c * rows] = in[c + r * cols];
    }
}

/// Compute a permutation to reorder an integer array into a sorted configuration
uint32_t jit_mkperm(const uint32_t *ptr, uint32_t size, uint32_t bucket_count,
                    uint32_t *perm, uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jit_fail("jit_mkperm(): bucket_count cannot be zero!");

    Stream *stream = active_stream;

    if (stream->cuda) {
        const Device &device = state.devices[stream->device];

        // Don't use more than 1 block/SM due to shared memory requirement
        const uint32_t warp_size = 32;
        uint32_t block_count, thread_count;
        device.get_launch_config(&block_count, &thread_count, size, 1024, 1);

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

        if (bucket_size_1 * warp_count <= device.shared_memory_bytes) {
            /* "Tiny" variant, which uses shared memory atomics to produce a stable
               permutation. Handles up to 512 buckets with 64KiB of shared memory. */

            phase_1 = jit_cuda_mkperm_phase_1_tiny[device.id];
            phase_4 = jit_cuda_mkperm_phase_4_tiny[device.id];
            shared_size = bucket_size_1 * warp_count;
            bucket_size_all *= warp_count;
            variant = "tiny";
        } else if (bucket_size_1 <= device.shared_memory_bytes) {
            /* "Small" variant, which uses shared memory atomics and handles up to
               16K buckets with 64KiB of shared memory. The permutation can be
               somewhat unstable due to scheduling variations when performing atomic
               operations (although some effort is made to keep it stable within
               each group of 32 elements by performing an intra-warp reduction.) */

            phase_1 = jit_cuda_mkperm_phase_1_small[device.id];
            phase_4 = jit_cuda_mkperm_phase_4_small[device.id];
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

            phase_1 = jit_cuda_mkperm_phase_1_large[device.id];
            phase_4 = jit_cuda_mkperm_phase_4_large[device.id];
            variant = "large";
            initialize_buckets = true;
        }

        bool needs_transpose = bucket_size_1 != bucket_size_all;
        uint32_t *buckets_1, *buckets_2, *counter = nullptr;
        buckets_1 = buckets_2 =
            (uint32_t *) jit_malloc(AllocType::Device, bucket_size_all);

        // Scratch space for matrix transpose operation
        if (needs_transpose)
            buckets_2 = (uint32_t *) jit_malloc(AllocType::Device, bucket_size_all);

        if (offsets) {
            counter = (uint32_t *) jit_malloc(AllocType::Device, sizeof(uint32_t)),
            cuda_check(cuMemsetD8Async((CUdeviceptr) counter, 0, sizeof(uint32_t),
                                       stream->handle));
        }

        if (initialize_buckets)
            cuda_check(cuMemsetD8Async((CUdeviceptr) buckets_1, 0,
                                       bucket_size_all, stream->handle));

        /* Determine the amount of work to be done per block, and ensure that it is
           divisible by the warp size (the kernel implementation assumes this.) */
        uint32_t size_per_block = (size + block_count - 1) / block_count;
        size_per_block = (size_per_block + warp_size - 1) / warp_size * warp_size;

        jit_log(Debug,
                "jit_mkperm(" ENOKI_PTR
                ", size=%u, bucket_count=%u, block_count=%u, thread_count=%u, "
                "size_per_block=%u, variant=%s, shared_size=%u)",
                (uintptr_t) ptr, size, bucket_count, block_count, thread_count,
                size_per_block, variant, shared_size);

        // Phase 1: Count the number of occurrences per block
        void *args_1[] = { &ptr, &buckets_1, &size, &size_per_block,
                           &bucket_count };

        cuda_check(cuLaunchKernel(phase_1, block_count, 1, 1, thread_count, 1, 1,
                                  shared_size, stream->handle, args_1, nullptr));

        // Phase 2: exclusive prefix sum over transposed buckets
        if (needs_transpose)
            jit_transpose(buckets_1, buckets_2, bucket_size_all / bucket_size_1,
                          bucket_count);

        jit_scan(buckets_2, buckets_2, bucket_size_all / sizeof(uint32_t));

        if (needs_transpose)
            jit_transpose(buckets_2, buckets_1, bucket_count,
                          bucket_size_all / bucket_size_1);

        // Phase 3: collect non-empty buckets (optional)
        if (likely(offsets)) {
            uint32_t block_count_3, thread_count_3;
            device.get_launch_config(&block_count_3, &thread_count_3,
                                     bucket_count * block_count);

            // Round up to a multiple of the thread count
            uint32_t bucket_count_rounded =
                (bucket_count + thread_count_3 - 1) / thread_count_3 * thread_count_3;

            void *args_3[] = { &buckets_1, &bucket_count, &bucket_count_rounded,
                               &size,      &counter,      &offsets };

            cuda_check(cuLaunchKernel(jit_cuda_mkperm_phase_3[device.id],
                                      block_count_3, 1, 1, thread_count_3, 1, 1,
                                      sizeof(uint32_t) * thread_count_3,
                                      stream->handle, args_3, nullptr));

            cuda_check(cuMemcpyAsync((CUdeviceptr) (offsets + 4 * bucket_count),
                                     (CUdeviceptr) counter, sizeof(uint32_t),
                                     stream->handle));

            cuda_check(cuEventRecord(stream->event, stream->handle));
        }

        // Phase 4: write out permutation based on bucket counts
        void *args_4[] = { &ptr, &buckets_1, &perm, &size, &size_per_block,
                           &bucket_count };
        cuda_check(cuLaunchKernel(phase_4, block_count, 1, 1, thread_count, 1, 1,
                                  shared_size, stream->handle, args_4, nullptr));

        if (likely(offsets)) {
            unlock_guard guard(state.mutex);
            cuda_check(cuEventSynchronize(stream->event));
        }

        jit_free(buckets_1);
        if (needs_transpose)
            jit_free(buckets_2);
        jit_free(counter);

        return offsets ? offsets[4 * bucket_count] : 0u;
    } else { // if (!stream->cuda)
        size_t bucket_size = sizeof(uint32_t) * (size_t) bucket_count;
        uint32_t *buckets = (uint32_t *) jit_malloc(AllocType::Host, bucket_size);
        memset(buckets, 0, bucket_size);
        uint32_t unique_count = 0;

        for (uint32_t i = 0; i < size; ++i)
            buckets[ptr[i]]++;

        uint32_t sum = 0;
        for (uint32_t i = 0; i < bucket_count; ++i) {
            if (buckets[i] > 0) {
                if (offsets) {
                    offsets[unique_count*4] = i;
                    offsets[unique_count*4 + 1] = sum;
                    offsets[unique_count*4 + 2] = buckets[i];
                    offsets[unique_count*4 + 3] = 0;
                }
                unique_count++;
            }
            uint32_t temp = buckets[i];
            buckets[i] = sum;
            sum += temp;
        }

        for (uint32_t i = 0; i < size; ++i) {
            uint32_t index = buckets[ptr[i]]++;
            perm[index] = i;
        }

        jit_free(buckets);
        return unique_count;
    }
}

// Compute a permutation to reorder an array of registered pointers
VCallBucket *jit_vcall(const char *domain, uint32_t index,
                       uint32_t *bucket_count_out) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_vcall(): you must invoke jit_device_set() to "
                  "choose a target device before calling this function.");
    bool cuda = stream->cuda;

    auto it = state.vcall_cache.find(index);
    if (it != state.vcall_cache.end()) {
        *bucket_count_out = it.value().first;
        return it.value().second;
    }

    jit_var_eval(index);
    Variable *v = jit_var(index);
    const void *ptr = v->data;
    uint32_t size = v->size;

    uint32_t bucket_count = jit_registry_get_max(domain) + 1;

    jit_log(Debug, "jit_vcall(%u, domain=\"%s\")", index, domain);

    size_t perm_size    = (size_t) size * (size_t) sizeof(uint32_t),
           offsets_size = (bucket_count * 4 + 1) * sizeof(uint32_t);

    uint32_t *offsets = (uint32_t *) jit_malloc(
        cuda ? AllocType::HostPinned : AllocType::Host, offsets_size);
    uint32_t *perm = (uint32_t *) jit_malloc(
        cuda ? AllocType::Device : AllocType::Host, perm_size);

    // Compute permutation
    uint32_t unique_count     = jit_mkperm((const uint32_t *) ptr, size, bucket_count, perm, offsets),
             unique_count_out = unique_count;

    // Register permutation variable with JIT backend and transfer ownership
    uint32_t perm_var = jit_var_map(VarType::UInt32, perm, size, 1);

    Variable v2;
    v2.type = (uint32_t) VarType::UInt32;
    v2.dep[0] = perm_var;
    v2.retain_data = true;
    v2.tsize = 1;
    v2.cuda = active_stream->cuda;
    v2.unaligned = 1;

    uint32_t *offsets_out = offsets;

    for (uint32_t i = 0; i < unique_count; ++i) {
        uint32_t bucket_id     = offsets[i * 4 + 0],
                 bucket_offset = offsets[i * 4 + 1],
                 bucket_size   = offsets[i * 4 + 2];

        if (bucket_id == 0) {
            --unique_count_out;
            continue;
        }

        /// Crete variable for permutation subrange
        v2.data = perm + bucket_offset;
        v2.size = bucket_size;

        uint32_t index;
        Variable *vo;
        std::tie(index, vo) = jit_trace_append(v2);

        jit_var_inc_ref_int(perm_var);
        jit_var_inc_ref_ext(index, vo);

        void *ptr = jit_registry_get_ptr(domain, bucket_id);
        memcpy(offsets_out, &ptr, sizeof(void *));
        memcpy(offsets_out + 2, &index, sizeof(uint32_t *));
        offsets_out += 4;

        jit_trace(
            "jit_vcall(): registered variable %u: bucket %u (%p) of size %u.",
            index, bucket_id, ptr, bucket_size);
    }

    jit_var_dec_ref_ext(perm_var);

    *bucket_count_out = unique_count_out;

    v = jit_var(index);
    v->vcall_cached = true;
    state.vcall_cache[index] =
        std::make_pair(unique_count_out, (VCallBucket *) offsets);

    return (VCallBucket *) offsets;
}
