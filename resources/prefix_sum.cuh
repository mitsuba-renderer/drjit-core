/*
    kernels/mkperm.cuh -- CUDA, exclusive prefix sum for 32 bit unsigned integers

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <type_traits>

template <typename T1, typename T2> struct is_same { static constexpr bool value = false; };
template <typename T> struct is_same<T, T> { static constexpr bool value = true; };

template <bool Inclusive, typename Type, typename Vector, uint32_t M>
DEVICE void prefix_sum_small(const Type *in, Type *out, uint32_t size) {
    Type *shared = SharedMemory<Type>::get();
    Type values[M];

    // Fetch M values at once with a vector load operation
    *(Vector *) values = ((const Vector *) in)[threadIdx.x];

    // Unrolled exclusive prefix sum over those M values
    Type sum_local = Type(0);
    for (uint32_t i = 0; i < M; ++i) {
        Type v = values[i];

        if constexpr (is_same<Type, float>::value || is_same<Type, double>::value) {
            /* Must be careful not to process uninitialized memory beyond the
               input array when performing arithmetic with floating point
               numbers (could produce a NaN that breaks the result even for
               in-bounds data) */

            if (threadIdx.x * M + i >= size)
                v = Type(0);
        }

        if constexpr (Inclusive) {
            sum_local += v;
            values[i] = sum_local;
        } else {
            values[i] = sum_local;
            sum_local += v;
        }
    }

    // Block-level inclusive prefix sum of 'sum_local' via shared memory
    uint32_t si = threadIdx.x;
    shared[si] = 0;
    si += blockDim.x;

    Type sum_block = sum_local;
    for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        shared[si] = sum_block;
        __syncthreads();
        sum_block = shared[si] + shared[si - offset];
        __syncthreads();
    }

    sum_block -= sum_local;

    for (uint32_t i = 0; i < M; ++i)
        values[i] += sum_block;

    ((Vector *) out)[threadIdx.x] = *(const Vector *) values;
}

/*
 * The two functions below store and read block-level partial reductions
 * and are used by prefix_sum_large.
 *
 * The CUDA backend implementation for *large* numeric types (double precision
 * floats, 64 bit integers) has the following technical limitation: when
 * reducing 64-bit integers, their values must be smaller than 2**62. When
 * reducing double precision arrays, the two least significant mantissa bits
 * are clamped to zero when forwarding the prefix from one 512-wide block to
 * the next (at a very minor loss in accuracy). The reason is that the
 * operations requires two status bits (in the `flags` variable)  to coordinate
 * the prefix and status of each 512-wide block, and those must each fit into a
 * single 64 bit value (128-bit writes aren't guaranteed to be atomic).
 */
template <typename T>
DEVICE FINLINE void store_pair(uint64_t *ptr, uint32_t flags, T value) {
    uint64_t combined;

    if constexpr (sizeof(T) == 4) {
        uint32_t value_32;
        memcpy(&value_32, &value, sizeof(T));
        combined = flags | (((uint64_t) value_32) << 32);
    } else if constexpr (sizeof(T) == 8) {
        if constexpr (is_same<T, uint64_t>::value) {
            combined = (value << 2) | flags;
        } else {
            uint64_t value_64;
            memcpy(&value_64, &value, sizeof(T));
            combined = (value_64 & ~3ull) | flags;
        }
    }

    asm volatile("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(combined));
}

template <typename T>
DEVICE FINLINE void load_pair(uint64_t *ptr, uint32_t *flags_out, T *value_out) {
    uint64_t tmp;
    asm volatile("ld.cg.u64 %0, [%1];" : "=l"(tmp) : "l"(ptr));

    if constexpr (sizeof(T) == 4) {
        *flags_out = (uint32_t) tmp;
        uint32_t value_32 = (uint32_t) (tmp >> 32);
        memcpy(value_out, &value_32, sizeof(T));
    } else if constexpr (sizeof(T) == 8) {
        *flags_out = ((uint32_t) tmp) & 3u;

        if constexpr (is_same<T, uint64_t>::value) {
            *value_out = tmp >> 2;
        } else {
            tmp &= ~3ull;
            memcpy(value_out, &tmp, sizeof(T));
        }
    }
}

/**
 * Prefix sum for large arrays, which are processed in chunks of 128*N*M
 * (usually M=N=4, so 2048) elements. This works as follows:
 *
 * 1. All 128 threads in a block cooperate to perform a coalesced M-wide vector
 *    load. This repeats N-1 more times to read a contiguous chunk of 128*N*M
 *    values. The loaded data is written to shared memory (in the same order).
 *
 * 2. Each thread fetches N*M *contiguous* elements from shared memory and
 *    performs a local inclusive or exclusive prefix sum using register memory.
 *    This process also produces the sum of all N*M values.
 *
 * 3. The 128 threads perform a tree reduction via shared memory to compute
 *    an inclusive prefix sum of the local sum computed in step 2.
 *
 * 4. The last thread stores the tentative block-wide sum in a global memory
 *    scratch pad.
 *
 * 5. All threads in the block look backwards to account for the prefix
 *    of predecessor blocks. This is based on the paper
 *      "Single-pass Parallel Prefix Scan with Decoupled Look-back"
 *       by Duane Merrill and Michael Garland
 *
 * 6. The last thread stores the final block-wide sum with correct prefix
 *    in the global memory scratch pad.
 */

template <bool Inclusive, typename Type, uint32_t N, typename Vector, uint32_t M>
DEVICE void prefix_sum_large(const Type *in, Type *out, uint32_t size, uint64_t *scratch) {
    Type *shared = SharedMemory<Type>::get();
    const uint32_t thread_count = 128;

    /* Copy a block of input data to shared memory */ {
        Vector v[N];
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t j = (blockIdx.x * N + i) * thread_count + threadIdx.x;
            Vector value = ((const Vector *) in)[j];

            /* Must be careful not to process uninitialized memory beyond the
               input array when performing arithmetic with floating point
               numbers (could produce a NaN that breaks the result even for
               in-bounds data) */
            if constexpr (is_same<Vector, float4>::value) {
                j *= M;
                if (j + 0 >= size) value.x = 0;
                if (j + 1 >= size) value.y = 0;
                if (j + 2 >= size) value.z = 0;
                if (j + 3 >= size) value.w = 0;
            }

            if constexpr (is_same<Vector, double2>::value) {
                j *= M;
                if (j + 0 >= size) value.x = 0;
                if (j + 1 >= size) value.y = 0;
            }

            v[i] = value;
        }

        for (uint32_t i = 0; i < N; ++i)
            ((Vector *) shared)[i * thread_count + threadIdx.x] = v[i];
    }

    __syncthreads();

    // Fetch input from shared memory
    Type values[N * M];
    for (uint32_t i = 0; i < N; ++i)
        ((Vector *) values)[i] = ((const Vector *) shared)[threadIdx.x * N + i];

    // Unrolled exclusive prefix sum
    Type sum_local = Type(0);
    for (uint32_t i = 0; i < N * M; ++i) {
        Type v = values[i];

        if constexpr (Inclusive) {
            sum_local += v;
            values[i] = sum_local;
        } else {
            values[i] = sum_local;
            sum_local += v;
        }
    }

    __syncthreads();

    // Block-level inclusive prefix sum of 'sum_local' via shared memory
    uint32_t si = threadIdx.x;
    shared[si] = 0;
    si += thread_count;

    Type sum_block = sum_local;
    for (uint32_t offset = 1; offset < thread_count; offset <<= 1) {
        shared[si] = sum_block;
        __syncthreads();
        sum_block = shared[si] + shared[si - offset];
        __syncthreads();
    }

    // Store tentative block-level inclusive prefix sum value in global memory
    // (still missing prefix from predecessors)
    scratch += blockIdx.x;
    if (threadIdx.x == thread_count - 1)
        store_pair(scratch, 1, sum_block);

    uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    Type prefix = Type(0);

    // Each thread looks back a different amount
    int32_t shift = lane - WARP_SIZE;

    // Decoupled loopback iteration
    while (true) {
        uint32_t flags;
        Type value;

        // Load with volatile inline assembly to prevent loop hoisting
        load_pair(scratch + shift, &flags, &value);

        // Retry if at least one of the predecessors hasn't made any progress yet
        if (__any_sync(FULL_MASK, flags == 0))
            continue;

        uint32_t mask = __ballot_sync(FULL_MASK, flags == 2);
        if (mask == 0) {
            // Sum partial results, look back further
            prefix += value;
            shift -= WARP_SIZE;
        } else {
            // Lane 'index' is done!
            uint32_t index = 31 - __clz(mask);

            // Sum up all the unconverged (higher) lanes *and* 'index'
            if (lane >= index)
                prefix += value;

            break;
        }
    }

    // Warp-level sum reduction of 'prefix'
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        prefix += __shfl_down_sync(FULL_MASK, prefix, offset, WARP_SIZE);

    // Broadcast the reduced 'prefix' value from lane 0
    prefix = __shfl_sync(FULL_MASK, prefix, 0);

    // Offset the local block sum with the final prefix
    sum_block += prefix;

    // Store block-level complete inclusive prefixnsum value in global memory
    if (threadIdx.x == thread_count - 1)
        store_pair(scratch, 2, sum_block);

    sum_block -= sum_local;
    for (uint32_t i = 0; i < N * M; ++i)
        values[i] += sum_block;

    // Store input into shared memory
    for (uint32_t i = 0; i < N; ++i)
        ((Vector *) shared)[threadIdx.x*N + i] = ((const Vector *) values)[i];

    __syncthreads();

    /* Copy shared memory back to global memory */ {
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t j = i * thread_count + threadIdx.x;
            Vector v = ((const Vector *) shared)[j];
            ((Vector *) out)[j + blockIdx.x * (N * thread_count)] = v;
        }
    }
}

KERNEL void prefix_sum_large_init(uint64_t *scratch, uint32_t size) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        scratch[i] = (i < 32) ? 2 : 0;
}

KERNEL void prefix_sum_exc_small_u32(const uint32_t *in, uint32_t *out, uint32_t size) {
    prefix_sum_small<false, uint32_t, uint4, 4>(in, out, size);
}

KERNEL void prefix_sum_exc_small_u64(const uint64_t *in, uint64_t *out, uint32_t size) {
    prefix_sum_small<false, uint64_t, ulonglong2, 2>(in, out, size);
}

KERNEL void prefix_sum_exc_small_f32(const float *in, float *out, uint32_t size) {
    prefix_sum_small<false, float, float4, 4>(in, out, size);
}

KERNEL void prefix_sum_exc_small_f64(const double *in, double *out, uint32_t size) {
    prefix_sum_small<false, double, double2, 2>(in, out, size);
}

KERNEL void prefix_sum_inc_small_u32(const uint32_t *in, uint32_t *out, uint32_t size) {
    prefix_sum_small<true, uint32_t, uint4, 4>(in, out, size);
}

KERNEL void prefix_sum_inc_small_u64(const uint64_t *in, uint64_t *out, uint32_t size) {
    prefix_sum_small<true, uint64_t, ulonglong2, 2>(in, out, size);
}

KERNEL void prefix_sum_inc_small_f32(const float *in, float *out, uint32_t size) {
    prefix_sum_small<true, float, float4, 4>(in, out, size);
}

KERNEL void prefix_sum_inc_small_f64(const double *in, double *out, uint32_t size) {
    prefix_sum_small<true, double, double2, 2>(in, out, size);
}

KERNEL void prefix_sum_exc_large_u32(const uint32_t *in, uint32_t *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<false, uint32_t, 4, uint4, 4>(in, out, size, scratch);
}

KERNEL void prefix_sum_exc_large_u64(const uint64_t *in, uint64_t *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<false, uint64_t, 4, ulonglong2, 2>(in, out, size, scratch);
}

KERNEL void prefix_sum_exc_large_f32(const float *in, float *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<false, float, 4, float4, 4>(in, out, size, scratch);
}

KERNEL void prefix_sum_exc_large_f64(const double *in, double *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<false, double, 4, double2, 2>(in, out, size, scratch);
}

KERNEL void prefix_sum_inc_large_u32(const uint32_t *in, uint32_t *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<true, uint32_t, 4, uint4, 4>(in, out, size, scratch);
}

KERNEL void prefix_sum_inc_large_u64(const uint64_t *in, uint64_t *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<true, uint64_t, 4, ulonglong2, 2>(in, out, size, scratch);
}

KERNEL void prefix_sum_inc_large_f32(const float *in, float *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<true, float, 4, float4, 4>(in, out, size, scratch);
}

KERNEL void prefix_sum_inc_large_f64(const double *in, double *out, uint32_t size, uint64_t *scratch) {
    prefix_sum_large<true, double, 4, double2, 2>(in, out, size, scratch);
}

