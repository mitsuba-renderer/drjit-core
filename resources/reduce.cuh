/*
    kernels/reduce.cuh -- CUDA parallel reduction kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

template <typename Reduce, typename Ts, typename Tv, uint32_t BlockSize>
__device__ void reduce(const Ts *data, uint32_t size, Ts *out) {
    Tv *shared = SharedMemory<Tv>::get();

    uint32_t tid    = threadIdx.x,
             bid    = blockIdx.x,
             nb     = gridDim.x,
             offset = BlockSize * 2 * bid + tid,
             stride = BlockSize * 2 * nb;

    Reduce red;
    Tv value = red.init();

    // Grid-stride loop to reduce elements
    for (uint32_t i = offset; i < size; i += stride) {
        value = red(value, Tv(data[i]));
        if (i + BlockSize < size)
            value = red(value, Tv(data[i + BlockSize]));
    }

    // Write to shared memory and wait for all threads to reach this point
    shared[tid] = value;
    __syncthreads();

    // Block-level reduction from nb*BlockSize -> nb*32 values
    if (BlockSize >= 1024 && tid < 512)
        shared[tid] = value = red(value, shared[tid + 512]);
    __syncthreads();

    if (BlockSize >= 512 && tid < 256)
        shared[tid] = value = red(value, shared[tid + 256]);
    __syncthreads();

    if (BlockSize >= 256 && tid < 128)
        shared[tid] = value = red(value, shared[tid + 128]);
    __syncthreads();

    if (BlockSize >= 128 && tid < 64)
       shared[tid] = value = red(value, shared[tid + 64]);
    __syncthreads();

    if (tid < 32) {
        if (BlockSize >= 64)
            value = red(value, shared[tid + 32]);

        // Block-level reduction from nb*32 -> nb values
        for (uint32_t i = 1; i < WARP_SIZE; i *= 2)
            value = red(value, __shfl_xor_sync(FULL_MASK, value, i));

        if (tid == 0)
            out[bid] = Ts(value);
    }
}

template <typename Value> struct reduction_add {
    __device__ Value init() { return (Value) 0; }
    __device__ Value operator()(Value a, Value b) const {
        return add_(a, b);
    }
};

template <typename Value> struct reduction_mul {
    __device__ Value init() { return (Value) 1; }
    __device__ Value operator()(Value a, Value b) const {
        return mul_(a, b);
    }
};

template <typename Value> struct reduction_max {
    __device__ Value init() {
        return std::is_integral<Value>::value
                   ?  std::numeric_limits<Value>::min()
                   : -std::numeric_limits<Value>::infinity();
    }
    __device__ Value operator()(Value a, Value b) const {
        return max_(a, b);
    }
};

template <> struct reduction_max<half> {
    __device__ half init() { return __ushort_as_half((unsigned short) 0xFC00U); }
    __device__ half operator()(half a, half b) const {
        return max_(a, b);
    }
};

template <typename Value> struct reduction_min {
    __device__ Value init() {
        return std::is_integral<Value>::value
                   ? std::numeric_limits<Value>::max()
                   : std::numeric_limits<Value>::infinity();
    }
    __device__ Value operator()(Value a, Value b) const {
        return min_(a, b);
    }
};

template <> struct reduction_min<half> {
    __device__ half init() { return __ushort_as_half((unsigned short) 0x7BFFU); }
    __device__ half operator()(half a, half b) const {
        return min_(a, b);
    }
};

template <typename Value> struct reduction_or {
    __device__ Value init() { return (Value) 0; }
    __device__ Value operator()(Value a, Value b) const {
        return a | b;
    }
};

template <typename Value> struct reduction_and {
    __device__ Value init() { return (Value) -1; }
    __device__ Value operator()(Value a, Value b) const {
        return a & b;
    }
};

// ----------------------------------------------------------------------------

#define RED(name, Ts, Tv, suffix)                                      \
    KERNEL void reduce_##name##_##suffix(const Ts *data,               \
                                         uint32_t size, Ts *out) {     \
        reduce<reduction_##name<Ts>, Ts, Tv, 1024>(data, size, out);   \
    }

#define RED_ALL_U(Name)                                                \
    RED(Name, uint32_t, uint32_t, u32)                                 \
    RED(Name, uint64_t, uint64_t, u64)                                 \
    RED(Name, half, float, f16)                                        \
    RED(Name, float, float, f32)                                       \
    RED(Name, double, double, f64)

#define RED_ALL(Name)                                                  \
    RED(Name, int32_t, int32_t, i32)                                   \
    RED(Name, uint32_t, uint32_t, u32)                                 \
    RED(Name, int64_t, int64_t, i64)                                   \
    RED(Name, uint64_t, uint64_t, u64)                                 \
    RED(Name, half, float, f16)                                        \
    RED(Name, float, float, f32)                                       \
    RED(Name, double, double, f64)

RED_ALL_U(add)
RED_ALL(mul)
RED_ALL(min)
RED_ALL(max)

// ----------------------------------------------------------------------------

RED(or,  uint32_t, uint32_t, u32)
RED(and, uint32_t, uint32_t, u32)

#undef RED
#undef RED_ALL
