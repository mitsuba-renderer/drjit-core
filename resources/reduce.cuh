/*
    kernels/reduce.cuh -- CUDA parallel reduction kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

template <typename Value, typename Reduce, uint32_t BlockSize>
__device__ void reduce(const Value *data, uint32_t size, Value *out) {
    Value *shared = SharedMemory<Value>::get();

    uint32_t tid    = threadIdx.x,
             bid    = blockIdx.x,
             nb     = gridDim.x,
             offset = BlockSize * 2 * bid + tid,
             stride = BlockSize * 2 * nb;

    Reduce red;
    Value value = red.init();

    // Grid-stride loop to reduce elements
    for (uint32_t i = offset; i < size; i += stride) {
        value = red(value, data[i]);
        if (i + BlockSize < size)
            value = red(value, data[i + BlockSize]);
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
            out[bid] = value;
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

#define HORIZ_OP(Name, Reduction, Type, Suffix)                                \
    KERNEL void Name##_##Suffix(const Type *data, uint32_t size, Type *out) {  \
        reduce<Type, Reduction<Type>, 1024>(data, size, out);                  \
    }

#define HORIZ_OP_ALL(Name, Reduction)                                          \
    HORIZ_OP(Name, Reduction, int32_t, i32)                                    \
    HORIZ_OP(Name, Reduction, uint32_t, u32)                                   \
    HORIZ_OP(Name, Reduction, int64_t, i64)                                    \
    HORIZ_OP(Name, Reduction, uint64_t, u64)                                   \
    HORIZ_OP(Name, Reduction, half, f16)                                       \
    HORIZ_OP(Name, Reduction, float, f32)                                      \
    HORIZ_OP(Name, Reduction, double, f64)

HORIZ_OP_ALL(reduce_sum, reduction_add)
HORIZ_OP_ALL(reduce_mul, reduction_mul)
HORIZ_OP_ALL(reduce_min, reduction_min)
HORIZ_OP_ALL(reduce_max, reduction_max)

// ----------------------------------------------------------------------------

HORIZ_OP(reduce_or,  reduction_or,  uint32_t, u32)
HORIZ_OP(reduce_and, reduction_and, uint32_t, u32)

#undef HORIZ_OP
#undef HORIZ_OP_ALL
