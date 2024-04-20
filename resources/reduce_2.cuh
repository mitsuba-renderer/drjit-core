/*
    kernels/reduce.cuh -- CUDA parallel reduction kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

template <typename Value, typename Reduce, uint32_t BlockSize>
__device__ void reduce_2(const Value *in_1, const Value *in_2, uint32_t size,
                         Value *out) {
    Value *shared = SharedMemory<Value>::get();

    uint32_t tid = threadIdx.x,
             bid = blockIdx.x,
             nb = gridDim.x,
             offset = BlockSize * 2 * bid + tid,
             stride = BlockSize * 2 * nb;

    Reduce red;
    Value value = red.init();

    // Grid-stride loop to reduce elements
    for (uint32_t i = offset; i < size; i += stride) {
        value = red(value, in_1[i], in_2[i]);
        uint32_t ib = i + BlockSize;
        if (ib < size)
            value = red(value, in_1[ib], in_2[ib]);
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
        for (uint32_t i = 1; i < 32; i *= 2)
            value = red(value, __shfl_xor_sync(FULL_MASK, value, i));

        if (tid == 0)
            out[bid] = value;
    }
}

template <typename Value> struct reduction_dot {
    __device__ Value init() { return (Value) 0; }
    __device__ Value operator()(Value accum, Value value) const {
        return add_(accum, value);
    }
    __device__ Value operator()(Value accum, Value in_1, Value in_2) const {
        return fma_(in_1, in_2, accum);
    }
};

#define HORIZ_OP(Name, Reduction, Type, Suffix)                                \
    KERNEL void Name##_##Suffix(const Type *in_1, const Type *in_2,            \
                                uint32_t size, Type *out) {                    \
        reduce_2<Type, Reduction<Type>, 1024>(in_1, in_2, size, out);          \
    }

#define HORIZ_OP_ALL(Name, Reduction)                                          \
    HORIZ_OP(Name, Reduction, half, f16)                                       \
    HORIZ_OP(Name, Reduction, float, f32)                                      \
    HORIZ_OP(Name, Reduction, double, f64)

HORIZ_OP_ALL(reduce_dot, reduction_dot)

#undef HORIZ_OP
#undef HORIZ_OP_ALL
