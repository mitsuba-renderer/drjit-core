/*
    kernels/block_reduce.cuh -- Warp/block-cooperative block reductions

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <cuda_fp16.h>

template <typename Reduce, typename Ts, typename Tv, uint32_t BlockSize>
__device__ void block_reduce(const Ts *in, Ts *out, uint32_t size) {
    Tv *shared = SharedMemory<Tv>::get();

    uint32_t tid = threadIdx.x,
             i   = blockIdx.x * blockDim.x + tid;

    if (i >= size)
        return;

    uint32_t tid_b = tid % BlockSize;

    Reduce red;
    Tv value = (Tv) in[i];

    // Potentially begin with a shared memory tree reduction ...
    if (BlockSize > WARP_SIZE) {
        shared[tid] = value;
        __syncthreads();

        for (uint32_t s = BlockSize / 2; s > WARP_SIZE; s >>= 1) {
            if (tid_b < s)
                shared[tid] = value = red(value, shared[tid + s]);
            __syncthreads();
        }
    }

    // .. then switch over to warp-local reduction
    if (tid_b < WARP_SIZE) {
        // Fetch final intermediate result from 2nd warp
        if (BlockSize > WARP_SIZE)
            value = red(value, shared[tid + WARP_SIZE]);

        constexpr uint32_t Active = BlockSize >= WARP_SIZE ? WARP_SIZE
                                                           : BlockSize;

        for (uint32_t k = 1; k < Active; k *= 2)
            value = red(value, __shfl_xor_sync((uint32_t) -1, value, k, Active));

        // Write reduced result back to global memory
        if (tid_b == 0)
            out[i / BlockSize] = Ts(value);
    }
}

// ----------------------------------------------------------------------------

#define BLOCK_RED_1(op, Ts, Tv, tname, bsize)                                  \
    KERNEL void block_reduce_##op##_##tname##_##bsize(const Ts *in, Ts *out,   \
                                                      uint32_t size) {         \
        block_reduce<reduction_##op<Tv>, Ts, Tv, bsize>(in, out, size);        \
    }

#define BLOCK_RED(op, Ts, Tv, tname)                                           \
    BLOCK_RED_1(op, Ts, Tv, tname, 1024)                                       \
    BLOCK_RED_1(op, Ts, Tv, tname, 512)                                        \
    BLOCK_RED_1(op, Ts, Tv, tname, 256)                                        \
    BLOCK_RED_1(op, Ts, Tv, tname, 128)                                        \
    BLOCK_RED_1(op, Ts, Tv, tname, 64)                                         \
    BLOCK_RED_1(op, Ts, Tv, tname, 32)                                         \
    BLOCK_RED_1(op, Ts, Tv, tname, 16)                                         \
    BLOCK_RED_1(op, Ts, Tv, tname, 8)                                          \
    BLOCK_RED_1(op, Ts, Tv, tname, 4)                                          \
    BLOCK_RED_1(op, Ts, Tv, tname, 2)

#define BLOCK_RED_ALL(op)                                                      \
    BLOCK_RED(op, half, float, f16)                                            \
    BLOCK_RED(op, float, float, f32)                                           \
    BLOCK_RED(op, double, double, f64)                                         \
    BLOCK_RED(op, uint32_t, uint32_t, u32)                                     \
    BLOCK_RED(op, uint32_t, uint64_t, u64)                                     \
    BLOCK_RED(op, int32_t, int32_t, i32)                                       \
    BLOCK_RED(op, int64_t, int64_t, i64)

#define BLOCK_RED_ALL_U(op)                                                    \
    BLOCK_RED(op, half, float, f16)                                            \
    BLOCK_RED(op, float, float, f32)                                           \
    BLOCK_RED(op, double, double, f64)                                         \
    BLOCK_RED(op, uint32_t, uint32_t, u32)                                     \
    BLOCK_RED(op, uint32_t, uint64_t, u64)

// ----------------------------------------------------------------------------

BLOCK_RED_ALL_U(add)
BLOCK_RED_ALL(min)
BLOCK_RED_ALL(max)
BLOCK_RED_ALL(mul)

// ----------------------------------------------------------------------------

BLOCK_RED(or,  uint32_t, uint32_t, u32)
BLOCK_RED(and, uint32_t, uint32_t, u32)

// ----------------------------------------------------------------------------
