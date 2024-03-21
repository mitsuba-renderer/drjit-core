/*
    kernels/block_sum.cuh -- Sum reduction within blocks

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <cuda_fp16.h>

/// Based on 'reduce4' example from the NVIDIA SDK
template <typename Ts, typename Tv, uint32_t BlockSize>
__device__ void block_sum(const Ts *in, Ts *out) {
    Tv *shared = SharedMemory<Tv>::get();

    uint32_t tid = threadIdx.x,
             i   = blockIdx.x * (BlockSize * 2) + tid;

    Tv sum = Tv(in[i]) + Tv(in[i + BlockSize]);

    if (BlockSize >= 64) {
        shared[tid] = sum;
        __syncthreads();
    }

    // Begin with tree reduction using shared memory
    for (uint32_t s = BlockSize / 2; s > 32; s >>= 1) {
        if (tid < s)
            shared[tid] = sum = sum + shared[tid + s];
        __syncthreads();
    }

    // .. then switch over to warp-local reduction
    if (tid < 32) {
        // Fetch final intermediate sum from 2nd warp
        if (BlockSize >= 64)
            sum += shared[tid + 32];

        constexpr uint32_t Active = BlockSize >= 32 ? 32 : BlockSize;
        constexpr uint32_t ActiveMask = (uint32_t) ((((size_t) 1) << Active) - 1);

        for (uint32_t i = 1; i < Active; i *= 2)
            sum += __shfl_xor_sync(ActiveMask, sum, i);
    }

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = Ts(sum);
}

#define BLOCK_SUM_1(Ts, Tv, tname, BlockSize)                                  \
    KERNEL void block_sum_##tname##_##BlockSize(const Ts *in, Ts *out) {       \
        block_sum<Ts, Tv, BlockSize>(in, out);                                 \
    }

#define BLOCK_SUM(Ts, Tv, tname)                                               \
    BLOCK_SUM_1(Ts, Tv, tname, 512)                                            \
    BLOCK_SUM_1(Ts, Tv, tname, 256)                                            \
    BLOCK_SUM_1(Ts, Tv, tname, 128)                                            \
    BLOCK_SUM_1(Ts, Tv, tname, 64)                                             \
    BLOCK_SUM_1(Ts, Tv, tname, 32)                                             \
    BLOCK_SUM_1(Ts, Tv, tname, 16)                                             \
    BLOCK_SUM_1(Ts, Tv, tname, 8)                                              \
    BLOCK_SUM_1(Ts, Tv, tname, 4)                                              \
    BLOCK_SUM_1(Ts, Tv, tname, 2)                                              \
    BLOCK_SUM_1(Ts, Tv, tname, 1)

BLOCK_SUM(half, float, f16)
BLOCK_SUM(float, float, f32)
BLOCK_SUM(double, double, f64)
BLOCK_SUM(uint32_t, uint32_t, u32)
BLOCK_SUM(uint32_t, uint64_t, u64)
