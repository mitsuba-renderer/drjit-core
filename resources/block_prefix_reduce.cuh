/*
    kernels/block_prefix_reduce.cuh -- Cooperative block prefix reduction kernel

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

/*
  This kernel uses a specified strategy ('Reduction') to prefix-reduce
  contiguous blocks of an input array 'in' and write the output to 'out'.

  The implementation is very similar to the block reduction in 'block_reduce.h'.
  Please start by reading the documentation there.

  The main difference is that reduced warps in the block reduction can simply exit,
  whereas every thread needs to write back a result here. This requires knowing
  the the reduced value of the predecessor.

  The communication scheme to resolve this dependency is based on

    "Single-pass Parallel Prefix Scan with Decoupled Look-back"
     Duane Merrill and Michael Garland

  This isn't the fastest possible CUDA prefix reduction kernel. But it is
  general and generates compact code. It can be instantiated in many variants
  that are shipped along with Dr.Jit.
*/

struct block_prefix_reduce_params {
    const void *in;
    void *scratch;
    void *out;
    uint32_t size;
    uint32_t block_size;
    uint32_t chunks_per_block;
    uint8_t x_is_block_id;
    uint8_t exclusive;
    uint8_t reverse;
};

/// ChunkSize is a power of two and potentially larger than the actual 'block_size'
template <typename Red, typename T, uint32_t ChunkSize>
__device__ void block_prefix_reduce(block_prefix_reduce_params params) {
    // If reducing chunks smaller than this threshold, do multiple of them per thread block
    constexpr uint32_t MultiChunkThreshold = 128;
    constexpr bool MultiChunk = ChunkSize < MultiChunkThreshold;

    // The thread block size is known at compile time
    constexpr uint32_t BlockDim = MultiChunk ? MultiChunkThreshold : ChunkSize;

    // When reducing large datasets, blocks may be split into multiple chunks
    constexpr bool Large = ChunkSize == 1024;

    // The reduction may use a higher intermediate precision
    using Value = typename Red::Value;

    // Shared memory storage uses this higher precision
    Value *shared = SharedMemory<Value>::get();
    const T * __restrict__ in = (T *) params.in;
    T * __restrict__ out = (T *) params.out;

    // Scratch buffer for communication between thread blocks
    using UInt = uint_with_size_t<sizeof(Value)>;
    UInt *scratch = (UInt *) params.scratch;

    // Query the thread and block index once
    uint32_t tid = threadIdx.x,
             bid_x = blockIdx.x,
             bid_y = blockIdx.y;

    uint32_t block;        // Thread's block index
    uint32_t rel_chunk;    // Thread's relative chunk index within block
    uint32_t chunk;        // Thread's chunk index
    uint32_t pos_in_chunk; // Thread's element index in chunk
    uint32_t tb_chunk;     // Chunk index of element 0 of the thread block

    if constexpr (Large) {
        // This is a large reduction with blocks that are further split into
        // chunks, whose IDs are encoded in the X/Y block index. The following
        // ensures compliance with CUDA launch requirements (Y grid size < 64K).

        block     = params.x_is_block_id ? bid_x : bid_y;
        rel_chunk = params.x_is_block_id ? bid_y : bid_x;

        chunk = block * params.chunks_per_block + rel_chunk;
        pos_in_chunk = tid;
        tb_chunk = chunk;
    } else {
        // This is a small reduction with 1 chunk per block

        if constexpr (MultiChunk) {
            // This kernel processes multiple chunks per thread block

            tb_chunk = bid_x * (BlockDim / ChunkSize);
            chunk = tb_chunk + tid / ChunkSize;
            pos_in_chunk = tid % ChunkSize;
        } else {
            // This kernel processes one chunk per block

            tb_chunk = bid_x;
            chunk = tb_chunk;
            pos_in_chunk = tid;
        }

        block = chunk;
        rel_chunk = 0;
    }

    // Index of this thread relative to the block, and input array
    uint32_t pos_in_block = rel_chunk * ChunkSize + pos_in_chunk;

    if (params.reverse)
        pos_in_block = params.block_size - 1 - pos_in_block;

    uint32_t pos_in_input = block * params.block_size + pos_in_block;

    // Are we in bounds within respect to the block and input array?
    bool is_valid = pos_in_block < params.block_size &&
                    pos_in_input < params.size;

    Red red;
    Value value = red.init();

    // Perform a coalesced load
    if (is_valid)
        value = (Value) in[pos_in_input];

    // Prefix sum using shared memory
    for (uint32_t i = 1; i < ChunkSize; i <<= 1) {
        shared[tid] = value;
        __syncthreads();

        if (pos_in_chunk - i < ChunkSize)
            value = red(value, shared[tid - i]);

        __syncthreads();
    }

    // Wait for the result of predecessor blocks
    Value prefix = red.init();
    if (Large && scratch) {
        // Advance to the churrent chunk's position within the scratch space
        scratch += chunk * 2;

        // The leader holds the chunk's reduced value
        bool is_leader = tid == 1023;

        if (is_leader)
            store_with_status(scratch, memcpy_cast<UInt>(value), 1);

        // Each thread looks back a different amount
        uint32_t lane = tid & (WarpSize - 1);
        int32_t shift = lane - WarpSize;

        // Decoupled look-back iteration
        while (true) {
            uint32_t status;
            Value pred;
            UInt pred_u;

            if ((int32_t) rel_chunk + shift >= 0) {
                load_with_status(scratch + 2*shift, pred_u, status);
                pred = memcpy_cast<Value>(pred_u);
            } else {
                pred = red.init();
                status = 2;
            }

            // Retry if at least one of the predecessors hasn't made any progress yet
            if (__any_sync(WarpMask, status == 0))
                continue;

            uint32_t mask = __ballot_sync(WarpMask, status == 2);
            if (mask == 0) {
                // Sum partial results, look back further
                prefix = red(prefix, pred);
                shift -= WarpSize;
            } else {
                // Lane 'index' is done!
                uint32_t index = 31 - __clz(mask);

                // Sum up all the unconverged (higher) lanes *and* 'index'
                if (lane >= index)
                    prefix = red(prefix, pred);

                break;
            }
        }

        // Warp-level sum reduction of 'prefix'
        for (uint32_t i = 1; i < WarpSize; i *= 2)
            prefix = red(prefix, __shfl_xor_sync(WarpMask, prefix, i));

        value = red(value, prefix);

        if (is_leader)
            store_with_status(scratch, memcpy_cast<UInt>(value), 2);
    }

    // Write reduced result back to global memory
    if (is_valid) {
        if (params.exclusive) {
            shared[tid] = value;
            __syncthreads();
            value = pos_in_chunk == 0 ? prefix : shared[tid - 1];
        }

        out[pos_in_input] = (T) value;
    }
}

// ----------------------------------------------------------------------------

#define BLOCK_P_RED_1(Op, T, TName, BSize)                                     \
    KERNEL void block_prefix_reduce_##Op##_##TName##_##BSize(                  \
        block_prefix_reduce_params params) {                                   \
        block_prefix_reduce<reduction_##Op<T>, T, BSize>(params);              \
    }

#define BLOCK_P_RED(Op, T, TName)                                              \
    BLOCK_P_RED_1(Op, T, TName, 1024)                                          \
    BLOCK_P_RED_1(Op, T, TName, 512)                                           \
    BLOCK_P_RED_1(Op, T, TName, 256)                                           \
    BLOCK_P_RED_1(Op, T, TName, 128)                                           \
    BLOCK_P_RED_1(Op, T, TName, 64)                                            \
    BLOCK_P_RED_1(Op, T, TName, 32)                                            \
    BLOCK_P_RED_1(Op, T, TName, 16)                                            \
    BLOCK_P_RED_1(Op, T, TName, 8)                                             \
    BLOCK_P_RED_1(Op, T, TName, 4)                                             \
    BLOCK_P_RED_1(Op, T, TName, 2)

#define BLOCK_P_RED_ALL(Op)                                                    \
    BLOCK_P_RED(Op, half, f16)                                                 \
    BLOCK_P_RED(Op, float, f32)                                                \
    BLOCK_P_RED(Op, double, f64)                                               \
    BLOCK_P_RED(Op, uint32_t, u32)                                             \
    BLOCK_P_RED(Op, uint32_t, u64)                                             \
    BLOCK_P_RED(Op, int32_t, i32)                                              \
    BLOCK_P_RED(Op, int64_t, i64)

/// Skip signed integer versions
#define BLOCK_P_RED_ALL_2(Op)                                                  \
    BLOCK_P_RED(Op, half, f16)                                                 \
    BLOCK_P_RED(Op, float, f32)                                                \
    BLOCK_P_RED(Op, double, f64)                                               \
    BLOCK_P_RED(Op, uint32_t, u32)                                             \
    BLOCK_P_RED(Op, uint64_t, u64)

// ----------------------------------------------------------------------------

BLOCK_P_RED_ALL_2(add)
BLOCK_P_RED_ALL_2(mul)
BLOCK_P_RED_ALL(min)
BLOCK_P_RED_ALL(max)

// ----------------------------------------------------------------------------

BLOCK_P_RED(or, uint32_t, u32)
BLOCK_P_RED(or, uint64_t, u64)
BLOCK_P_RED(and, uint32_t, u32)
BLOCK_P_RED(and, uint64_t, u64)

// ----------------------------------------------------------------------------
