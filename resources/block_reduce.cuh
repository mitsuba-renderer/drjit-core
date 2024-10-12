/*
    kernels/block_reduce.cuh -- Cooperative block reduction kernel

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

/*
  This kernel uses a specified strategy (``Red``) to reduce contiguous
  blocks of an input array ``in`` to single values that are written to ``out``.
  The higher-level task is to reduce blocks of size ``block_size`` within an
  array of size ``size``, and to do so with reasonable efficiently in all cases.

  Note the distinction between *blocks* (of the reduction) and *thread blocks*
  (CUDA cooperative work units) in the following.

  Several aspects are noteworthy:

  - Large blocks (e.g. ``block_size == size``) are too big to be efficiently
    handled by a single thread block. Each block is therefore considered to be
    further split into *chunks* that can be mapped to thread blocks. The output
    written by this kernel is proportional to the chunk count, and a subsequent
    reduction is needed if there are several chunks per block.

    The ``ChunkSize`` template parameter denotes the maximum chunk size
    supported by the kernel. It must be a power of two, as the kernel uses a
    tree-based reduction strategy. The code below instantiates versions with
    chunk sizes ranging from 2 to 1024.

  - Tiny blocks (e.g. ``block_size == 2``) can be directly handled (i.e., there
    is no difference between chunks and blocks), but will lead to poor GPU
    occupancy if mapped 1:1 onto thread blocks. In this case, each thread block
    processes multiple chunks in parallel (potentially up to 128).

  - Things may not divide up cleanly: ``size`` may not be divisible by
    ``block_size``, and ``block_size`` may not be divisible by ``ChunkSize``.
    In this case, trailing chunks/blocks are considered shorter.

  - When both ``size`` and ``block_size`` multipled by the element size are
    divisible by 16 bytes, it can be advantageous if each thread loads a
    packet of values. This is realized for large chunks (`ChunkSize = 1024`)
    by a separate vectorized version (with ``_vec`` suffix).

    The vectorized version maxes out the GPU bandwidth (tested on an A6000,
    where it was as fast as CUB). The non-vectorized version is ~2x slower.

  Two examples:

    - ``size=1000847``, ``block_size=3`` (both prime). The reduction will
      process ``333615`` chunks/blocks with ``ChunkSize = 4``.

      Since a thread block with only 4 threads would yield too little occpancy,
      Dr.Jit will assign 64 chunks to each thread block and launch a total of
      524 thread blocks with 256 threads each.

      Some of the thread blocks contain chunks that are out of bounds and must
      be ignored. Furthermore, each chunk will only read 3 instead of 4 entries,
      and the last one only has two entries.

    - ``size=1000847``, ``block_size=11923`` (both prime). We will process
      ``84`` blocks, which are each split into 12 chunks with ``ChunkSize =
      1024``. However, the last chunk in each block has fewer elements, and the
      final block has a trailing chunk that doesn't overlap with the input data.
      A subsequent reduction over 1008 chunk outputs with block size 12 produces
      the result.

  Relevant CUDA configuration parameters of this kernel are:

  - ChunkSize: a power of two
  - Thread count: >= max(ChunkSize / VectorWidth, 32)
  - X/Y block count: number of blocks and chunks. To comply with CUDA
    launch size requirements, their position may be swapped.

  In comparison to CUB, the implementation is better-adapted to the needs of
  Dr.Jit (contiguous block reduction), while generating significantly less code
  when instantiated in the many necessary variants.
*/

struct block_reduce_params {
    const void *in;
    void *out;
    uint32_t size;
    uint32_t block_size;
    uint32_t chunks_per_block;
    uint32_t chunk_count;
    uint8_t x_is_block_id;
};

template <typename Red, typename T, typename Tl, uint32_t ChunkSize_>
__device__ void block_reduce(block_reduce_params params) {
    // Vector loads fetch multiple values at once
    constexpr uint32_t VectorWidth = sizeof(Tl) / sizeof(T);

    // How many load instructions does one thread block issue in total?
    constexpr uint32_t ChunkSize = ChunkSize_ / VectorWidth;

    // If reducing chunks smaller than this threshold, do multiple of them per thread block
    constexpr uint32_t MultiChunkThreshold = 128;
    constexpr bool MultiChunk = ChunkSize < MultiChunkThreshold;

    // The thread block size is known at compile time
    constexpr uint32_t BlockDim = MultiChunk ? MultiChunkThreshold : ChunkSize;

    // When reducing large datasets, blocks may be split into multiple chunks
    constexpr bool Large = ChunkSize_ == 1024;

    // The reduction may use a higher intermediate precision
    using Value = typename Red::Value;

    // Shared memory storage uses this higher precision
    Value *shared = SharedMemory<Value>::get();
    const T * __restrict__ in = (T *) params.in;
    T * __restrict__ out = (T *) params.out;

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
    uint32_t pos_in_block = rel_chunk * ChunkSize + pos_in_chunk,
             pos_in_input = block * params.block_size + pos_in_block;

    // Are we in bounds within respect to the block and input array?
    bool is_valid = pos_in_block < params.block_size &&
                    pos_in_input < params.size;

    Red red;
    Value value = red.init();

    // Perform a coalesced load, potentially with vectorization
    if (is_valid) {
        if constexpr (std::is_same<T, Tl>::value) {
            value = (Value) in[pos_in_input];
        } else {
            uint4 vec_load = ((uint4 *) in)[pos_in_input];
            value = memcpy_cast<Tl>(vec_load).reduce(red);
        }
    }

    // Number of contiguous threads in this warp that process the same chunk
    constexpr uint32_t Active = ChunkSize >= WarpSize ? WarpSize : ChunkSize;

    // This first warp reduction brings the chunk size down to 'ReducedChunkSize'
    constexpr uint32_t ReducedChunkSize = ChunkSize / Active;
    for (uint32_t k = 1; k < Active; k *= 2)
        value = red(value, __shfl_xor_sync(WarpMask, value, k, Active));

    // To reduce further, we need shared memory (happens when ChunkSize > 32)
    if constexpr (ReducedChunkSize > 1) {
        // Lane 0 in each warp writes back its result
        if ((tid & (WarpSize - 1)) == 0)
            shared[tid / Active] = value;

        // Most of the thread block can quit after this step
        if (tid >= BlockDim / Active)
            return;

        // Load the data right back
        __syncthreads();

        value = shared[tid];

        for (uint32_t k = 1; k < ReducedChunkSize; k *= 2)
            value = red(value, __shfl_xor_sync(WarpMask, value, k, ReducedChunkSize));

        // Different threads handle different chunks, and we moved their state around
        // Need to update the indices.
        if constexpr (MultiChunk) {
            pos_in_chunk = tid % ReducedChunkSize;
            chunk = tb_chunk + tid / ReducedChunkSize;
        }
    }

    if (pos_in_chunk == 0 && (!MultiChunk || chunk < params.chunk_count))
        out[chunk] = (T) value;
}

// ----------------------------------------------------------------------------

#define BLOCK_RED_1(Op, T, Vec, TName, BSize)                                  \
    KERNEL void block_reduce_##Op##_##TName##_##BSize(block_reduce_params p) { \
        block_reduce<reduction_##Op<T>, T, Vec, BSize>(p);                     \
    }

#define BLOCK_RED(Op, T, Vec, TName)                                           \
    BLOCK_RED_1(Op, T, T, TName, 1024)                                         \
    BLOCK_RED_1(Op, T, T, TName, 512)                                          \
    BLOCK_RED_1(Op, T, T, TName, 256)                                          \
    BLOCK_RED_1(Op, T, T, TName, 128)                                          \
    BLOCK_RED_1(Op, T, T, TName, 64)                                           \
    BLOCK_RED_1(Op, T, T, TName, 32)                                           \
    BLOCK_RED_1(Op, T, T, TName, 16)                                           \
    BLOCK_RED_1(Op, T, T, TName, 8)                                            \
    BLOCK_RED_1(Op, T, T, TName, 4)                                            \
    BLOCK_RED_1(Op, T, T, TName, 2)                                            \
    BLOCK_RED_1(Op, T, Vec<T>, TName##_vec, 1024)                              \

#define BLOCK_RED_ALL(Op)                                                      \
    BLOCK_RED(Op, half, Vec8, f16)                                             \
    BLOCK_RED(Op, float, Vec4, f32)                                            \
    BLOCK_RED(Op, double, Vec2, f64)                                           \
    BLOCK_RED(Op, uint32_t, Vec4, u32)                                         \
    BLOCK_RED(Op, uint64_t, Vec2, u64)                                         \
    BLOCK_RED(Op, int32_t, Vec4, i32)                                          \
    BLOCK_RED(Op, int64_t, Vec2, i64)

/// Skip signed integer versions
#define BLOCK_RED_ALL_2(Op)                                                    \
    BLOCK_RED(Op, half, Vec8, f16)                                             \
    BLOCK_RED(Op, float, Vec4, f32)                                            \
    BLOCK_RED(Op, double, Vec2, f64)                                           \
    BLOCK_RED(Op, uint32_t, Vec4, u32)                                         \
    BLOCK_RED(Op, uint64_t, Vec2, u64)

// ----------------------------------------------------------------------------

BLOCK_RED_ALL_2(add)
BLOCK_RED_ALL_2(mul)
BLOCK_RED_ALL(min)
BLOCK_RED_ALL(max)

// ----------------------------------------------------------------------------

BLOCK_RED(or, uint32_t, Vec4, u32)
BLOCK_RED(or, uint64_t, Vec2, u64)
BLOCK_RED(and, uint32_t, Vec4, u32)
BLOCK_RED(and, uint64_t, Vec2, u64)

// ----------------------------------------------------------------------------
