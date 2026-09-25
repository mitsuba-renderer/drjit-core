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

  Each CUDA thread block processes a tile of 2048 elements using a coalesced
  load, a sequential per-thread scan, and warp shuffles. Blocks that are
  smaller than a tile are packed into tiles (as many whole blocks as fit), and
  the scan carries segment-head flags so that the running value restarts at
  each block boundary. Blocks larger than a tile are split into multiple
  tiles that communicate through a scratch buffer using a variant of

    "Single-pass Parallel Prefix Scan with Decoupled Look-back"
     Duane Merrill and Michael Garland

  The classic decoupled look-back produces results that depend on the timing
  of thread blocks: each tile combines whatever mix of aggregates and
  inclusive prefixes its predecessors have published so far. For
  non-associative operations (floating point addition and multiplication),
  this makes the output non-deterministic. The variant here instead publishes
  partial sums organized as a radix-32 tree and always combines them in the
  same order, which yields bitwise reproducible results at the same cost.
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

// Launch configuration. The host code in 'cuda_ts.cpp' depends on these values.
#define BlockPrefixReduceThreads 256
#define BlockPrefixReduceItems 8
#define BlockPrefixReduceTile (BlockPrefixReduceThreads * BlockPrefixReduceItems)

/// Scratch entries needed per block for the radix-32 partial sum tree
__device__ inline uint32_t block_prefix_reduce_scratch_stride(uint32_t chunks_per_block) {
    uint32_t stride = 0;
    for (uint32_t n = chunks_per_block;; n = (n + 31) >> 5) {
        stride += n;
        if (n <= 32)
            break;
    }
    return stride;
}

/// Padded shared memory index that avoids bank conflicts in the transposes
__device__ inline uint32_t block_prefix_reduce_pad(uint32_t i) {
    return i + (i >> 5);
}

template <typename Red, typename T>
__device__ void block_prefix_reduce(block_prefix_reduce_params params) {
    constexpr uint32_t Threads = BlockPrefixReduceThreads,
                       Items = BlockPrefixReduceItems,
                       Tile = BlockPrefixReduceTile,
                       Warps = Threads / WarpSize;

    using Value = typename Red::Value;
    using UInt = uint_with_size_t<sizeof(Value)>;

    // Shared memory: per-warp head flags, then the padded tile and per-warp values
    uint32_t *warp_flag = (uint32_t *) SharedMemory<Value>::get();
    uint32_t *warp_fexcl = warp_flag + Warps;
    Value *shared = (Value *) (warp_fexcl + Warps);
    Value *warp_tot = shared + block_prefix_reduce_pad(Tile);
    Value *warp_excl = warp_tot + Warps;
    Value *tile_prefix_s = warp_excl + Warps;

    Red red;
    uint32_t tid = threadIdx.x,
             lane = tid & (WarpSize - 1),
             warp = tid / WarpSize;

    // Block and chunk IDs are encoded in the X/Y block index. The following
    // ensures compliance with CUDA launch requirements (Y grid size < 64K).
    uint32_t bid       = params.x_is_block_id ? blockIdx.x : blockIdx.y,
             rel_chunk = params.x_is_block_id ? blockIdx.y : blockIdx.x;

    uint32_t block_size = params.block_size;
    bool large = block_size > Tile, reverse = params.reverse;

    // Segment structure of the tile. Small blocks form 'seg_count' segments of
    // 'seg_len' elements each, and tile 'bid' covers blocks starting at
    // 'bid * seg_count'. Large blocks consist of multiple tiles, and tile
    // 'rel_chunk' of block 'bid' is a single segment.
    uint32_t seg_len = large ? Tile : block_size,
             seg_count = large ? 1 : Tile / block_size;

    // The tile loads and stores a contiguous memory range 'in[base + lstart +
    // j]' for j < Tile, of which entries with 'lstart + j < limit' are valid.
    // In reverse mode, the scan order runs against memory order within each
    // segment, which is handled by mirroring indices in shared memory.
    size_t base;
    uint32_t lstart, limit;
    if (large) {
        uint32_t tile_start = rel_chunk * Tile;
        base = (size_t) bid * block_size;
        lstart = reverse ? block_size - tile_start - Tile : tile_start;
        limit = min(block_size, params.size - bid * block_size);
    } else {
        uint32_t tile_base = bid * seg_count * block_size;
        base = tile_base;
        lstart = 0;
        limit = min(seg_count * block_size, params.size - tile_base);
    }

    const T * __restrict__ in = (const T *) params.in + base;
    T * __restrict__ out = (T *) params.out + base;

    // Coalesced load, staged through shared memory
    #pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
        uint32_t j = i * Threads + tid, off = lstart + j;
        Value v = red.init();
        if (off < limit)
            v = (Value) in[off];
        shared[block_prefix_reduce_pad(j)] = v;
    }
    __syncthreads();

    // Each thread scans 'Items' consecutive elements in scan order and writes
    // the local result back in place. Position 'p' lies in segment 'seg' at
    // offset 'pos'. The running value restarts at segment heads, and
    // 'first_head' records the first head within the thread (elements before
    // it continue the segment of the preceding threads). This loop and the
    // one applying the prefix are not unrolled to keep the kernel small.
    uint32_t p0 = tid * Items, seg = p0 / seg_len, pos = p0 - seg * seg_len;
    Value agg = red.init();
    uint32_t first_head = Items;
    #pragma unroll 1
    for (uint32_t i = 0; i < Items; ++i) {
        uint32_t r = block_prefix_reduce_pad(
            (reverse && seg < seg_count) ? seg * seg_len + (seg_len - 1 - pos) : p0 + i);
        Value v = shared[r], prev = agg;
        bool head = !large && pos == 0;

        agg = head ? v : red(agg, v);
        shared[r] = params.exclusive ? (head ? red.init() : prev) : agg;
        if (head)
            first_head = min(first_head, i);

        if (++pos == seg_len) {
            pos = 0;
            seg++;
        }
    }

    // Segmented inclusive warp scan of the thread aggregates. After the loop,
    // 'inc' holds the reduction since the last head, and 'flag' indicates
    // whether a head occurred at or before this lane.
    Value inc = agg;
    bool flag = first_head != Items;
    #pragma unroll
    for (uint32_t d = 1; d < WarpSize; d *= 2) {
        Value n = __shfl_up_sync(WarpMask, inc, d);
        bool nf = __shfl_up_sync(WarpMask, (uint32_t) flag, d);
        if (lane >= d) {
            if (!flag)
                inc = red(n, inc);
            flag |= nf;
        }
    }
    if (lane == WarpSize - 1) {
        warp_tot[warp] = inc;
        warp_flag[warp] = flag;
    }
    __syncthreads();

    Value tile_prefix = red.init();
    if (warp == 0) {
        // Segmented scan of the warp totals and the tile aggregate
        Value w_inc = lane < Warps ? warp_tot[lane] : red.init();
        bool w_flag = lane < Warps ? warp_flag[lane] : false;
        #pragma unroll
        for (uint32_t d = 1; d < WarpSize; d *= 2) {
            Value n = __shfl_up_sync(WarpMask, w_inc, d);
            bool nf = __shfl_up_sync(WarpMask, (uint32_t) w_flag, d);
            if (lane >= d) {
                if (!w_flag)
                    w_inc = red(n, w_inc);
                w_flag |= nf;
            }
        }
        Value w_excl = __shfl_up_sync(WarpMask, w_inc, 1);
        bool w_fexcl = __shfl_up_sync(WarpMask, (uint32_t) w_flag, 1);
        if (lane < Warps) {
            warp_excl[lane] = lane == 0 ? red.init() : w_excl;
            warp_fexcl[lane] = lane == 0 ? false : w_fexcl;
        }
        Value tile_agg = __shfl_sync(WarpMask, w_inc, Warps - 1);

        if (params.scratch) {
            UInt *scratch = (UInt *) params.scratch +
                            (size_t) bid * block_prefix_reduce_scratch_stride(params.chunks_per_block) * 2;

            // Publish the tile aggregate at level 0 as early as possible
            if (lane == 0)
                store_with_status(scratch + rel_chunk * 2, memcpy_cast<UInt>(tile_agg), 1);

            // Deterministic look-back. The scratch buffer holds a radix-32
            // tree of partial sums: level 0 has one entry per tile, level l+1
            // one entry per complete group of 32 level-l entries. The tile's
            // exclusive prefix is the sum over its base-32 digits of the
            // preceding siblings at each level. Each level is reduced by a
            // fixed warp butterfly and the levels are accumulated in a fixed
            // order, so the result does not depend on timing. A tile publishes
            // the level-(l+1) entry of its group if its digits at all levels up
            // to l equal 31, i.e., if it is the last tile of that group.
            Value own = tile_agg;
            uint32_t g = rel_chunk, n = params.chunks_per_block, level_base = 0;
            bool closes = true;

            while (true) {
                uint32_t digit = g & 31, group = g >> 5, status;
                Value v;

                // Wait until the preceding siblings at this level are available
                do {
                    v = red.init();
                    status = 1;
                    if (lane < digit) {
                        UInt u;
                        load_with_status(scratch + (level_base + group * 32 + lane) * 2, u, status);
                        v = memcpy_cast<Value>(u);
                    }
                } while (__any_sync(WarpMask, status == 0));

                #pragma unroll
                for (uint32_t i = 1; i < WarpSize; i *= 2)
                    v = red(v, __shfl_xor_sync(WarpMask, v, i));

                tile_prefix = red(tile_prefix, v);
                own = red(v, own);
                closes = closes && digit == 31;

                if (n <= 32)
                    break;

                // Advance to the next level and publish the group total if needed
                level_base += n;
                n = (n + 31) >> 5;
                if (closes && lane == 0)
                    store_with_status(scratch + (level_base + group) * 2, memcpy_cast<UInt>(own), 1);

                if (group == 0)
                    break;

                g = group;
            }
        }

        if (lane == 0)
            *tile_prefix_s = tile_prefix;
    }
    __syncthreads();

    // Prefix of the segment that continues into this thread's first elements
    Value thread_excl = __shfl_up_sync(WarpMask, inc, 1);
    bool thread_fexcl = __shfl_up_sync(WarpMask, (uint32_t) flag, 1);
    if (lane == 0) {
        thread_excl = red.init();
        thread_fexcl = false;
    }
    Value prefix = warp_fexcl[warp] ? warp_excl[warp]
                                    : red(*tile_prefix_s, warp_excl[warp]);
    prefix = thread_fexcl ? thread_excl : red(prefix, thread_excl);

    // Apply the prefix to the elements before the first head, then store
    seg = p0 / seg_len;
    pos = p0 - seg * seg_len;
    #pragma unroll 1
    for (uint32_t i = 0; i < first_head; ++i) {
        uint32_t r = block_prefix_reduce_pad(
            (reverse && seg < seg_count) ? seg * seg_len + (seg_len - 1 - pos) : p0 + i);
        shared[r] = red(prefix, shared[r]);
        if (++pos == seg_len) {
            pos = 0;
            seg++;
        }
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
        uint32_t j = i * Threads + tid, off = lstart + j;
        if (off < limit)
            out[off] = (T) shared[block_prefix_reduce_pad(j)];
    }
}

// ----------------------------------------------------------------------------

#define BLOCK_P_RED(Op, T, TName)                                              \
    KERNEL void __launch_bounds__(BlockPrefixReduceThreads)                    \
        block_prefix_reduce_##Op##_##TName(block_prefix_reduce_params params) {\
        block_prefix_reduce<reduction_##Op<T>, T>(params);                     \
    }

#define BLOCK_P_RED_ALL(Op)                                                    \
    BLOCK_P_RED(Op, half, f16)                                                 \
    BLOCK_P_RED(Op, float, f32)                                                \
    BLOCK_P_RED(Op, double, f64)                                               \
    BLOCK_P_RED(Op, uint32_t, u32)                                             \
    BLOCK_P_RED(Op, uint64_t, u64)                                             \
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
