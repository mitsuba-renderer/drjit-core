/*
    resources/metal_kernels.metal — Precompiled MSL utility kernels.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    Supports: float, half, int, uint, ulong/long.
    Excluded: double (no float64 on Metal GPU).
*/

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_atomic>
using namespace metal;

// MSL doesn't provide simd_shuffle_xor for 64-bit types. Emulate via two
// 32-bit shuffles.
inline ulong simd_shuffle_xor(ulong v, ushort mask) {
    uint lo = (uint)(v & 0xFFFFFFFF);
    uint hi = (uint)(v >> 32);
    lo = simd_shuffle_xor(lo, mask);
    hi = simd_shuffle_xor(hi, mask);
    return ((ulong)hi << 32) | (ulong)lo;
}

inline long simd_shuffle_xor(long v, ushort mask) {
    return (long)simd_shuffle_xor((ulong)v, mask);
}

// ============================================================================
//  Reduction operators
// ============================================================================

// Only bitwise Or/And reductions still run as custom kernels; sum / product /
// minimum / maximum are dispatched through MPSGraph (see metal_ts.mm).

template <typename T> struct reduction_or {
    static T init() { return T(0); }
    static T apply(T a, T b) { return a | b; }
};

template <typename T> struct reduction_and {
    static T init() { return ~T(0); }
    static T apply(T a, T b) { return a & b; }
};

// ============================================================================
//  block_reduce kernel
//
//  Tree reduction using SIMD group shuffles + threadgroup shared memory.
//
//  1. Each thread loads one element (or identity if OOB).
//  2. SIMD-group reduction via simd_shuffle_xor.
//  3. If chunk > SIMD width, lane 0 writes to shared, then a second
//     SIMD reduction over the partials.
//  4. Thread 0 writes the scalar result.
// ============================================================================

struct block_reduce_params {
    ulong  in;          // gpuAddress of input buffer
    ulong  out;         // gpuAddress of output buffer
    uint   size;        // total number of input elements
    uint   block_size;  // how many elements per logical block
    uint   chunk_count; // total number of output chunks
};

template <typename T, typename Red, uint ChunkSize>
kernel void block_reduce_kernel(
    device const block_reduce_params &p [[buffer(0)]],
    threadgroup T *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]])
{
    constexpr uint SimdWidth = 32;
    constexpr bool MultiChunk = (ChunkSize < 128);
    constexpr uint ThreadsPerGroup = MultiChunk ? 128 : ChunkSize;

    device const T *in  = (device const T *) p.in;
    device T       *out = (device T *)       p.out;

    uint chunk, pos_in_chunk;
    if (MultiChunk) {
        uint chunks_per_tg = ThreadsPerGroup / ChunkSize;
        chunk        = tg_id * chunks_per_tg + tid / ChunkSize;
        pos_in_chunk = tid % ChunkSize;
    } else {
        chunk        = tg_id;
        pos_in_chunk = tid;
    }

    uint chunks_per_block = (p.block_size + ChunkSize - 1) / ChunkSize;
    uint block_id        = chunk / chunks_per_block;
    uint chunk_in_block  = chunk % chunks_per_block;
    uint pos_in_block    = chunk_in_block * ChunkSize + pos_in_chunk;
    uint pos_in_input    = block_id * p.block_size + pos_in_block;
    bool valid = (pos_in_block < p.block_size) && (pos_in_input < p.size);

    T value = valid ? in[pos_in_input] : Red::init();

    // SIMD-group reduction
    constexpr uint Active = (ChunkSize >= SimdWidth) ? SimdWidth : ChunkSize;
    for (uint k = 1; k < Active; k *= 2)
        value = Red::apply(value, simd_shuffle_xor(value, (ushort)k));

    constexpr uint ReducedChunkSize = ChunkSize / Active;

    if (ReducedChunkSize > 1) {
        if ((tid % SimdWidth) == 0)
            shared[tid / Active] = value;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid >= ThreadsPerGroup / Active)
            return;

        value = shared[tid];

        for (uint k = 1; k < ReducedChunkSize; k *= 2)
            value = Red::apply(value, simd_shuffle_xor(value, (ushort)k));

        if (MultiChunk) {
            pos_in_chunk = tid % ReducedChunkSize;
            uint chunks_per_tg = ThreadsPerGroup / ChunkSize;
            chunk = tg_id * chunks_per_tg + tid / ReducedChunkSize;
        }
    }

    if (pos_in_chunk == 0 && chunk < p.chunk_count)
        out[chunk] = value;
}

// ============================================================================
//  Instantiations (type × op × ChunkSize)
// ============================================================================

#define INSTANTIATE_RED(Name, T, Red)                                         \
    template [[host_name("block_reduce_" Name "_2")]]                         \
    kernel void block_reduce_kernel<T, Red<T>, 2>(                            \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_4")]]                         \
    kernel void block_reduce_kernel<T, Red<T>, 4>(                            \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_8")]]                         \
    kernel void block_reduce_kernel<T, Red<T>, 8>(                            \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_16")]]                        \
    kernel void block_reduce_kernel<T, Red<T>, 16>(                           \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_32")]]                        \
    kernel void block_reduce_kernel<T, Red<T>, 32>(                           \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_64")]]                        \
    kernel void block_reduce_kernel<T, Red<T>, 64>(                           \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_128")]]                       \
    kernel void block_reduce_kernel<T, Red<T>, 128>(                          \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_256")]]                       \
    kernel void block_reduce_kernel<T, Red<T>, 256>(                          \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_512")]]                       \
    kernel void block_reduce_kernel<T, Red<T>, 512>(                          \
        device const block_reduce_params &, threadgroup T *, uint, uint);     \
    template [[host_name("block_reduce_" Name "_1024")]]                      \
    kernel void block_reduce_kernel<T, Red<T>, 1024>(                         \
        device const block_reduce_params &, threadgroup T *, uint, uint);

// Only bitwise Or/And remain as custom kernels (MPSGraph's reductionAnd/Or
// accept boolean tensors only). uint8 covers boolean any()/all() reductions.
INSTANTIATE_RED("or_u32",   uint,   reduction_or)
INSTANTIATE_RED("and_u32",  uint,   reduction_and)
INSTANTIATE_RED("or_u64",   ulong,  reduction_or)
INSTANTIATE_RED("and_u64",  ulong,  reduction_and)
INSTANTIATE_RED("or_u8",    uchar,  reduction_or)
INSTANTIATE_RED("and_u8",   uchar,  reduction_and)

// ============================================================================
//  compress_scatter — materialize a compacted index list from a prefix sum.
//
//  ``MetalThreadState::compress`` first builds the inclusive uint32 prefix sum
//  of the boolean mask with MPSGraph; this kernel then writes the indices:
//  for every ``tid`` whose mask is set, store ``tid`` at ``out[prefix[tid]-1]``
//  (the inclusive prefix is 1-based, so the ``-1`` makes it 0-based).
// ============================================================================

struct compress_scatter_params {
    ulong in;      // device const uint8_t * (mask)
    ulong prefix;  // device const uint32_t * (inclusive prefix of the mask)
    ulong out;     // device uint32_t *       (compacted indices)
    uint  size;
};

kernel void compress_scatter(
    device const compress_scatter_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.size) return;
    device const uint8_t *in = (device const uint8_t *) p.in;
    if (!in[tid]) return;
    device const uint *prefix = (device const uint *) p.prefix;
    device uint *out          = (device uint *)       p.out;
    out[prefix[tid] - 1] = tid;
}

// ============================================================================
//  mkperm — compute a permutation that groups equal-valued uint32 entries
//
//  Simplified 3-phase approach (vs CUDA's 4-phase with transposition):
//    Phase 1: Global-memory atomic histogram
//    Phase 2: Exclusive prefix sum of histogram (via block_prefix_reduce,
//             called from C++)
//    Phase 3: Scatter indices into permutation array using atomics
//
//  This avoids shared-memory histogram complexity and is correct for all
//  bucket counts. Performance is adequate for typical Dr.Jit workloads
//  (virtual function dispatch with tens of buckets). A shared-memory
//  variant can be added later if profiling warrants it.
// ============================================================================

struct mkperm_params {
    ulong values;       // device const uint32_t *
    ulong buckets;      // device uint32_t * (histogram / prefix sums)
    ulong perm;         // device uint32_t * (output permutation)
    uint  size;
    uint  bucket_count;
};

kernel void mkperm_phase_1(
    device const mkperm_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.size) return;

    device const uint *values  = (device const uint *) p.values;
    device atomic_uint *buckets = (device atomic_uint *) p.buckets;

    uint val = values[tid];
    atomic_fetch_add_explicit(&buckets[val], 1u, memory_order_relaxed);
}

kernel void mkperm_phase_3(
    device const mkperm_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.size) return;

    device const uint *values  = (device const uint *) p.values;
    device atomic_uint *buckets = (device atomic_uint *) p.buckets;
    device uint *perm           = (device uint *)        p.perm;

    uint val = values[tid];
    uint pos = atomic_fetch_add_explicit(&buckets[val], 1u, memory_order_relaxed);
    perm[pos] = tid;
}

// ============================================================================
//  mkperm_offsets — detect non-empty buckets (Phase 2.5, optional)
//
//  Scans the prefix-summed histogram and writes (bucket_id, offset, count)
//  tuples for every non-empty bucket.
// ============================================================================

struct mkperm_offsets_params {
    ulong buckets;       // device const uint32_t * (prefix sums)
    ulong offsets;       // device uint4 * (output)
    ulong counter;       // device atomic_uint * (running count of non-empty)
    uint  bucket_count;
    uint  perm_size;
};

kernel void mkperm_detect_offsets(
    device const mkperm_offsets_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.bucket_count) return;

    device const uint *buckets = (device const uint *) p.buckets;
    device uint4 *offsets      = (device uint4 *)      p.offsets;
    device atomic_uint *counter = (device atomic_uint *) p.counter;

    uint a = buckets[tid];
    uint b = (tid + 1 < p.bucket_count) ? buckets[tid + 1] : p.perm_size;

    if (a != b) {
        uint idx = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        offsets[idx] = uint4(tid, a, b - a, 0);
    }
}

// ============================================================================
//  aggregate — populate a per-vcall data buffer from a list of entries.
//
//  Mirrors the host-side ``AggregationEntry`` (16 bytes, see jit.h):
//    {int32 size; uint32 offset; const void *src}
//
//  Each thread handles one entry. Sign of ``size`` selects mode:
//    - size > 0 : ``src`` packs the literal value; copy its low ``size`` bytes
//                 to ``dst + offset``.
//    - size < 0 : ``src`` is a device pointer; copy ``-size`` bytes from
//                 ``*src`` to ``dst + offset``.
//
//  Supported sizes: ±1, ±2, ±4, ±8 (matches the CUDA implementation in
//  resources/misc.cuh).
// ============================================================================

struct AggregationEntry {
    int   size;     // 4 bytes, offset 0
    uint  offset;   // 4 bytes, offset 4
    ulong src;      // 8 bytes, offset 8 (value if size>0, device addr if size<0)
};

kernel void aggregate_kernel(
    device uint8_t *dst [[buffer(0)]],
    device const AggregationEntry *entries [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    AggregationEntry e = entries[tid];
    device uint8_t *d = dst + e.offset;
    int sz = e.size;
    switch (sz) {
        case  1: *d                 = (uchar)  e.src; break;
        case  2: *(device ushort*)d = (ushort) e.src; break;
        case  4: *(device uint*)d   = (uint)   e.src; break;
        case  8: *(device ulong*)d  = (ulong)  e.src; break;
        case -1: *d                 = *(device const uchar*)  (e.src); break;
        case -2: *(device ushort*)d = *(device const ushort*) (e.src); break;
        case -4: *(device uint*)d   = *(device const uint*)   (e.src); break;
        case -8: *(device ulong*)d  = *(device const ulong*)  (e.src); break;
    }
}

// ============================================================================
//  memset — fill a buffer with a replicated 2/4/8-byte pattern (one element per
//  thread). The 1-byte case is handled by a blit ``fillBuffer`` on the host
//  side; a dedicated kernel per element width avoids a per-thread branch.
// ============================================================================

struct memset_params {
    ulong dst;     // device pointer (buffer gpuAddress + byte offset)
    ulong value;   // replicated pattern; only the low sizeof(T) bytes are used
};

template <typename T>
kernel void memset_kernel(
    device const memset_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    ((device T *) p.dst)[tid] = (T) p.value;
}

template [[host_name("memset_u16")]]
kernel void memset_kernel<ushort>(device const memset_params &, uint);
template [[host_name("memset_u32")]]
kernel void memset_kernel<uint>(device const memset_params &, uint);
template [[host_name("memset_u64")]]
kernel void memset_kernel<ulong>(device const memset_params &, uint);
