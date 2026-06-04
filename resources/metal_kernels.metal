/*
    resources/metal_kernels.metal — Precompiled MSL utility kernels.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_atomic>
using namespace metal;

// ============================================================================
//  any() / all() — reduce a boolean mask to a single value
//
//  ``reduce_*_init`` seeds the result with the identity;
//  ``reduce_all``/``reduce_any`` then flip it via an atomic store on a hit. Each
//  thread tests one coalesced 16-byte (uint4) packet, and a per-threadgroup
//  reduction issues at most one store.
// ============================================================================

struct reduce_bool_params {
    ulong in;
    ulong out;
    uint  size;
};

template <bool IsAll>
kernel void reduce_bool_init(constant reduce_bool_params &p [[buffer(0)]]) {
    *(device uint *) p.out = IsAll ? 1u : 0u;
}

template <bool IsAll>
kernel void reduce_bool_kernel(
    constant reduce_bool_params &p [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
    device const uchar *in  = (device const uchar *) p.in;
    device atomic_uint *out = (device atomic_uint *) p.out;

    threadgroup atomic_uint tg_found;
    if (lid == 0)
        atomic_store_explicit(&tg_found, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One coalesced 16-byte (uint4) packet per thread.
    bool found = false;
    uint n16 = p.size >> 4;
    if (gid < n16) {
        uint4 w = ((device const uint4 *) in)[gid];
        found = IsAll ? !all(w == 0x01010101u) : any(w != 0u);
    }

    // Thread 0 also mops up the < 16 tail bytes that don't fill a packet.
    if (gid == 0) {
        for (uint i = n16 << 4; i < p.size; ++i)
            found = found || (IsAll ? (in[i] == 0u) : (in[i] != 0u));
    }

    // OR 'found' across the threadgroup so lid 0 issues a single atomic store.
    if (found)
        atomic_store_explicit(&tg_found, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0 && atomic_load_explicit(&tg_found, memory_order_relaxed))
        atomic_store_explicit(out, IsAll ? 0u : 1u, memory_order_relaxed);
}

template [[host_name("reduce_all_init")]]
kernel void reduce_bool_init<true>(constant reduce_bool_params &);
template [[host_name("reduce_any_init")]]
kernel void reduce_bool_init<false>(constant reduce_bool_params &);

template [[host_name("reduce_all")]]
kernel void reduce_bool_kernel<true>(
    constant reduce_bool_params &, uint, uint);
template [[host_name("reduce_any")]]
kernel void reduce_bool_kernel<false>(
    constant reduce_bool_params &, uint, uint);

// ============================================================================
//  compress_scatter — materialize a compacted index list from a prefix sum.
// ============================================================================

struct compress_scatter_params {
    ulong in;      // device const uint8_t * (mask)
    ulong prefix;  // device const uint32_t * (inclusive prefix of the mask)
    ulong out;     // device uint32_t *       (compacted indices)
    uint  size;
};

kernel void compress_scatter(
    constant compress_scatter_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    device const uint8_t *in = (device const uint8_t *) p.in;
    if (!in[tid]) return;
    device const uint *prefix = (device const uint *) p.prefix;
    device uint *out          = (device uint *)       p.out;
    out[prefix[tid] - 1] = tid;
}

// ============================================================================
//  mkperm — group equal-valued uint32 entries into a permutation.
//
//  Global-atomic fallback (unstable) used when bucket_count is too large for
//  the stable "tiny" variant below. Phases: atomic histogram, exclusive prefix
//  sum (block_prefix_reduce, from C++), atomic scatter.
// ============================================================================

struct mkperm_params {
    ulong values;       // device const uint32_t *
    ulong buckets;      // device uint32_t * (histogram / prefix sums)
    ulong perm;         // device uint32_t * (output permutation)
    uint  size;
    uint  bucket_count;
};

kernel void mkperm_phase_1(
    constant mkperm_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    device const uint *values  = (device const uint *) p.values;
    device atomic_uint *buckets = (device atomic_uint *) p.buckets;

    uint val = values[tid];
    atomic_fetch_add_explicit(&buckets[val], 1u, memory_order_relaxed);
}

kernel void mkperm_phase_3(
    constant mkperm_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
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
    uint  row_stride;    // stride between consecutive bucket bases
};

kernel void mkperm_detect_offsets(
    constant mkperm_offsets_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    device const uint *buckets = (device const uint *) p.buckets;
    device uint4 *offsets      = (device uint4 *)      p.offsets;
    device atomic_uint *counter = (device atomic_uint *) p.counter;

    uint a = buckets[tid * p.row_stride];
    uint b = (tid + 1 < p.bucket_count) ? buckets[(tid + 1) * p.row_stride]
                                        : p.perm_size;

    if (a != b) {
        uint idx = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        offsets[idx] = uint4(tid, a, b - a, 0);
    }
}

// ============================================================================
//  mkperm (stable "tiny" variant) — per-SIMD-group threadgroup histograms.
//
//  Each SIMD group keeps a private histogram in threadgroup memory and
//  processes a contiguous element range, so a per-group segmented exclusive
//  prefix sum over the bucket-major layout
//
//      buckets[group*seg + bucket*rows_per_group + (sub_block*warp_count + warp)]
//
//  (seg = rows_per_group * bucket_count) yields stable per-row base offsets.
// ============================================================================

struct mkperm_tiny_params {
    ulong values;          // device const uint *
    ulong buckets;         // device uint *
    ulong perm;            // device uint *
    uint  size;
    uint  size_per_block;  // elements per sub-block (warp-aligned)
    uint  bucket_count;
    uint  block_size;      // user block size
    uint  rows_per_group;  // gpu_blocks_per_group * warp_count
};

// Bitmask of active SIMD lanes that share 'value', found in O(key_bits) ballots
// by intersecting per-bit agreement masks across the group.
static inline uint mkperm_match_any(uint value, uint key_bits) {
    uint peers = (uint) ((ulong) simd_active_threads_mask());
    for (uint b = 0; b < key_bits; ++b) {
        uint bit  = (value >> b) & 1u;
        uint ones = (uint) ((ulong) simd_ballot(bit != 0u));
        peers &= bit ? ones : ~ones;
    }
    return peers;
}

// Add 'value' to a warp-private histogram (one update per peer group).
static inline void mkperm_count(uint value, uint lane, uint key_bits,
                                threadgroup uint *buckets) {
    uint peers = mkperm_match_any(value, key_bits);
    if (lane == ctz(peers))
        buckets[value] += popcount(peers);
}

// Accumulate 'value' into a warp-private histogram with a single update per
// peer group; returns this lane's slot (base offset + rank among its peers).
static inline uint mkperm_reduce(uint value, uint lane, uint key_bits,
                                 threadgroup uint *buckets) {
    uint peers  = mkperm_match_any(value, key_bits);
    uint leader = ctz(peers);
    uint rel    = popcount(peers & ((1u << lane) - 1u));
    uint base   = 0u;
    if (lane == leader) {
        base = buckets[value];
        buckets[value] = base + popcount(peers);
    }
    base = simd_shuffle(base, ushort(leader)); // per-group leader -> shuffle
    return base + rel;
}

kernel void mkperm_phase_1_tiny(
    constant mkperm_tiny_params &p [[buffer(0)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint3 lid3       [[thread_position_in_threadgroup]],
    uint3 tg3        [[threads_per_threadgroup]],
    uint  warp_id    [[simdgroup_index_in_threadgroup]],
    uint  lane       [[thread_index_in_simdgroup]],
    uint  warp_count [[simdgroups_per_threadgroup]],
    uint  warp_size  [[threads_per_simdgroup]]) {

    device const uint *values = (device const uint *) p.values;
    device uint *buckets       = (device uint *)       p.buckets;

    uint lid = lid3.x, tcount = tg3.x;
    uint group = tgid.y, sub_block = tgid.x, bc = p.bucket_count;

    uint user_block_start = group * p.block_size;
    uint user_block_end   = min(user_block_start + p.block_size, p.size);
    uint block_start      = user_block_start + sub_block * p.size_per_block;
    uint block_end        = min(block_start + p.size_per_block, user_block_end);

    // Zero this threadgroup's per-warp histograms
    for (uint i = lid; i < bc * warp_count; i += tcount)
        shared[i] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint *shared_warp = shared + warp_id * bc;
    uint key_bits = (bc <= 1u) ? 0u : (32u - clz(bc - 1u));

    // Each warp processes a contiguous range for stable ordering
    uint total = (block_end > block_start) ? (block_end - block_start) : 0u;
    uint epw   = ((total + warp_count - 1u) / warp_count + warp_size - 1u) &
                 ~(warp_size - 1u);
    uint warp_start = block_start + warp_id * epw;
    uint warp_end   = min(warp_start + epw, block_end);

    // Uniform loop bound so the simdgroup barrier is reached by every lane.
    for (uint base = warp_start; base < warp_start + epw; base += warp_size) {
        uint i = base + lane;
        if (i < warp_end)
            mkperm_count(values[i], lane, key_bits, shared_warp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write per-warp histograms out in bucket-major layout
    uint rows = p.rows_per_group, gbase = group * rows * bc;
    for (uint i = lid; i < bc * warp_count; i += tcount) {
        uint w = i / bc, b = i - w * bc;
        uint row = sub_block * warp_count + w;
        buckets[gbase + b * rows + row] = shared[i];
    }
}

kernel void mkperm_phase_4_tiny(
    constant mkperm_tiny_params &p [[buffer(0)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint3 lid3       [[thread_position_in_threadgroup]],
    uint3 tg3        [[threads_per_threadgroup]],
    uint  warp_id    [[simdgroup_index_in_threadgroup]],
    uint  lane       [[thread_index_in_simdgroup]],
    uint  warp_count [[simdgroups_per_threadgroup]],
    uint  warp_size  [[threads_per_simdgroup]]) {

    device const uint *values = (device const uint *) p.values;
    device const uint *buckets = (device const uint *) p.buckets;
    device uint *perm          = (device uint *)       p.perm;

    uint lid = lid3.x, tcount = tg3.x;
    uint group = tgid.y, sub_block = tgid.x, bc = p.bucket_count;

    uint user_block_start = group * p.block_size;
    uint user_block_end   = min(user_block_start + p.block_size, p.size);
    uint block_start      = user_block_start + sub_block * p.size_per_block;
    uint block_end        = min(block_start + p.size_per_block, user_block_end);

    // Load this threadgroup's per-warp base offsets from the bucket-major layout
    uint rows = p.rows_per_group, gbase = group * rows * bc;
    for (uint i = lid; i < bc * warp_count; i += tcount) {
        uint w = i / bc, b = i - w * bc;
        uint row = sub_block * warp_count + w;
        shared[i] = buckets[gbase + b * rows + row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint *shared_warp = shared + warp_id * bc;
    uint key_bits = (bc <= 1u) ? 0u : (32u - clz(bc - 1u));

    uint total = (block_end > block_start) ? (block_end - block_start) : 0u;
    uint epw   = ((total + warp_count - 1u) / warp_count + warp_size - 1u) &
                 ~(warp_size - 1u);
    uint warp_start = block_start + warp_id * epw;
    uint warp_end   = min(warp_start + epw, block_end);

    for (uint base = warp_start; base < warp_start + epw; base += warp_size) {
        uint i = base + lane;
        if (i < warp_end) {
            uint off = mkperm_reduce(values[i], lane, key_bits, shared_warp);
            perm[user_block_start + off] = i;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
//  aggregate — populate a per-vcall data buffer from a list of entries.
//  Refer to the CUDA version of this kernel in misc.cuh for more details.
// ============================================================================

struct alignas(16) AggregationEntry {
    int16_t size;
    uint16_t resource_kind; // ignored
    uint32_t offset;
    ulong src;
};

static_assert(sizeof(AggregationEntry) == 16, "Alignment issue");

kernel void aggregate_kernel(
    device uint8_t *dst [[buffer(0)]],
    device const AggregationEntry *in [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {

    AggregationEntry rec = in[tid];

    device uint8_t *d = dst + rec.offset;

    switch (rec.size) {
        case  1: *d                 = (uchar)  rec.src; break;
        case  2: *(device ushort*)d = (ushort) rec.src; break;
        case  4: *(device uint*)d   = (uint)   rec.src; break;
        case  8: *(device ulong*)d  = (ulong)  rec.src; break;
        case -1: *d                 = *(device const uchar*)  (rec.src); break;
        case -2: *(device ushort*)d = *(device const ushort*) (rec.src); break;
        case -4: *(device uint*)d   = *(device const uint*)   (rec.src); break;
        case -8: *(device ulong*)d  = *(device const ulong*)  (rec.src); break;
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
    constant memset_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    ((device T *) p.dst)[tid] = (T) p.value;
}

template [[host_name("memset_u16")]]
kernel void memset_kernel<ushort>(constant memset_params &, uint);
template [[host_name("memset_u32")]]
kernel void memset_kernel<uint>(constant memset_params &, uint);
template [[host_name("memset_u64")]]
kernel void memset_kernel<ulong>(constant memset_params &, uint);

// ============================================================================
// Narrow a float32 buffer to float16 (For Metal float16 scatter-reductions)
// ============================================================================

struct convert_params {
    ulong src; // device const float*
    ulong dst; // device half*
};

kernel void convert_f32_f16(constant convert_params &p [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    ((device half *) p.dst)[tid] = (half) ((device const float *) p.src)[tid];
}

// ============================================================================
// Metal has no 3-channel texture format, so a logical texture is split into
// 1/2/4-channel sub-textures; these kernels rearrange the data on the GPU.
// ============================================================================

struct channel_pack_params {
    ulong src;        // device const T* (source buffer base + byte offset)
    ulong dst;        // device T*       (destination buffer base + byte offset)
    uint  n_channels; // channels per texel in the interleaved layout
    uint  ci;         // channels per texel in the packed layout (1, 2, or 4)
    uint  tex_base;   // first interleaved channel of this sub-texture (tex * 4)
    uint  c_valid;    // number of real channels to copy (1..4; rest are padding)
};

template <typename T>
kernel void deinterleave_kernel(constant channel_pack_params &p [[buffer(0)]],
                                uint tid [[thread_position_in_grid]]) {
    uint pix = tid / p.ci;
    uint c   = tid % p.ci;
    device T *dst = (device T *) p.dst;
    if (c < p.c_valid)
        dst[tid] = ((device const T *) p.src)[pix * p.n_channels + p.tex_base + c];
    else
        dst[tid] = (T) 0;
}

template <typename T>
kernel void interleave_kernel(constant channel_pack_params &p [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
    uint pix = tid / p.c_valid;
    uint c   = tid % p.c_valid;
    ((device T *) p.dst)[pix * p.n_channels + p.tex_base + c] =
        ((device const T *) p.src)[pix * p.ci + c];
}

template [[host_name("deinterleave_u16")]]
kernel void deinterleave_kernel<ushort>(constant channel_pack_params &, uint);
template [[host_name("deinterleave_u32")]]
kernel void deinterleave_kernel<uint>(constant channel_pack_params &, uint);
template [[host_name("interleave_u16")]]
kernel void interleave_kernel<ushort>(constant channel_pack_params &, uint);
template [[host_name("interleave_u32")]]
kernel void interleave_kernel<uint>(constant channel_pack_params &, uint);
