/*
    resources/metal_kernels.metal — Precompiled MSL utility kernels.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <metal_stdlib>
using namespace metal;

// ============================================================================
//  compress_scatter — materialize a compacted index list from a prefix sum.
// ============================================================================

struct compress_scatter_params {
    ulong in;      // device const uint8_t * (mask)
    ulong prefix;  // device const uint32_t * (inclusive prefix of the mask)
    ulong out;     // device uint32_t *       (compacted indices)
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

// Per-threadgroup/warp range and histogram geometry shared by the two tiny
// kernels: this warp's contiguous element range within its sub-block, plus
// the threadgroup's base offsets into the bucket-major global layout.
struct mkperm_tiny_setup {
    uint user_block_start; // first element of the enclosing user block
    uint warp_start, epw;  // warp range start / warp-aligned span
    uint warp_end;         // one past this warp's last valid element
    uint key_bits;         // bits needed to tell buckets apart
    uint gbase;            // group base offset into the bucket-major layout
    uint row_base;         // first histogram row of this sub-block
};

static inline mkperm_tiny_setup mkperm_tiny_init(
    constant mkperm_tiny_params &p, uint2 tgid, uint warp_id,
    uint warp_count, uint warp_size) {
    uint group = tgid.y, sub_block = tgid.x;

    uint user_block_start = group * p.block_size;
    uint user_block_end   = min(user_block_start + p.block_size, p.size);
    uint block_start      = user_block_start + sub_block * p.size_per_block;
    uint block_end        = min(block_start + p.size_per_block, user_block_end);

    // Each warp processes a contiguous range for stable ordering
    uint total = (block_end > block_start) ? (block_end - block_start) : 0u;
    uint epw   = ((total + warp_count - 1u) / warp_count + warp_size - 1u) &
                 ~(warp_size - 1u);

    mkperm_tiny_setup s;
    s.user_block_start = user_block_start;
    s.warp_start = block_start + warp_id * epw;
    s.warp_end   = min(s.warp_start + epw, block_end);
    s.epw        = epw;
    s.key_bits   = (p.bucket_count <= 1u) ? 0u
                                          : (32u - clz(p.bucket_count - 1u));
    s.gbase      = group * p.rows_per_group * p.bucket_count;
    s.row_base   = sub_block * warp_count;
    return s;
}

kernel void mkperm_phase_1_tiny(
    constant mkperm_tiny_params &p [[buffer(0)]],
    threadgroup uint *shared [[threadgroup(0)]],
    // MSL requires uniform scalar/vector widths across position attributes
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint2 lid2       [[thread_position_in_threadgroup]],
    uint2 tg2        [[threads_per_threadgroup]],
    uint  warp_id    [[simdgroup_index_in_threadgroup]],
    uint  lane       [[thread_index_in_simdgroup]],
    uint  warp_count [[simdgroups_per_threadgroup]],
    uint  warp_size  [[threads_per_simdgroup]]) {

    device const uint *values = (device const uint *) p.values;
    device uint *buckets       = (device uint *)       p.buckets;

    uint lid = lid2.x, tcount = tg2.x, bc = p.bucket_count;
    mkperm_tiny_setup s =
        mkperm_tiny_init(p, tgid, warp_id, warp_count, warp_size);

    // Zero this threadgroup's per-warp histograms
    for (uint i = lid; i < bc * warp_count; i += tcount)
        shared[i] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint *shared_warp = shared + warp_id * bc;

    // Uniform loop bound so the simdgroup barrier is reached by every lane.
    for (uint base = s.warp_start; base < s.warp_start + s.epw;
         base += warp_size) {
        uint i = base + lane;
        if (i < s.warp_end)
            mkperm_count(values[i], lane, s.key_bits, shared_warp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write per-warp histograms out in bucket-major layout
    uint rows = p.rows_per_group;
    for (uint i = lid; i < bc * warp_count; i += tcount) {
        uint w = i / bc, b = i - w * bc;
        buckets[s.gbase + b * rows + s.row_base + w] = shared[i];
    }
}

kernel void mkperm_phase_4_tiny(
    constant mkperm_tiny_params &p [[buffer(0)]],
    threadgroup uint *shared [[threadgroup(0)]],
    // MSL requires uniform scalar/vector widths across position attributes
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint2 lid2       [[thread_position_in_threadgroup]],
    uint2 tg2        [[threads_per_threadgroup]],
    uint  warp_id    [[simdgroup_index_in_threadgroup]],
    uint  lane       [[thread_index_in_simdgroup]],
    uint  warp_count [[simdgroups_per_threadgroup]],
    uint  warp_size  [[threads_per_simdgroup]]) {

    device const uint *values = (device const uint *) p.values;
    device const uint *buckets = (device const uint *) p.buckets;
    device uint *perm          = (device uint *)       p.perm;

    uint lid = lid2.x, tcount = tg2.x, bc = p.bucket_count;
    mkperm_tiny_setup s =
        mkperm_tiny_init(p, tgid, warp_id, warp_count, warp_size);

    // Load this threadgroup's per-warp base offsets from the bucket-major layout
    uint rows = p.rows_per_group;
    for (uint i = lid; i < bc * warp_count; i += tcount) {
        uint w = i / bc, b = i - w * bc;
        shared[i] = buckets[s.gbase + b * rows + s.row_base + w];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint *shared_warp = shared + warp_id * bc;

    for (uint base = s.warp_start; base < s.warp_start + s.epw;
         base += warp_size) {
        uint i = base + lane;
        if (i < s.warp_end) {
            uint off = mkperm_reduce(values[i], lane, s.key_bits, shared_warp);
            perm[s.user_block_start + off] = i;
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

template [[host_name("deinterleave_u8")]]
kernel void deinterleave_kernel<uchar>(constant channel_pack_params &, uint);
template [[host_name("deinterleave_u16")]]
kernel void deinterleave_kernel<ushort>(constant channel_pack_params &, uint);
template [[host_name("deinterleave_u32")]]
kernel void deinterleave_kernel<uint>(constant channel_pack_params &, uint);
template [[host_name("interleave_u8")]]
kernel void interleave_kernel<uchar>(constant channel_pack_params &, uint);
template [[host_name("interleave_u16")]]
kernel void interleave_kernel<ushort>(constant channel_pack_params &, uint);
template [[host_name("interleave_u32")]]
kernel void interleave_kernel<uint>(constant channel_pack_params &, uint);

// ============================================================================
//  block_reduce — reduce contiguous blocks of an input array
//
//  The host-side logic in MetalThreadState::block_reduce() picks one of two
//  kernels per (op, type) combination:
//
//  - ``block_reduce_small_*``: one thread serially accumulates one entire
//    block. On Apple GPUs, per-lane streaming through a contiguous range
//    runs at (or near) full memory bandwidth, which makes this the fastest
//    mapping for small blocks — no threadgroup memory, simd ops, or
//    recursion needed.
//
//  - ``block_reduce_*``: one threadgroup reduces one *chunk* (a span of up
//    to ``chunk_size`` elements of one block). Each thread accumulates a
//    contiguous grain of 4 elements per loop iteration; a simdgroup
//    shuffle tree plus one threadgroup-memory exchange combine the result.
//    Blocks larger than ``chunk_size`` are split into multiple chunks
//    whose partial results are reduced by a recursive invocation.
//
//  The per-thread grain of 4 contiguous elements is essential: it amortizes
//  load-instruction issue (the bottleneck at 1 element/thread) and the cost
//  of the final reduction tree. Out-of-bounds positions contribute the
//  reduction identity, so inputs need no padding.
//
//  Accumulation may use a wider type ``A`` than the storage type ``T``
//  (f16 sums/products accumulate in f32, matching the CUDA backend).
// ============================================================================

struct block_reduce_params {
    ulong in;               // device const T *
    ulong out;              // device T *
    uint  size;             // total number of input elements
    uint  block_size;       // elements per reduction block
    uint  chunk_size;       // elements per chunk (chunked kernel only)
    uint  chunks_per_block; // chunks per block (chunked kernel only)
};

// Numeric limits used as min/max reduction identities (infinities for
// floating point types, matching ``jitc_reduce_identity``)
template <typename T> struct red_limits { };
template <> struct red_limits<half> {
    static half  lo() { return -numeric_limits<half>::infinity(); }
    static half  hi() { return  numeric_limits<half>::infinity(); }
};
template <> struct red_limits<float> {
    static float lo() { return -numeric_limits<float>::infinity(); }
    static float hi() { return  numeric_limits<float>::infinity(); }
};
template <> struct red_limits<uint> {
    static uint  lo() { return 0u; }
    static uint  hi() { return 0xFFFFFFFFu; }
};
template <> struct red_limits<int> {
    static int   lo() { return int(0x80000000u); }
    static int   hi() { return 0x7FFFFFFF; }
};
template <> struct red_limits<ulong> {
    static ulong lo() { return 0ul; }
    static ulong hi() { return 0xFFFFFFFFFFFFFFFFul; }
};
template <> struct red_limits<long> {
    static long  lo() { return long(0x8000000000000000ul); }
    static long  hi() { return 0x7FFFFFFFFFFFFFFFl; }
};

// min/max with the same NaN behavior as the CUDA backend: fmin/fmax return
// the non-NaN operand, so NaNs are ignored unless every value is NaN.
static inline half  min_(half a, half b)   { return fmin(a, b); }
static inline float min_(float a, float b) { return fmin(a, b); }
static inline uint  min_(uint a, uint b)   { return min(a, b); }
static inline int   min_(int a, int b)     { return min(a, b); }
static inline ulong min_(ulong a, ulong b) { return a < b ? a : b; }
static inline long  min_(long a, long b)   { return a < b ? a : b; }

static inline half  max_(half a, half b)   { return fmax(a, b); }
static inline float max_(float a, float b) { return fmax(a, b); }
static inline uint  max_(uint a, uint b)   { return max(a, b); }
static inline int   max_(int a, int b)     { return max(a, b); }
static inline ulong max_(ulong a, ulong b) { return a > b ? a : b; }
static inline long  max_(long a, long b)   { return a > b ? a : b; }

// simd_shuffle_xor for the 64-bit types lacking hardware simd-reduction
// instructions; the values move as two 32-bit halves
static inline ulong shfl_xor_(ulong v, ushort m) {
    return as_type<ulong>(simd_shuffle_xor(as_type<uint2>(v), m));
}
static inline long  shfl_xor_(long v, ushort m) {
    return as_type<long>(simd_shuffle_xor(as_type<uint2>(v), m));
}

template <typename A> struct red_add {
    static A init() { return A(0); }
    A operator()(A a, A b) const { return a + b; }
};

template <typename A> struct red_mul {
    static A init() { return A(1); }
    A operator()(A a, A b) const { return a * b; }
};

template <typename A> struct red_min {
    static A init() { return red_limits<A>::hi(); }
    A operator()(A a, A b) const { return min_(a, b); }
};

template <typename A> struct red_max {
    static A init() { return red_limits<A>::lo(); }
    A operator()(A a, A b) const { return max_(a, b); }
};

template <typename A> struct red_or {
    static A init() { return A(0); }
    A operator()(A a, A b) const { return a | b; }
};

template <typename A> struct red_and {
    static A init() { return ~A(0); }
    A operator()(A a, A b) const { return a & b; }
};

/// Reduce ``v`` across the 32 lanes of a simdgroup (every lane participates).
/// This generic shuffle-based fallback only handles the 64-bit types; the
/// overloads below route everything else to the hardware simd-reduction
/// instructions, which benchmark measurably faster under sustained load.
/// Note: simd_min/simd_max ignore NaNs like fmin/fmax (verified empirically
/// on Apple silicon), matching the CUDA backend's reduction semantics.
template <typename Op, typename A>
static inline A simd_reduce_op(Op op, A v) {
    v = op(v, shfl_xor_(v, ushort(1)));
    v = op(v, shfl_xor_(v, ushort(2)));
    v = op(v, shfl_xor_(v, ushort(4)));
    v = op(v, shfl_xor_(v, ushort(8)));
    v = op(v, shfl_xor_(v, ushort(16)));
    return v;
}

static inline half  simd_reduce_op(red_add<half>,  half v)  { return simd_sum(v); }
static inline float simd_reduce_op(red_add<float>, float v) { return simd_sum(v); }
static inline uint  simd_reduce_op(red_add<uint>,  uint v)  { return simd_sum(v); }
static inline float simd_reduce_op(red_mul<float>, float v) { return simd_product(v); }
static inline uint  simd_reduce_op(red_mul<uint>,  uint v)  { return simd_product(v); }
static inline half  simd_reduce_op(red_min<half>,  half v)  { return simd_min(v); }
static inline float simd_reduce_op(red_min<float>, float v) { return simd_min(v); }
static inline uint  simd_reduce_op(red_min<uint>,  uint v)  { return simd_min(v); }
static inline int   simd_reduce_op(red_min<int>,   int v)   { return simd_min(v); }
static inline half  simd_reduce_op(red_max<half>,  half v)  { return simd_max(v); }
static inline float simd_reduce_op(red_max<float>, float v) { return simd_max(v); }
static inline uint  simd_reduce_op(red_max<uint>,  uint v)  { return simd_max(v); }
static inline int   simd_reduce_op(red_max<int>,   int v)   { return simd_max(v); }
static inline uint  simd_reduce_op(red_or<uint>,   uint v)  { return simd_or(v); }
static inline uint  simd_reduce_op(red_and<uint>,  uint v)  { return simd_and(v); }

template <typename T, typename A, typename Op>
kernel void block_reduce_small(constant block_reduce_params &p [[buffer(0)]],
                               uint t [[thread_position_in_grid]]) {
    device const T *in = (device const T *) p.in;
    device T *out      = (device T *) p.out;
    Op op;

    // The grid is sized to the block count; only the final block may be short
    ulong base = (ulong) t * p.block_size;
    ulong rem  = p.size - base;
    uint  n    = rem < p.block_size ? (uint) rem : p.block_size;

    device const T *row = in + base;
    A acc = Op::init();
    uint i = 0;
    for (; i + 4u <= n; i += 4u) {
        A v01 = op((A) row[i],      (A) row[i + 1u]);
        A v23 = op((A) row[i + 2u], (A) row[i + 3u]);
        acc = op(acc, op(v01, v23));
    }
    for (; i < n; ++i)
        acc = op(acc, (A) row[i]);

    out[t] = (T) acc;
}

// ``OutT`` is normally ``T``; the f16 sum/product prefix reductions use a
// variant that keeps the chunk totals in f32 (``OutT = A``) so that the
// carried prefixes match the CUDA backend's intermediate precision.
template <typename T, typename A, typename Op, typename OutT>
kernel void block_reduce_chunk(constant block_reduce_params &p [[buffer(0)]],
                               // Position attributes must share a vector width
                               uint2 tg_id  [[threadgroup_position_in_grid]],
                               uint2 tid_2  [[thread_position_in_threadgroup]],
                               uint2 tg_w_2 [[threads_per_threadgroup]],
                               uint lane    [[thread_index_in_simdgroup]],
                               uint sgid    [[simdgroup_index_in_threadgroup]],
                               uint nsg     [[simdgroups_per_threadgroup]]) {
    device const T *in = (device const T *) p.in;
    device OutT *out   = (device OutT *) p.out;
    Op op;

    uint tid = tid_2.x, tg_w = tg_w_2.x;
    uint rel_chunk = tg_id.x, block = tg_id.y;
    ulong block_start = (ulong) block * p.block_size,
          chunk_start = (ulong) rel_chunk * p.chunk_size;

    // Accumulate a contiguous grain of 4 elements per loop iteration.
    // Out-of-bounds positions (a short trailing chunk/block) are skipped,
    // which makes input padding unnecessary.
    A acc = Op::init();
    for (uint off = tid * 4u; off < p.chunk_size; off += tg_w * 4u) {
        ulong pos_in_block = chunk_start + off,
              pos          = block_start + pos_in_block;
        if (pos_in_block + 3u < p.block_size && pos + 3u < p.size) {
            A v01 = op((A) in[pos],      (A) in[pos + 1u]);
            A v23 = op((A) in[pos + 2u], (A) in[pos + 3u]);
            acc = op(acc, op(v01, v23));
        } else {
            for (uint j = 0; j < 4u; ++j) {
                if (pos_in_block + j < p.block_size && pos + j < p.size)
                    acc = op(acc, (A) in[pos + j]);
            }
        }
    }

    // Reduce within each simdgroup, then once more across simdgroups
    acc = simd_reduce_op(op, acc);

    threadgroup A sh[32];
    if (nsg > 1u) {
        if (lane == 0u)
            sh[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            acc = lane < nsg ? sh[lane] : Op::init();
            acc = simd_reduce_op(op, acc);
        }
    }

    if (tid == 0u)
        out[(ulong) block * p.chunks_per_block + rel_chunk] = (OutT) acc;
}

#define DRJIT_BLOCK_RED(op, tname, T, A)                                       \
    template [[host_name("block_reduce_small_" #op "_" tname)]]                \
    kernel void block_reduce_small<T, A, red_##op<A>>(                         \
        constant block_reduce_params &, uint);                                 \
    template [[host_name("block_reduce_" #op "_" tname)]]                      \
    kernel void block_reduce_chunk<T, A, red_##op<A>, T>(                      \
        constant block_reduce_params &, uint2, uint2, uint2, uint, uint, uint);

// Signed integer sums/products/or/and are dispatched to the unsigned kernels
DRJIT_BLOCK_RED(add, "f16", half,  float)
DRJIT_BLOCK_RED(add, "f32", float, float)
DRJIT_BLOCK_RED(add, "u32", uint,  uint)
DRJIT_BLOCK_RED(add, "u64", ulong, ulong)

DRJIT_BLOCK_RED(mul, "f16", half,  float)
DRJIT_BLOCK_RED(mul, "f32", float, float)
DRJIT_BLOCK_RED(mul, "u32", uint,  uint)
DRJIT_BLOCK_RED(mul, "u64", ulong, ulong)

DRJIT_BLOCK_RED(min, "f16", half,  half)
DRJIT_BLOCK_RED(min, "f32", float, float)
DRJIT_BLOCK_RED(min, "u32", uint,  uint)
DRJIT_BLOCK_RED(min, "u64", ulong, ulong)
DRJIT_BLOCK_RED(min, "i32", int,   int)
DRJIT_BLOCK_RED(min, "i64", long,  long)

DRJIT_BLOCK_RED(max, "f16", half,  half)
DRJIT_BLOCK_RED(max, "f32", float, float)
DRJIT_BLOCK_RED(max, "u32", uint,  uint)
DRJIT_BLOCK_RED(max, "u64", ulong, ulong)
DRJIT_BLOCK_RED(max, "i32", int,   int)
DRJIT_BLOCK_RED(max, "i64", long,  long)

DRJIT_BLOCK_RED(or,  "u8",  uchar, uint)
DRJIT_BLOCK_RED(or,  "u32", uint,  uint)
DRJIT_BLOCK_RED(or,  "u64", ulong, ulong)

DRJIT_BLOCK_RED(and, "u8",  uchar, uint)
DRJIT_BLOCK_RED(and, "u32", uint,  uint)
DRJIT_BLOCK_RED(and, "u64", ulong, ulong)

// Widened variants for f16 sum/product prefix reductions (chunk totals in f32)
template [[host_name("block_reduce_wide_add_f16")]]
kernel void block_reduce_chunk<half, float, red_add<float>, float>(
    constant block_reduce_params &, uint2, uint2, uint2, uint, uint, uint);
template [[host_name("block_reduce_wide_mul_f16")]]
kernel void block_reduce_chunk<half, float, red_mul<float>, float>(
    constant block_reduce_params &, uint2, uint2, uint2, uint, uint, uint);

// ============================================================================
//  block_prefix_reduce — prefix-reduce contiguous blocks of an input array
//
//  The host decomposes the operation following the classic *reduce-then-scan*
//  strategy. (CUDA's single-pass decoupled lookback does not port: it needs
//  untorn tagged stores, device-scope release/acquire ordering, and a
//  forward-progress guarantee between threadgroups — none of which Metal
//  provides at our feature baseline.)
//
//  1. When a block exceeds one threadgroup's span (``chunk_size``, at most
//     4 x 1024 elements), ``block_reduce_chunk`` first computes per-chunk
//     totals, and a recursive prefix reduction over those totals yields each
//     chunk's exclusive prefix (``p.prefixes``).
//  2. This kernel then scans each chunk independently: a register scan of
//     each thread's grain of 4, an exclusive simdgroup scan of the thread
//     totals, one threadgroup-memory exchange across simdgroups, plus the
//     chunk prefix.
//
//  Reverse scans traverse each chunk from its upper end downward; the chunk
//  decomposition itself remains memory-aligned, and the recursion (invoked
//  with the same ``reverse`` flag) orders the chunk prefixes accordingly.
//  Out-of-bounds scan slots contribute the identity and are not written, so
//  ragged trailing blocks need no padding.
// ============================================================================

struct block_prefix_reduce_params {
    ulong in;               // device const T *
    ulong out;              // device T *
    ulong prefixes;         // device const T * (exclusive per-chunk prefixes), or 0
    uint  size;             // total number of input elements
    uint  block_size;       // elements per reduction block
    uint  chunk_size;       // elements per chunk; at most 4x the threadgroup width
    uint  chunks_per_block; // chunks per block
    uint  exclusive;        // exclusive (vs. inclusive) prefix reduction?
    uint  reverse;          // scan from the end of each block?
};

// simd_shuffle_up with support for 64-bit types (moved as two 32-bit halves).
// Lanes < delta receive undefined values; all callers guard on the lane index.
template <typename A>
static inline A     shfl_up_(A v, ushort d)     { return simd_shuffle_up(v, d); }
static inline ulong shfl_up_(ulong v, ushort d) {
    return as_type<ulong>(simd_shuffle_up(as_type<uint2>(v), d));
}
static inline long  shfl_up_(long v, ushort d) {
    return as_type<long>(simd_shuffle_up(as_type<uint2>(v), d));
}

/// Exclusive prefix reduction of ``v`` across the 32 lanes of a simdgroup.
/// Generic shuffle-based fallback; the overloads below route sums/products
/// to the hardware prefix-scan instructions.
template <typename Op, typename A>
static inline A simd_prefix_excl(Op op, A v, uint lane) {
    for (ushort d = 1; d < 32; d <<= 1) {
        A n = shfl_up_(v, d);
        if (lane >= d)
            v = op(v, n);
    }
    A e = shfl_up_(v, ushort(1));
    return lane == 0u ? Op::init() : e;
}

static inline float simd_prefix_excl(red_add<float>, float v, uint) {
    return simd_prefix_exclusive_sum(v);
}
static inline uint simd_prefix_excl(red_add<uint>, uint v, uint) {
    return simd_prefix_exclusive_sum(v);
}
static inline float simd_prefix_excl(red_mul<float>, float v, uint) {
    return simd_prefix_exclusive_product(v);
}
static inline uint simd_prefix_excl(red_mul<uint>, uint v, uint) {
    return simd_prefix_exclusive_product(v);
}

// ``OutT`` is normally ``T``; the u8 -> u32 prefix sum that powers compress()
// uses a widened variant.
template <typename T, typename A, typename Op, typename OutT>
kernel void block_prefix_reduce_kernel(
    constant block_prefix_reduce_params &p [[buffer(0)]],
    // Position attributes must share a vector width
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tid_2 [[thread_position_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint nsg    [[simdgroups_per_threadgroup]]) {
    device const T *in = (device const T *) p.in;
    device OutT *out   = (device OutT *) p.out;
    Op op;

    uint tid = tid_2.x;
    uint rel_chunk = tg_id.x, block = tg_id.y;
    ulong block_start = (ulong) block * p.block_size;

    // Map this thread's 4 consecutive scan-space slots to memory positions.
    // Forward scans walk the chunk upward from rel_chunk * chunk_size;
    // reverse scans walk it downward from (rel_chunk + 1) * chunk_size - 1.
    A v[4];
    bool valid[4];
    ulong pos[4];
    for (uint j = 0; j < 4u; ++j) {
        uint s = tid * 4u + j;
        ulong pib = p.reverse
            ? ((ulong) (rel_chunk + 1u) * p.chunk_size - 1u - s)
            : ((ulong) rel_chunk * p.chunk_size + s);
        pos[j] = block_start + pib;
        valid[j] = s < p.chunk_size && pib < p.block_size && pos[j] < p.size;
    }

    for (uint j = 0; j < 4u; ++j)
        v[j] = valid[j] ? (A) in[pos[j]] : Op::init();

    // Inclusive scan of the grain in registers
    v[1] = op(v[1], v[0]);
    v[2] = op(v[2], v[1]);
    v[3] = op(v[3], v[2]);

    // Exclusive prefix of the per-thread totals within each simdgroup
    A prefix = simd_prefix_excl(op, v[3], lane);

    // Combine across simdgroups: one exclusive scan of the simdgroup totals
    threadgroup A sh[32];
    if (nsg > 1u) {
        if (lane == 31u)
            sh[sgid] = op(prefix, v[3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            A t = lane < nsg ? sh[lane] : Op::init();
            sh[lane] = simd_prefix_excl(op, t, lane);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = op(prefix, sh[sgid]);
    }

    // Account for preceding chunks of the same block (multi-chunk blocks).
    // The prefixes are stored in the accumulator type so that e.g. f16 sums
    // carry f32 chunk prefixes, matching the CUDA backend's precision.
    if (p.prefixes) {
        device const A *pre = (device const A *) p.prefixes;
        prefix = op(prefix, pre[(ulong) block * p.chunks_per_block + rel_chunk]);
    }

    // Final values; an exclusive reduction shifts everything by one slot
    A r[4];
    if (p.exclusive) {
        r[0] = prefix;
        r[1] = op(v[0], prefix);
        r[2] = op(v[1], prefix);
        r[3] = op(v[2], prefix);
    } else {
        r[0] = op(v[0], prefix);
        r[1] = op(v[1], prefix);
        r[2] = op(v[2], prefix);
        r[3] = op(v[3], prefix);
    }

    for (uint j = 0; j < 4u; ++j) {
        if (valid[j])
            out[pos[j]] = (OutT) r[j];
    }
}

#define DRJIT_BLOCK_PRED(op, tname, T, A)                                      \
    template [[host_name("block_prefix_reduce_" #op "_" tname)]]               \
    kernel void block_prefix_reduce_kernel<T, A, red_##op<A>, T>(              \
        constant block_prefix_reduce_params &, uint2, uint2, uint, uint, uint);

// Signed integer sums/products/or/and are dispatched to the unsigned kernels
DRJIT_BLOCK_PRED(add, "f16", half,  float)
DRJIT_BLOCK_PRED(add, "f32", float, float)
DRJIT_BLOCK_PRED(add, "u32", uint,  uint)
DRJIT_BLOCK_PRED(add, "u64", ulong, ulong)

DRJIT_BLOCK_PRED(mul, "f16", half,  float)
DRJIT_BLOCK_PRED(mul, "f32", float, float)
DRJIT_BLOCK_PRED(mul, "u32", uint,  uint)
DRJIT_BLOCK_PRED(mul, "u64", ulong, ulong)

DRJIT_BLOCK_PRED(min, "f16", half,  half)
DRJIT_BLOCK_PRED(min, "f32", float, float)
DRJIT_BLOCK_PRED(min, "u32", uint,  uint)
DRJIT_BLOCK_PRED(min, "u64", ulong, ulong)
DRJIT_BLOCK_PRED(min, "i32", int,   int)
DRJIT_BLOCK_PRED(min, "i64", long,  long)

DRJIT_BLOCK_PRED(max, "f16", half,  half)
DRJIT_BLOCK_PRED(max, "f32", float, float)
DRJIT_BLOCK_PRED(max, "u32", uint,  uint)
DRJIT_BLOCK_PRED(max, "u64", ulong, ulong)
DRJIT_BLOCK_PRED(max, "i32", int,   int)
DRJIT_BLOCK_PRED(max, "i64", long,  long)

DRJIT_BLOCK_PRED(or,  "u32", uint,  uint)
DRJIT_BLOCK_PRED(or,  "u64", ulong, ulong)

DRJIT_BLOCK_PRED(and, "u32", uint,  uint)
DRJIT_BLOCK_PRED(and, "u64", ulong, ulong)

// Widened u8 -> u32 inclusive prefix sum (with the matching pass-1 chunk
// reduction); this powers compress(), which scans a 0/1 boolean mask
template [[host_name("block_reduce_wide_add_u8")]]
kernel void block_reduce_chunk<uchar, uint, red_add<uint>, uint>(
    constant block_reduce_params &, uint2, uint2, uint2, uint, uint, uint);
template [[host_name("block_prefix_reduce_add_u8")]]
kernel void block_prefix_reduce_kernel<uchar, uint, red_add<uint>, uint>(
    constant block_prefix_reduce_params &, uint2, uint2, uint, uint, uint);

// ============================================================================
//  reduce_dot — out[0] = sum(a * b), the fused dot-product reduction
//
//  Mirrors ``block_reduce_chunk`` with two input streams and a fused
//  multiply-add; the host reduces the per-threadgroup partial results with a
//  regular sum block reduction. Like the CUDA backend, accumulation happens
//  in the storage type (f16 dots accumulate in f16).
// ============================================================================

struct reduce_dot_params {
    ulong in1;       // device const T *
    ulong in2;       // device const T *
    ulong out;       // device T * (one partial result per threadgroup)
    uint  size;      // total number of elements
    uint  chunk_size;// elements per threadgroup
};

template <typename T>
kernel void reduce_dot_kernel(constant reduce_dot_params &p [[buffer(0)]],
                              uint tg_id [[threadgroup_position_in_grid]],
                              uint tid   [[thread_position_in_threadgroup]],
                              uint tg_w  [[threads_per_threadgroup]],
                              uint lane  [[thread_index_in_simdgroup]],
                              uint sgid  [[simdgroup_index_in_threadgroup]],
                              uint nsg   [[simdgroups_per_threadgroup]]) {
    device const T *a = (device const T *) p.in1;
    device const T *b = (device const T *) p.in2;
    device T *out     = (device T *) p.out;
    red_add<T> op;

    T acc = T(0);
    for (uint off = tid * 4u; off < p.chunk_size; off += tg_w * 4u) {
        ulong pos = (ulong) tg_id * p.chunk_size + off;
        if (pos + 3u < p.size) {
            acc = fma(a[pos],      b[pos],      acc);
            acc = fma(a[pos + 1u], b[pos + 1u], acc);
            acc = fma(a[pos + 2u], b[pos + 2u], acc);
            acc = fma(a[pos + 3u], b[pos + 3u], acc);
        } else {
            for (uint j = 0; j < 4u; ++j) {
                if (pos + j < p.size)
                    acc = fma(a[pos + j], b[pos + j], acc);
            }
        }
    }

    acc = simd_reduce_op(op, acc);

    threadgroup T sh[32];
    if (nsg > 1u) {
        if (lane == 0u)
            sh[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            acc = lane < nsg ? sh[lane] : T(0);
            acc = simd_reduce_op(op, acc);
        }
    }

    if (tid == 0u)
        out[tg_id] = acc;
}

template [[host_name("reduce_dot_f16")]]
kernel void reduce_dot_kernel<half>(
    constant reduce_dot_params &, uint, uint, uint, uint, uint, uint);
template [[host_name("reduce_dot_f32")]]
kernel void reduce_dot_kernel<float>(
    constant reduce_dot_params &, uint, uint, uint, uint, uint, uint);
