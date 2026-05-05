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

// MSL allows `simd_shuffle_xor(float2, ushort)` in principle, but on some
// devices the codegen miscompiles or the shader hangs. Provide an explicit
// componentwise shuffle for `float2` (= dd_t) so the DD reduction kernels
// behave consistently.
inline float2 simd_shuffle_xor(float2 v, ushort mask) {
    return float2(simd_shuffle_xor(v.x, mask),
                  simd_shuffle_xor(v.y, mask));
}

// ============================================================================
//  Float64 emulation via double-double (DD) -- two-float pair (hi, lo).
//  Mirrors the helpers injected from src/metal_dd_preamble.h, but lives here
//  so the precompiled utility kernels (block_reduce, etc.) can operate on
//  Float64 when JitFlag::MetalEmulateFloat64 is enabled.
// ============================================================================
typedef float2 dd_t;

inline dd_t dd_two_sum(float a, float b) {
    float s  = a + b;
    float bb = s - a;
    float e  = (a - (s - bb)) + (b - bb);
    return dd_t(s, e);
}
inline dd_t dd_fast_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return dd_t(s, e);
}
inline dd_t dd_neg(dd_t a) { return dd_t(-a.x, -a.y); }
inline dd_t dd_add(dd_t a, dd_t b) {
    dd_t s = dd_two_sum(a.x, b.x);
    dd_t t = dd_two_sum(a.y, b.y);
    s.y += t.x;
    s   = dd_fast_two_sum(s.x, s.y);
    s.y += t.y;
    s   = dd_fast_two_sum(s.x, s.y);
    return s;
}
inline dd_t dd_sub(dd_t a, dd_t b) { return dd_add(a, dd_neg(b)); }
inline bool dd_lt(dd_t a, dd_t b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
inline bool dd_gt(dd_t a, dd_t b) { return a.x > b.x || (a.x == b.x && a.y > b.y); }
inline dd_t dd_min(dd_t a, dd_t b) { return dd_lt(a, b) ? a : b; }
inline dd_t dd_max(dd_t a, dd_t b) { return dd_gt(a, b) ? a : b; }

// ============================================================================
//  Reduction operators
// ============================================================================

template <typename T> struct reduction_add {
    static T init() { return T(0); }
    static T apply(T a, T b) { return a + b; }
};

template <typename T> struct reduction_mul {
    static T init() { return T(1); }
    static T apply(T a, T b) { return a * b; }
};

template <typename T> struct reduction_min {
    static T init();
    static T apply(T a, T b) { return min(a, b); }
};
template <> float  reduction_min<float>::init()  { return INFINITY; }
template <> half   reduction_min<half>::init()   { return half(INFINITY); }
template <> int    reduction_min<int>::init()    { return INT_MAX; }
template <> uint   reduction_min<uint>::init()   { return UINT_MAX; }
template <> long   reduction_min<long>::init()   { return LONG_MAX; }
template <> ulong  reduction_min<ulong>::init()  { return ULONG_MAX; }

template <typename T> struct reduction_max {
    static T init();
    static T apply(T a, T b) { return max(a, b); }
};
template <> float  reduction_max<float>::init()  { return -INFINITY; }
template <> half   reduction_max<half>::init()   { return half(-INFINITY); }
template <> int    reduction_max<int>::init()    { return INT_MIN; }
template <> uint   reduction_max<uint>::init()   { return 0u; }
template <> long   reduction_max<long>::init()   { return LONG_MIN; }
template <> ulong  reduction_max<ulong>::init()  { return 0ull; }

template <typename T> struct reduction_or {
    static T init() { return T(0); }
    static T apply(T a, T b) { return a | b; }
};

template <typename T> struct reduction_and {
    static T init() { return ~T(0); }
    static T apply(T a, T b) { return a & b; }
};

// DD specializations: Float64 emulation. reduction_add uses the precise
// dd_add helper (not raw float2 + float2). min/max use lex comparison.
//
// Precision note: dd_add follows Bailey's Algorithm 6 (Hida et al. 2007),
// which is accurate to ~2 ulp DD when the operands are well-separated in
// magnitude. Long reductions over inputs with mixed scales (e.g. 1e16 + 1.0)
// can still lose the small contributions during the residual-merge step
// `s.y += t.x`. For exact sums regardless of scale, a Kahan-Babuska-Neumaier
// accumulator would be needed -- intentionally not implemented here because
// it doubles the per-step cost.
template <> struct reduction_add<dd_t> {
    static dd_t init() { return dd_t(0.0f, 0.0f); }
    static dd_t apply(dd_t a, dd_t b) { return dd_add(a, b); }
};
template <> struct reduction_min<dd_t> {
    static dd_t init() { return dd_t(INFINITY, 0.0f); }
    static dd_t apply(dd_t a, dd_t b) { return dd_min(a, b); }
};
template <> struct reduction_max<dd_t> {
    static dd_t init() { return dd_t(-INFINITY, 0.0f); }
    static dd_t apply(dd_t a, dd_t b) { return dd_max(a, b); }
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

// float32
INSTANTIATE_RED("add_f32",  float,  reduction_add)
INSTANTIATE_RED("mul_f32",  float,  reduction_mul)
INSTANTIATE_RED("min_f32",  float,  reduction_min)
INSTANTIATE_RED("max_f32",  float,  reduction_max)

// float16
INSTANTIATE_RED("add_f16",  half,   reduction_add)
INSTANTIATE_RED("mul_f16",  half,   reduction_mul)
INSTANTIATE_RED("min_f16",  half,   reduction_min)
INSTANTIATE_RED("max_f16",  half,   reduction_max)

// uint32
INSTANTIATE_RED("add_u32",  uint,   reduction_add)
INSTANTIATE_RED("mul_u32",  uint,   reduction_mul)
INSTANTIATE_RED("min_u32",  uint,   reduction_min)
INSTANTIATE_RED("max_u32",  uint,   reduction_max)
INSTANTIATE_RED("or_u32",   uint,   reduction_or)
INSTANTIATE_RED("and_u32",  uint,   reduction_and)

// int32
INSTANTIATE_RED("min_i32",  int,    reduction_min)
INSTANTIATE_RED("max_i32",  int,    reduction_max)

// uint64
INSTANTIATE_RED("add_u64",  ulong,  reduction_add)
INSTANTIATE_RED("mul_u64",  ulong,  reduction_mul)
INSTANTIATE_RED("min_u64",  ulong,  reduction_min)
INSTANTIATE_RED("max_u64",  ulong,  reduction_max)
INSTANTIATE_RED("or_u64",   ulong,  reduction_or)
INSTANTIATE_RED("and_u64",  ulong,  reduction_and)

// int64
INSTANTIATE_RED("min_i64",  long,   reduction_min)
INSTANTIATE_RED("max_i64",  long,   reduction_max)

// uint8 (used for boolean and/or reductions)
INSTANTIATE_RED("or_u8",    uchar,  reduction_or)
INSTANTIATE_RED("and_u8",   uchar,  reduction_and)

// float64 (DD emulation; no mul -- the DD product would be a separate kernel)
INSTANTIATE_RED("add_f64dd", dd_t, reduction_add)
INSTANTIATE_RED("min_f64dd", dd_t, reduction_min)
INSTANTIATE_RED("max_f64dd", dd_t, reduction_max)

// ============================================================================
//  block_prefix_reduce — inclusive / exclusive prefix scan within blocks
//
//  Each threadgroup handles one block of up to ChunkSize elements.
//  Uses a work-efficient Hillis-Steele parallel prefix scan in
//  threadgroup shared memory. For blocks larger than 1024, the C++
//  dispatch splits into chunks and uses block_reduce to compute
//  inter-chunk offsets (two-pass approach).
//
//  Parameters are passed via a struct matching the C++ layout.
// ============================================================================

struct block_prefix_reduce_params {
    ulong  in;               // device const T *
    ulong  out;              // device T *
    ulong  offsets;          // device const T * (inter-chunk offsets, or 0)
    uint   size;             // total number of input elements
    uint   block_size;       // how many elements per logical block
    uint   exclusive;        // 1 = exclusive prefix sum, 0 = inclusive
    uint   reverse;          // 1 = reverse direction
    uint   chunks_per_block; // number of chunks per logical block (1 for small blocks)
};

template <typename T, typename Red, uint ChunkSize>
kernel void block_prefix_reduce_kernel(
    device const block_prefix_reduce_params &p [[buffer(0)]],
    threadgroup T *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]])
{
    constexpr uint SimdWidth = 32;
    constexpr bool MultiChunk = (ChunkSize < 128);
    constexpr uint ThreadsPerGroup = MultiChunk ? 128 : ChunkSize;

    device const T *in  = (device const T *) p.in;
    device T       *out = (device T *)       p.out;
    device const T *offsets = (device const T *) p.offsets;

    uint chunk, pos_in_chunk;
    if (MultiChunk) {
        uint chunks_per_tg = ThreadsPerGroup / ChunkSize;
        chunk        = tg_id * chunks_per_tg + tid / ChunkSize;
        pos_in_chunk = tid % ChunkSize;
    } else {
        chunk        = tg_id;
        pos_in_chunk = tid;
    }

    uint block_id       = chunk / p.chunks_per_block;
    uint chunk_in_block = chunk % p.chunks_per_block;
    uint pos_in_block   = chunk_in_block * ChunkSize + pos_in_chunk;
    if (p.reverse)
        pos_in_block = p.block_size - 1 - pos_in_block;

    uint pos_in_input = block_id * p.block_size + pos_in_block;
    bool valid = (pos_in_block < p.block_size) && (pos_in_input < p.size);

    T value = valid ? in[pos_in_input] : Red::init();

    // Inclusive prefix scan via shared memory (Hillis-Steele)
    for (uint stride = 1; stride < ChunkSize; stride <<= 1) {
        shared[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pos_in_chunk >= stride)
            value = Red::apply(value, shared[tid - stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Add inter-chunk offset if provided (from two-pass large reduction)
    if (offsets && chunk_in_block > 0) {
        T off = offsets[chunk];
        value = Red::apply(value, off);
    }

    // Write result
    if (valid) {
        if (p.exclusive) {
            // Shift right: exclusive[i] = inclusive[i-1], exclusive[0] = identity
            shared[tid] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            T chunk_offset = Red::init();
            if (offsets && chunk_in_block > 0)
                chunk_offset = offsets[chunk];
            value = (pos_in_chunk == 0) ? chunk_offset : shared[tid - 1];
        }
        out[pos_in_input] = value;
    }
}

// Instantiation macro
#define INSTANTIATE_PREFIX(Name, T, Red)                                       \
    template [[host_name("block_prefix_" Name "_2")]]                         \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 2>(                     \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_4")]]                         \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 4>(                     \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_8")]]                         \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 8>(                     \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_16")]]                        \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 16>(                    \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_32")]]                        \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 32>(                    \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_64")]]                        \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 64>(                    \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_128")]]                       \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 128>(                   \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_256")]]                       \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 256>(                   \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_512")]]                       \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 512>(                   \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint); \
    template [[host_name("block_prefix_" Name "_1024")]]                      \
    kernel void block_prefix_reduce_kernel<T, Red<T>, 1024>(                  \
        device const block_prefix_reduce_params &, threadgroup T *, uint, uint);

// float32
INSTANTIATE_PREFIX("add_f32", float, reduction_add)
INSTANTIATE_PREFIX("mul_f32", float, reduction_mul)
INSTANTIATE_PREFIX("min_f32", float, reduction_min)
INSTANTIATE_PREFIX("max_f32", float, reduction_max)

// float16
INSTANTIATE_PREFIX("add_f16", half,  reduction_add)
INSTANTIATE_PREFIX("mul_f16", half,  reduction_mul)
INSTANTIATE_PREFIX("min_f16", half,  reduction_min)
INSTANTIATE_PREFIX("max_f16", half,  reduction_max)

// uint32
INSTANTIATE_PREFIX("add_u32", uint,  reduction_add)
INSTANTIATE_PREFIX("mul_u32", uint,  reduction_mul)
INSTANTIATE_PREFIX("min_u32", uint,  reduction_min)
INSTANTIATE_PREFIX("max_u32", uint,  reduction_max)
INSTANTIATE_PREFIX("or_u32",  uint,  reduction_or)
INSTANTIATE_PREFIX("and_u32", uint,  reduction_and)

// int32
INSTANTIATE_PREFIX("min_i32", int,   reduction_min)
INSTANTIATE_PREFIX("max_i32", int,   reduction_max)

// uint64
INSTANTIATE_PREFIX("add_u64", ulong, reduction_add)
INSTANTIATE_PREFIX("mul_u64", ulong, reduction_mul)
INSTANTIATE_PREFIX("min_u64", ulong, reduction_min)
INSTANTIATE_PREFIX("max_u64", ulong, reduction_max)
INSTANTIATE_PREFIX("or_u64",  ulong, reduction_or)
INSTANTIATE_PREFIX("and_u64", ulong, reduction_and)

// int64
INSTANTIATE_PREFIX("min_i64", long,  reduction_min)
INSTANTIATE_PREFIX("max_i64", long,  reduction_max)

// uint8 (boolean prefix scans)
INSTANTIATE_PREFIX("or_u8",   uchar, reduction_or)
INSTANTIATE_PREFIX("and_u8",  uchar, reduction_and)

// ============================================================================
//  extract_chunk_totals — gather the last element of each chunk from a
//  pass-1 inclusive scan, on-device.
//
//  Replaces the CPU gather inside ``MetalThreadState::block_prefix_reduce``
//  for blocks > chunk_size: previously the path was
//      memcpy GPU→CPU(size)  →  CPU index loop  →  memcpy CPU→GPU(total_chunks)
//  costing two stalls. The kernel below dispatches one thread per chunk
//  and writes a flat ``(total_chunks,)`` device buffer the recursive
//  ``block_prefix_reduce`` call then scans.
//
//  The kernel is byte-copy in disguise — the storage type ``T`` only
//  determines the load/store width, not any arithmetic. Four
//  instantiations cover every dtype (1, 2, 4, 8 bytes); the C++ caller
//  picks by ``type_size[vt]``.
// ============================================================================

struct extract_chunk_totals_params {
    ulong  in;               // device const T *  (pass-1 inclusive scan)
    ulong  out;              // device T *        (flat (total_chunks,) dst)
    uint   size;             // total elements in `in`
    uint   block_size;
    uint   chunk_size;
    uint   chunks_per_block;
    uint   total_chunks;
    uint   reverse;          // 0 = forward, 1 = reverse
};

template <typename T>
kernel void extract_chunk_totals_kernel(
    device const extract_chunk_totals_params &p [[buffer(0)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.total_chunks) return;

    device const T *in  = (device const T *) p.in;
    device       T *out = (device       T *) p.out;

    uint blk = c / p.chunks_per_block;
    uint cib = c % p.chunks_per_block;
    uint cap = (cib + 1) * p.chunk_size;
    if (cap > p.block_size)
        cap = p.block_size;

    uint last;
    if (p.reverse == 0)
        last = blk * p.block_size + cap - 1;
    else
        last = blk * p.block_size + (p.block_size - cap);

    out[c] = (last < p.size) ? in[last] : T(0);
}

template [[host_name("extract_chunk_totals_b1")]]
kernel void extract_chunk_totals_kernel<uchar>(
    device const extract_chunk_totals_params &, uint);
template [[host_name("extract_chunk_totals_b2")]]
kernel void extract_chunk_totals_kernel<ushort>(
    device const extract_chunk_totals_params &, uint);
template [[host_name("extract_chunk_totals_b4")]]
kernel void extract_chunk_totals_kernel<uint>(
    device const extract_chunk_totals_params &, uint);
template [[host_name("extract_chunk_totals_b8")]]
kernel void extract_chunk_totals_kernel<ulong>(
    device const extract_chunk_totals_params &, uint);

// ============================================================================
//  compress_total — sum the last exclusive prefix entry with the last
//  per-block count to produce the total compacted-output count, on-device.
//
//  Replaces the two ``memcpy GPU→CPU + CPU add`` round-trips inside
//  ``MetalThreadState::compress``. Single-thread kernel; the C++ caller
//  still issues one final sync before reading ``count_buf->contents()``.
// ============================================================================

struct compress_total_params {
    ulong  prefix;       // device const uint* (exclusive prefix array)
    ulong  counts;       // device const uint* (saved per-block counts)
    ulong  out;          // device uint*       (count_buf — single value)
    uint   block_count;
};

kernel void compress_total_kernel(
    device const compress_total_params &p [[buffer(0)]])
{
    device const uint *prefix = (device const uint *) p.prefix;
    device const uint *counts = (device const uint *) p.counts;
    device       uint *out    = (device       uint *) p.out;
    uint last = p.block_count - 1;
    out[0] = prefix[last] + counts[last];
}

// ============================================================================
//  compress — convert a boolean mask to a compacted index list
//
//  Two-pass approach:
//    Pass 1 (compress_count): Each threadgroup counts true values in its
//          chunk and writes the partial count to a scratch buffer.
//    Between passes the CPU prefix-sums the per-block counts via
//          block_prefix_reduce (or a simple CPU loop for small arrays).
//    Pass 2 (compress_write): Each threadgroup does an intra-group
//          exclusive prefix scan and writes indices offset by the
//          block-level prefix.
//
//  For small arrays (≤ 4096 elements) the single-pass kernel
//  compress_small performs the entire operation in one threadgroup.
// ============================================================================

struct compress_params {
    ulong in;           // device const uint8_t *
    ulong out;          // device uint32_t *
    ulong scratch;      // device uint32_t * (per-block counts / prefix)
    uint  size;
    uint  count_offset; // offset into scratch where total count is stored
};

kernel void compress_small(
    device const compress_params &p [[buffer(0)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    device const uint8_t *in = (device const uint8_t *) p.in;
    device uint *out         = (device uint *)          p.out;
    device uint *scratch     = (device uint *)          p.scratch;

    // Each thread processes 4 elements
    uint base = tid * 4;
    uint local_count = 0;
    uint local_prefix[4];
    for (uint i = 0; i < 4; i++) {
        local_prefix[i] = local_count;
        if (base + i < p.size)
            local_count += in[base + i] ? 1 : 0;
    }

    // Exclusive scan via shared memory (Hillis-Steele)
    uint si = tid;
    shared[si] = 0;
    si += tg_size;
    shared[si] = local_count;

    for (uint offset = 1; offset < tg_size; offset <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sum = shared[si] + shared[si - offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[si] = sum;
    }

    // Total count
    if (tid == tg_size - 1)
        scratch[p.count_offset] = shared[si];

    uint block_prefix = shared[si] - local_count;
    for (uint i = 0; i < 4; i++) {
        if (base + i < p.size && in[base + i])
            out[block_prefix + local_prefix[i]] = base + i;
    }
}

kernel void compress_count(
    device const compress_params &p [[buffer(0)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    device const uint8_t *in = (device const uint8_t *) p.in;
    device uint *scratch     = (device uint *)          p.scratch;

    uint items_per_thread = 16;
    uint block_start = tg_id * tg_size * items_per_thread;

    uint local_count = 0;
    for (uint i = 0; i < items_per_thread; i++) {
        uint idx = block_start + tid * items_per_thread + i;
        if (idx < p.size)
            local_count += in[idx] ? 1 : 0;
    }

    // SIMD reduction
    for (uint k = 1; k < 32; k *= 2)
        local_count += simd_shuffle_xor(local_count, (ushort)k);

    // Write lane-0 to shared, then reduce across SIMD groups
    if ((tid % 32) == 0)
        shared[tid / 32] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint val = (tid < tg_size / 32) ? shared[tid] : 0;
        for (uint k = 1; k < 32; k *= 2)
            val += simd_shuffle_xor(val, (ushort)k);
        if (tid == 0)
            scratch[tg_id] = val;
    }
}

kernel void compress_write(
    device const compress_params &p [[buffer(0)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    device const uint8_t *in = (device const uint8_t *) p.in;
    device uint *out         = (device uint *)          p.out;
    device uint *scratch     = (device uint *)          p.scratch;

    const uint items_per_thread = 16;
    uint block_start = tg_id * tg_size * items_per_thread;
    uint block_prefix = scratch[tg_id]; // prefix-summed by prior pass

    // Per-thread local exclusive scan over its items
    uint local_prefix[items_per_thread];
    uint local_count = 0;
    for (uint i = 0; i < items_per_thread; i++) {
        local_prefix[i] = local_count;
        uint idx = block_start + tid * items_per_thread + i;
        if (idx < p.size && in[idx])
            local_count++;
    }

    // Block-level exclusive scan of per-thread counts (Hillis-Steele,
    // double-buffered in shared memory; mirrors `compress_small`).
    uint si = tid;
    shared[si] = 0;
    si += tg_size;
    shared[si] = local_count;

    for (uint offset = 1; offset < tg_size; offset <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sum = shared[si] + shared[si - offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[si] = sum;
    }

    uint thread_prefix = shared[si] - local_count;  // exclusive

    for (uint i = 0; i < items_per_thread; i++) {
        uint idx = block_start + tid * items_per_thread + i;
        if (idx < p.size && in[idx])
            out[block_prefix + thread_prefix + local_prefix[i]] = idx;
    }
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
//  reduce_dot — dot product of two arrays
//
//  Each threadgroup multiplies + reduces a tile, writing one partial result.
//  The C++ dispatch then calls block_reduce(Add) on the partials.
// ============================================================================

struct reduce_dot_params {
    ulong ptr_1;
    ulong ptr_2;
    uint  size;
    ulong out;
};

template <typename T>
kernel void reduce_dot_kernel(
    device const reduce_dot_params &p [[buffer(0)]],
    threadgroup T *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    device const T *a = (device const T *) p.ptr_1;
    device const T *b = (device const T *) p.ptr_2;
    device T *out     = (device T *)       p.out;

    // Each thread reduces two elements (standard trick for dot products)
    uint idx = tg_id * tg_size * 2 + tid;
    T sum = T(0);
    if (idx < p.size)
        sum = a[idx] * b[idx];
    if (idx + tg_size < p.size)
        sum += a[idx + tg_size] * b[idx + tg_size];

    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0)
        out[tg_id] = shared[0];
}

template [[host_name("reduce_dot_f32")]]
kernel void reduce_dot_kernel<float>(
    device const reduce_dot_params &, threadgroup float *, uint, uint, uint);

template [[host_name("reduce_dot_f16")]]
kernel void reduce_dot_kernel<half>(
    device const reduce_dot_params &, threadgroup half *, uint, uint, uint);

template [[host_name("reduce_dot_u32")]]
kernel void reduce_dot_kernel<uint>(
    device const reduce_dot_params &, threadgroup uint *, uint, uint, uint);

template [[host_name("reduce_dot_i32")]]
kernel void reduce_dot_kernel<int>(
    device const reduce_dot_params &, threadgroup int *, uint, uint, uint);

// ============================================================================
//  reduce_expanded — combine multiple per-thread copies of a buffer.
//
//  Used by ReduceMode::Expand to avoid lossy float atomic_fetch_add. The
//  scatter target is allocated as ``exp`` consecutive copies of ``size``
//  elements, with each Metal thread routing its update to copy
//  ``thread_position_in_grid % exp``. After the scatter, this kernel
//  collapses the ``exp`` copies into the first one in-place:
//      data[i] = data[i] OP data[i + size] OP ... OP data[i + (exp-1)*size]
//  for ``i`` in [0, size). One thread per output element; the per-output
//  reduction is sequential. Mirrors LLVMThreadState::reduce_expanded.
// ============================================================================

struct reduce_expanded_params {
    ulong data;     // device T * — points at the expanded scatter buffer
    uint  exp;      // number of copies
    uint  size;     // logical size of one copy
};

template <typename T, typename Red>
kernel void reduce_expanded_kernel(
    device const reduce_expanded_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.size) return;

    device T *data = (device T *) p.data;
    T value = data[tid];
    for (uint j = 1; j < p.exp; ++j)
        value = Red::apply(value, data[tid + j * p.size]);
    data[tid] = value;
}

#define INSTANTIATE_REDEXP(Name, T, Red)                                       \
    template [[host_name("reduce_expanded_" Name)]]                            \
    kernel void reduce_expanded_kernel<T, Red<T>>(                             \
        device const reduce_expanded_params &, uint);

// float32
INSTANTIATE_REDEXP("add_f32",  float,  reduction_add)
INSTANTIATE_REDEXP("mul_f32",  float,  reduction_mul)
INSTANTIATE_REDEXP("min_f32",  float,  reduction_min)
INSTANTIATE_REDEXP("max_f32",  float,  reduction_max)

// float16
INSTANTIATE_REDEXP("add_f16",  half,   reduction_add)
INSTANTIATE_REDEXP("mul_f16",  half,   reduction_mul)
INSTANTIATE_REDEXP("min_f16",  half,   reduction_min)
INSTANTIATE_REDEXP("max_f16",  half,   reduction_max)

// uint32
INSTANTIATE_REDEXP("add_u32",  uint,   reduction_add)
INSTANTIATE_REDEXP("mul_u32",  uint,   reduction_mul)
INSTANTIATE_REDEXP("min_u32",  uint,   reduction_min)
INSTANTIATE_REDEXP("max_u32",  uint,   reduction_max)
INSTANTIATE_REDEXP("or_u32",   uint,   reduction_or)
INSTANTIATE_REDEXP("and_u32",  uint,   reduction_and)

// int32
INSTANTIATE_REDEXP("min_i32",  int,    reduction_min)
INSTANTIATE_REDEXP("max_i32",  int,    reduction_max)

// uint64
INSTANTIATE_REDEXP("add_u64",  ulong,  reduction_add)
INSTANTIATE_REDEXP("mul_u64",  ulong,  reduction_mul)
INSTANTIATE_REDEXP("min_u64",  ulong,  reduction_min)
INSTANTIATE_REDEXP("max_u64",  ulong,  reduction_max)
INSTANTIATE_REDEXP("or_u64",   ulong,  reduction_or)
INSTANTIATE_REDEXP("and_u64",  ulong,  reduction_and)

// int64
INSTANTIATE_REDEXP("min_i64",  long,   reduction_min)
INSTANTIATE_REDEXP("max_i64",  long,   reduction_max)

// uint8 (booleans)
INSTANTIATE_REDEXP("or_u8",    uchar,  reduction_or)
INSTANTIATE_REDEXP("and_u8",   uchar,  reduction_and)

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
