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
    constant compress_scatter_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
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
    constant mkperm_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.size) return;

    device const uint *values  = (device const uint *) p.values;
    device atomic_uint *buckets = (device atomic_uint *) p.buckets;

    uint val = values[tid];
    atomic_fetch_add_explicit(&buckets[val], 1u, memory_order_relaxed);
}

kernel void mkperm_phase_3(
    constant mkperm_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
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
    constant mkperm_offsets_params &p [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
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
    uint tid [[thread_position_in_grid]]) {
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
