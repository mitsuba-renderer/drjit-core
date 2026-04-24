/*
    src/util.cpp -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <condition_variable>
#include "internal.h"
#include "util.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "call.h"
#include "profile.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

const char *red_name[(int) ReduceOp::Count] = { "identity", "add", "mul",
                                                "min", "max", "and", "or" };

/// Fill a device memory region with constants of a given type
void jitc_memset_async(JitBackend backend, void *ptr, uint32_t size_,
                       uint32_t isize, const void *src) {
    thread_state(backend)->memset_async(ptr, size_, isize, src);
}

/// Perform a synchronous copy operation
void jitc_memcpy(JitBackend backend, void *dst, const void *src, size_t size) {
    ThreadState *ts = thread_state(backend);

    // Temporarily release the lock while copying
    jitc_sync_thread(ts);
    ts->memcpy(dst, src, size);
}

/// Perform an asynchronous copy operation
void jitc_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size) {
    thread_state(backend)->memcpy_async(dst, src, size);
}

void jitc_reduce(JitBackend backend, VarType type, ReduceOp op,
                 uint32_t size, const void *in, void *out) {
    thread_state(backend)->block_reduce(type, op, size, size, in, out);
}

/// Reduce over elements within blocks
void jitc_block_reduce(JitBackend backend, VarType vt, ReduceOp op,
                       uint32_t size, uint32_t block_size, const void *in,
                       void *out) {
    thread_state(backend)->block_reduce(vt, op, size, block_size, in, out);
}

/// Implements various kinds of prefix reductions
void jitc_block_prefix_reduce(JitBackend backend, VarType vt, ReduceOp op,
                              uint32_t size, uint32_t block_size,
                              bool exclusive, bool reverse, const void *in,
                              void *out) {
    thread_state(backend)->block_prefix_reduce(vt, op, size, block_size,
                                               exclusive, reverse, in, out);
}

void jitc_reduce_dot(JitBackend backend, VarType type,
                     const void *ptr_1, const void *ptr_2,
                     uint32_t size, void *out) {
    thread_state(backend)->reduce_dot(type, ptr_1, ptr_2, size, out);
}

void jitc_batched_gemm(JitBackend backend, VarType type, bool At, bool Bt,
                 uint32_t M, uint32_t N, uint32_t K,
                 const GemmBatch *batch,
                 const void *A, const void *B, void *C) {
    if (M == 0 || N == 0 || K == 0)
        return;
    if (batch && batch->n_bdims + batch->n_rdims > DRJIT_GEMM_MAX_BDIMS)
        jitc_raise("jit_batched_gemm(): n_bdims + n_rdims exceeds "
                   "DRJIT_GEMM_MAX_BDIMS.");

    // Beyond pruning trivial (``extent == 1``) dimensions, some batch
    // shapes are mathematically equivalent to a non-batched (or
    // less-batched) GEMM whose ``M`` or ``K`` dimension has been
    // enlarged: the per-batch matrices already lie in memory as if
    // they were a single larger matrix, so the batch dimension can be
    // folded away by adjusting ``M`` or ``K``.
    //
    // Two folds are valid:
    //
    // (1) Fold a grid dimension into ``M``, available when ``At`` is
    //     false. Required: ``b_stride == 0`` (the same ``B`` is reused
    //     for every grid step) and ``a_stride == M*K`` (per-step ``A``
    //     matrices are contiguous, concatenated along their first
    //     axis). Output ``C`` is already laid out as ``E`` consecutive
    //     ``(M, N)`` blocks in memory, which is bit-identical to a
    //     single ``(E*M, N)`` matrix in row-major order, so no data
    //     has to move.
    //
    // (2) Fold a reduce dimension into ``K``, available when ``At`` is
    //     true and ``Bt`` is false. Required: ``a_stride == K*M`` and
    //     ``b_stride == K*N`` (each per-reduce slab is the contiguous
    //     ``(K, M)`` / ``(K, N)`` slab the kernel expects). The
    //     reduce-dim sum ``sum_r A[r]^T @ B[r]`` then equals one
    //     larger GEMM whose operands have been concatenated along the
    //     ``K`` axis.
    //
    // Mirrored folds (folding a grid dim into ``N``, or folding a
    // reduce dim under ``At == false``) are not performed: the output
    // layout always concatenates along the outermost batch axis, which
    // lines up with ``M`` but not ``N``, and the reduce-fold geometry
    // only works when ``A``'s ``K`` axis is contiguous in memory.
    GemmBatch b = batch ? *batch : GemmBatch{};
    auto drop = [&](uint32_t d) {
        for (uint32_t i = d; i + 1 < b.n_bdims + b.n_rdims; ++i) {
            b.extent[i]   = b.extent[i + 1];
            b.a_stride[i] = b.a_stride[i + 1];
            b.b_stride[i] = b.b_stride[i + 1];
        }
        if (d < b.n_bdims) b.n_bdims--;
        else               b.n_rdims--;
    };

    for (uint32_t d = 0; d < b.n_bdims + b.n_rdims; )
        if (b.extent[d] == 1) drop(d); else ++d;

    while (!At && b.n_bdims > 0 &&
           b.b_stride[0] == 0 &&
           (uint64_t) b.a_stride[0] == (uint64_t) M * K &&
           (uint64_t) M * b.extent[0] <= UINT32_MAX) {
        M *= b.extent[0];
        drop(0);
    }

    while (At && !Bt && b.n_rdims > 0 &&
           (uint64_t) b.a_stride[b.n_bdims] == (uint64_t) K * M &&
           (uint64_t) b.b_stride[b.n_bdims] == (uint64_t) K * N &&
           (uint64_t) K * b.extent[b.n_bdims] <= UINT32_MAX) {
        K *= b.extent[b.n_bdims];
        drop(b.n_bdims);
    }

    const GemmBatch *b_eff = (b.n_bdims || b.n_rdims) ? &b : nullptr;
    thread_state(backend)->batched_gemm(type, At, Bt, M, N, K, b_eff, A, B, C);
}

/// 'All' reduction for boolean arrays (internal)
void jitc_all_async_4(JitBackend backend, uint8_t *values, uint32_t size, uint8_t *out) {
    thread_state(backend)->block_reduce_bool(values, size, out, ReduceOp::And);
}

/// 'Any' reduction for boolean arrays (asynchronous)
void jitc_any_async_4(JitBackend backend, uint8_t *values, uint32_t size, uint8_t *out) {
    thread_state(backend)->block_reduce_bool(values, size, out, ReduceOp::Or);
}

void jitc_any_async(JitBackend backend, uint8_t *values, uint32_t size, uint8_t *out) {
    jitc_any_async_4(backend, values, size, out);
    jitc_block_reduce(backend, VarType::UInt8, ReduceOp::Or, 4, 4, out, out);
}

void jitc_all_async(JitBackend backend, uint8_t *values, uint32_t size, uint8_t *out) {
    jitc_all_async_4(backend, values, size, out);
    jitc_block_reduce(backend, VarType::UInt8, ReduceOp::And, 4, 4, out, out);
}

/// 'All' reduction for boolean arrays
bool jitc_all(JitBackend backend, uint8_t *values, uint32_t size) {
    uint8_t buf[4], *tmp;
    if (backend == JitBackend::CUDA)
        tmp = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
    else
        tmp = buf;

    jitc_all_async_4(backend, values, size, tmp);
    jitc_sync_thread();

    bool result = (tmp[0] & tmp[1] & tmp[2] & tmp[3]) != 0;

    if (backend == JitBackend::CUDA)
        jitc_free(tmp);

    return result;
}

/// 'Any' reduction for boolean arrays
bool jitc_any(JitBackend backend, uint8_t *values, uint32_t size) {
    uint8_t buf[4], *tmp;
    if (backend == JitBackend::CUDA)
        tmp = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
    else
        tmp = buf;

    jitc_any_async_4(backend, values, size, tmp);
    jitc_sync_thread();

    bool result = (tmp[0] | tmp[1] | tmp[2] | tmp[3]) != 0;

    if (backend == JitBackend::CUDA)
        jitc_free(tmp);

    return result;
}

/// Mask compression
uint32_t jitc_compress(JitBackend backend, const uint8_t *in, uint32_t size, uint32_t *out) {
    return thread_state(backend)->compress(in, size, out);
}

static ProfilerRegion profiler_region_mkperm("jit_mkperm");

/// Compute a permutation to reorder an integer array into a sorted configuration
uint32_t jitc_mkperm(JitBackend backend, const uint32_t *ptr, uint32_t size,
                     uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {

    ProfilerPhase profiler(profiler_region_mkperm);
    return thread_state(backend)->mkperm(ptr, size, bucket_count, perm, offsets);
}

/// Asynchronously update a single element in memory
void jitc_poke(JitBackend backend, void *dst, const void *src, uint32_t size) {
    thread_state(backend)->poke(dst, src, size);
}

void jitc_aggregate(JitBackend backend, void *dst_, AggregationEntry *agg,
                    uint32_t size) {
    thread_state(backend)->aggregate(dst_, agg, size);
}

void jitc_enqueue_host_func(JitBackend backend, void (*callback)(void *),
                            void *payload) {
    thread_state(backend)->enqueue_host_func(callback, payload);
}

void jitc_reduce_expanded(VarType vt, ReduceOp op, void *ptr, uint32_t exp, uint32_t size) {
    thread_state_llvm->reduce_expanded(vt, op, ptr, exp, size);
}
