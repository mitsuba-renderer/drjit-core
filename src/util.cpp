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
                 const void *ptr, uint32_t size, void *out) {
    thread_state(backend)->reduce(type, op, ptr, size, out);
}

/// Sum over elements within blocks
void jitc_block_reduce(JitBackend backend, VarType vt, ReduceOp op, const void *in,
                       uint32_t size, uint32_t block_size, void *out) {
    thread_state(backend)->block_reduce(vt, op, in, size, block_size, out);
}


void jitc_reduce_dot(JitBackend backend, VarType type,
                     const void *ptr_1, const void *ptr_2,
                     uint32_t size, void *out) {
    thread_state(backend)->reduce_dot(type, ptr_1, ptr_2, size, out);
}

/// 'All' reduction for boolean arrays
bool jitc_all(JitBackend backend, uint8_t *values, uint32_t size) {
    return thread_state(backend)->all(values, size);
}

/// 'Any' reduction for boolean arrays
bool jitc_any(JitBackend backend, uint8_t *values, uint32_t size) {
    return thread_state(backend)->any(values, size);
}

/// Exclusive prefix sum
void jitc_prefix_sum(JitBackend backend, VarType vt, bool exclusive,
                     const void *in, uint32_t size, void *out) {
    thread_state(backend)->prefix_sum(vt, exclusive, in, size, out);
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
