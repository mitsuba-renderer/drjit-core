/*
    src/util.h -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include "cuda.h"

/// Descriptive names for the various reduction operations
extern const char *red_name[(int) ReduceOp::Count];

/// Fill a device memory region with constants of a given type
extern void jitc_memset_async(JitBackend backend, void *ptr, uint32_t size,
                              uint32_t isize, const void *src);

/// Reduce the given array to a single value
extern void jitc_reduce(JitBackend backend, VarType type, ReduceOp rtype,
                        const void *ptr, uint32_t size, void *out);

/// Reduce elements within blocks
extern void jitc_block_reduce(JitBackend backend, VarType type, ReduceOp op,
                              const void *in, uint32_t size, uint32_t block_size,
                              void *out);

/// Dot product reduction
extern void jitc_reduce_dot(JitBackend backend, VarType type,
                            const void *ptr_1, const void *ptr_2,
                            uint32_t size, void *out);

/// 'All' reduction for boolean arrays
extern bool jitc_all(JitBackend backend, uint8_t *values, uint32_t size);

/// 'Any' reduction for boolean arrays
extern bool jitc_any(JitBackend backend, uint8_t *values, uint32_t size);

/// Exclusive prefix sum
extern void jitc_prefix_sum(JitBackend backend, VarType vt, bool exclusive,
                            const void *in, uint32_t size, void *out);

/// Mask compression
extern uint32_t jitc_compress(JitBackend backend, const uint8_t *in, uint32_t size,
                              uint32_t *out);

/// Compute a permutation to reorder an integer array into discrete groups
extern uint32_t jitc_mkperm(JitBackend backend, const uint32_t *values, uint32_t size,
                            uint32_t bucket_count, uint32_t *perm,
                            uint32_t *offsets);

/// Perform a synchronous copy operation
extern void jitc_memcpy(JitBackend backend, void *dst, const void *src, size_t size);

/// Perform an assynchronous copy operation
extern void jitc_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size);

/// Asynchronously update a single element in memory
extern void jitc_poke(JitBackend backend, void *dst, const void *src, uint32_t size);

extern void jitc_aggregate(JitBackend backend, void *dst,
                           AggregationEntry *agg, uint32_t size);

// Enqueue a function to be run on the host once backend computation is done
extern void jitc_enqueue_host_func(JitBackend backend, void (*callback)(void *),
                                   void *payload);

/// LLVM: reduce a variable that was previously expanded due to dr.ReduceOp.Expand
extern void jitc_reduce_expanded(VarType vt, ReduceOp op, void *data, uint32_t exp, uint32_t size);
