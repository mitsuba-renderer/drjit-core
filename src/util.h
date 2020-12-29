/*
    src/util.h -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>
#include "cuda_api.h"

/// Descriptive names for the various reduction operations
extern const char *reduction_name[(int) ReductionType::Count];

/// Fill a device memory region with constants of a given type
extern void jitc_memset_async(int cuda, void *ptr, uint32_t size, uint32_t isize,
                              const void *src);

/// Reduce the given array to a single value
extern void jitc_reduce(int cuda, VarType type, ReductionType rtype,
                        const void *ptr, uint32_t size, void *out);

/// 'All' reduction for boolean arrays
extern uint8_t jitc_all(int cuda, uint8_t *values, uint32_t size);

/// 'Any' reduction for boolean arrays
extern uint8_t jitc_any(int cuda, uint8_t *values, uint32_t size);

/// Exclusive prefix sum
extern void jitc_scan_u32(int cuda, const uint32_t *in, uint32_t size,
                          uint32_t *out);

/// Mask compression
extern uint32_t jitc_compress(int cuda, const uint8_t *in, uint32_t size,
                              uint32_t *out);

/// Compute a permutation to reorder an integer array into discrete groups
extern uint32_t jitc_mkperm(int cuda, const uint32_t *values, uint32_t size,
                            uint32_t bucket_count, uint32_t *perm,
                            uint32_t *offsets);

/// Perform a synchronous copy operation
extern void jitc_memcpy(int cuda, void *dst, const void *src, size_t size);

/// Perform an assynchronous copy operation
extern void jitc_memcpy_async(int cuda, void *dst, const void *src, size_t size);

// Compute a permutation to reorder an array of registered pointers
extern VCallBucket *jitc_vcall(int cuda, const char *domain, uint32_t index,
                               uint32_t *bucket_count_out);

/// Replicate individual input elements to larger blocks
extern void jitc_block_copy(int cuda, enum VarType type, const void *in,
                            void *out, uint32_t size, uint32_t block_size);

/// Sum over elements within blocks
extern void jitc_block_sum(int cuda, enum VarType type, const void *in,
                           void *out, uint32_t size, uint32_t block_size);

/// Asynchronously update a single element in memory
extern void jitc_poke(int cuda, void *dst, const void *src, uint32_t size);
