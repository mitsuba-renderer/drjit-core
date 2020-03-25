#pragma once

#include <enoki/jit.h>
#include "cuda_api.h"

/// Descriptive names for the various reduction operations
extern const char *reduction_name[(int) ReductionType::Count];

/// Fill a device memory region with constants of a given type
extern void jit_fill(VarType type, void *ptr, uint32_t size, const void *src);

/// Reduce the given array to a single value
extern void jit_reduce(VarType type, ReductionType rtype, const void *ptr,
                       uint32_t size, void *out);

/// 'All' reduction for boolean arrays
extern bool jit_all(bool *values, uint32_t size);

/// 'Any' reduction for boolean arrays
extern bool jit_any(bool *values, uint32_t size);

/// Exclusive prefix sum
extern void jit_scan(const uint32_t *in, uint32_t *out, uint32_t size);

/// Compute a permutation to reorder an integer array into discrete groups
extern void jit_mkperm(const uint32_t *values, uint32_t size,
                       uint32_t bucket_count, uint32_t *perm,
                       uint32_t *offsets);
