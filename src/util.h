#pragma once

#include <enoki/jit.h>
#include "cuda_api.h"

/// Descriptive names for the various reduction operations
extern const char *reduction_name[(int) ReductionType::Count];

/// Fill a device memory region with constants of a given type
extern void jit_fill(VarType type, void *ptr, size_t size, const void *src);

/// Reduce the given array to a single value
extern void jit_reduce(VarType type, ReductionType rtype, const void *ptr,
                       size_t size, void *out);

/// Exclusive prefix sum
extern void jit_scan(const uint32_t *in, uint32_t *out, uint32_t size);
