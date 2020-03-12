#pragma once

#include <enoki/jit.h>
#include "cuda_api.h"

enum class ReductionType {
    Add, Mul, Min, Max, And, Or
};

/// Fill a device memory region with constants of a given type
extern void jit_fill(VarType type, void *ptr, size_t size, const void *src);
