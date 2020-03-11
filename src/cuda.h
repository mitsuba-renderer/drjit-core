#pragma once

#include <enoki/jit.h>
#include "cuda_api.h"

/// Fill a device memory region with 'size' 8-bit values.
extern void jit_fill_8(void *ptr, size_t size, uint8_t value);

/// Fill a device memory region with 'size' 16-bit values.
extern void jit_fill_16(void *ptr, size_t size, uint16_t value);

/// Fill a device memory region with 'size' 32-bit values.
extern void jit_fill_32(void *ptr, size_t size, uint32_t value);

/// Fill a device memory region with 'size' 64-bit values.
extern void jit_fill_64(void *ptr, size_t size, uint64_t value);

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
extern void cuda_check_impl(CUresult errval, const char *file, const int line);
