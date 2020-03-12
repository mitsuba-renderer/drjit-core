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
