/*
    src/malloc.h -- Asynchronous memory allocation system + cache

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <drjit-core/containers.h>
#include "hash.h"

using AllocInfo = uint64_t;

inline AllocInfo alloc_info_encode(size_t size, AllocType type, int device) {
    return (((uint64_t) size) << 16) + (((uint64_t) type) << 8) +
           ((uint64_t) device);
}

inline drjit::dr_tuple<size_t, AllocType, int> alloc_info_decode(AllocInfo value) {
    return drjit::dr_tuple((size_t)(value >> 16),
                           (AllocType)((value >> 8) & 0xFF),
                           (int) (value & 0xFF));
}

using AllocInfoMap = tsl::robin_map<AllocInfo, std::vector<void *>, UInt64Hasher>;
using AllocUsedMap = tsl::robin_map<uintptr_t, AllocInfo, UInt64Hasher>;

/// Round to the next power of two
extern size_t round_pow2(size_t x);
extern uint32_t round_pow2(uint32_t x);

/// Descriptive names for the various allocation types
extern const char *alloc_type_name[(int) AllocType::Count];
extern const char *alloc_type_name_short[(int) AllocType::Count];

/// Allocate the given flavor of memory
extern void *jitc_malloc(AllocType type, size_t size) JIT_MALLOC;

/// Release the given pointer
extern void jitc_free(void *ptr);

/// Change the flavor of an allocated memory region
extern void* jitc_malloc_migrate(void *ptr, AllocType type, int move);

/// Release all unused memory to the GPU / OS
extern void jitc_flush_malloc_cache(bool warn);

/// Shut down the memory allocator (calls \ref jitc_flush_malloc_cache() and reports leaks)
extern void jitc_malloc_shutdown();

/// Query the flavor of a memory allocation made using \ref jitc_malloc()
extern AllocType jitc_malloc_type(void *ptr);

/// Query the device associated with a memory allocation made using \ref jitc_malloc()
extern int jitc_malloc_device(void *ptr);

/// Clear the peak memory usage statistics
extern void jitc_malloc_clear_statistics();
