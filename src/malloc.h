/*
    src/malloc.h -- Asynchronous memory allocation system + cache

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>
#include "hash.h"

/// Data structure characterizing a memory allocation
struct AllocInfo {
    AllocType type = AllocType::Host;
    uint32_t device = 0;
    size_t size = 0;

    AllocInfo() { }

    AllocInfo(AllocType type, uint32_t device, size_t size)
        : type(type), device(device), size(size) { }

    bool operator==(const AllocInfo &at) const {
        return type == at.type && device == at.device &&
               size == at.size;
    }

    bool operator!=(const AllocInfo &at) const {
        return type != at.type || device != at.device ||
               size != at.size;
    }
};

/// Custom hasher for \ref AllocInfo
struct AllocInfoHasher {
    size_t operator()(const AllocInfo &at) const {
        size_t result = std::hash<size_t>()(at.size);
        hash_combine(result, std::hash<uint32_t>()(at.device));
        hash_combine(result, std::hash<uint32_t>()((uint32_t) at.type));
        return result;
    }
};

using AllocInfoMap = tsl::robin_map<AllocInfo, std::vector<void *>, AllocInfoHasher>;

/// Round to the next power of two
extern size_t round_pow2(size_t x);
extern uint32_t round_pow2(uint32_t x);

/// Descriptive names for the various allocation types
extern const char *alloc_type_name[(int) AllocType::Count];

/// Allocate the given flavor of memory
extern void *jit_malloc(AllocType type, size_t size) JITC_MALLOC;

/// Release the given pointer
extern void jit_free(void *ptr);

/// Schedule a function that will reclaim memory from pending jit_free()s
extern void jit_free_flush();

/// Change the flavor of an allocated memory region
extern void* jit_malloc_migrate(void *ptr, AllocType type);

/// Asynchronously prefetch a memory region
extern void jit_malloc_prefetch(void *ptr, int device);

/// Release all unused memory to the GPU / OS
extern void jit_malloc_trim(bool warn = true);

/// Shut down the memory allocator (calls \ref jit_malloc_trim() and reports leaks)
extern void jit_malloc_shutdown();
