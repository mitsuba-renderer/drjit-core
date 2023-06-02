/*
    src/malloc.h -- Asynchronous memory allocator + cache

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "core.h"
#include "hash.h"

struct AllocInfo {
    uint64_t backend : 2;
    uint64_t device : 8;
    uint64_t shared : 1;
    uint64_t size : 53;

    AllocInfo(JitBackend backend, int device, size_t size, bool shared)
        : backend((uint64_t) backend), device((uint64_t) device),
          shared((uint64_t) shared), size((uint64_t) size) { }

    bool operator==(AllocInfo a) const {
        return backend == a.backend && shared == a.shared &&
               device == a.device && size == a.size;
    }

    bool operator!=(AllocInfo a) const { return !operator==(a); }

    int type() const { return 2 * (int) backend + (int) shared; }
};

static_assert(sizeof(AllocInfo) == sizeof(uint64_t));

struct AllocInfoHasher {
    size_t operator()(AllocInfo a) const {
        uint64_t u = 0;
        memcpy(&u, &a, sizeof(uint64_t));
        return UInt64Hasher()(u);
    }
};

using AllocInfoMap = tsl::robin_map<AllocInfo, std::vector<void *>, AllocInfoHasher>;
using AllocUsedMap = tsl::robin_map<uintptr_t, AllocInfo, UInt64Hasher>;

/// Round to the next power of two
extern size_t round_pow2(size_t x);
extern uint32_t round_pow2(uint32_t x);

/// Allocate the given flavor of memory
extern void *jitc_malloc(JitBackend backend, size_t size, bool shared = 0) JIT_MALLOC;
inline void *jitc_malloc_shared(JitBackend backend, size_t size) {
    return jitc_malloc(backend, size, true);
}

/// Release the given pointer
extern void jitc_free(void *ptr);

/// Change the flavor of an allocated memory region
extern void* jitc_malloc_migrate(void *ptr, JitBackend dst, int move);

/// Release all unused memory to the GPU / OS
extern void jitc_flush_malloc_cache(bool warn);

/// Shut down the memory allocator (calls \ref jitc_flush_malloc_cache() and reports leaks)
extern void jitc_malloc_shutdown();

/// Query the device associated with a memory allocation made using \ref jitc_malloc()
extern int jitc_malloc_device(void *ptr);

/// Clear the peak memory usage statistics
extern void jitc_malloc_clear_statistics();
