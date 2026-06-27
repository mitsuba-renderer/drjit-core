/*
    src/malloc.h -- Asynchronous memory allocation system + cache

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <drjit-core/nanostl.h>
#include <drjit-core/half.h>
#include "hash.h"

using AllocInfo = uint64_t;

/// Bit layout: [size : 52][shared : 1][backend : 3][device : 8]
inline AllocInfo alloc_info_encode(size_t size, JitBackend backend, bool shared,
                                   int device) {
    return (((uint64_t) size)    << 12) |
           (((uint64_t) shared)  << 11) |
           (((uint64_t) backend) <<  8) |
            ((uint64_t) device);
}

inline drjit::tuple<size_t, JitBackend, bool, int>
alloc_info_decode(AllocInfo v) {
    return drjit::make_tuple((size_t)     (v >> 12),
                             (JitBackend)((v >>  8) & 0x7),
                             (bool)      ((v >> 11) & 0x1),
                             (int)       ( v        & 0xFF));
}

using AllocInfoMap = tsl::robin_map<AllocInfo, std::vector<void *>, UInt64Hasher>;
using AllocUsedMap = tsl::robin_map<uintptr_t, AllocInfo, UInt64Hasher>;

/// Round to the next power of two
extern size_t round_pow2(size_t x);
extern uint32_t round_pow2(uint32_t x);

/// Human-readable name of a JIT backend
inline const char *jitc_backend_name(JitBackend b) {
    switch (b) {
        case JitBackend::None:  return "host";
        case JitBackend::LLVM:  return "LLVM";
        case JitBackend::CUDA:  return "CUDA";
        case JitBackend::AMD:   return "AMD";
        case JitBackend::Metal: return "Metal";
        default: return "?";
    }
}

/// Allocate a buffer of memory. See \ref jit_malloc() for the meaning of the
/// ``(backend, shared)`` parameters.
extern void *jitc_malloc(JitBackend backend, size_t size,
                         bool shared = false) JIT_MALLOC;

/// Release the given pointer
extern void jitc_free(void *ptr);

/// Return a no-longer-used allocation ``(info, ptr)`` to the free list. Safe to
/// call from a GPU completion handler.
extern void jitc_malloc_release(AllocInfo info, void *ptr);

/// Release a heap-owned batch of deferred ``(AllocInfo, ptr)`` frees.
extern void jitc_malloc_release_batch(void *batch);

/// Migrate an allocated memory region to a different backend
extern void *jitc_malloc_migrate(void *ptr, JitBackend backend, int move = 1);

/// Release all unused memory to the GPU / OS
extern void jitc_flush_malloc_cache(bool warn);

/// Shut down the memory allocator (calls \ref jitc_flush_malloc_cache() and reports leaks)
extern void jitc_malloc_shutdown();

/// Query the device associated with a memory allocation made using \ref jitc_malloc()
extern int jitc_malloc_device(void *ptr);

/// Return the peak memory usage (watermark) for a given backend
extern size_t jitc_malloc_watermark(JitBackend backend);

/// Clear the peak memory usage statistics
extern void jitc_malloc_clear_statistics();
