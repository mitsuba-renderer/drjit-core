/*
    src/malloc.cpp -- Asynchronous memory allocator + cache

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "malloc.h"
#include "log.h"
#include "util.h"
#include "llvm.h"
#include "state.h"
#include "thread_state.h"
#include "profiler.h"
#include "backends.h"

#if !defined(_WIN32)
#  include <sys/mman.h>
#endif

// Try to use huge pages for allocations > 2M (only on Linux)
#if defined(__linux__)
#  define DRJIT_HUGEPAGE 1
#else
#  define DRJIT_HUGEPAGE 0
#endif

#define DRJIT_HUGEPAGE_SIZE (2 * 1024 * 1024)

static_assert(
    sizeof(tsl::detail_robin_hash::bucket_entry<AllocUsedMap::value_type, false>) == 24,
    "AllocUsedMap: incorrect bucket size, likely an issue with padding/packing!");

/// Identifiers for various types of memory allocations
constexpr size_t AllocTypeCount = 2 * (size_t) JitBackend::Count;

const char *alloc_type_name[AllocTypeCount] = {
    "host",  "???", // shared memory not used for host allocations
    "LLVM",  "LLVM shared",
    "CUDA",  "CUDA shared",
    "Metal", "Metal shared"
};

// Round an unsigned integer up to the next power of two (64 bit version)
size_t round_pow2(size_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;  x |= x >> 32;
    return x + 1;
}

// Round an unsigned integer up to the next power of two (32 bit version)
uint32_t round_pow2(uint32_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// Perform a potentially large allocation with 64-byte alignment for AVX512
static void *aligned_malloc(size_t size) {
#if !defined(_WIN32)
    // Use posix_memalign for small allocations and mmap() for big ones
    if (size < DRJIT_HUGEPAGE_SIZE) {
        void *ptr = nullptr;
        int rv = posix_memalign(&ptr, 64, size);
        return rv == 0 ? ptr : nullptr;
    } else {
        void *ptr;

#if DRJIT_HUGEPAGE
        // Attempt to allocate a 2M page directly
        ptr = mmap(0, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_HUGETLB, -1, 0);
        if (ptr != MAP_FAILED)
            return ptr;
#endif

        // Allocate 4K pages
        ptr = mmap(0, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);

#if DRJIT_HUGEPAGE
        // .. and advise the OS to convert to 2M pages
        if (ptr != MAP_FAILED)
            madvise(ptr, size, MADV_HUGEPAGE);
#endif

        return ptr;
    }
#else
    return _aligned_malloc(size, 64);
#endif
}

static void aligned_free(void *ptr, size_t size) {
#if !defined(_WIN32)
    if (size < DRJIT_HUGEPAGE_SIZE)
        free(ptr);
    else
        munmap(ptr, size);
#else
    (void) size;
    _aligned_free(ptr);
#endif
}

static ProfilerRegion profiler_region_malloc("jit_malloc");

void* jitc_malloc(JitBackend backend, size_t size, bool shared) {
    if (size == 0)
        return nullptr;
    ProfilerPhase profiler(profiler_region_malloc);

#if defined(DRJIT_ENABLE_CUDA)
    // Round up to the next multiple of 64 bytes
    if (backend == JitBackend::CUDA)
        size = (size + 63) / 64 * 64;
#endif

#if defined(DRJIT_ENABLE_LLVM)
    if (backend == JitBackend::LLVM) {
        // Increase allocation to a multiple of the biggest packet size
        size_t packet_size = std::max(jitc_llvm_vector_width, 8u) * sizeof(double);
        size = (size + packet_size - 1) / packet_size * packet_size;
    }
#endif

    /* Round 'size' to the next larger power of two. This is somewhat
       wasteful, but reduces the number of different sizes that an allocation
       can have to a manageable amount that facilitates re-use. */
    size = round_pow2(size);

    ThreadState *ts = nullptr;
    int device = 0;

    if (backend == JitBackend::None) {
        // No distinction between shared/normal memory
        shared = false;
    } else {
        ts = thread_state(backend);
#if defined(DRJIT_ENABLE_CUDA) || defined(DRJIT_ENABLE_METAL)
        device = ts->device;
#endif
    }

    AllocInfo ai(backend, device, size, shared);
    bool reused = false;
    void *ptr = nullptr;

    /* Try to reuse a freed allocation */ {
        lock_guard guard(state.alloc_free_lock);
        auto it = state.alloc_free.find(ai);

        if (it != state.alloc_free.end()) {
            std::vector<void *> &list = it.value();
            if (!list.empty()) {
                ptr = list.back();
                list.pop_back();
                reused = true;
            }
        }
    }

    // Otherwise, allocate memory
    if (unlikely(!ptr)) {
        for (int i = 0; i < 2; ++i) {
            /* Temporarily release the main lock */ {
                unlock_guard guard(state.lock);

                if (backend == JitBackend::None || backend == JitBackend::LLVM)
                    ptr = aligned_malloc(size);
                else
                    ptr = ts->malloc(size, shared);
            }

            if (ptr) // success
                break;

            if (i == 0) // free up some memory, then retry
                jitc_flush_malloc_cache(true);
        }

        size_t &allocated = state.alloc_allocated[ai.type()],
               &watermark = state.alloc_watermark[ai.type()];

        allocated += size;
        watermark = std::max(allocated, watermark);
    }

    if (unlikely(!ptr))
        jitc_raise("jit_malloc(): could not allocate %zu bytes of %s memory.",
                   size, alloc_type_name[ai.type()]);

    state.alloc_used.emplace((uintptr_t) ptr, ai);
    state.alloc_usage[ai.type()] += size;

    (void) reused; // don't warn if tracing is disabled

    jitc_trace("jit_malloc(type=%s, size=%zu): " DRJIT_PTR " (%s)",
               alloc_type_name[ai.type()], size, (uintptr_t) ptr,
               reused ? "reused" : "new allocation");

    return ptr;
}

void jitc_free(void *ptr) {
    if (!ptr)
        return;

    auto it = state.alloc_used.find((uintptr_t) ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_fail("jit_free(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);

    AllocInfo ai = it->second;
    state.alloc_used.erase(it);
    state.alloc_usage[ai.type()] -= ai.size;

    if (!ai.shared) {
        // Immediately make allocation available for asynchronous reuse
        lock_guard guard(state.alloc_free_lock);
        state.alloc_free[ai].push_back(ptr);
    } else {
        /* Pinned memory is accessible from both host and device. Safe reuse of
           this allocation requires waiting for any computation using it to finish.
           To avoid synchronizing host and device (which would be costly), we
           instead insert a callback into the device command queue */

        struct ReleaseRecord {
            AllocInfo ai;
            void *ptr;
        };

        ReleaseRecord *r =
            (ReleaseRecord *) malloc_check(sizeof(ReleaseRecord));
        r->ai = ai;
        r->ptr = ptr;

        auto callback = [](void *p) {
            ReleaseRecord *r2 = (ReleaseRecord *) p;
            /* In critical section */ {
                lock_guard guard(state.alloc_free_lock);
                state.alloc_free[r2->ai].push_back(r2->ptr);
            }
            free(r2);
        };

        thread_state(ai.backend)->enqueue_callback(callback, r);
    }

    jitc_trace("jit_free(" DRJIT_PTR ", type=%s, size=%zu)",
               (uintptr_t) ptr, alloc_type_name[ai.type()], size);
}

void jitc_malloc_clear_statistics() {
    for (size_t i = 0; i < AllocTypeCount; ++i)
        state.alloc_watermark[i] = state.alloc_allocated[i];
}

void* jitc_malloc_migrate(void *src, JitBackend dst_b, int move) {
    if (!src)
        return nullptr;

    auto it = state.alloc_used.find((uintptr_t) src);
    if (unlikely(it == state.alloc_used.end()))
        jitc_fail("jit_malloc_migrate(): unknown address " DRJIT_PTR "!",
                  (uintptr_t) src);

    AllocInfo src_ai = it->second;
    JitBackend src_b = (JitBackend) src_ai.backend;
    size_t size = (size_t) src_ai.size;

    if (src_b != dst_b && src_b != JitBackend::None && dst_b != JitBackend::None)
        jitc_raise("jit_malloc_migrate(): migrations between different "
                   "backends are not supported.");

    ThreadState *ts = nullptr;
    if (src_b != JitBackend::None)
        ts = thread_state(src_b);
    else if (dst_b != JitBackend::None)
        ts = thread_state(dst_b);

    AllocInfo dst_ai(dst_b, ts ? ts->device : 0, size, 0);
    void *dst = nullptr;

#if defined(DRJIT_ENABLE_LLVM)
    if (move && (src_b == JitBackend::LLVM || dst_b == JitBackend::LLVM)) {
        /* The 'None' and 'LLVM' backends both store data in host memory.
           Move migrations can be done by changing the pointer type. */

        state.alloc_usage[src_ai.type()] -= size;
        state.alloc_usage[dst_ai.type()] += size;
        state.alloc_allocated[src_ai.type()] -= size;
        state.alloc_allocated[dst_ai.type()] += size;
        it.value() = dst_ai;

        dst = src;
    }
#endif

    // Simple case if the source and backends are the same
    if (!dst && src_b == dst_b) {
        if (move && src_ai.device == dst_ai.device &&
                    src_ai.shared == dst_ai.shared) {
            dst = src;
        } else {
            dst = jitc_malloc(src_b, size);
            jitc_memcpy_async(src_b, dst, src, size);
        }
    }

    if (!dst) {
        dst = jitc_malloc(dst_b, size);

        if (src_b == JitBackend::None) {
            // Host -> Device copies performed via an intermediate shared buffer
            void *tmp = jitc_malloc(dst_b, size, /* shared = */ true);
            memcpy(ts->host_ptr(tmp), src, size);
            ts->enqueue_memcpy(dst, tmp, size);
            jitc_free(tmp);
        } else {
            ts->enqueue_memcpy(dst, src, size);
        }
    }

    jitc_trace("jit_malloc_migrate(" DRJIT_PTR " -> " DRJIT_PTR ", %s -> %s)",
               (uintptr_t) src, (uintptr_t) dst, alloc_type_name[ai.type()],
               alloc_type_name[2 * (int) dst_b]);

    if (move)
        jitc_free(src);

    return dst;
}

static bool jitc_flush_malloc_cache_warned = false;

static ProfilerRegion profiler_region_flush_malloc_cache("jit_flush_malloc_cache");

/// Release all unused memory to the GPU / OS
void jitc_flush_malloc_cache(bool warn) {
    if (warn && !jitc_flush_malloc_cache_warned) {
        jitc_log(
            Warn,
            "jit_flush_malloc_cache(): Dr.Jit exhausted the available memory and had "
            "to flush its allocation cache to free up additional memory. This "
            "is an expensive operation and will have a negative effect on "
            "performance. You may want to change your computation so that it "
            "uses less memory. This warning will only be displayed once.");

        jitc_flush_malloc_cache_warned = true;
    }
    ProfilerPhase profiler(profiler_region_flush_malloc_cache);
    AllocInfoMap alloc_free;

    /* Wait for currently running computation to finish. Do this without
       releasing the central lock so that no further computation can be
       enqueued in the meantime. This ensures that the allocations in
       the 'alloc_free' map are safe to be released */

    for (ThreadState *ts : state.tss)
        ts->sync();

    /* Critical section */ {
        lock_guard guard(state.alloc_free_lock);
        alloc_free.swap(state.alloc_free);
    }

    size_t trim_count[AllocTypeCount] = { },
           trim_size [AllocTypeCount] = { };

    /* Temporarily release the main lock */ {
        unlock_guard guard(state.lock);

        for (auto& kv : alloc_free) {
            const AllocInfo ai = kv.first;
            const std::vector<void *> &entries = kv.second;

            trim_count[ai.type()] += entries.size();
            trim_size[ai.type()] += ai.size * entries.size();

            switch ((JitBackend) ai.backend) {
#if defined(DRJIT_ENABLE_LLVM)
                case JitBackend::LLVM:
                    [[fallthrough]];
#endif

                case JitBackend::None:
                    for (void *ptr : entries)
                        aligned_free(ptr, ai.size);
                    break;

#if defined(DRJIT_ENABLE_CUDA)
                case JitBackend::CUDA:
                    for (void *ptr : entries)
                        jitc_cuda_free(ai.device, ai.shared, ptr);
                    break;
#endif

#if defined(DRJIT_ENABLE_METAL)
                case JitBackend::Metal:
                    for (void *ptr : entries)
                        jitc_metal_free(ai.device, ai.shared, ptr);
                    break;
#endif

                default:
                    jitc_fail("jit_flush_malloc_cache(): unhandled backend!");
            }
        }
    }

    size_t total = 0;
    for (size_t i = 0; i < AllocTypeCount; ++i) {
        state.alloc_allocated[i] -= trim_size[i];
        total += trim_count[i];
    }

    if (total) {
        jitc_log(Debug, "jit_flush_malloc_cache(): freed");
        for (size_t i = 0; i < AllocTypeCount; ++i) {
            if (trim_count[i] == 0)
                continue;

            jitc_log(Debug, " - %s: %s in %zu allocation%s",
                    alloc_type_name[i], jitc_mem_string(trim_size[i]),
                    trim_count[i], trim_count[i] > 1 ? "s" : "");
        }
    }
}

void jitc_malloc_shutdown() {
    jitc_flush_malloc_cache(false);

    size_t leak_count[AllocTypeCount] = { },
           leak_size [AllocTypeCount] = { };
    for (auto kv : state.alloc_used) {
        int type = kv.second.type();
        leak_count[type]++;
        leak_size[type] += kv.second.size;
    }

    size_t total = 0;
    for (size_t i = 0; i < AllocTypeCount; ++i)
        total += leak_count[i];

    if (total) {
        jitc_log(Warn, "jit_malloc_shutdown(): leaked");

        for (size_t i = 0; i < AllocTypeCount; ++i) {
            if (leak_count[i] == 0)
                continue;

            jitc_log(Warn, " - %s: %s in %zu allocation%s",
                    alloc_type_name[i], jitc_mem_string(leak_size[i]),
                    leak_count[i], leak_count[i] > 1 ? "s" : "");
        }
    }
}
