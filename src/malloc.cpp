/*
    src/malloc.cpp -- Asynchronous memory allocation system + cache

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "util.h"
#include "profiler.h"

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

const char *alloc_type_name[(int) AllocType::Count] = {
    "host",   "host-async", "host-pinned", "device"
};

const char *alloc_type_name_short[(int) AllocType::Count] = {
    "host       ",
    "host-async ",
    "host-pinned",
    "device     "
};

// Round an unsigned integer up to a power of two
size_t round_pow2(size_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;  x |= x >> 32;
    return x + 1;
}

uint32_t round_pow2(uint32_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}


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

void* jitc_malloc(AllocType type, size_t size) {
    if (size == 0)
        return nullptr;

    if ((type != AllocType::Host && type != AllocType::HostAsync) ||
        jitc_llvm_vector_width < 16) {
        // Round up to the next multiple of 64 bytes
        size = (size + 63) / 64 * 64;
    } else {
        size_t packet_size = jitc_llvm_vector_width * sizeof(double);
        size = (size + packet_size - 1) / packet_size * packet_size;
    }

    /* Round 'size' to the next larger power of two. This is somewhat
       wasteful, but reduces the number of different sizes that an allocation
       can have to a manageable amount that facilitates re-use. */
    size = round_pow2(size);

    JitBackend backend =
        (type == AllocType::Device || type == AllocType::HostPinned)
            ? JitBackend::CUDA
            : JitBackend::LLVM;
    ThreadState *ts = nullptr;

    int device = 0;
    if (backend == JitBackend::CUDA) {
        ts = thread_state(backend);
        device = ts->device;
    }

    AllocInfo ai = alloc_info_encode(size, type, device);
    const char *descr = nullptr;
    void *ptr = nullptr;

    /* Try to reuse a freed allocation */ {
        lock_guard guard(state.alloc_free_lock);
        auto it = state.alloc_free.find(ai);

        if (it != state.alloc_free.end()) {
            std::vector<void *> &list = it.value();
            if (!list.empty()) {
                ptr = list.back();
                list.pop_back();
                descr = "reused";
            }
        }
    }

    // Otherwise, allocate memory
    if (unlikely(!ptr)) {
        for (int i = 0; i < 2; ++i) {
            unlock_guard guard(state.lock);
            /* Temporarily release the main lock */ {
                if (backend != JitBackend::CUDA) {
                    ptr = aligned_malloc(size);
                } else {
                    scoped_set_context guard_2(ts->context);
                    CUresult ret;

                    if (type == AllocType::HostPinned)
                        ret = cuMemAllocHost(&ptr, size);
                    else if (ts->memory_pool)
                        ret = cuMemAllocAsync((CUdeviceptr*) &ptr, size, ts->stream);
                    else
                        ret = cuMemAlloc((CUdeviceptr*) &ptr, size);

                    if (ret)
                        ptr = nullptr;
                }
            }
            if (ptr)
                break;
            if (i == 0) // free memory, then retry
                jitc_flush_malloc_cache(true);
        }
        descr = "new allocation";

        size_t &allocated = state.alloc_allocated[(int) type],
               &watermark = state.alloc_watermark[(int) type];

        allocated += size;
        watermark = std::max(allocated, watermark);
    }

    if (unlikely(!ptr))
        jitc_raise("jit_malloc(): out of memory! Could not allocate %zu bytes "
                   "of %s memory.", size, alloc_type_name[(int) type]);

    state.alloc_used.emplace((uintptr_t) ptr, ai);
    state.alloc_usage[(int) type] += size;

    (void) descr; // don't warn if tracing is disabled
    if (ts)
        jitc_trace("jit_malloc(type=%s, device=%u, size=%zu): " DRJIT_PTR " (%s)",
                  alloc_type_name[(int) type], device, size, (uintptr_t) ptr, descr);
    else
        jitc_trace("jit_malloc(type=%s, size=%zu): " DRJIT_PTR " (%s)",
                   alloc_type_name[(int) type], size, (uintptr_t) ptr, descr);

    return ptr;
}

void jitc_free(void *ptr) {
    if (!ptr)
        return;

    auto it = state.alloc_used.find((uintptr_t) ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_free(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);
    AllocInfo info = it->second;
    state.alloc_used.erase(it);

    auto [size, type, device] = alloc_info_decode(info);
    state.alloc_usage[(int) type] -= size;

    if (type != AllocType::HostPinned) {
        lock_guard guard(state.alloc_free_lock);
        state.alloc_free[info].push_back(ptr);
    } else {
        /* Host-pinned memory is released asynchronously by inserting
           an event into the CUDA stream */
        struct ReleaseRecord {
            AllocInfo info;
            void *ptr;
        };
        ReleaseRecord *r =
            (ReleaseRecord *) malloc_check(sizeof(ReleaseRecord));
        r->info = info;
        r->ptr = ptr;
        cuda_check(cuLaunchHostFunc(
            thread_state_cuda->stream,
            [](void *p) {
                ReleaseRecord *r2 = (ReleaseRecord *) p;
                {
                    lock_guard guard(state.alloc_free_lock);
                    state.alloc_free[r2->info].push_back(r2->ptr);
                }
                free(r2);
            },
            r));
    }

    if (type == AllocType::Device || type == AllocType::HostPinned)
        jitc_trace("jit_free(" DRJIT_PTR ", type=%s, device=%i, size=%zu)",
                   (uintptr_t) ptr, alloc_type_name[(int) type], device, size);
    else
        jitc_trace("jit_free(" DRJIT_PTR ", type=%s, size=%zu)",
                   (uintptr_t) ptr, alloc_type_name[(int) type], size);
}

void jitc_malloc_clear_statistics() {
    for (int i = 0; i < (int) AllocType::Count; ++i)
        state.alloc_watermark[i] = state.alloc_allocated[i];
}

void* jitc_malloc_migrate(void *ptr, AllocType dst_type, int move) {
    if (!ptr)
        return nullptr;

    auto it = state.alloc_used.find((uintptr_t) ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_malloc_migrate(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);

    auto [size, src_type, device] = alloc_info_decode(it->second);

    JitBackend src_backend =
        (src_type == AllocType::Device || src_type == AllocType::HostPinned)
            ? JitBackend::CUDA
            : JitBackend::LLVM;

    // Maybe nothing needs to be done..
    if (src_type == dst_type &&
        (dst_type != AllocType::Device ||
         device == thread_state(JitBackend::CUDA)->device)) {
        if (move) {
            return ptr;
        } else {
            void *ptr_new = jitc_malloc(dst_type, size);
            if (dst_type == AllocType::Host)
                memcpy(ptr_new, ptr, size);
            else
                jitc_memcpy_async(src_backend, ptr_new, ptr, size);
            return ptr_new;
        }
    }

    if ((src_type == AllocType::Host && dst_type == AllocType::HostAsync) ||
        (src_type == AllocType::HostAsync && dst_type == AllocType::Host)) {
        if (move) {
            state.alloc_usage[(int) src_type] -= size;
            state.alloc_usage[(int) dst_type] += size;
            state.alloc_allocated[(int) src_type] -= size;
            state.alloc_allocated[(int) dst_type] += size;
            it.value() = alloc_info_encode(size, dst_type, device);
            return ptr;
        } else {
            void *ptr_new = jitc_malloc(dst_type, size);
            jitc_memcpy_async(src_backend, ptr_new, ptr, size);

            // When copying from the host, wait for the operation to finish
            if (src_type == AllocType::Host)
                jitc_sync_thread();
            return ptr_new;
        }
    }

    if (dst_type == AllocType::HostAsync || src_type == AllocType::HostAsync)
        jitc_raise("jit_malloc_migrate(): migrations between CUDA and "
                   "host-asynchronous memory are not supported.");

    /// At this point, source or destination is a GPU array, get assoc. state
    ThreadState *ts = thread_state(JitBackend::CUDA);

    if (dst_type == AllocType::Host) // Upgrade from host to host-pinned memory
        dst_type = AllocType::HostPinned;

    void *ptr_new = jitc_malloc(dst_type, size);
    jitc_trace("jit_malloc_migrate(" DRJIT_PTR " -> " DRJIT_PTR ", %s -> %s)",
              (uintptr_t) ptr, (uintptr_t) ptr_new,
              alloc_type_name[(int) src_type],
              alloc_type_name[(int) dst_type]);

    scoped_set_context guard(ts->context);
    if (src_type == AllocType::Host) {
        // Host -> Device memory, create an intermediate host-pinned array
        void *tmp = jitc_malloc(AllocType::HostPinned, size);
        memcpy(tmp, ptr, size);
        cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                 (CUdeviceptr) ptr, size,
                                 ts->stream));
        jitc_free(tmp);
    } else {
        cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                 (CUdeviceptr) ptr, size,
                                 ts->stream));
    }

    if (move)
        jitc_free(ptr);

    return ptr_new;
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

    // Another synchronization to be sure that 'alloc_free' can be released
    jitc_sync_all_devices();

    /* Critical section */ {
        lock_guard guard(state.alloc_free_lock);
        alloc_free.swap(state.alloc_free);
    }

    size_t trim_count[(int) AllocType::Count] = { 0 },
           trim_size [(int) AllocType::Count] = { 0 };

    /* Temporarily release the main lock */ {
        unlock_guard guard(state.lock);

        for (auto& kv : alloc_free) {
            auto [size, type, device] = alloc_info_decode(kv.first);
            const std::vector<void *> &entries = kv.second;

            trim_count[(int) type] += entries.size();
            trim_size[(int) type] += size * entries.size();

            switch ((AllocType) type) {
                case AllocType::Device:
                    if (state.backends & (uint32_t) JitBackend::CUDA) {
                        const Device &dev = state.devices[device];
                        scoped_set_context guard2(dev.context);
                        if (dev.memory_pool) {
                            for (void *ptr : entries)
                                cuda_check(cuMemFreeAsync((CUdeviceptr) ptr, dev.stream));
                        } else {
                            for (void *ptr : entries)
                                cuda_check(cuMemFree((CUdeviceptr) ptr));
                        }
                    }
                    break;

                case AllocType::HostPinned:
                    if (state.backends & (uint32_t) JitBackend::CUDA) {
                        const Device &dev = state.devices[device];
                        scoped_set_context guard2(dev.context);
                        for (void *ptr : entries)
                            cuda_check(cuMemFreeHost(ptr));
                    }
                    break;

                case AllocType::Host:
                case AllocType::HostAsync:
                    for (void *ptr : entries)
                        aligned_free(ptr, size);
                    break;

                default:
                    jitc_fail("jit_flush_malloc_cache(): unsupported allocation type!");
            }
        }
    }

    for (int i = 0; i < (int) AllocType::Count; ++i)
        state.alloc_allocated[i] -= trim_size[i];

    size_t total = 0;
    for (int i = 0; i < (int) AllocType::Count; ++i)
        total += trim_count[i];

    if (total > 0) {
        jitc_log(Debug, "jit_flush_malloc_cache(): freed");
        for (int i = 0; i < (int) AllocType::Count; ++i) {
            if (trim_count[i] == 0)
                continue;
            jitc_log(Debug, " - %s memory: %s in %zu allocation%s",
                    alloc_type_name[i], jitc_mem_string(trim_size[i]),
                    trim_count[i], trim_count[i] > 1 ? "s" : "");
        }
    }
}

/// Query the flavor of a memory allocation made using \ref jitc_malloc()
AllocType jitc_malloc_type(void *ptr) {
    auto it = state.alloc_used.find((uintptr_t) ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_malloc_type(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);
    auto [size, type, device] = alloc_info_decode(it->second);
    (void) size; (void) device;
    return type;
}

/// Query the device associated with a memory allocation made using \ref jitc_malloc()
int jitc_malloc_device(void *ptr) {
    auto it = state.alloc_used.find((uintptr_t) ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jitc_malloc_device(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);
    auto [size, type, device] = alloc_info_decode(it->second);
    (void) size;

    if (type == AllocType::Host || type == AllocType::HostAsync)
        return -1;
    else
        return device;
}

void jitc_malloc_shutdown() {
    jitc_flush_malloc_cache(false);

    size_t leak_count[(int) AllocType::Count] = { 0 },
           leak_size [(int) AllocType::Count] = { 0 };
    for (auto kv : state.alloc_used) {
        auto [size, type, device] = alloc_info_decode(kv.second);
        (void) device;
        leak_count[(int) type]++;
        leak_size[(int) type] += size;
    }

    size_t total = 0;
    for (int i = 0; i < (int) AllocType::Count; ++i)
        total += leak_count[i];

    if (total > 0) {
        jitc_log(Warn, "jit_malloc_shutdown(): leaked");
        for (int i = 0; i < (int) AllocType::Count; ++i) {
            if (leak_count[i] == 0)
                continue;

            jitc_log(Warn, " - %s memory: %s in %zu allocation%s",
                    alloc_type_name[i], jitc_mem_string(leak_size[i]),
                    leak_count[i], leak_count[i] > 1 ? "s" : "");
        }
    }
}
