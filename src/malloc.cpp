/*
    src/malloc.cpp -- Asynchronous memory allocation system + cache

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "llvm.h"
#include "log.h"
#include "util.h"
#include "profile.h"

#if !defined(_WIN32)
#  include <sys/mman.h>
#endif

#if defined(DRJIT_ENABLE_METAL)
#  include "metal.h"
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

void* jitc_malloc(JitBackend backend, size_t size, bool shared) {
    if (size == 0)
        return nullptr;

    // The 'shared' flag is meaningless for backend=None
    if (backend == JitBackend::None)
        shared = false;

    bool host_alloc = (backend == JitBackend::None) ||
                      (jitc_is_llvm(backend));

    if (!host_alloc || jitc_llvm_vector_width < 16) {
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

    ThreadState *ts = nullptr;
    int device = 0;
#if defined(DRJIT_ENABLE_CUDA)
    if (jitc_is_cuda(backend)) {
        ts = thread_state(backend);
        device = ts->device;
    }
#endif

#if defined(DRJIT_ENABLE_METAL)
    if (jitc_is_metal(backend)) {
        ts = thread_state(backend);
        device = ts->device;
    }
#endif

    AllocInfo ai = alloc_info_encode(size, backend, shared, device);
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
#if defined(DRJIT_ENABLE_METAL)
        void *metal_buf = nullptr; // opaque MTLBuffer handle
#endif
        for (int i = 0; i < 2; ++i) {
            {
                unlock_guard guard(state.lock);
                if (host_alloc) {
                    ptr = aligned_malloc(size);

#if defined(DRJIT_ENABLE_METAL)
                } else if (jitc_is_metal(backend)) {
                    metal_buf =
                        metal_buffer_new(ts->metal_device, size, shared, &ptr);
#endif

#if defined(DRJIT_ENABLE_CUDA)
                } else if (jitc_is_cuda(backend)) {
                    scoped_set_context guard_2(ts->context);
                    CUresult ret;

                    if (shared)
                        ret = cuMemAllocHost(&ptr, size);
                    else if (ts->memory_pool)
                        ret = cuMemAllocAsync((CUdeviceptr*) &ptr, size, ts->stream);
                    else
                        ret = cuMemAlloc((CUdeviceptr*) &ptr, size);

                    if (ret)
                        ptr = nullptr;
#endif
                }
            }
            if (ptr)
                break;
            if (i == 0) // free memory, then retry
                jitc_flush_malloc_cache(true);
        }

#if defined(DRJIT_ENABLE_METAL)
        // Register metal buffer given that we're holding the JIT lock again
        if (metal_buf)
            jitc_metal_register_buffer(ptr, metal_buf, size);
#endif

        descr = "new allocation";

        state.alloc_allocated[(int) backend] += size;
        state.alloc_watermark[(int) backend] =
            std::max(state.alloc_allocated[(int) backend],
                     state.alloc_watermark[(int) backend]);
    }

    if (unlikely(!ptr))
        jitc_raise("jit_malloc(): out of memory! Could not allocate %zu bytes "
                   "(backend=%s, shared=%i).", size,
                   jitc_backend_name(backend), (int) shared);

    state.alloc_used.emplace((uintptr_t) ptr, ai);
    state.alloc_usage[(int) backend] += size;

    (void) descr;
    jitc_trace("jit_malloc(backend=%s, shared=%i, device=%u, size=%zu): "
               DRJIT_PTR " (%s)",
               jitc_backend_name(backend), (int) shared, device, size,
               (uintptr_t) ptr, descr);

#if defined(DRJIT_SANITIZE_INTENSE)
    // Optional: intense internal sanitation instrumentation
    jitc_sanitation_checkpoint();
#endif

    return ptr;
}

/// Return a no-longer-used allocation to the free list
void jitc_malloc_release(AllocInfo info, void *ptr) {
    lock_guard guard(state.alloc_free_lock);
    state.alloc_free[info].push_back(ptr);
}

void jitc_malloc_release_batch(void *p) {
    auto *batch = (std::vector<std::pair<uint64_t, void *>> *) p;
    {
        lock_guard guard(state.alloc_free_lock);
        for (const auto &e : *batch)
            state.alloc_free[e.first].push_back(e.second);
    }
    delete batch;
}

void jitc_free(void *ptr) {
    if (!ptr)
        return;

    uintptr_t key = (uintptr_t) ptr;
    size_t hash = UInt64Hasher()(key);

    auto it = state.alloc_used.find(key, hash);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_free(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);
    AllocInfo info = it->second;
    state.alloc_used.erase_fast(it);

    auto [size, backend, shared, device] = alloc_info_decode(info);
    state.alloc_usage[(int) backend] -= size;

    // Look up the thread-local record once and access all fields through it
    ThreadLocal &tl = jitc_thread_local();
    ThreadState *ts = nullptr;

    switch (backend) {
#if defined(DRJIT_ENABLE_CUDA)
        case JitBackend::CUDA: ts = tl.ts_cuda; break;
#endif
#if defined(DRJIT_ENABLE_METAL)
        case JitBackend::Metal: ts = tl.ts_metal; break;
#endif
        case JitBackend::LLVM: ts = tl.ts_llvm; break;
        default: break;
    }

    if (ts)
        ts->notify_free(ptr);

    if (shared && !ts) // Leak allocation if Dr.Jit has already shut down
        return;

    // Free regular/host allocations immediately. Shared allocations are parked
    // for later release via ThreadState::flush_deferred_free()
    if (!shared || backend == JitBackend::None)
        jitc_malloc_release(info, ptr);
    else
        ts->actual_state()->deferred_free.push_back({ info, ptr });

    jitc_trace("jit_free(" DRJIT_PTR ", backend=%s, shared=%i, device=%i, size=%zu)",
               (uintptr_t) ptr, jitc_backend_name(backend), (int) shared,
               device, size);
}

void jitc_malloc_clear_statistics() {
    for (int i = 0; i < (int) JitBackend::Count; ++i)
        state.alloc_watermark[i] = state.alloc_allocated[i];
}


void* jitc_malloc_migrate(void *ptr, JitBackend dst_backend, int move) {
    if (!ptr)
        return nullptr;

    auto it = state.alloc_used.find((uintptr_t) ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_malloc_migrate(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);

    auto [size, src_backend, src_shared, device] = alloc_info_decode(it->second);

    if (src_backend != dst_backend &&
        src_backend != JitBackend::None &&
        dst_backend != JitBackend::None)
        jitc_raise("jit_malloc_migrate(): direct migration between distinct "
                   "backends (%s and %s) is not supported; first migrate to "
                   "the host (backend=None).",
                   jitc_backend_name(src_backend),
                   jitc_backend_name(dst_backend));

    if (!jitc_is_gpu(src_backend) && !jitc_is_gpu(dst_backend)) {
        // The LLVM backend can directly use CPU buffers and vice versa
        if (move) {
            state.alloc_usage[(int) src_backend] -= size;
            state.alloc_usage[(int) dst_backend] += size;
            state.alloc_allocated[(int) src_backend] -= size;
            state.alloc_allocated[(int) dst_backend] += size;
            it.value() = alloc_info_encode(size, dst_backend, 0, 0);
            return ptr;
        }

        bool shared = false;

        // Prefer LLVM-backend output buffer to use async copies if the source is from there
        if (jitc_is_llvm(src_backend))
            dst_backend = JitBackend::LLVM;

        // Synchronous memcpy in case the source is a CPU buffer (might be freed after this operation)
        if (src_backend == JitBackend::None &&
            jitc_is_llvm(dst_backend))
            shared = true;

        void *ptr_new = jitc_malloc(dst_backend, size, shared);
        if (jitc_is_llvm(dst_backend) && !shared)
            jitc_memcpy_async(dst_backend, ptr_new, ptr, size);
        else
            memcpy(ptr_new, ptr, size);

        return ptr_new;
    }

    // From here on at least one side is a GPU backend, and (by the rejection
    // above) the other side is either the same GPU backend or None.
    JitBackend gpu_backend = jitc_is_gpu(src_backend) ? src_backend : dst_backend;
    ThreadState *ts = thread_state(gpu_backend);
    bool same_device =
        !jitc_is_gpu(dst_backend) || device == thread_state(dst_backend)->device;

    // Same GPU backend, non-shared source, same device: move=1 is a no-op;
    // move=0 reduces to an async same-backend memcpy.
    if (src_backend == dst_backend && !src_shared && same_device) {
        if (move)
            return ptr;
        void *ptr_new = jitc_malloc(dst_backend, size);
        jitc_memcpy_async(dst_backend, ptr_new, ptr, size);
        return ptr_new;
    }

    // GPU → host: allocate a CPU-visible GPU buffer so the device → host
    // transfer can run asynchronously on the backend's queue/stream.
    bool dst_is_none = (dst_backend == JitBackend::None);
    JitBackend alloc_backend = dst_is_none ? gpu_backend : dst_backend;
    bool alloc_shared = dst_is_none;

    void *ptr_new = jitc_malloc(alloc_backend, size, alloc_shared);
    jitc_trace("jit_malloc_migrate(" DRJIT_PTR " -> " DRJIT_PTR
               ", %s (shared=%i) -> %s)",
               (uintptr_t) ptr, (uintptr_t) ptr_new,
               jitc_backend_name(src_backend), (int) src_shared,
               jitc_backend_name(dst_backend));

#if defined(DRJIT_ENABLE_METAL)
    if (jitc_is_metal(gpu_backend))
        jitc_memcpy_async(JitBackend::Metal, ptr_new, ptr, size);
#endif
#if defined(DRJIT_ENABLE_CUDA)
    if (jitc_is_cuda(gpu_backend)) {
        scoped_set_context guard(ts->context);
        if (src_backend == JitBackend::None) {
            // Host → device: cuMemcpyAsync from pageable host memory is slow;
            // stage through a pinned buffer instead.
            void *tmp = jitc_malloc(JitBackend::CUDA, size, /*shared=*/true);
            memcpy(tmp, ptr, size);
            cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                     (CUdeviceptr) tmp, size,
                                     ts->stream));
            jitc_free(tmp);
        } else {
            cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                     (CUdeviceptr) ptr, size,
                                     ts->stream));
        }
    }
#endif
    (void) ts;

    if (move)
        jitc_free(ptr);

    return ptr_new;
}

static bool jitc_flush_malloc_cache_warned = false;

static ProfilerRegion profiler_region_flush_malloc_cache("jit_flush_malloc_cache");

/// Release all unused memory to the GPU / OS
void jitc_flush_malloc_cache(bool warn) {
    if (jitc_flags() & (uint32_t) JitFlag::FreezingScope)
        jitc_raise(
            "jit_flush_malloc_cache(): Tried to free the allocation cache "
            "while recording a frozen function. This is not supported.");
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

    size_t trim_count[(int) JitBackend::Count] = { 0 },
           trim_size [(int) JitBackend::Count] = { 0 };

    /* Temporarily release the main lock */ {
        unlock_guard guard(state.lock);

        for (auto& kv : alloc_free) {
            auto [size, backend, shared, device] = alloc_info_decode(kv.first);
            const std::vector<void *> &entries = kv.second;

            trim_count[(int) backend] += entries.size();
            trim_size [(int) backend] += size * entries.size();

            if (backend == JitBackend::None || jitc_is_llvm(backend)) {
                for (void *ptr : entries)
                    aligned_free(ptr, size);
                continue;
            }

#if defined(DRJIT_ENABLE_CUDA)
            if (jitc_is_cuda(backend) &&
                (state.backends & (1u << (uint32_t) JitBackend::CUDA))) {
                const CUDADevice &dev = state.devices[device];
                scoped_set_context guard2(dev.context);
                if (shared) {
                    for (void *ptr : entries)
                        cuda_check(cuMemFreeHost(ptr));
                } else if (dev.memory_pool) {
                    for (void *ptr : entries)
                        cuda_check(cuMemFreeAsync((CUdeviceptr) ptr, dev.stream));
                } else {
                    for (void *ptr : entries)
                        cuda_check(cuMemFree((CUdeviceptr) ptr));
                }
                continue;
            }
#endif

#if defined(DRJIT_ENABLE_METAL)
            if (jitc_is_metal(backend) &&
                (state.backends & (1u << (uint32_t) JitBackend::Metal))) {
                std::vector<void *> bufs(entries.size());
                {
                    lock_guard guard_map(state.lock);
                    for (size_t i = 0; i < entries.size(); ++i)
                        bufs[i] = jitc_metal_unregister_buffer(entries[i]);
                }
                for (void *buf : bufs)
                    if (buf)
                        metal_buffer_free(buf);
                continue;
            }
#endif
        }
    }

    for (int i = 0; i < (int) JitBackend::Count; ++i)
        state.alloc_allocated[i] -= trim_size[i];

    size_t total = 0;
    for (int i = 0; i < (int) JitBackend::Count; ++i)
        total += trim_count[i];

    if (total > 0) {
        jitc_log(Debug, "jit_flush_malloc_cache(): freed");
        for (int i = 0; i < (int) JitBackend::Count; ++i) {
            if (trim_count[i] == 0)
                continue;
            jitc_log(Debug, " - %s memory: %s in %zu allocation%s",
                    jitc_backend_name((JitBackend) i),
                    jitc_mem_string(trim_size[i]),
                    trim_count[i], trim_count[i] > 1 ? "s" : "");
        }
    }
}

/// Query the device associated with a memory allocation made using \ref jitc_malloc()
int jitc_malloc_device(void *ptr) {
    auto it = state.alloc_used.find((uintptr_t) ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jitc_malloc_device(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);
    auto [size, backend, shared, device] = alloc_info_decode(it->second);
    (void) size; (void) shared;

    if (backend == JitBackend::None || jitc_is_llvm(backend))
        return -1;
    else
        return device;
}

size_t jitc_malloc_watermark(JitBackend backend) {
    return state.alloc_watermark[(int) backend];
}

void jitc_malloc_shutdown() {
    jitc_flush_malloc_cache(false);

    size_t leak_count[(int) JitBackend::Count] = { 0 },
           leak_size [(int) JitBackend::Count] = { 0 };
    for (auto kv : state.alloc_used) {
        auto [size, backend, shared, device] = alloc_info_decode(kv.second);
        (void) shared; (void) device;
        leak_count[(int) backend]++;
        leak_size [(int) backend] += size;
    }

    size_t total = 0;
    for (int i = 0; i < (int) JitBackend::Count; ++i)
        total += leak_count[i];

    if (jit_leak_warnings()) {
        if (total > 0) {
            jitc_log(Warn, "jit_malloc_shutdown(): leaked");
            for (int i = 0; i < (int) JitBackend::Count; ++i) {
                if (leak_count[i] == 0)
                    continue;
                jitc_log(Warn, " - %s memory: %s in %zu allocation%s",
                        jitc_backend_name((JitBackend) i),
                        jitc_mem_string(leak_size[i]),
                        leak_count[i], leak_count[i] > 1 ? "s" : "");
            }
        }
    }
}
