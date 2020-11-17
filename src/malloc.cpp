/*
    src/malloc.cpp -- Asynchronous memory allocation system + cache

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "tbb.h"
#include "util.h"

static_assert(
    sizeof(tsl::detail_robin_hash::bucket_entry<AllocUsedMap::value_type, false>) == 24,
    "AllocUsedMap: incorrect bucket size, likely an issue with padding/packing!");

const char *alloc_type_name[(int) AllocType::Count] = {
    "host",   "host-async", "host-pinned",
    "device", "managed",    "managed-read-mostly"
};

const char *alloc_type_name_short[(int) AllocType::Count] = {
    "host       ",
    "host-async ",
    "host-pinned",
    "device     ",
    "managed    ",
    "managed/rm "
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

void* jit_malloc(AllocType type, size_t size) {
    if (size == 0)
        return nullptr;

    if ((type != AllocType::Host && type != AllocType::HostAsync) ||
        jit_llvm_vector_width < 16) {
        // Round up to the next multiple of 64 bytes
        size = (size + 63) / 64 * 64;
    } else {
        size_t packet_size = jit_llvm_vector_width * sizeof(double);
        size = (size + packet_size - 1) / packet_size * packet_size;
    }

#if !defined(ENOKI_JIT_ENABLE_TBB)
    /// There are no streams / host-asynchronous allocations when TBB is disabled
    if (type == AllocType::HostAsync)
        type = AllocType::Host;
#endif

    /* Round 'size' to the next larger power of two. This is somewhat
       wasteful, but reduces the number of different sizes that an allocation
       can have to a manageable amount that facilitates re-use. */
    size = round_pow2(size);

    AllocInfo ai(size, type, 0);

    const char *descr = nullptr;
    void *ptr = nullptr;
    Stream *stream = active_stream;

    /* Acquire lock protecting stream->release_chain and state.alloc_free */ {
        lock_guard guard(state.malloc_mutex);

        if (type == AllocType::Device || type == AllocType::HostAsync) {
            if (unlikely(!stream))
                jit_raise(
                    "jit_malloc(): you must specify an active device using "
                    "jit_set_device() before allocating a device/host-async memory!");
            else if (unlikely(!stream->cuda == (type == AllocType::Device)))
                jit_raise("jit_malloc(): you must specify the right backend "
                          "via jit_set_device() before allocating a "
                          "device/host-async memory!");
            ai.device = stream->cuda ? stream->device : 0;
        }

        if (type == AllocType::Device || type == AllocType::HostAsync) {
            /* Check for arrays with a pending free operation on the current
               stream. This only works for device or host-async memory, as other
               allocation flavors (host-pinned, shared, shared-read-mostly) can be
               accessed from both CPU & GPU and might still be used. */

            ReleaseChain *chain = stream->release_chain;
            while (chain) {
                auto it = chain->entries.find(ai);
                if (it != chain->entries.end()) {
                    auto &list = it.value();
                    if (!list.empty()) {
                        ptr = list.back();
                        list.pop_back();
                        descr = "reused local";
                        break;
                    }
                }

                chain = chain->next;
            }
        }

        // Look globally. Are there suitable freed arrays?
        if (ptr == nullptr) {
            auto it = state.alloc_free.find(ai);

            if (it != state.alloc_free.end()) {
                std::vector<void *> &list = it.value();
                if (!list.empty()) {
                    ptr = list.back();
                    list.pop_back();
                    descr = "reused global";
                }
            }
        }
    }

    // 3. Looks like we will have to allocate some memory..
    if (unlikely(ptr == nullptr)) {
        if (type == AllocType::Host || type == AllocType::HostAsync) {
            int rv;
            /* Temporarily release the main lock */ {
                unlock_guard guard(state.mutex);
#if !defined(_WIN32)
                rv = posix_memalign(&ptr, 64, ai.size);
#else
                ptr = _aligned_malloc(ai.size, 64);
                rv = ptr == nullptr ? ENOMEM : 0;
#endif
            }
            if (rv == ENOMEM) {
                jit_malloc_trim();
                /* Temporarily release the main lock */ {
                    unlock_guard guard(state.mutex);
#if !defined(_WIN32)
                    rv = posix_memalign(&ptr, 64, ai.size);
#else
                    ptr = _aligned_malloc(ai.size, 64);
                    rv = ptr == nullptr ? ENOMEM : 0;
#endif
                }
            }
            if (rv != 0)
                ptr = nullptr;
        } else {
            scoped_set_context guard(stream->context);
            CUresult (*alloc) (CUdeviceptr *, size_t) = nullptr;

            auto cuMemAllocManaged_ = [](CUdeviceptr *ptr_, size_t size_) {
                return cuMemAllocManaged(ptr_, size_, CU_MEM_ATTACH_GLOBAL);
            };

            auto cuMemAllocManagedReadMostly_ = [](CUdeviceptr *ptr_, size_t size_) {
                CUresult ret = cuMemAllocManaged(ptr_, size_, CU_MEM_ATTACH_GLOBAL);
                if (ret == CUDA_SUCCESS)
                    cuda_check(cuMemAdvise(*ptr_, size_, CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
                return ret;
            };

            switch (type) {
                case AllocType::HostPinned:        alloc = (decltype(alloc)) cuMemAllocHost; break;
                case AllocType::Device:            alloc = cuMemAlloc; break;
                case AllocType::Managed:           alloc = cuMemAllocManaged_; break;
                case AllocType::ManagedReadMostly: alloc = cuMemAllocManagedReadMostly_; break;
                default:
                    jit_fail("jit_malloc(): internal-error unsupported allocation type!");
            }

            CUresult ret;

            /* Temporarily release the main lock */ {
                unlock_guard guard(state.mutex);
                ret = alloc((CUdeviceptr *) &ptr, ai.size);
            }

            if (ret != CUDA_SUCCESS) {
                jit_malloc_trim();

                /* Temporarily release the main lock */ {
                    unlock_guard guard(state.mutex);
                    ret = alloc((CUdeviceptr *) &ptr, ai.size);
                }

                if (ret != CUDA_SUCCESS)
                    ptr = nullptr;
            }

            size_t &allocated = state.alloc_allocated[ai.type],
                   &watermark = state.alloc_watermark[ai.type];

            allocated += ai.size;
            watermark = std::max(allocated, watermark);
        }
        descr = "new allocation";
    }

    if (unlikely(ptr == nullptr))
        jit_raise("jit_malloc(): out of memory! Could not "
                  "allocate %zu bytes of %s memory.",
                  size, alloc_type_name[ai.type]);

    state.alloc_used.emplace(ptr, ai);

    if ((AllocType) ai.type == AllocType::Device)
        jit_trace("jit_malloc(type=%s, device=%u, size=%zu): " ENOKI_PTR " (%s)",
                  alloc_type_name[ai.type], (uint32_t) ai.device, (size_t) ai.size,
                  (uintptr_t) ptr, descr);
    else
        jit_trace("jit_malloc(type=%s, size=%zu): " ENOKI_PTR " (%s)",
                  alloc_type_name[ai.type], (size_t) ai.size, (uintptr_t) ptr,
                  descr);

    state.alloc_usage[ai.type] += ai.size;

    return ptr;
}

void jit_free(void *ptr) {
    if (ptr == nullptr)
        return;

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jit_raise("jit_free(): unknown address " ENOKI_PTR "!", (uintptr_t) ptr);

    AllocInfo ai = it.value();

    if ((AllocType) ai.type == AllocType::Host) {
        // Acquire lock protecting 'state.alloc_free'
        lock_guard guard(state.malloc_mutex);
        state.alloc_free[ai].push_back(ptr);
    } else {
        Stream *stream = active_stream;
        bool cuda = (AllocType) ai.type != AllocType::HostAsync;
        if (likely(stream && cuda == stream->cuda)) {
            /* Acquire lock protecting 'stream->release_chain' */ {
                lock_guard guard(state.malloc_mutex);
                ReleaseChain *chain = stream->release_chain;
                if (unlikely(!chain))
                    chain = stream->release_chain = new ReleaseChain();
                chain->entries[ai].push_back(ptr);
            }
        } else {
            /* This is bad -- freeing a pointer outside of an active
               stream, or with the wrong backend activated. That pointer may
               still be used in a kernel that is currently being executed
               asynchronously. The only thing we can do at this point is to
               flush all streams. */
            jit_sync_all_devices();
            lock_guard guard(state.malloc_mutex);
            state.alloc_free[ai].push_back(ptr);
        }
    }

    if ((AllocType) ai.type == AllocType::Device)
        jit_trace("jit_free(" ENOKI_PTR ", type=%s, device=%u, size=%zu)",
                  (uintptr_t) ptr, alloc_type_name[ai.type],
		  (uint32_t) ai.device, (size_t) ai.size);
    else
        jit_trace("jit_free(" ENOKI_PTR ", type=%s, size=%zu)", (uintptr_t) ptr,
                  alloc_type_name[ai.type], (size_t) ai.size);

    state.alloc_usage[ai.type] -= ai.size;
    state.alloc_used.erase(it);
}

void jit_free_flush() {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        return;

    ReleaseChain *chain = stream->release_chain;
    if (chain == nullptr || chain->entries.empty())
        return;

    size_t n_dealloc = 0;
    for (auto &kv: chain->entries)
        n_dealloc += kv.second.size();

    if (n_dealloc == 0)
        return;

    ReleaseChain *chain_new = new ReleaseChain();
    chain_new->next = chain;
    stream->release_chain = chain_new;

    jit_trace("jit_free_flush(): scheduling %zu deallocation%s",
              n_dealloc, n_dealloc > 1 ? "s" : "");

    if (stream->cuda) {
        scoped_set_context guard(stream->context);
        cuda_check(cuLaunchHostFunc(
            stream->handle,
            [](void *ptr) -> void {
                /* Acquire lock protecting stream->release_chain and
                   state.alloc_free */
                lock_guard guard(state.malloc_mutex);
                ReleaseChain *chain0 = (ReleaseChain *) ptr,
                             *chain1 = chain0->next;

                for (auto &kv : chain1->entries) {
                    const AllocInfo &ai = kv.first;
                    std::vector<void *> &target = state.alloc_free[ai];
                    target.insert(target.end(), kv.second.begin(),
                                  kv.second.end());
                }

                delete chain1;
                chain0->next = nullptr;
            },
            chain_new));
    } else {
#if defined(ENOKI_JIT_ENABLE_TBB)
        tbb_stream_enqueue_func(
            stream,
            [](void *ptr_) -> void {
                void *ptr = *((void **) ptr_);
                /* Acquire lock protecting stream->release_chain and
                   state.alloc_free */
                lock_guard guard(state.malloc_mutex);
                ReleaseChain *chain0 = (ReleaseChain *) ptr,
                             *chain1 = chain0->next;

                for (auto &kv : chain1->entries) {
                    const AllocInfo &ai = kv.first;
                    std::vector<void *> &target = state.alloc_free[ai];
                    target.insert(target.end(), kv.second.begin(),
                                  kv.second.end());
                }

                delete chain1;
                chain0->next = nullptr;
            },
            &chain_new, sizeof(void *));
#else
        jit_fail("jit_free_flush(): should never get here!");
#endif
    }
}

void* jit_malloc_migrate(void *ptr, AllocType type, int move) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_malloc_migrate(): you must invoke jitc_set_device() to "
                  "choose a target device before evaluating expressions using "
                  "the JIT compiler.");

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jit_raise("jit_malloc_migrate(): unknown address " ENOKI_PTR "!", (uintptr_t) ptr);

    AllocInfo ai = it.value();

#if defined(ENOKI_JIT_ENABLE_TBB)
    if (((AllocType) ai.type == AllocType::Host && type == AllocType::HostAsync) ||
        ((AllocType) ai.type == AllocType::HostAsync && type == AllocType::Host)) {
        if (move) {
            it.value().type = (uint32_t) type;
            return ptr;
        }
    }
#else
    if (type == AllocType::HostAsync)
        type = AllocType::Host;
#endif

    // Maybe nothing needs to be done..
    if ((AllocType) ai.type == type && (type != AllocType::Device || ai.device == stream->device))
        return ptr;

    /// At this point, source or destination is a GPU array, so let's check for CUDA
    if (!stream->cuda)
        jit_raise(
            "jit_malloc_migrate(): you must specify an active CUDA device "
            "using jit_set_device() before invoking this function with a "
            "device/managed/host-pinned pointer!");

    if (type == AllocType::HostAsync || (AllocType) ai.type == AllocType::HostAsync)
        jit_raise("jit_malloc_migrate(): migrations between CUDA and "
                  "host-asynchronous memory are not supported.");

    if (type == AllocType::Host) // Upgrade from host to host-pinned memory
        type = AllocType::HostPinned;

    void *ptr_new = jit_malloc(type, ai.size);
    jit_trace("jit_malloc_migrate(" ENOKI_PTR " -> " ENOKI_PTR ", %s -> %s)",
              (uintptr_t) ptr, (uintptr_t) ptr_new,
              alloc_type_name[ai.type], alloc_type_name[(int) type]);

    scoped_set_context guard(stream->context);
    if ((AllocType) ai.type == AllocType::Host) {
        /// Host -> Device memory, create an intermediate host-pinned array
        void *tmp = jit_malloc(AllocType::HostPinned, ai.size);
        memcpy(tmp, ptr, ai.size);
        cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                 (CUdeviceptr) ptr, ai.size,
                                 stream->handle));
        jit_free(tmp);
    } else {
        cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                 (CUdeviceptr) ptr, ai.size,
                                 stream->handle));

    }

    if (move)
        jit_free(ptr);

    return ptr_new;
}

/// Asynchronously prefetch a memory region
void jit_malloc_prefetch(void *ptr, int device) {
    Stream *stream = active_stream;
    if (unlikely(!stream || !stream->cuda))
        jit_raise(
            "jit_malloc_prefetch(): you must specify an active CUDA device "
            "using jit_set_device() before invoking this function!");

    if (device == -1) {
        device = CU_DEVICE_CPU;
    } else {
        if ((size_t) device >= state.devices.size())
            jit_raise("jit_malloc_prefetch(): invalid device ID!");
        device = state.devices[device].id;
    }

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jit_raise("jit_malloc_prefetch(): unknown address " ENOKI_PTR "!",
                  (uintptr_t) ptr);

    AllocInfo ai = it.value();

    if ((AllocType) ai.type != AllocType::Managed &&
        (AllocType) ai.type != AllocType::ManagedReadMostly)
        jit_raise("jit_malloc_prefetch(): invalid memory type, expected "
                  "Managed or ManagedReadMostly.");

    scoped_set_context guard(stream->context);
    if (device == -2) {
        for (const Device &d : state.devices)
            cuda_check(cuMemPrefetchAsync((CUdeviceptr) ptr, ai.size, d.id,
                                          stream->handle));
    } else {
        cuda_check(cuMemPrefetchAsync((CUdeviceptr) ptr, ai.size, device,
                                      stream->handle));
    }
}

static bool jit_malloc_trim_warned = false;

/// Release all unused memory to the GPU / OS
void jit_malloc_trim(bool warn) {
    if (warn && !jit_malloc_trim_warned) {
        jit_log(
            Warn,
            "jit_malloc_trim(): Enoki exhausted the available memory and had "
            "to flush its allocation cache to free up additional memory. This "
            "is an expensive operation and will have a negative effect on "
            "performance. You may want to change your computation so that it "
            "uses less memory. This warning will only be displayed once.");

        jit_malloc_trim_warned = true;
    }

    AllocInfoMap alloc_free;

    /* Critical section */ {
        lock_guard guard(state.malloc_mutex);
        alloc_free = std::move(state.alloc_free);
    }

    // Ensure that all computation using this memory has indeed completed.
    jit_sync_all_devices();

    size_t trim_count[(int) AllocType::Count] = { 0 },
           trim_size [(int) AllocType::Count] = { 0 };

    /* Temporarily release the main lock */ {
        unlock_guard guard(state.mutex);

        for (auto& kv : alloc_free) {
            const std::vector<void *> &entries = kv.second;

            trim_count[(int) kv.first.type] += entries.size();
            trim_size[(int) kv.first.type] += kv.first.size * entries.size();

            switch ((AllocType) kv.first.type) {
                case AllocType::Device:
                case AllocType::Managed:
                case AllocType::ManagedReadMostly:
                    if (state.has_cuda) {
                        for (void *ptr : entries)
                            cuda_check(cuMemFree((CUdeviceptr) ptr));
                    }
                    break;

                case AllocType::HostPinned:
                    if (state.has_cuda) {
                        for (void *ptr : entries)
                            cuda_check(cuMemFreeHost(ptr));
                    }
                    break;

                case AllocType::Host:
                case AllocType::HostAsync:
#if !defined(_WIN32)
                    for (void *ptr : entries)
                        free(ptr);
#else
                    for (void* ptr : entries)
                        _aligned_free(ptr);
#endif
                    break;

                default:
                    jit_fail("jit_malloc_trim(): unsupported allocation type!");
            }
        }
    }

    for (auto& kv : alloc_free)
        state.alloc_allocated[kv.first.type] -= kv.first.size;

    size_t total = 0;
    for (int i = 0; i < (int) AllocType::Count; ++i)
        total += trim_count[i];

    if (total > 0) {
        jit_log(Debug, "jit_malloc_trim(): freed");
        for (int i = 0; i < (int) AllocType::Count; ++i) {
            if (trim_count[i] == 0)
                continue;
            jit_log(Debug, " - %s memory: %s in %zu allocation%s",
                    alloc_type_name[i], jit_mem_string(trim_size[i]),
                    trim_count[i], trim_count[i] > 1 ? "s" : "");
        }
    }
}

/// Query the flavor of a memory allocation made using \ref jit_malloc()
AllocType jit_malloc_type(void *ptr) {
    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jit_raise("jit_malloc_type(): unknown address " ENOKI_PTR "!", (uintptr_t) ptr);
    return (AllocType) it->second.type;
}

/// Query the device associated with a memory allocation made using \ref jit_malloc()
int jit_malloc_device(void *ptr) {
    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jit_raise("jit_malloc_type(): unknown address " ENOKI_PTR "!", (uintptr_t) ptr);
    const AllocInfo &ai = it.value();
    if (ai.type == (int) AllocType::Host || ai.type == (int) AllocType::HostAsync)
        return -1;
    else
        return ai.device;
}

void jit_malloc_shutdown() {
    jit_malloc_trim(false);

    size_t leak_count[(int) AllocType::Count] = { 0 },
           leak_size [(int) AllocType::Count] = { 0 };
    for (auto kv : state.alloc_used) {
        leak_count[(int) kv.second.type]++;
        leak_size[(int) kv.second.type] += kv.second.size;
    }

    size_t total = 0;
    for (int i = 0; i < (int) AllocType::Count; ++i)
        total += leak_count[i];

    if (total > 0) {
        jit_log(Warn, "jit_malloc_shutdown(): leaked");
        for (int i = 0; i < (int) AllocType::Count; ++i) {
            if (leak_count[i] == 0)
                continue;

            jit_log(Warn, " - %s memory: %s in %zu allocation%s",
                    alloc_type_name[i], jit_mem_string(leak_size[i]),
                    leak_count[i], leak_count[i] > 1 ? "s" : "");
        }
    }
}
