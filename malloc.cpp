#include "jit.h"
#include "log.h"

const char *alloc_type_names[5] = { "host", "host-pinned", "device",
                                    "managed", "managed-read-mostly" };

// Round an unsigned integer up to a power of two
static size_t round_pow2(size_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;  x |= x >> 32;
    return x + 1;
}

void* jit_malloc(AllocType type, size_t size) {
    if (size == 0)
        return nullptr;

    /* Round 'size' to the next larger power of two. This is somewhat
       wasteful, but reduces the number of different sizes that an allocation
       can have to a manageable amount that facilitates re-use. */
    size = round_pow2(size);

    AllocInfo ai(type, 0, size);

    const char *descr = nullptr;
    void *ptr = nullptr;

#if defined(ENOKI_CUDA)
    Stream *stream = active_stream;
    if (type == AllocType::Device) {
        if (unlikely(!stream))
            jit_fail("jit_malloc(): device must be set using jit_device_set() "
                     "before allocating device pointer!");
        ai.device = stream->device;
    }

    // 1. Check for suitable recently freed arrays on the current stream
    if (type != AllocType::Host) {
        auto it = stream->alloc_pending.find(ai);

        if (it != state.alloc_free.end()) {
            std::vector<void *> &list = it.value();
            if (!list.empty()) {
                ptr = list.back();
                list.pop_back();
                descr = "reused local";
            }
        }
    }
#endif

    // 2. Look globally. Are there suitable freed arrays?
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

    // 3. Looks like we will have to allocate some memory..
    if (ptr == nullptr) {
        if (type == AllocType::Host) {
            unlock_guard guard(state.mutex);
            int rv = posix_memalign(&ptr, 64, ai.size);
            if (rv == ENOMEM) {
                jit_malloc_trim();
                rv = posix_memalign(&ptr, 64, ai.size);
            }
            if (rv != 0)
                ptr = nullptr;
        } else {
            #if defined(ENOKI_CUDA)
                cudaError_t (*alloc) (void **, size_t) = nullptr;

                auto cudaMallocManaged_ = [](void **ptr_, size_t size_) {
                    return cudaMallocManaged(ptr_, size_);
                };

                auto cudaMallocManagedReadMostly_ = [](void **ptr_, size_t size_) {
                    cudaError_t ret = cudaMallocManaged(ptr_, size_);
                    if (ret == cudaSuccess)
                        cuda_check(cudaMemAdvise(*ptr_, size_, cudaMemAdviseSetReadMostly, 0));
                    return ret;
                };

                switch (type) {
                    case AllocType::HostPinned:        alloc = cudaMallocHost; break;
                    case AllocType::Device:            alloc = cudaMalloc; break;
                    case AllocType::Managed:           alloc = cudaMallocManaged_; break;
                    case AllocType::ManagedReadMostly: alloc = cudaMallocManagedReadMostly_; break;
                    default:
                        jit_fail("jit_malloc(): internal-error unsupported allocation type!");
                }

                cudaError_t ret;

                /* Temporarily release the lock */ {
                    unlock_guard guard(state.mutex);
                    ret = alloc(&ptr, ai.size);
                }

                if (ret != cudaSuccess) {
                    jit_malloc_trim();

                    /* Temporarily release the lock */ {
                        unlock_guard guard(state.mutex);
                        ret = alloc(&ptr, ai.size);
                    }

                    if (ret != cudaSuccess)
                        ptr = nullptr;
                }

            #else
                jit_fail("jit_malloc(): unsupported array type! (CUDA support was disabled.)");
            #endif
        }
        descr = "new allocation";
    }

    if (unlikely(ptr == nullptr))
        jit_raise("jit_malloc(): out of memory! Could not "
                  "allocate %zu bytes of %s memory.",
                  size, alloc_type_names[(int) ai.type]);

    state.alloc_used.insert({ ptr, ai });

    if (ai.type == AllocType::Device)
        jit_log(Trace, "jit_malloc(type=%s, device=%u, size=%zu) -> " PTR " (%s)",
                alloc_type_names[(int) ai.type], ai.device, ai.size, ptr, descr);
    else
        jit_log(Trace, "jit_malloc(type=%s, size=%zu) -> " PTR " (%s)",
                alloc_type_names[(int) ai.type], ai.size, ptr, descr);

    size_t &usage     = state.alloc_usage[(int) ai.type],
           &watermark = state.alloc_watermark[(int) ai.type];

    usage += ai.size;
    watermark = std::max(watermark, usage);

    return ptr;
}

void jit_free(void *ptr) {
    if (ptr == nullptr)
        return;

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jit_raise("jit_free(): unknown address " PTR "!", ptr);

    AllocInfo ai = it.value();
    if (ai.type == AllocType::Device)
        jit_log(Trace, "jit_free(" PTR ", type=%s, device=%u, size=%zu)", ptr,
                alloc_type_names[(int) ai.type], ai.device, ai.size);
    else
        jit_log(Trace, "jit_free(" PTR ", type=%s, size=%zu)", ptr,
                alloc_type_names[(int) ai.type], ai.size);

    state.alloc_used.erase(it);
    if (ai.type == AllocType::Host) {
        state.alloc_free[ai].push_back(ptr);
    } else {
#if defined(ENOKI_CUDA)
        Stream *stream = active_stream;
        if (unlikely(!stream))
            jit_fail("jit_free(): device and stream must be set! (call "
                     "jit_device_set() beforehand)!");
        active_stream->alloc_pending[ai].push_back(ptr);
#else
        jit_fail("jit_free(): unsupported array type! (CUDA support was disabled.)");
#endif
    }
}

void jit_free_flush() {
#if defined(ENOKI_CUDA)
    Stream *stream = active_stream;

    if (unlikely(stream == nullptr))
        jit_fail("jit_free_flush(): device and stream must be set! (call "
                 "jit_device_set() beforehand)!");

    AllocInfoMap *pending_tmp = new AllocInfoMap(std::move(stream->alloc_pending));

    jit_log(Trace, "jit_free_flush(): scheduling %zu deallocation%s.",
            pending_tmp->size(), pending_tmp->size() > 1 ? "s" : "");

    cuda_check(cudaLaunchHostFunc(
        stream->handle,
        [](void *ptr) {
            lock_guard guard(state.mutex);
            AllocInfoMap *pending = (AllocInfoMap *) ptr;

            jit_log(Trace, "jit_free_flush(): performing %zu deallocation%s.",
                    pending->size(), pending->size() > 1 ? "s" : "");

            for (auto &kv: *pending) {
                const AllocInfo &ai = kv.first;
                state.alloc_usage[(int) ai.type] -= ai.size * kv.second.size();
                std::vector<void *> &target = state.alloc_free[ai];
                target.insert(target.end(), kv.second.begin(), kv.second.end());
            }

            delete pending;
        },

        pending_tmp
    ));
#endif
}

void* jit_malloc_migrate(void *ptr, AllocType type) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_fail("jit_malloc_migrate(): device and stream must be set! "
                 "(call jit_device_set() beforehand)!");

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jit_raise("jit_malloc_migrate(): unknown address " PTR "!", ptr);

    AllocInfo ai = it.value();

    if (ai.type != type &&
        (ai.type == AllocType::Host || type == AllocType::Host))
        jit_raise("jit_malloc_migrate(): non-pinned host memory does not "
                  "support asynchronous migration!");

    // Maybe nothing needs to be done..
    if (ai.type == type && (type != AllocType::Device || ai.device == stream->device))
        return ptr;

    jit_log(Trace, "jit_malloc_migrate(" PTR "): %s -> %s", ptr,
            alloc_type_names[(int) ai.type],
            alloc_type_names[(int) type]) ;

    void *ptr_new = jit_malloc(type, ai.size);
    cuda_check(cudaMemcpyAsync(ptr_new, ptr, ai.size,
                               cudaMemcpyDefault,
                               stream->handle));
    jit_free(ptr);

    return ptr_new;
}

void jit_malloc_trim() {
    AllocInfoMap alloc_free(std::move(state.alloc_free));
    unlock_guard guard(state.mutex);

    size_t trim_count[5] = { 0 }, trim_size[5] = { 0 };
    for (auto kv : alloc_free) {
        const std::vector<void *> &entries = kv.second;
        trim_count[(int) kv.first.type] += entries.size();
        trim_size[(int) kv.first.type] += kv.first.size * entries.size();
        switch (kv.first.type) {
#if defined(ENOKI_CUDA)
            case AllocType::Device:
            case AllocType::Managed:
            case AllocType::ManagedReadMostly:
                for (void *ptr : entries)
                    cuda_check(cudaFree(ptr));
                break;

            case AllocType::HostPinned:
                for (void *ptr : entries)
                    cuda_check(cudaFreeHost(ptr));
                break;
#endif

            case AllocType::Host:
                for (void *ptr : entries)
                    free(ptr);
                break;

            default:
                jit_fail("jit_malloc_trim(): unsupported allocation type!");
        }
    }

    size_t total = trim_count[0] + trim_count[1] + trim_count[2] +
                   trim_count[3] + trim_count[4];
    if (total > 0) {
        jit_log(Debug, "jit_malloc_trim(): freed");
        for (int i = 0; i < 5; ++i) {
            if (trim_count[i] == 0)
                continue;
            jit_log(Debug, "%22s memory: %s in %zu allocation%s.",
                    alloc_type_names[i], jit_mem_string(trim_size[i]),
                    trim_count[i], trim_count[i] > 1 ? "s" : "");
        }
    }
}

void jit_malloc_shutdown() {
    jit_malloc_trim();

    size_t leak_count[5] { 0 }, leak_size[5] { 0 };
    for (auto kv : state.alloc_used) {
        leak_count[(int) kv.second.type]++;
        leak_size[(int) kv.second.type] += kv.second.size;
    }

    size_t total = leak_count[0] + leak_count[1] + leak_count[2] +
                   leak_count[3] + leak_count[4];
    if (total > 0) {
        jit_log(Warn, "jit_malloc_shutdown(): leaked");
        for (int i = 0; i < 5; ++i) {
            if (leak_count[i] == 0)
                continue;

            jit_log(Warn, "%22s memory: %s in %zu allocation%s.",
                    alloc_type_names[i], jit_mem_string(leak_size[i]),
                    leak_count[i], leak_count[i] > 1 ? "s" : "");
        }
    }
}
