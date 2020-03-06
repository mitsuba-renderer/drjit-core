#include "jit.h"
#include "log.h"

static const char *alloc_type_names[5] = { "host", "host-pinned", "device",
                                           "managed", "managed-read-mostly" };

// Round an unsigned integer up to a power of two
size_t round_pow2(size_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;  x |= x >> 32;
    return x + 1;
}

void* jit_malloc(AllocType type, size_t size) {
    if (unlikely(!state.initialized))
        jit_fail("jit_malloc(): JIT compiler is uninitialized!");

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
            jit_fail("jit_malloc(): device must be set using jit_set_context() "
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
            int rv = posix_memalign(&ptr, 64, ai.size);
            if (rv == ENOMEM) {
                jit_malloc_trim();
                rv = posix_memalign(&ptr, 64, ai.size);
            }
            if (rv != 0)
                jit_raise("jit_malloc(): out of memory!");
        } else {
            #if defined(ENOKI_CUDA)
                cudaError_t (*alloc) (void **, size_t) = nullptr;

                auto cudaMallocManaged_ = [](void **ptr_, size_t size_) {
                    return cudaMallocManaged(ptr_, size_);
                };

                switch (type) {
                    case AllocType::HostPinned:        alloc = cudaMallocHost; break;
                    case AllocType::Device:            alloc = cudaMalloc; break;
                    case AllocType::Managed:
                    case AllocType::ManagedReadMostly: alloc = cudaMallocManaged_; break;
                    default:
                        jit_fail("jit_malloc(): internal-error unsupported allocation type!");
                }

                cudaError_t ret = alloc(&ptr, ai.size);
                if (ret != cudaSuccess) {
                    jit_malloc_trim();
                    cudaError_t ret = alloc(&ptr, ai.size);
                    if (ret != cudaSuccess)
                        throw std::runtime_error("jit_malloc(): out of memory!");
                }

                if (type == AllocType::ManagedReadMostly)
                    cuda_check(cudaMemAdvise(ptr, ai.size, cudaMemAdviseSetReadMostly, 0));
            #else
                jit_fail("jit_malloc(): internal error --- unsupported allocation type!");
            #endif
        }
        descr = "new allocation";
    }

    state.alloc_used.insert({ ptr, ai });

    if (ai.type == AllocType::Device)
        jit_log(Trace, "jit_malloc(type=%s, device=%u, size=%zu) -> %p (%s)",
                alloc_type_names[(int) ai.type], ai.device, ai.size, ptr, descr);
    else
        jit_log(Trace, "jit_malloc(type=%s, size=%zu) -> %p (%s)",
                alloc_type_names[(int) ai.type], ai.size, ptr, descr);

    size_t &usage     = state.alloc_usage[(int) ai.type],
           &watermark = state.alloc_watermark[(int) ai.type];
    usage += ai.size;
    watermark = std::max(watermark, usage);

    return ptr;
}

void jit_free(void *ptr) {
    if (unlikely(!state.initialized))
        jit_fail("jit_malloc(): JIT compiler is uninitialized!");

    if (ptr == nullptr)
        return;

    auto it = state.alloc_used.find(ptr);
    if (it == state.alloc_used.end())
        jit_raise("jit_free(): unknown address %p!", ptr);

    AllocInfo ai = it->second;
    if (ai.type == AllocType::Device)
        jit_log(Trace, "jit_free(%p, type=%s, device=%u, size=%zu)", ptr,
                alloc_type_names[(int) ai.type], ai.device, ai.size);
    else
        jit_log(Trace, "jit_free(%p, type=%s, size=%zu)", ptr,
                alloc_type_names[(int) ai.type], ai.size);

    state.alloc_used.erase(it);
    if (ai.type == AllocType::Host) {
        state.alloc_free[ai].push_back(ptr);
    } else {
#if defined(ENOKI_CUDA)
        Stream *stream = active_stream;
        active_stream->alloc_pending[ai].push_back(ptr);
#else
        jit_fail("jit_free(): internal error -- unsupported array type! (CUDA support was disabled.)");
#endif
    }
}

void jit_flush_free() {
#if defined(ENOKI_CUDA)
    Stream *stream = active_stream;

    if (!state.initialized || stream == nullptr)
        return;

    AllocInfoMap *pending_tmp = new AllocInfoMap(std::move(stream->alloc_pending));

    jit_log(Trace, "jit_flush_free(): scheduling %zu deallocation%s.",
            pending_tmp->size(), pending_tmp->size() > 1 ? "s" : "");

    cuda_check(cudaLaunchHostFunc(
        stream->handle,
        [](void *ptr) {
            lock_guard guard(state.mutex);
            AllocInfoMap *pending = (AllocInfoMap *) ptr;

            jit_log(Debug, "jit_flush_free(): performing %zu deallocation%s.",
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
                jit_fail("jit_malloc_trim(): internal error -- unsupported allocation type!");
        }
    }

    size_t total = trim_count[0] + trim_count[1] + trim_count[2] +
                   trim_count[3] + trim_count[4];
    if (total > 0) {
        jit_log(Trace, "jit_malloc_trim(): freed");
        for (int i = 0; i < 5; ++i) {
            if (trim_count[i] == 0)
                continue;
            jit_log(Trace, "%22s memory: %zu bytes in %zu allocation%s.",
                    alloc_type_names[i], trim_size[i], trim_count[i],
                    trim_count[i] > 1 ? "s" : "");
        }
    }
}

void jit_malloc_shutdown() {
    // Clear the memory cache
    jit_malloc_trim();

    size_t leak_count[5] { 0 }, leak_size[5] { 0 };
    for (auto kv : state.alloc_used) {
        leak_count[(int) kv.second.type]++;
        leak_size[(int) kv.second.type] += kv.second.size;
    }

    size_t total = leak_count[0] + leak_count[1] + leak_count[2] +
                   leak_count[3] + leak_count[4];
    if (total > 0) {
        jit_log(Trace, "jit_malloc_shutdown(): leaked");
        for (int i = 0; i < 5; ++i) {
            if (leak_count[i] == 0)
                continue;
            jit_log(Trace, "%22s memory: %zu bytes in %zu allocation%s.",
                    alloc_type_names[i], leak_size[i], leak_count[i],
                    leak_count[i] > 1 ? "s" : "");
        }
    }
}
