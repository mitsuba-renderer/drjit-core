/*
    src/registry.cpp -- Pointer registry for vectorized method calls

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"

static_assert(sizeof(void*) == 8, "32 bit architectures are not supported!");

/// Register a pointer with Enoki's pointer registry
uint32_t jit_registry_put(const char *domain, void *ptr) {
    if (unlikely(ptr == nullptr))
        jit_raise("jit_registry_put(): cannot register the null pointer!");

    // Create the rev. map. first and throw if the pointer is already registered
    auto it_rev = state.registry_rev.try_emplace(ptr, RegistryKey(domain, 0));
    if (unlikely(!it_rev.second))
        jit_raise("jit_registry_put(): pointer %p was already registered!", ptr);

    // Get or create the bookkeeping record associated with the domain
    auto it_head = state.registry_fwd.try_emplace(RegistryKey(domain, 0), nullptr);
    uintptr_t &value_head = (uintptr_t &) it_head.first.value();

    uint32_t next_avail = (uint32_t) (value_head >> 32),
             counter    = (uint32_t)  value_head;

    if (next_avail) {
        // Case 1: some previously released IDs are available, reuse them
        auto it_next = state.registry_fwd.find(RegistryKey(domain, next_avail));
        if (unlikely(it_next == state.registry_fwd.end()))
            jit_fail("jit_registry_put(): data structure corrupted (1)!");

        uintptr_t &value_next = (uintptr_t &) it_next.value();

        uint32_t next_avail_2 = (uint32_t) (value_next >> 32),
                 unused       = (uint32_t)  value_next;

        if (unlikely(unused != 0))
            jit_fail("jit_registry_put(): data structure corrupted (2)!");

        // Update bookkeeping record with next element from linked list
        value_head = ((uintptr_t) next_avail_2 << 32) | counter;

        // Initialize reused entry
        value_next = (uintptr_t) ptr;

        // Finally, update reverse mapping
        it_rev.first.value().id = next_avail;

        jit_trace("jit_registry_put(" ENOKI_PTR ", domain=\"%s\"): %u (reused)",
                  (uintptr_t) ptr, domain, next_avail);

        return next_avail;
    } else {
        // Case 2: need to create a new record

        // Increment counter and update bookkeeping record
        value_head = ++counter;

        // Create new record
        auto it_new =
            state.registry_fwd.try_emplace(RegistryKey(domain, counter), ptr);
        if (unlikely(!it_new.second))
            jit_fail("jit_registry_put(): data structure corrupted (3)!");

        // Finally, update reverse mapping
        it_rev.first.value().id = counter;

        jit_trace("jit_registry_put(" ENOKI_PTR ", domain=\"%s\"): %u (new)",
                  (uintptr_t) ptr, domain, counter);

        return counter;
    }
}

/// Remove a pointer from the registry
void jit_registry_remove(void *ptr) {
    if (ptr == nullptr)
        return;

    jit_trace("jit_registry_remove(" ENOKI_PTR ")", (uintptr_t) ptr);

    auto it_rev = state.registry_rev.find(ptr);
    if (unlikely(it_rev == state.registry_rev.end()))
        jit_raise("jit_registry_remove(): pointer %p could not be found!", ptr);

    RegistryKey key = it_rev.value();

    // Get the forward record associated with the pointer
    auto it_fwd = state.registry_fwd.find(RegistryKey(key.domain, key.id));
    if (unlikely(it_fwd == state.registry_fwd.end()))
        jit_raise("jit_registry_remove(): data structure corrupted (1)!");

    // Get the bookkeeping record associated with the domain
    auto it_head = state.registry_fwd.find(RegistryKey(key.domain, 0));
    if (unlikely(it_head == state.registry_fwd.end()))
        jit_raise("jit_registry_remove(): data structure corrupted (2)!");

    // Update the head node
    uintptr_t &value_head = (uintptr_t &) it_head.value();
    uint32_t next_avail = (uint32_t) (value_head >> 32),
             counter    = (uint32_t)  value_head;
    value_head = ((uintptr_t) key.id << 32) | counter;

    // Update the current node
    uintptr_t &value_fwd = (uintptr_t &) it_fwd.value();
    value_fwd = (uintptr_t) next_avail << 32;

    // Remove reverse mapping
    state.registry_rev.erase(it_rev);
}

/// Query the ID associated a registered pointer
uint32_t jit_registry_get_id(const void *ptr) {
    if (ptr == nullptr)
        return 0;

    auto it = state.registry_rev.find(ptr);
    if (unlikely(it == state.registry_rev.end()))
        jit_raise("jit_registry_get_id(): pointer %p could not be found!", ptr);
    return it.value().id;
}

/// Query the domain associated a registered pointer
const char *jit_registry_get_domain(const void *ptr) {
    if (ptr == nullptr)
        return nullptr;

    auto it = state.registry_rev.find(ptr);
    if (unlikely(it == state.registry_rev.end()))
        jit_raise("jit_registry_get_domain(): pointer %p could not be found!", ptr);
    return it.value().domain;
}

/// Query the pointer associated a given domain and ID
void *jit_registry_get_ptr(const char *domain, uint32_t id) {
    if (id == 0)
        return nullptr;

    auto it = state.registry_fwd.find(RegistryKey(domain, id));
    if (unlikely(it == state.registry_fwd.end()))
        jit_raise("jit_registry_get_ptr(): entry with domain=\"%s\", id=%u "
                  "could not be found!", domain, id);

    return it.value();
}

/// Compact the registry and release unused IDs and attributes
void jit_registry_trim() {
    RegistryFwdMap registry_fwd;

    for (auto &kv : state.registry_fwd) {
        const char *domain = kv.first.domain;
        uint32_t id = kv.first.id;
        void *ptr = kv.second;

        if (id != 0 &&
            ((uint32_t) (uintptr_t) ptr != 0u ||
             state.registry_rev.find(ptr) != state.registry_rev.end())) {
            registry_fwd.insert(kv);

            auto it_head =
                registry_fwd.try_emplace(RegistryKey(domain, 0), nullptr);

            uintptr_t &value_head = (uintptr_t &) it_head.first.value();
            value_head = std::max(value_head, (uintptr_t) id);
        }
    }

    if (state.registry_fwd.size() != registry_fwd.size()) {
        jit_trace("jit_registry_trim(): removed %zu / %zu entries.",
                  state.registry_fwd.size() - registry_fwd.size(),
                  state.registry_fwd.size());

        state.registry_fwd = std::move(registry_fwd);
    }

    AttributeMap attributes;
    for (auto &kv : state.attributes) {
        if (state.registry_fwd.find(RegistryKey(kv.first.domain, 0)) != state.registry_fwd.end()) {
            attributes.insert(kv);
        } else {
            if (state.has_cuda)
                cuda_check(cuMemFree((CUdeviceptr) kv.second.ptr));
            else
                free(kv.second.ptr);
        }
    }

    if (state.attributes.size() != attributes.size()) {
        jit_trace("jit_registry_trim(): removed %zu / %zu attributes.",
                  state.attributes.size() - attributes.size(),
                  state.attributes.size());
        state.attributes = std::move(attributes);
    }
}

/// Provide a bound (<=) on the largest ID associated with a domain
uint32_t jit_registry_get_max(const char *domain) {
    // Get the bookkeeping record associated with the domain
    auto it_head = state.registry_fwd.find(RegistryKey(domain, 0));
    if (unlikely(it_head == state.registry_fwd.end()))
        return 0;

    uintptr_t value_head = (uintptr_t) it_head.value();
    return (uint32_t) value_head; // extract counter field
}

void jit_registry_shutdown() {
    jit_registry_trim();

    if (!state.registry_fwd.empty() || !state.registry_rev.empty())
        jit_log(Warn, "jit_registry_shutdown(): leaked %zu forward "
                "and %zu reverse mappings!", state.registry_fwd.size(),
                state.registry_rev.size());

    if (!state.attributes.empty())
        jit_log(Warn, "jit_registry_shutdown(): leaked %zu attributes!",
                state.attributes.size());
}

void jit_registry_set_attr(void *ptr, const char *name,
                           const void *value, size_t isize) {
    auto it = state.registry_rev.find(ptr);
    if (unlikely(it == state.registry_rev.end()))
        jit_raise("jit_registry_set_attr(): pointer %p could not be found!", ptr);

    const char *domain = it.value().domain;
    uint32_t id = it.value().id;

    jit_trace("jit_registry_set_attr(" ENOKI_PTR ", id=%u, name=\"%s\", size=%zu)",
              (uintptr_t) ptr, id, name, isize);

    AttributeValue &attr = state.attributes[AttributeKey(domain, name)];
    if (attr.isize == 0)
        attr.isize = isize;
    else if (attr.isize != isize)
        jit_raise("jit_registry_set_attr(): incompatible size!");

    if (id >= attr.count) {
        uint32_t new_count = std::max(id + 1, std::max(8u, attr.count * 2u));
        size_t old_size = (size_t) attr.count * (size_t) isize;
        size_t new_size = (size_t) new_count * (size_t) isize;
        void *ptr;

        if (state.has_cuda) {
            CUcontext ctx;
            cuda_check(cuCtxGetCurrent(&ctx));
            if (!ctx)
                cuda_check(cuCtxSetCurrent(state.devices[0].context));
            CUresult ret = cuMemAllocManaged((CUdeviceptr *) &ptr, new_size, CU_MEM_ATTACH_GLOBAL);
            if (ret != CUDA_SUCCESS) {
                jit_malloc_trim();
                cuda_check(cuMemAllocManaged((CUdeviceptr *) &ptr, new_size, CU_MEM_ATTACH_GLOBAL));
            }
            cuda_check(cuMemAdvise((CUdeviceptr) ptr, new_size, CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
            if (!ctx)
                cuda_check(cuCtxSetCurrent(ctx));
        } else {
            ptr = malloc_check(new_size);
        }

        if (old_size != 0)
            memcpy(ptr, attr.ptr, old_size);
        memset((uint8_t *) ptr + old_size, 0, new_size - old_size);

        if (state.has_cuda)
            cuda_check(cuMemFree((CUdeviceptr) attr.ptr));
        else
            free(attr.ptr);

        attr.ptr = ptr;
        attr.count = new_count;
    }

    memcpy((uint8_t *) attr.ptr + id * isize, value, isize);
}

const void *jit_registry_attr_data(const char *domain, const char *name) {
    auto it = state.attributes.find(AttributeKey(domain, name));
    if (unlikely(it == state.attributes.end()))
        jit_raise("jit_registry_attr_data(): entry with domain=\"%s\", "
                  "name=\"%s\" not found!", domain, name);
    return it.value().ptr;
}
