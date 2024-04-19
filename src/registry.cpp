/*
    src/registry.cpp -- Pointer registry for vectorized method calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "registry.h"
#include "log.h"
#include <queue>

// Dr.Jit maintains an ID registry per backend and class. This class separates
// multiple parallel data structures maintaining this information.
struct DomainKey {
    JitBackend backend;
    const char *domain;

    struct Eq {
        bool operator()(DomainKey k1, DomainKey k2) const {
            return k1.backend == k2.backend && strcmp(k1.domain, k2.domain) == 0;
        }
    };

    struct Hash {
        size_t operator()(DomainKey k) const {
            return hash_str(k.domain, (size_t) k.backend);
        }
    };
};

struct Ptr {
    void *ptr;
    bool active;
    Ptr(void *ptr, bool active) : ptr(ptr), active(active) { }
};

// Per-domain information: a forward map (index-> pointer) and a list of unused entries
struct Domain {
    const char *name;
    uint32_t id_bound;
    std::vector<Ptr> fwd_map;
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>
        free_pq;
};

struct ReverseKey {
    uint32_t domain_id;
    uint32_t index;
};

/// Main registry data structure
struct Registry {
    tsl::robin_map<DomainKey, uint32_t, DomainKey::Hash, DomainKey::Eq> domain_ids;
    std::vector<Domain> domains;
    tsl::robin_map<void *, ReverseKey, PointerHasher> rev_map;
};

static Registry registry;

/// Register a pointer with Dr.Jit's pointer registry
uint32_t jitc_registry_put(JitBackend backend, const char *domain_name, void *ptr) {
    Registry &r = registry;

    auto [it1, result1] =
        r.rev_map.try_emplace(ptr, ReverseKey{});
    if (unlikely(!result1))
        jitc_raise("jit_registry_put(domain=\"%s\", ptr=%p): pointer is "
                   "already registered!", domain_name, ptr);

    // Allocate a domain entry for the key (backend, domain) if unregistered
    auto [it2, result2] = r.domain_ids.try_emplace(DomainKey{ backend, domain_name },
                                                   (uint32_t) r.domains.size());
    if (result2) {
        r.domains.emplace_back();
        Domain &domain = r.domains.back();
        domain.name = domain_name;
        domain.id_bound = 0;
    }

    uint32_t domain_id = it2->second;

    // Allocate the lowest-valued ID
    Domain &domain = r.domains[domain_id];
    uint32_t index;
    if (!domain.free_pq.empty()) {
        index = domain.free_pq.top();
        domain.free_pq.pop();
        domain.fwd_map[index] = Ptr(ptr, true);
    } else {
        index = (uint32_t) domain.fwd_map.size();
        domain.fwd_map.emplace_back(ptr, true);
    }

    domain.id_bound = std::max(domain.id_bound, index + 1);

    jitc_log(Debug, "jit_registry_put(domain=\"%s\", ptr=%p): id_bound=%u",
             domain_name, ptr, domain.id_bound);

    // Create a reverse mapping
    it1.value() = ReverseKey { domain_id, index };

    return index + 1;
}

/// Remove a pointer from the registry
void jitc_registry_remove(const void *ptr) {
    Registry &r = registry;
    size_t ptr_hash = PointerHasher()(ptr);

    auto it = r.rev_map.find((void *) ptr, ptr_hash);
    if (it == r.rev_map.end())
        jitc_raise("jit_registry_remove(ptr=%p): pointer is not registered!", ptr);

    ReverseKey rk = it->second;
    r.rev_map.erase_fast(it);

    Domain &domain = r.domains[rk.domain_id];
    domain.free_pq.push(rk.index);
    if (domain.fwd_map[rk.index].ptr != ptr)
        jitc_fail("jit_registry_remove(ptr=%p): data structure corrupt!", ptr);

    domain.fwd_map[rk.index] = Ptr(nullptr, false);

    if (domain.id_bound == rk.index + 1) {
        while (domain.id_bound > 0 && !domain.fwd_map[domain.id_bound - 1].ptr)
            domain.id_bound--;
    }

    jitc_log(Debug, "jit_registry_remove(domain=\"%s\", ptr=%p): id_bound=%u",
             domain.name, ptr, domain.id_bound);
}

uint32_t jitc_registry_id(const void *ptr) {
    if (!ptr)
        return 0;

    Registry &r = registry;
    auto it = r.rev_map.find((void *) ptr);
    if (it == r.rev_map.end())
        jitc_raise("jit_registry_id(ptr=%p): pointer is "
                   "not registered!", ptr);

    return it->second.index + 1;
}

uint32_t jitc_registry_id_bound(JitBackend backend, const char *domain) {
    Registry &r = registry;
    auto it = r.domain_ids.find(DomainKey{ backend, domain });
    if (it == r.domain_ids.end())
        return 0;
    else
        return r.domains[it->second].id_bound;
}

void *jitc_registry_ptr(JitBackend backend, const char *domain_name, uint32_t id) {
    if (id == 0)
        return nullptr;

    Registry &r = registry;
    auto it = r.domain_ids.find(DomainKey{ backend, domain_name });
    void *ptr = nullptr;

    if (it != r.domain_ids.end()) {
        Domain &domain = r.domains[it->second];
        if (id - 1 >= domain.fwd_map.size())
            jitc_raise("jit_registry_ptr(domain=\"%s\", id=%u): instance is "
                       "not registered!",
                       domain_name, id);
        Ptr entry = domain.fwd_map[id - 1];
        if (entry.active)
            ptr = entry.ptr;
    }

    return ptr;
}

void *jitc_registry_peek(JitBackend backend, const char *domain_name) {
    Registry &r = registry;
    auto it = r.domain_ids.find(DomainKey{ backend, domain_name });
    void *ptr = nullptr;

    if (it != r.domain_ids.end()) {
        Domain &domain = r.domains[it->second];
        uint32_t bound = domain.id_bound;
        if (bound > 0) {
            Ptr entry = domain.fwd_map[bound - 1];
            return entry.ptr;
        }
    }

    return ptr;
}

void jitc_registry_clear() {
    jitc_log(Debug, "jit_registry_clear()");
    Registry &r = registry;
    for (Domain &d : r.domains) {
        for (Ptr &p : d.fwd_map)
            p.active = false;
        d.id_bound = 0;
    }
}

void jitc_registry_shutdown() {
    Registry &r = registry;
    for (Domain &d : r.domains) {
        if (d.fwd_map.size() == d.free_pq.size())
            continue;
        jitc_log(Warn,
                 "jit_registry_shutdown(): leaking %zu instances of type \"%s\".",
                 d.fwd_map.size() - d.free_pq.size(), d.name);
    }
}
