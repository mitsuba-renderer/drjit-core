/*
    src/registry.h -- Pointer registry for vectorized method calls

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "core.h"
#include "hash.h"


struct AttributeKey {
    const char *domain;
    const char *name;

    AttributeKey(const char *domain, const char *name) : domain(domain), name(name) { }

    bool operator==(const AttributeKey &k) const {
        return strcmp(domain, k.domain) == 0 && strcmp(name, k.name) == 0;
    }

    /// Helper class to hash AttributeKey instances
    struct Hasher {
        size_t operator()(const AttributeKey &k) const {
            return hash_str(k.domain, hash_str(k.name));
        }
    };
};

struct AttributeValue {
    uint32_t isize = 0;
    uint32_t count = 0;
    void *ptr = nullptr;
};

// Key associated with a pointer registered in DrJit's pointer registry
struct RegistryKey {
    const char *domain;
    uint32_t id;
    RegistryKey(const char *domain, uint32_t id) : domain(domain), id(id) { }

    bool operator==(const RegistryKey &k) const {
        return id == k.id && strcmp(domain, k.domain) == 0;
    }

    /// Helper class to hash RegistryKey instances
    struct Hasher {
        size_t operator()(const RegistryKey &k) const {
            return hash_str(k.domain, k.id);
        }
    };
};

struct Registry {
    using RegistryFwdMap =
        tsl::robin_map<RegistryKey, void *, RegistryKey::Hasher,
                       std::equal_to<RegistryKey>,
                       std::allocator<std::pair<RegistryKey, void *>>,
                       /* StoreHash = */ true>;

    using RegistryRevMap = tsl::robin_pg_map<const void *, RegistryKey>;

    using AttributeMap =
        tsl::robin_map<AttributeKey, AttributeValue, AttributeKey::Hasher,
                       std::equal_to<AttributeKey>,
                       std::allocator<std::pair<AttributeKey, AttributeValue>>,
                       /* StoreHash = */ true>;

    /// Two-way mapping that can be used to associate pointers with unique 32 bit IDs
    RegistryFwdMap fwd;
    RegistryRevMap rev;

    /// Per-pointer attributes provided by the pointer registry
    AttributeMap attributes;
};

/// Register a pointer with Dr.Jit's pointer registry
extern uint32_t jitc_registry_put(JitBackend backend, const char *domain,
                                  void *ptr);

/// Remove a pointer from the registry
extern void jitc_registry_remove(JitBackend backend, void *ptr);

/// Provide a bound (<=) on the largest ID associated with a domain
extern uint32_t jitc_registry_get_max(JitBackend backend, const char *domain);

/// Query the ID associated a registered pointer
extern uint32_t jitc_registry_get_id(JitBackend backend, const void *ptr);

/// Query the domain associated a registered pointer
extern const char *jitc_registry_get_domain(JitBackend backend,
                                            const void *ptr);

/// Query the pointer associated a given domain and ID
extern void *jitc_registry_get_ptr(JitBackend backend, const char *domain,
                                   uint32_t id);

/// Compact the registry and release unused IDs
extern void jitc_registry_trim();

/// Clear the registry and release all IDs and attributes
extern void jitc_registry_clean();

/// Shut down the pointer registry (reports leaks)
extern void jitc_registry_shutdown();

/// Set a custom per-pointer attribute
extern void jitc_registry_set_attr(JitBackend backend, void *ptr,
                                   const char *name, const void *value,
                                   size_t size);

/// Retrieve a pointer to a buffer storing a specific attribute
extern const void *jitc_registry_attr_data(JitBackend backend,
                                           const char *domain,
                                           const char *name);
