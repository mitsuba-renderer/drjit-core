/*
    src/registry.h -- Pointer registry for vectorized method calls

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>

/// Register a pointer with Enoki's pointer registry
extern uint32_t jitc_registry_put(const char *domain, void *ptr);

/// Remove a pointer from the registry
extern void jitc_registry_remove(void *ptr);

/// Provide a bound (<=) on the largest ID associated with a domain
extern uint32_t jitc_registry_get_max(const char *domain);

/// Query the ID associated a registered pointer
extern uint32_t jitc_registry_get_id(const void *ptr);

/// Query the domain associated a registered pointer
extern const char *jitc_registry_get_domain(const void *ptr);

/// Query the pointer associated a given domain and ID
extern void *jitc_registry_get_ptr(const char *domain, uint32_t id);

/// Compact the registry and release unused IDs
extern void jitc_registry_trim();

/// Shut down the pointer registry (reports leaks)
extern void jitc_registry_shutdown();

/// Set a custom per-pointer attribute
extern void jitc_registry_set_attr(void *ptr, const char *name,
                                   const void *value, size_t size);

/// Retrieve a pointer to a buffer storing a specific attribute
extern const void *jitc_registry_attr_data(const char *domain, const char *name);
