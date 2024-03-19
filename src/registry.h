/*
    src/registry.h -- Pointer registry for vectorized method calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

/// Register a pointer with Dr.Jit's pointer registry
extern uint32_t jitc_registry_put(JitBackend backend, const char *domain,
                                  void *ptr);

/// Remove a pointer from the registry
extern void jitc_registry_remove(const void *ptr);

/// Get the instance ID associated with the given pointer
extern uint32_t jitc_registry_id(const void *ptr);

/// Return the largest instance ID for the given domain
extern uint32_t jitc_registry_id_bound(JitBackend backend, const char *domain);

/// Return the pointer value associated with a given instance ID
extern void *jitc_registry_ptr(JitBackend backend, const char *domain, uint32_t id);

/// Return an arbitrary pointer value associated with a given domain
extern void *jitc_registry_peek(JitBackend backend, const char *domain);

/// Check for leaks in the registry
extern void jitc_registry_shutdown();

/// Disable any instances that are currently registered in the registry
extern void jitc_registry_clear();
