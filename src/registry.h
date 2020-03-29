#pragma once

#include <stdint.h>

/// Register a pointer with Enoki's pointer registry
extern uint32_t jit_registry_put(const char *domain, void *ptr);

/// Remove a pointer from the registry
extern void jit_registry_remove(void *ptr);

/// Provide a bound (<=) on the largest ID associated with a domain
extern uint32_t jit_registry_get_max(const char *domain);

/// Query the ID associated a registered pointer
extern uint32_t jit_registry_get_id(const void *ptr);

/// Query the domain associated a registered pointer
extern const char *jit_registry_get_domain(const void *ptr);

/// Query the pointer associated a given domain and ID
extern void *jit_registry_get_ptr(const char *domain, uint32_t id);

/// Compact the registry and release unused IDs
extern void jit_registry_trim();

/// Shut down the pointer registry (reports leaks)
extern void jit_registry_shutdown();
