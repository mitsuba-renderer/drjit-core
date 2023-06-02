#pragma once

#include "core.h"

/// Initialize core data structures of the JIT compiler
extern void jitc_init(uint32_t backends);

/// Release all resources used by the JIT compiler, and report reference leaks.
extern void jitc_shutdown(int light);

