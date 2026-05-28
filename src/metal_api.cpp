/*
    src/metal_api.cpp -- Low-level interface to the Metal API

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Apple's Metal frameworks are linked statically (not via dlopen) on macOS,
    so this file is much shorter than the equivalent CUDA/LLVM API loaders.
    Its only responsibilities are:

      * Initialize / shut down the metal-cpp wrapper (a thin C++ shim around
        the Objective-C Metal API).
      * Provide an autorelease pool RAII helper that bookkeeping-wraps every
        Metal call, in line with the LuisaCompute reference design.
      * Centralize Metal error reporting via ``DRJIT_METAL_CHECK``.
*/

#if defined(DRJIT_ENABLE_METAL)

#include "metal_api.h"
#include "log.h"

// metal-cpp single-header. The macros below select the precise framework
// implementation header that contains the inline definitions. We include this
// from a single .cpp so the symbols are emitted exactly once.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

bool jitc_metal_api_init() {
    // metal-cpp is header-only; nothing global to initialize here. We simply
    // sanity-check that at least one Metal device exists -- this is the same
    // check that ``jitc_metal_init`` will repeat with a more detailed log.
    DRJIT_METAL_SCOPED_POOL;
    NS::Array *devices = MTL::CopyAllDevices();
    bool ok = devices && devices->count() > 0;
    if (devices)
        devices->release();
    return ok;
}

void jitc_metal_api_shutdown() {
    // No global state to release. The autorelease pools created during
    // execution take care of releasing all transient objects.
}

NS::AutoreleasePool *jitc_metal_pool_create() {
    return NS::AutoreleasePool::alloc()->init();
}

ScopedMetalPool::ScopedMetalPool()
    : pool(jitc_metal_pool_create()) { }

ScopedMetalPool::~ScopedMetalPool() {
    if (pool)
        pool->release();
}

void jitc_metal_check_error(NS::Error *error, const char *expr,
                            const char *file, int line) {
    if (!error)
        return;

    const char *desc =
        error->localizedDescription()
            ? error->localizedDescription()->utf8String()
            : "<no description>";

    jitc_fail("Metal API call failed at %s:%d (%s): %s",
              file, line, expr, desc);
}

#endif // defined(DRJIT_ENABLE_METAL)
