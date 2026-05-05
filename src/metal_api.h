/*
    src/metal_api.h -- Low-level interface to the Metal API

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    This header sits between the Dr.Jit codebase and the metal-cpp wrapper. We
    intentionally keep the metal-cpp headers out of widely-included headers
    (such as internal.h) by exposing Metal handles as ``void *`` pointers in
    the public-facing data structures and only ``#include`` ``metal-cpp`` from
    the small number of ``.cpp`` files that genuinely need it (``metal_*.cpp``).

    On platforms without Apple's Metal frameworks (i.e. anything other than
    macOS / iOS) the entire Metal backend is conditionally disabled through the
    ``DRJIT_ENABLE_METAL`` preprocessor symbol that is set by the build system.
*/

#pragma once

#if !defined(DRJIT_ENABLE_METAL)
#  error "metal_api.h must only be included when DRJIT_ENABLE_METAL is defined"
#endif

#include <cstdint>
#include <cstddef>

// Forward declarations of Metal handles (kept opaque -- see header comment)
namespace MTL {
    class Device;
    class CommandQueue;
    class CommandBuffer;
    class ComputeCommandEncoder;
    class BlitCommandEncoder;
    class AccelerationStructureCommandEncoder;
    class ComputePipelineState;
    class Library;
    class Function;
    class Buffer;
    class Texture;
    class SharedEvent;
    class BinaryArchive;
    class AccelerationStructure;
    class IntersectionFunctionTable;
    class VisibleFunctionTable;
}

namespace NS {
    class AutoreleasePool;
    class String;
    class Error;
}

/// Initialize the Metal API layer (returns false on non-Apple platforms or
/// when no Metal-capable GPU could be found)
extern bool jitc_metal_api_init();

/// Shutdown the Metal API layer and release any global state
extern void jitc_metal_api_shutdown();

/// Allocate a fresh autorelease pool. The caller must call ``release()`` on
/// the returned pool when finished.
extern NS::AutoreleasePool *jitc_metal_pool_create();

/// RAII helper that wraps a block of Metal API calls with an autorelease pool.
/// Following the LuisaCompute pattern, every Metal interaction is bracketed
/// to keep the Objective-C reference counts under control.
struct ScopedMetalPool {
    NS::AutoreleasePool *pool;

    ScopedMetalPool();
    ~ScopedMetalPool();

    ScopedMetalPool(const ScopedMetalPool &) = delete;
    ScopedMetalPool &operator=(const ScopedMetalPool &) = delete;
};

/// Convenience macro: wrap a block of code in an autorelease pool.
#define DRJIT_METAL_SCOPED_POOL ScopedMetalPool _drjit_metal_pool

/// Helper used by ``DRJIT_METAL_CHECK`` to format and raise Metal errors. The
/// implementation lives in ``metal_api.cpp``. The function takes the
/// ``NS::Error`` produced by Metal, the originating expression, and the file
/// + line at which the error occurred.
extern void jitc_metal_check_error(NS::Error *error, const char *expr,
                                   const char *file, int line);

/// Execute an expression that returns an ``NS::Error*`` and raise an exception
/// if it indicates failure. The output of the expression must be assigned to a
/// local ``NS::Error *err = nullptr;`` declared by the caller.
#define DRJIT_METAL_CHECK(expr_)                                              \
    do {                                                                      \
        if (unlikely(_drjit_metal_err))                                       \
            jitc_metal_check_error(_drjit_metal_err, #expr_, __FILE__,        \
                                   __LINE__);                                 \
    } while (0)
