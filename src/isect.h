/*
    src/isect.h -- Custom intersection functions for ray tracing backends

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include "hash.h"
#include "io.h"
#include "unit.h"

struct CallData;

/// A recorded intersection function, one per instance
struct IsectFunc {
    /// Symbolic graph and captured-data layout
    CallData *call = nullptr;

    /// Index of the handle variable
    uint32_t id = 0;

    /// Hash of the rendered body per variant, which identifies the unit in
    /// every kernel. Metal scenes with instance motion use variant 1.
    XXH128_hash_t hash[2] { };

    ~IsectFunc();
};

/**
 * Connects a recorded intersection routine to custom geometry in a scene
 *
 * Recording produces an IsectFunc, but traversal needs compiled code and
 * the geometry's captured parameters. jit_isect_bind() creates this object
 * during scene setup and fills record.data with those parameters. When a
 * kernel traces the scene, its bindings identify the functions to compile.
 * At launch, func->hash selects the compiled unit to install in the Embree
 * callback record, OptiX hit record, or Metal intersection function table.
 *
 * The public JitIsectBinding record is the first member so that the API and
 * Embree callback can use its address directly. The binding retains func
 * and owns record.data until jit_isect_unbind(). Both allocations defer
 * reclamation until kernels in flight have completed.
 */
struct JitIsectBindingExt {
    JitIsectBinding record;

    IsectFunc *func;

    /// Scene key: the RTCScene pointer on LLVM, the OptixShaderBindingTable
    /// pointer on CUDA, and the MetalScene pointer on Metal
    uintptr_t scene;

    /// CUDA: the SBT variable whose literal is 'scene', on which the binding
    /// holds a reference (zero elsewhere)
    uint32_t scene_var;

    /// CUDA: index of the hit record in the shader binding table. Metal:
    /// index of the intersection function table entry.
    uint32_t record_index;

    /// Position in 'isect_bindings'
    uint32_t list_index;

    /// CUDA: the program group currently written to the hit record header
    void *optix_pg;

    /// Metal: the scene has instance motion (see ``IsectFunc::hash``)
    bool motion;
};

/// Live bindings
extern std::vector<JitIsectBindingExt *> isect_bindings;

/// Set while rendering a Metal intersection function for a scene with
/// instance motion, which selects the ``instance_motion`` variant
extern bool isect_motion;

extern void jitc_isect_begin(JitBackend backend, VarType float_type,
                             uint32_t *in);
extern uint32_t jitc_isect_end(JitBackend backend, const char *name,
                               const uint32_t *out);
extern JitIsectBinding *jitc_isect_bind(uint32_t func, uintptr_t scene,
                                        uint32_t record_index, void *user);
extern void jitc_isect_unbind(JitIsectBinding *binding);

/// Find the kernel's unit for a body hash, or null
extern const KernelIsectUnit *jitc_isect_find_unit(const Kernel &kernel,
                                                   XXH128_hash_t hash);

/**
 * Assemble the functions bound to 'scene' (or to any scene, when zero) into
 * the kernel under construction. Called while rendering a ray tracing
 * operation.
 */
extern void jitc_isect_assemble_scene(JitBackend backend, uintptr_t scene);

/// Record the intersection function units of the kernel being compiled.
template <typename Handle>
void jitc_isect_kernel_units(Kernel &kernel, Handle handle) {
    uint32_t n = (uint32_t) callable_units.size() - unit_dispatch_count;
    if (!n)
        return;
    kernel.isect = (KernelIsectUnit *) malloc_check(sizeof(KernelIsectUnit) * n);
    kernel.isect_count = n;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t unit = unit_dispatch_count + i;
        kernel.isect[i] = { callable_units[unit].hash, handle(unit) };
    }
}

/**
 * Install the compiled code of the kernel's intersection functions into the
 * bindings that use them. Runs at every launch, including replay of frozen
 * functions.
 *
 * LLVM: stores the code address in the binding record. CUDA: writes the hit
 * group header and data pointer of each binding's record in the launched
 * shader binding table 'optix_sbt', ordered on 'stream'. Metal launches
 * resolve their bindings via jitc_metal_scene_ift() instead.
 */
extern void jitc_isect_launch(JitBackend backend, const Kernel &kernel,
                              void *optix_sbt, void *stream);

/// Forget compiled code pointers (the unit cache was flushed)
extern void jitc_isect_invalidate();

/// Release remaining bindings at shutdown
extern void jitc_isect_shutdown();
