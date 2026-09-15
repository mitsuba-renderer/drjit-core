/*
    src/isect.h -- Custom intersection functions for ray tracing backends

    A recorded intersection function is an indirect callable with a fixed
    signature and no caller: the body is retained as a symbolic graph and
    assembled into every kernel that traces a scene it is bound to. See the
    documentation in jit.h for the application-facing contract.

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

/// A recorded intersection function, owned by a handle variable
struct IsectFunc {
    /// Retained symbolic graph and captured-data layout (one instance)
    CallData *call = nullptr;

    JitBackend backend;

    /// Index of the handle variable
    uint32_t id = 0;

    /// Body hash. The body depends only on the graph and the capture layout,
    /// both of which are fixed when the function is created, so the hash is
    /// computed once and identifies the function's unit in every kernel.
    XXH128_hash_t hash { 0, 0 };

    ~IsectFunc();
};

/**
 * One geometry's binding of an intersection function. The record comes
 * first: its address is the handle that the application holds. The whole
 * struct is a 'shared' allocation that is freed only once the kernels in
 * flight have completed (see jitc_isect_bind).
 */
struct IsectBinding {
    JitIsectBinding record;

    IsectFunc *func;

    /// Scene key: the RTCScene pointer on LLVM, the OptixShaderBindingTable
    /// pointer on CUDA (the literal of the SBT variable 'sbt_var', on which
    /// the binding holds a reference). Used on CUDA to write the hit record
    /// of the launched scene's table.
    uintptr_t scene;
    uint32_t sbt_var;

    /// CUDA: index of the hit record in the shader binding table, the
    /// program group currently written to its header, and the host copy of
    /// the packed header
    uint32_t record_index;
    void *optix_pg;
    alignas(16) uint8_t optix_header[32];
};

extern void jitc_isect_begin(JitBackend backend, VarType float_type,
                             uint32_t *in);
extern uint32_t jitc_isect_end(const char *name, const uint32_t *out);
extern JitIsectBinding *jitc_isect_bind(uint32_t func, uintptr_t scene,
                                        uint32_t record_index, uint32_t flags,
                                        void *user);
extern void jitc_isect_unbind(JitIsectBinding *binding);

/**
 * Assemble the functions bound to 'scene' (or to any scene, when zero) into
 * the kernel under construction. Called while rendering a ray tracing
 * operation.
 */
extern void jitc_isect_assemble_scene(JitBackend backend, uintptr_t scene);

/// Record the intersection function units of the kernel being compiled,
/// which follow the dispatchable callables, along with the handle that
/// 'handle(unit_index)' returns for each (code address or program group)
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
 * Point the bindings whose function the kernel contains at the compiled code.
 * Runs at each launch, including kernel cache hits and frozen replay. On CUDA,
 * 'optix_sbt' is the launched shader binding table (only its bindings are
 * written) and 'stream' orders those writes; on LLVM both are null and every
 * matching binding's code pointer is set (a binding is only dereferenced by a
 * launch that traces its scene, and that launch resolves it here first).
 */
extern void jitc_isect_launch(JitBackend backend, const Kernel &kernel,
                              void *optix_sbt, void *stream);

/**
 * Check that every intersection unit of the kernel still has a live binding,
 * i.e. that the traced scenes were not rebuilt with different intersection
 * code since the kernel was recorded. Used by the dry run of a frozen
 * function to request a new trace instead of launching a stale kernel.
 */
extern bool jitc_isect_check(JitBackend backend, const Kernel &kernel);

/// Forget compiled code pointers (the unit cache was flushed)
extern void jitc_isect_invalidate();

/// Release remaining bindings at shutdown
extern void jitc_isect_shutdown();
