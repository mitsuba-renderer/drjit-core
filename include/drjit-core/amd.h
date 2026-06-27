/*
    drjit-core/amd.h -- Public API for the AMD/HIP backend.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

/// Return the no. of available AMD GPU devices that are compatible with Dr.Jit.
extern JIT_EXPORT int jit_amd_device_count();

/// Set the active AMD GPU device for the calling thread.
extern JIT_EXPORT void jit_amd_set_device(int device);

/// Return the Dr.Jit AMD device ID associated with the current thread.
extern JIT_EXPORT int jit_amd_device();

/// Return the raw HIP device ID associated with the current thread.
extern JIT_EXPORT int jit_amd_device_raw();

/// Return the AMDGPU architecture name, e.g. "gfx1201".
extern JIT_EXPORT const char *jit_amd_arch();

/// Return the HIP stream associated with the current thread.
extern JIT_EXPORT void *jit_amd_stream();

/// Return the HIP context associated with the current thread.
extern JIT_EXPORT void *jit_amd_context();

/// Return the HIP runtime version encoded as reported by hipRuntimeGetVersion().
extern JIT_EXPORT int jit_amd_runtime_version();

/// Wait for all computation on the current AMD device to finish.
extern JIT_EXPORT void jit_amd_sync_device();

/// Add HIP event synchronization between Dr.Jit's stream and an external HIP stream.
extern JIT_EXPORT void jit_amd_sync_stream(uintptr_t stream);

/**
 * \brief Wrap a HIPRT scene handle in a Dr.Jit scene variable
 *
 * The returned variable keeps backend-side bookkeeping alive while pending
 * kernels or frozen functions reference the scene. The HIPRT scene itself and
 * all geometry buffers remain owned by the caller.
 *
 * \param scene
 *     Opaque ``hiprtScene`` handle.
 *
 * \param geometry_types_mask
 *     Bit 0: triangle geometry present.
 *     Bit 3: triangle backface culling required for at least one instance.
 */
extern JIT_EXPORT uint32_t jit_amd_configure_scene(
    void *scene, uint32_t geometry_types_mask);

/**
 * \brief Wrap a HIPRT scene handle with optional custom intersection functions
 *
 * This extended form is used by renderers that build HIPRT AABB-list
 * geometries or triangle filters. ``intersect_function_names`` and
 * ``filter_function_names`` contain ``num_geom_types * num_ray_types``
 * entries, either null or naming device functions defined by ``device_source``.
 * ``func_table`` is an optional ``hiprtFuncTable`` configured by the caller.
 */
extern JIT_EXPORT uint32_t jit_amd_configure_scene_ex(
    void *scene, void *func_table, uint32_t num_geom_types,
    uint32_t num_ray_types, const char **intersect_function_names,
    const char **filter_function_names, const char *device_source,
    uint32_t geometry_types_mask);

/**
 * \brief Perform an inline HIPRT ray intersection in an AMD kernel
 *
 * Creates a ``VarKind::TraceRay`` IR node that emits HIPRT closest-hit or
 * any-hit traversal. Inputs and outputs match the Metal backend.
 *
 * Inputs ``args`` must contain 8 Float32 variables:
 * [ox, oy, oz, dx, dy, dz, tmin, tmax].
 *
 * Outputs ``out`` receives:
 * [valid, distance, bary_u, bary_v, instance_id, primitive_id,
 *  geometry_id, user_instance_id].
 */
extern JIT_EXPORT void jit_amd_ray_trace(uint32_t n_args, uint32_t *args,
                                         uint32_t mask, uint32_t *out,
                                         uint32_t n_out, uint32_t scene,
                                         int shadow);

/// Register a cleanup callback that runs when the scene variable dies.
extern JIT_EXPORT void jit_amd_scene_set_cleanup(uint32_t scene_index,
                                                 void (*callback)(void *),
                                                 void *payload);

/// Create an owner-token variable exposing a scene as a frozen-function input.
extern JIT_EXPORT uint32_t jit_amd_scene_owner_handle(uint32_t scene_index);

/// Create an owner-token variable exposing a scene's HIPRT function table.
extern JIT_EXPORT uint32_t jit_amd_scene_func_table_handle(
    uint32_t scene_index);

#if defined(__cplusplus)
}
#endif
