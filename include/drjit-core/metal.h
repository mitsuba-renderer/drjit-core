/*
    drjit-core/metal.h -- Public API for the Metal backend, including ray
    tracing primitives.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

/// Return an opaque pointer to the active Metal device (``id<MTLDevice>``).
/// Useful for application code that wishes to interoperate with raw Metal
/// APIs (e.g. acceleration structure construction).
extern JIT_EXPORT void *jit_metal_context();

/// Return the Metal command queue handle (``id<MTLCommandQueue>``) that Dr.Jit
/// uses for kernel submission.
extern JIT_EXPORT void *jit_metal_command_queue();

/**
 * \brief Inform Dr.Jit about a per-scene Metal ray-tracing configuration.
 *
 * The application calls this once per scene to register its acceleration
 * structure (TLAS), the list of child resources the TLAS references (BLAS
 * handles, vertex/index buffers), and (optionally) a library of custom
 * intersection functions plus an intersection-function-table specification.
 *
 * The function returns a JIT variable index that owns the lifetime of the
 * scene state on the drjit side. This index must be passed as the trailing
 * argument to subsequent ``jit_metal_ray_trace`` calls in order to bind
 * this scene's TLAS / IFT at launch time. When the variable's reference
 * count reaches zero (i.e. the application releases its handle), Dr.Jit
 * releases its retained references on the Metal library, the intersection
 * function table, and any other owned per-scene resources.
 *
 * Multiple scenes can be live simultaneously — each call returns a fresh
 * index, and ray-tracing kernels select the correct TLAS / IFT per launch
 * based on which scene_index was attached to the corresponding TraceRay
 * IR node at recording time.
 *
 * Each call builds a fresh scene variable; a geometry edit registers a new
 * scene, whose owner handle the next frozen-function traversal rebinds as an
 * input (see \ref jit_metal_scene_owner_handle).
 *
 * \param acceleration_structure
 *     The ``id<MTLAccelerationStructure>`` TLAS.
 *
 * \param resources / n_resources
 *     List of ``id<MTLResource>`` pointers that the TLAS references.
 *
 * \param intersection_fn_library
 *     Optional ``id<MTLLibrary>`` with custom intersection functions.
 *
 * \param n_ift_entries
 *     Number of entries in the intersection function table. Must be 0 if
 *     ``intersection_fn_library`` is null.
 *
 * \param ift_function_names
 *     Array of length ``n_ift_entries``. Each is a C-string name of an MSL
 *     intersection function in ``intersection_fn_library``. The names are
 *     copied internally.
 *
 * \param n_ift_buffers / ift_buffers / ift_buffer_slots Buffer bindings of the
 *     intersection function table: ``ift_buffers[i]`` (an ``id<MTLBuffer>``) is
 *     bound at the MSL ``[[buffer(ift_buffer_slots[i])]]`` slot. These bindings
 *     are scene-wide: intersection functions must locate per-geometry data
 *     through indexing (e.g. by instance/geometry/primitive ID) rather than
 *     through entry-specific buffers or offsets.
 *
 * \param geometry_types_mask
 *     Bit 0: triangle geometry present.
 *     Bit 1: bounding-box (custom-intersection) geometry present.
 *     Bit 2: curve geometry present.
 *     Bit 3: triangle backface culling required for at least one instance.
 */
extern JIT_EXPORT uint32_t jit_metal_configure_scene(
    void *acceleration_structure,
    void **resources,
    uint32_t n_resources,
    void *intersection_fn_library,
    uint32_t n_ift_entries,
    const char **ift_function_names,
    uint32_t n_ift_buffers,
    void **ift_buffers,
    const uint32_t *ift_buffer_slots,
    uint32_t geometry_types_mask);

/**
 * \brief Perform an inline ray intersection in a Metal compute kernel
 *
 * Creates a ``VarKind::TraceRay`` IR node that, when evaluated, emits an
 * MSL ``intersector<triangle_data, instancing>::intersect()`` call against
 * the scene identified by \c scene. For lanes that are masked off or that miss
 * the geometry, the distance output is set to +infinity, the validity flag to
 * ``false``, and every other output to zero, so callers need not separately
 * clear them.
 *
 * \param n_args
 *     Number of ray input arguments. Must be 8.
 *
 * \param args
 *     Array of 8 JIT variable indices:
 *       [0] ox     (Float32) — ray origin X
 *       [1] oy     (Float32) — ray origin Y
 *       [2] oz     (Float32) — ray origin Z
 *       [3] dx     (Float32) — ray direction X
 *       [4] dy     (Float32) — ray direction Y
 *       [5] dz     (Float32) — ray direction Z
 *       [6] tmin   (Float32) — minimum ray distance
 *       [7] tmax   (Float32) — maximum ray distance
 *
 * \param mask
 *     JIT variable index of the active lane mask (Bool).
 *
 * \param out
 *     Array of 8 JIT variable indices (output, written by this function):
 *       [0] valid        (Bool)    — true if a hit was found
 *       [1] distance     (Float32) — distance to the closest hit
 *       [2] bary_u       (Float32) — barycentric U coordinate
 *       [3] bary_v       (Float32) — barycentric V coordinate
 *       [4] instance_id  (UInt32)  — instance index in the TLAS
 *       [5] primitive_id  (UInt32)  — triangle index in the mesh
 *       [6] geometry_id   (UInt32)  — geometry index within the instance
 *       [7] user_instance_id (UInt32) — per-instance userID
 *
 * \param n_out
 *     Number of output variables. Must be 8.
 *
 * \param scene
 *     JIT variable index returned by \c jit_metal_configure_scene. Selects
 *     which scene's TLAS / IFT this trace operation will run against.
 *
 * \param shadow
 *     If nonzero, performs a shadow ray test. In this case, only
 *     output 0 (the hit flag) is computed; outputs 1-7 are left untouched.
 */
extern JIT_EXPORT void jit_metal_ray_trace(uint32_t n_args, uint32_t *args,
                                           uint32_t mask, uint32_t *out,
                                           uint32_t n_out, uint32_t scene,
                                           int shadow);

/**
 * \brief Look up the id<MTLBuffer> containing the given pointer.
 *
 * Returns the ``id<MTLBuffer>`` whose address range covers ``ptr``, or
 * ``nullptr`` if no registered buffer contains that address.
 * If found, ``*offset`` is set to the byte offset from the buffer start.
 */
extern JIT_EXPORT void *jit_metal_lookup_buffer(void *ptr, size_t *offset);

/**
 * \brief Register a cleanup callback that runs when the scene variable dies
 *
 * The application's Metal objects (TLAS, BLAS, buffers) must outlive the scene
 * variable, which can outlast the application's own use of the scene since
 * unevaluated kernels and frozen-function recordings reference it through their
 * TraceRay nodes. ``callback`` runs once, right after the MetalScene is
 * destroyed, so the application can release the scene by dropping its reference
 * rather than freeing those objects directly. ``scene_index`` must be a
 * variable representing a scene, as returned by \ref jit_metal_configure_scene.
 */
extern JIT_EXPORT void jit_metal_scene_set_cleanup(uint32_t scene_index,
                                                   void (*callback)(void *),
                                                   void *payload);

/**
 * \brief Create a handle exposing a scene as a frozen-function input
 *
 * Returns an ``UInt64`` variable whose data pointer is drjit-core's internal
 * per-scene bookkeeping object (an opaque C++ struct tracking the TLAS and
 * referenced resources). It is used as an identity token so that traversal by
 * ``dr.freeze`` can capture and correctly bind the scene to kernel launches.
 * The handle does not own the scene.
 * \c scene_index must be a variable representing a scene, as returned by \ref
 * jit_metal_configure_scene.
 */
extern JIT_EXPORT uint32_t jit_metal_scene_owner_handle(uint32_t scene_index);

#if defined(__cplusplus)
}
#endif
