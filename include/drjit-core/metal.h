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
 * handles, vertex/index buffers), and the size of the intersection function
 * table that its bounding-box geometry indexes.
 *
 * The function returns a JIT variable index that owns the lifetime of the scene
 * state on the drjit side. This index must be passed to ``jit_metal_ray_trace``
 * to bind this scene's TLAS / IFT at launch time. When the variable's reference
 * count reaches zero (i.e. the application releases its handle), Dr.Jit
 * releases the intersection function tables and any other owned per-scene
 * resources.
 *
 * \param acceleration_structure
 *     The ``id<MTLAccelerationStructure>`` TLAS.
 *
 * \param resources / n_resources
 *     List of ``id<MTLResource>`` pointers that the TLAS references.
 *
 * \param n_isect_entries
 *     Number of entries in the scene's intersection function table.
 *
 * \param geometry_types_mask
 *     Bit 0: triangle geometry present.
 *     Bit 1: bounding-box (custom-intersection) geometry present.
 *     Bit 2: curve geometry present.
 *     Bit 3: triangle backface culling required for at least one instance.
 *     Bit 4: motion (instance motion) present.
 */
extern JIT_EXPORT uint32_t jit_metal_configure_scene(
    void *acceleration_structure,
    void **resources,
    uint32_t n_resources,
    uint32_t n_isect_entries,
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
 *     Number of ray input arguments. Must be 9, or 10 when a per-lane
 *     visibility mask is supplied.
 *
 * \param args
 *     Array of 9 (or 10) JIT variable indices:
 *       [0] ox     (Float32) — ray origin X
 *       [1] oy     (Float32) — ray origin Y
 *       [2] oz     (Float32) — ray origin Z
 *       [3] dx     (Float32) — ray direction X
 *       [4] dy     (Float32) — ray direction Y
 *       [5] dz     (Float32) — ray direction Z
 *       [6] tmin   (Float32) — minimum ray distance
 *       [7] tmax   (Float32) — maximum ray distance
 *       [8] time   (Float32) — ray evaluation time
 *       [9] vmask  (UInt32)  — optional visibility mask
 *
 * \param mask
 *     JIT variable index of the active lane mask (Bool).
 *
 * \param out
 *     Array of 8 JIT variable indices (output, written by this function):
 *       [0] valid        (Bool)    — true if a hit was found
 *       [1] distance     (Float32) — distance to the closest hit
 *       [2] bary_u       (Float32) — barycentric U coordinate, or the first
 *                                    attribute of a bounding-box hit
 *       [3] bary_v       (Float32) — barycentric V coordinate, or the second
 *                                    attribute of a bounding-box hit
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
 *     If nonzero, performs a shadow ray test that accepts any intersection
 *     and terminates traversal early. Only outputs 0 (the hit flag) and 7
 *     (the user instance ID of whichever hit ended the traversal) are
 *     computed; callers may use the latter to classify the hit. Outputs
 *     1-6 keep their miss values.
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
 * The application's Metal objects (TLAS, BLAS, buffers) and intersection
 * bindings must outlive the scene variable. ``callback`` runs once when the
 * variable is freed, right before the scene is destroyed, so the application
 * can release the scene by dropping its reference. ``scene_index`` must be a
 * variable representing a scene, as returned by \ref jit_metal_configure_scene.
 */
extern JIT_EXPORT void jit_metal_scene_set_cleanup(uint32_t scene_index,
                                                   void (*callback)(void *),
                                                   void *payload);

/**
 * \brief Create a handle exposing a scene as a frozen-function input
 *
 * Returns an ``UInt64`` variable whose data pointer is drjit-core's internal
 * per-scene bookkeeping object. It is used as an identity token for
 * ``dr.freeze``. The handle does not own the scene. \c scene_index must be a
 * variable representing a scene, as returned by \ref jit_metal_configure_scene.
 */
extern JIT_EXPORT uint32_t jit_metal_scene_owner_handle(uint32_t scene_index);

#if defined(__cplusplus)
}
#endif
