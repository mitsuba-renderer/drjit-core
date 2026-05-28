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

/// Return an opaque pointer to the active Metal device (``MTL::Device*``).
/// Useful for application code that wishes to interoperate with raw Metal
/// APIs (e.g. acceleration structure construction).
extern JIT_EXPORT void *jit_metal_context();

/// Return the Metal command queue handle (``MTL::CommandQueue*``) that Dr.Jit
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
 * count reaches zero (i.e. the application releases its handle), drjit
 * releases its retained references on the Metal library, the intersection
 * function table, and any other owned per-scene resources.
 *
 * Multiple scenes can be live simultaneously — each call returns a fresh
 * index, and ray-tracing kernels select the correct TLAS / IFT per launch
 * based on which scene_index was attached to the corresponding TraceRay
 * IR node at recording time.
 *
 * \param acceleration_structure
 *     ``MTL::AccelerationStructure*`` — the TLAS to bind at ``[[buffer(1)]]``
 *     for every kernel that traces against this scene.
 *
 * \param resources / n_resources
 *     List of ``MTL::Resource*`` pointers that the TLAS references (child
 *     BLAS, vertex/index buffers, etc.). All entries are marked resident
 *     via ``useResource()`` whenever a kernel that uses this scene is
 *     launched.
 *
 * \param intersection_fn_library
 *     Optional ``MTL::Library*`` with custom intersection functions
 *     (``[[intersection(bounding_box, instancing)]]``). May be ``nullptr``
 *     for triangle-only scenes.
 *
 * \param n_ift_entries
 *     Number of entries in the intersection function table. Must be 0 if
 *     ``intersection_fn_library`` is null.
 *
 * \param ift_function_names
 *     Array of length ``n_ift_entries``. Each is a C-string name of an MSL
 *     intersection function in ``intersection_fn_library``.
 *
 * \param ift_buffers
 *     Array of length ``n_ift_entries``. Each is an ``MTL::Buffer*``
 *     bound to the IFT at the slot indicated by ``ift_buffer_slots[i]``.
 *     Individual entries may be null.
 *
 * \param ift_buffer_slots
 *     Array of length ``n_ift_entries`` — the MSL ``[[buffer(N)]]`` slot
 *     indices for each IFT entry's per-entry buffer.
 *
 * \param ift_buffer_offsets
 *     Optional array of length ``n_ift_entries`` — per-entry byte offsets
 *     forwarded to ``setBuffer``. Two IFT entries can share the same
 *     underlying buffer at different starting positions, which the
 *     multi-shape shape-group path uses to point each child at its own
 *     slice of the per-type combined buffer without needing a per-geometry
 *     uniform. May be null (treated as all zeros).
 *
 * \param geometry_types_mask
 *     Bit 0: triangle geometry present.
 *     Bit 1: bounding-box (custom-intersection) geometry present.
 *     Bit 2: curve geometry present.
 */
extern JIT_EXPORT uint32_t jit_metal_configure_scene(
    void *acceleration_structure,
    void **resources,
    uint32_t n_resources,
    void *intersection_fn_library,
    uint32_t n_ift_entries,
    const char **ift_function_names,
    void **ift_buffers,
    const uint32_t *ift_buffer_slots,
    const uint64_t *ift_buffer_offsets,
    uint32_t geometry_types_mask);

/**
 * \brief Perform an inline ray intersection in a Metal compute kernel
 *
 * Creates a ``VarKind::TraceRay`` IR node that, when evaluated, emits an
 * MSL ``intersector<triangle_data, instancing>::intersect()`` call against
 * the scene identified by \c scene.
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
 *     Array of 7 JIT variable indices (output, written by this function):
 *       [0] valid        (Bool)    — true if a hit was found
 *       [1] distance     (Float32) — distance to the closest hit
 *       [2] bary_u       (Float32) — barycentric U coordinate
 *       [3] bary_v       (Float32) — barycentric V coordinate
 *       [4] instance_id  (UInt32)  — instance index in the TLAS
 *       [5] primitive_id  (UInt32)  — triangle index in the mesh
 *       [6] geometry_id   (UInt32)  — geometry index within the instance
 *
 * \param n_out
 *     Number of output variables. Must be 7.
 *
 * \param scene
 *     JIT variable index returned by \c jit_metal_configure_scene. Selects
 *     which scene's TLAS / IFT this trace operation will run against.
 */
extern JIT_EXPORT void jit_metal_ray_trace(uint32_t n_args, uint32_t *args,
                                           uint32_t mask, uint32_t *out,
                                           uint32_t n_out, uint32_t scene);

/**
 * \brief Look up the MTL::Buffer containing the given pointer.
 *
 * Returns the ``MTL::Buffer*`` whose address range covers ``ptr``, or
 * ``nullptr`` if no registered buffer contains that address.
 * If found, ``*offset`` is set to the byte offset from the buffer start.
 */
extern JIT_EXPORT void *jit_metal_lookup_buffer(void *ptr, size_t *offset);

/**
 * \brief Invalidate a scene's TLAS pointer ahead of releasing it.
 *
 * Call this from ``accel_release_metal`` *before* releasing the
 * MTL::AccelerationStructure handle. If the underlying MetalScene is
 * still alive (e.g. captured by a frozen-function recording's TraceRay
 * dependency), this nulls out its ``tlas`` field so subsequent kernel
 * launches that resolve back to this MetalScene see "no TLAS" and fall
 * back to the most recently configured live scene at launch time —
 * preventing a use-after-free in ``setAccelerationStructure``.
 *
 * No-op if ``scene_index`` is 0 or the MetalScene is already released.
 */
extern JIT_EXPORT void jit_metal_invalidate_scene_tlas(uint32_t scene_index);

#if defined(__cplusplus)
}
#endif
