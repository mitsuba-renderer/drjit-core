/*
    src/metal.h -- Metal backend declarations

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

#include <cstdint>
#include <vector>

#if defined(DRJIT_ENABLE_METAL)

struct ThreadState;
struct Kernel;
enum class MetalKernel : uint32_t;

/// Initialize the Metal backend
extern bool jitc_metal_init();

/// Lazily create/look up a block (prefix) reduction compute pipeline.
/// Returns nullptr if the (op, type) combination is unsupported.
extern void *jitc_metal_block_reduce_pipeline(int device, MetalReduceKind kind,
                                              ReduceOp op, VarType vt);

/// Release all resources held by the Metal backend.
extern void jitc_metal_shutdown();

/// Return the active ``id<MTLDevice>`` for the current thread.
extern void *jitc_metal_context_impl();

/// Return the active ``id<MTLCommandQueue>`` for the current thread.
extern void *jitc_metal_command_queue_impl();

/// Begin/end a Metal GPU capture scope
extern void jitc_metal_profile_start();
extern void jitc_metal_profile_stop();

/// Wait for all Metal work submitted on the current thread to complete.
extern void jitc_metal_sync(ThreadState *ts);

/// Submit the current thread's pending command buffer without waiting for it.
extern void jitc_metal_flush(ThreadState *ts);

/// Wait until the command buffers committed by all threads have completed.
extern void jitc_metal_sync_devices();

/// Event API functions for the Metal backend
extern JitEvent jitc_metal_event_create(bool enable_timing);
extern void jitc_metal_event_destroy(JitEvent event);
extern void jitc_metal_event_record(JitEvent event);
extern int jitc_metal_event_query(JitEvent event);
extern void jitc_metal_event_wait(JitEvent event);

/// Retain an extra reference to a kernel-history command buffer
extern void jitc_metal_history_retain(void *cb);

/// Wait for a kernel-history command buffer and return its GPU time (ms).
/// Borrows the reference (i.e. does not release it).
extern float jitc_metal_history_wait(void *cb);

/// Release a kernel-history command buffer without waiting for it
extern void jitc_metal_history_release(void *cb);

// ---------------------------------------------------------------------

/// Allocate a new ``MTLBuffer`` of ``size`` bytes (shared or private storage).
/// Returns an owned (+1) ``id<MTLBuffer>`` handle. The ``ptr_out`` argument
/// returns the CPU pointer (shared) or GPU address (private).
///
/// The ``metal_*`` functions lack the ``jitc_`` prefix to indicate that they
/// are called *without* holding the Dr.Jit lock.
extern void *metal_buffer_new(void *dev, size_t size, bool shared,
                              void **ptr_out);

/// Release a buffer handle allocated by ``metal_buffer_new``.
extern void metal_buffer_free(void *buffer);

/// Register a MTLBuffer so for use with the find_buffer API shown below
extern void jitc_metal_register_buffer(void *ptr, void *mtl_buffer,
                                       size_t size);

/// Find the ``MTLBuffer`` for a given pointer adress. Returns the
/// ``id<MTLBuffer>`` and the byte offset from its start.
/// Returns {nullptr, 0} if no registered allocation contains ``ptr``.
extern void *jitc_metal_find_buffer(void *ptr, size_t *offset_out);

/// Unregister a previously registered MTLBuffer
extern void *jitc_metal_unregister_buffer(void *ptr);

// ---------------------------------------------------------------------

/// Compile the just-assembled kernel (one unit per indirect callable plus the
/// entry point, see unit.h) into a Kernel object. Units are compiled
/// concurrently, cached by content hash, and linked into a pipeline state.
extern bool jitc_metal_kernel_compile(ThreadState *ts, Kernel &kernel);

/// Free a previously compiled Metal kernel.
extern void jitc_metal_kernel_free(Kernel &kernel);

// ---------------------------------------------------------------------
//  Command-buffer / encoder helpers
// ---------------------------------------------------------------------

/// An intersection function table of a scene, built for one compute
/// pipeline. Function handles are pipeline-specific, hence a scene needs one
/// table per pipeline that traces it.
struct MetalSceneIFT {
    void *pso;
    void *ift;
};

/// Per-scene Metal ray-tracing state. One instance is allocated per call
/// to ``jitc_metal_configure_scene`` and wrapped in a JIT variable; its
/// lifetime is then driven by Dr.Jit's reference counting. When all
/// external references go away, Dr.Jit invokes a destruction callback
/// which releases the per-scene Metal objects.
struct MetalScene {
    /// id<MTLAccelerationStructure> (TLAS). Reconstructed in-shader from its
    /// ``gpuResourceID`` in ``params.args[]`` and made resident via
    /// useResource() at each launch. Not retained — the caller owns the TLAS
    /// lifetime.
    void *tlas = nullptr;

    /// Cached gpuResourceID of ``tlas``
    void *tlas_rid_for = nullptr;
    uint64_t tlas_rid = 0;

    /// Resources referenced by the TLAS (BLAS handles, vertex/index
    /// buffers, etc.). useResource()'d at every launch. Not retained.
    std::vector<void *> resources;

    /// Number of intersection function table entries, each bound to a
    /// recorded intersection function via jit_isect_bind()
    uint32_t isect_count = 0;

    /// Host-visible table with the data block pointer of each entry's
    /// binding. It is bound at buffer slot 0, and the generated intersection
    /// functions index it with their table offset.
    uint64_t *isect_table = nullptr;

    /// Bit 0 = triangle, bit 1 = bounding_box, bit 2 = curves, bit 3 =
    /// triangle backface culling, bit 4 = instance motion. Used to
    /// specialize the MSL ``intersector<...>`` template and the generated
    /// intersection functions at codegen time.
    uint32_t geometry_types_mask = 0;

    /// Cached handle variables for the scene's TLAS and (optionally) IFT
    uint32_t accel_handle = 0;
    uint32_t ift_handle = 0;

    /// Intersection function tables, one per pipeline that traced the scene.
    /// Each ``ift`` is an owned (+1) handle. The entries reflect the
    /// bindings at creation time, and a binding change drops the tables.
    std::vector<MetalSceneIFT> ift_cache;

    /// Invoked when the scene variable is freed, before the scene is
    /// destroyed, letting the application release its Metal objects and
    /// intersection bindings (see jit_metal_scene_set_cleanup()).
    void (*cleanup)(void *) = nullptr;
    void *cleanup_payload = nullptr;
};

/// Look up the ``MetalScene`` attached to a JIT variable returned by
/// ``jit_metal_configure_scene``. Aborts via ``jitc_fail()`` if the variable
/// does not wrap a Metal scene (e.g. destroyed already, wrong type/kind, ...).
extern MetalScene *jitc_metal_get_scene(uint32_t scene_index);

/// Wrap a raw owner pointer that is not itself a JIT variable (a scene / IFT)
/// as a resource handle: mem-maps a fresh backing for ``ptr`` and delegates to
/// jitc_var_resource_pointer(). Returns 0 if ``ptr`` is null.
extern uint32_t jitc_metal_make_resource_handle(void *ptr, ResourceKind kind);

/// Return the cached resource handle (Accel or IFT) for ``scene``, creating it
/// on first use. The returned index carries a reference the caller must release
extern uint32_t jitc_metal_scene_resource_handle(MetalScene *scene,
                                                 ResourceKind kind);

/// Create a UInt64 whose data pointer is the ``MetalScene`` owner of
/// ``scene_index``, surfacing the scene as a rebindable freeze input. Returns
/// 0 if ``scene_index`` does not identify a live scene.
extern uint32_t jitc_metal_scene_owner_handle(uint32_t scene_index);

/// Resolve the live ``gpuResourceID`` of an opaque-resource ``owner`` of the
/// given ``kind`` (an acceleration structure, texture, or sampler), writing it
/// to ``*value_out`` and returning true. Returns false for IFT handles
/// (PSO-dependent, refreshed at launch) and for ordinary buffers.
extern bool jitc_metal_resource_id(void *owner, ResourceKind kind,
                                   void **value_out);

/// Render-discovered set of distinct ``MetalScene*`` referenced by this
/// kernel's ``VarKind::TraceRay`` nodes (top-level schedule + callable bodies).
/// Populated during code generation and folded into the kernel source.
extern std::vector<MetalScene *> metal_kernel_scenes;

/// Append ``scene`` to ``metal_kernel_scenes`` if not already present (linear
/// dedup; at most a handful of scenes per kernel). ``nullptr`` is ignored.
extern void metal_register_kernel_scene(MetalScene *scene);

/// Return the scene's intersection function table for the pipeline of
/// ``kernel`` (an opaque ``id<MTLIntersectionFunctionTable>``). The first
/// call for a pipeline creates the table and points its entries at the
/// kernel's intersection function units. Returns null if the scene has no
/// intersection function entries.
extern void *jitc_metal_scene_ift(MetalScene *scene, const Kernel &kernel);

/// Drop the scene's intersection function tables (a binding changed)
extern void jitc_metal_scene_release_ifts(MetalScene *scene);

/// Append the resources that the kernel's intersection functions bound to
/// ``scene`` dereference (the pointer table, the data blocks, and the buffers
/// and textures they point to) to ``ro`` for residency
extern void jitc_metal_isect_resources(MetalScene *scene, const Kernel &kernel,
                                       std::vector<void *> &ro);

/// Build a MetalScene with the given configuration and wrap it in a JIT
/// variable. Returns the variable index; the caller is expected to hold
/// the reference for the scene's lifetime and dec_ref it on destruction.
extern uint32_t jitc_metal_configure_scene(void *accel, void **resources,
                                           uint32_t n_resources,
                                           uint32_t n_isect_entries,
                                           uint32_t geometry_types_mask);

/// Trace a batch of rays against the active scene. Mirrors the signature
/// of the public ``jit_metal_ray_trace`` (see drjit-core/metal.h).
extern void jitc_metal_ray_trace(uint32_t n_args, uint32_t *args,
                                 uint32_t mask, uint32_t *out,
                                 uint32_t n_out, uint32_t scene, int shadow);

#endif // defined(DRJIT_ENABLE_METAL)
