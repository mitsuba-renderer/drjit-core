/*
    src/metal.h -- Metal backend declarations

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_METAL)

struct ThreadState;
struct Kernel;

/// Initialize the Metal backend
extern bool jitc_metal_init();

/// Release all resources held by the Metal backend.
extern void jitc_metal_shutdown();

/// Wait for all Metal work submitted on the current thread to complete.
extern void jitc_metal_sync(ThreadState *ts);

// ---------------------------------------------------------------------

/// Allocate a new ``MTLBuffer`` of ``size`` bytes (shared or private storage).
/// Returns an owned (+1) ``id<MTLBuffer>`` handle. The ``ptr_out`` argument
/// returns the CPU pointer (shared) or GPU address (private).
///
/// ``metal_buffer_new``/``metal_buffer_free`` are called from
/// ``jit_malloc``/``jit_free`` *without* holding the Dr.Jit lock — hence the
/// ``metal_`` (rather than ``jitc_``) prefix.
extern void *metal_buffer_new(void *device, size_t size, bool shared,
                              void **ptr_out);

/// Release a buffer handle allocated by ``metal_buffer_new``.
extern void metal_buffer_free(void *buffer);

/// Schedule a deferred free: once ``cmdbuf`` finishes executing on the GPU,
/// return the allocation ``(info, ptr)`` to the malloc free list.
extern void jitc_metal_cmdbuf_free_on_complete(void *cmdbuf, uint64_t info,
                                               void *ptr);

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

/// Compile a piece of MSL source code into a Kernel object
extern bool jitc_metal_kernel_compile(const char *source, size_t source_size,
                                      const char *kernel_name, Kernel &kernel);

/// Free a previously compiled Metal kernel.
extern void jitc_metal_kernel_free(Kernel &kernel);

/// Retrieve a precompiled compute pipeline state from the utility kernel
/// library by kernel name (e.g. "block_reduce_add_f32_1024"). Returns
/// nullptr if the kernel is not found. Pipeline states are cached.
/// (opaque ``id<MTLComputePipelineState>``)
extern void *
jitc_metal_get_pipeline(int device_id, const char *name);

// ---------------------------------------------------------------------
//  Command-buffer / encoder helpers
// ---------------------------------------------------------------------
//
// The per-thread command buffer / encoder lifecycle lives on
// ``MetalThreadState`` (see ``ensure_cmdbuf`` / ``close_encoder`` /
// ``ensure_compute_encoder`` / ``ensure_blit_encoder`` / ``flush`` in
// metal_ts.h). ``jitc_metal_sync`` remains a free function because it is also
// called with a generic ``ThreadState *`` (e.g. from malloc.cpp / init.cpp /
// the record thread state).

/// Flush the thread's pending command buffer and wait for GPU completion so
/// the CPU can read back results. Routes a RecordThreadState to its internal
/// thread state, then delegates to ``MetalThreadState::flush(true)``.
extern void jitc_metal_sync(ThreadState *ts);

/// Commit + wait on an ad-hoc (standalone) command buffer.
extern void jitc_metal_commit_and_wait(void *cb_ptr);

/// Live-MetalScene registry. Used by ``MetalThreadState::launch`` during
/// frozen-function replay to detect stale ``MetalScene*`` pointers (the
/// scene was rebuilt mid-recording) and fall back to the most-recently
/// configured one.
extern void jitc_metal_register_live_scene(void *scene);
extern void jitc_metal_unregister_live_scene(void *scene);
extern bool jitc_metal_is_live_scene(void *scene);
extern void *jitc_metal_last_live_scene();

#include <string>
#include <unordered_map>
#include <cstdint>
#include <vector>


/// Per-scene Metal ray-tracing state. One instance is allocated per call
/// to ``jitc_metal_configure_scene`` and wrapped in a JIT variable; its
/// lifetime is then driven by Dr.Jit's reference counting. When all
/// external references go away, Dr.Jit invokes a destruction callback
/// which releases the per-scene Metal objects.
///
/// All fields are owned by the MetalScene — releasing the scene releases
/// the IFT, the linked MTLLibrary, and any cached compute pipelines /
/// per-entry buffers we hold strong references to.
struct MetalScene {
    /// id<MTLAccelerationStructure> (TLAS). Bound at [[buffer(1)]] for
    /// every kernel that traces against this scene. Not retained — the
    /// caller owns the TLAS lifetime.
    void *tlas = nullptr;

    /// Resources referenced by the TLAS (BLAS handles, vertex/index
    /// buffers, etc.). useResource()'d at every launch. Not retained.
    std::vector<void *> resources;

    /// Optional id<MTLLibrary> with custom intersection functions.
    /// Retained so the linker can find functions during pipeline
    /// creation.
    void *intersection_fn_library = nullptr;

    /// Per-IFT-entry MSL function names (heap-owned C strings).
    std::vector<std::string> intersection_fn_names;

    /// Per-IFT-entry data buffer (id<MTLBuffer>, may be null). NOT
    /// retained — caller manages buffer lifetime.
    std::vector<void *> intersection_fn_buffers;

    /// Per-IFT-entry MSL [[buffer(N)]] slot indices.
    std::vector<uint32_t> intersection_fn_buffer_slots;

    /// Per-IFT-entry byte offset into the buffer (added to
    /// ``setBuffer``'s offset arg). Lets two IFT entries bind the same
    /// underlying combined buffer at different starting positions —
    /// used by the multi-shape shape-group path so each child can index
    /// into its own slice without needing a per-geometry uniform.
    std::vector<uint64_t> intersection_fn_buffer_offsets;

    /// Bit 0 = triangle, bit 1 = bounding_box, bit 2 = curves. Used to
    /// specialize the MSL ``intersector<...>`` template at codegen time.
    uint32_t geometry_types_mask = 0;

    /// Lazily created intersection function tables, keyed by the MSL
    /// compute pipeline they were built for. We need separate IFT
    /// instances per pipeline because each MTLIntersectionFunctionTable
    /// is bound to function handles obtained from a specific
    /// MTLComputePipelineState.
    std::unordered_map<void *, void *> ift_cache;
};

/// Look up the ``MetalScene`` attached to a JIT variable returned by
/// ``jit_metal_configure_scene``. Returns nullptr if the variable is not
/// a scene (e.g. has been destroyed already, wrong type/kind, ...).
extern MetalScene *jitc_metal_get_scene(uint32_t scene_index);

/// Read the active scene that the current ThreadState's kernel is being
/// assembled / launched against. Returns the MetalScene determined by
/// the schedule walk in ``jitc_metal_assemble`` (or nullptr if no
/// TraceRay node was found in this kernel).
extern MetalScene *jitc_metal_active_scene();

/// Lazily build (and cache) an ``MTLIntersectionFunctionTable`` for the
/// given scene + compute pipeline. The function handles are derived from
/// the pipeline so each pipeline needs its own IFT instance. Returns
/// nullptr if the scene has no custom intersection functions configured.
/// (returns an opaque ``id<MTLIntersectionFunctionTable>``; ``pso`` is an
/// opaque ``id<MTLComputePipelineState>``)
extern void *
jitc_metal_get_or_create_ift_for_scene(MetalScene *scene, void *pso);

/// Build a MetalScene with the given configuration and wrap it in a JIT
/// variable. Returns the variable index; the caller is expected to hold
/// the reference for the scene's lifetime and dec_ref it on destruction.
extern uint32_t jitc_metal_configure_scene(void *accel, void **resources,
                                           uint32_t n_resources,
                                           void *intersection_fn_library,
                                           uint32_t n_ift_entries,
                                           const char **ift_function_names,
                                           void **ift_buffers,
                                           const uint32_t *ift_buffer_slots,
                                           const uint64_t *ift_buffer_offsets,
                                           uint32_t geometry_types_mask);

/// Trace a batch of rays against the active scene. Mirrors the signature
/// of the public ``jit_metal_ray_trace`` (see drjit-core/metal.h).
extern void jitc_metal_ray_trace(uint32_t n_args, uint32_t *args,
                                 uint32_t mask, uint32_t *out,
                                 uint32_t n_out, uint32_t scene);

/// Return the active ``id<MTLDevice>`` for the current thread.
extern void *jitc_metal_context_impl();

/// Return the active ``id<MTLCommandQueue>`` for the current thread.
extern void *jitc_metal_command_queue_impl();

#endif // defined(DRJIT_ENABLE_METAL)
