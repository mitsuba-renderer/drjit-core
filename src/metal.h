/*
    src/metal.h -- Metal backend declarations

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

#include <string>
#include <cstdint>
#include <vector>

#if defined(DRJIT_ENABLE_METAL)

struct ThreadState;
struct Kernel;
enum class MetalKernel : uint32_t;

/// Precompiled utility-kernel ``.metallib`` archive, embedded as a byte array.
extern "C" const unsigned char metal_kernels_metallib[];
extern "C" const size_t metal_kernels_metallib_size;

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

/// Resolve a Metal kernel-history entry's execution_time. The entry's task
/// slot holds an owned id<MTLCommandBuffer>: wait for it and release it.
extern float jitc_metal_finalize_kernel_history_entry(void *task);

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

/// Compile a piece of MSL source code into a Kernel object
extern bool jitc_metal_kernel_compile(const char *source, size_t source_size,
                                      const char *kernel_name, Kernel &kernel);

/// Free a previously compiled Metal kernel.
extern void jitc_metal_kernel_free(Kernel &kernel);

// ---------------------------------------------------------------------
//  Command-buffer / encoder helpers
// ---------------------------------------------------------------------

/// One buffer binding of an intersection function table. The argument table
/// of an ``MTLIntersectionFunctionTable`` is *shared by all entries* (there
/// is no per-entry binding in Metal), so bindings are specified per scene.
struct IFTBinding {
    /// MSL ``[[buffer(N)]]`` slot the data buffer binds to.
    uint32_t slot = 0;

    /// id<MTLBuffer> data buffer. Not retained — the caller manages buffer
    /// lifetime.
    void *buffer = nullptr;
};

/// Per-scene Metal ray-tracing state. One instance is allocated per call
/// to ``jitc_metal_configure_scene`` and wrapped in a JIT variable; its
/// lifetime is then driven by Dr.Jit's reference counting. When all
/// external references go away, Dr.Jit invokes a destruction callback
/// which releases the per-scene Metal objects.
///
/// All fields are owned by the MetalScene — releasing the scene releases
/// the IFT, the linked MTLLibrary, and any cached compute pipelines we
/// hold strong references to.
struct MetalScene {
    /// id<MTLAccelerationStructure> (TLAS). Reconstructed in-shader from its
    /// ``gpuResourceID`` in ``params.args[]`` and made resident via
    /// useResource() at each launch. Not retained — the caller owns the TLAS
    /// lifetime.
    void *tlas = nullptr;

    /// Resources referenced by the TLAS (BLAS handles, vertex/index
    /// buffers, etc.). useResource()'d at every launch. Not retained.
    std::vector<void *> resources;

    /// Optional id<MTLLibrary> with custom intersection functions.
    /// Retained so the linker can find functions during pipeline
    /// creation.
    void *intersection_fn_library = nullptr;

    /// MSL intersection-function names, one per IFT entry (linked into the
    /// PSO, looked up by name).
    std::vector<std::string> intersection_fns;

    /// Buffer bindings shared by all IFT entries (see IFTBinding).
    std::vector<IFTBinding> ift_bindings;

    /// Bit 0 = triangle, bit 1 = bounding_box, bit 2 = curves. Used to
    /// specialize the MSL ``intersector<...>`` template at codegen time.
    uint32_t geometry_types_mask = 0;

    /// Lazily created intersection function tables, paired with the MSL
    /// compute pipeline they were built for. We need separate IFT instances
    /// per pipeline because each MTLIntersectionFunctionTable is bound to
    /// function handles obtained from a specific MTLComputePipelineState. At
    /// most one entry per live PSO; each ``second`` is an owned (+1) IFT handle.
    std::vector<std::pair<void *, void *>> ift_cache;

    /// Invoked once after this MetalScene is destroyed, letting the application
    /// release its Metal objects then (see jit_metal_scene_set_cleanup()).
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
/// Populated during code generation (no separate pre-walk) and consumed at
/// compile time for PSO intersection-function linking (metal_core.mm).
extern std::vector<MetalScene *> metal_kernel_scenes;

/// Append ``scene`` to ``metal_kernel_scenes`` if not already present (linear
/// dedup; at most a handful of scenes per kernel). ``nullptr`` is ignored.
extern void metal_register_kernel_scene(MetalScene *scene);

/// Hashes of this kernel's indirect-callable functions (named ``func_<hash>``),
/// ordered by callable index. Populated during code generation and consumed to
/// build the kernel's visible function table.
extern std::vector<XXH128_hash_t> metal_kernel_callables;

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
                                           uint32_t n_ift_buffers,
                                           void **ift_buffers,
                                           const uint32_t *ift_buffer_slots,
                                           uint32_t geometry_types_mask);

/// Trace a batch of rays against the active scene. Mirrors the signature
/// of the public ``jit_metal_ray_trace`` (see drjit-core/metal.h).
extern void jitc_metal_ray_trace(uint32_t n_args, uint32_t *args,
                                 uint32_t mask, uint32_t *out,
                                 uint32_t n_out, uint32_t scene, int shadow);

#endif // defined(DRJIT_ENABLE_METAL)
