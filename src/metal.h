/*
    src/metal.h -- Metal backend declarations

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_METAL)

#include <cstdint>
#include <vector>

struct ThreadState;
struct Kernel;
struct ScheduledGroup;
struct MetalDevice;
struct KernelHistoryEntry;

/// Initialize the Metal backend. Enumerates the available Apple Silicon GPUs
/// (Metal 3 is required, i.e. M1+) and registers them in ``state.metal_devices``.
extern bool jitc_metal_init();

/// Release all resources held by the Metal backend.
extern void jitc_metal_shutdown();

/// Compile a piece of MSL source code into a compute pipeline state.
/// Returns the (opaque) ``MTL::ComputePipelineState*`` together with the
/// associated library, both stored within the kernel object.
///
/// The kernel name (entry-point function) is taken from ``kernel_name`` (set
/// by jitc_assemble) so the same convention as PTX/LLVM is reused.
extern bool jitc_metal_compile(const char *source, size_t source_size,
                               const char *kernel_name, Kernel &kernel);

/// Free a previously compiled Metal kernel.
extern void jitc_metal_free(Kernel &kernel);

/// Wait for all Metal work submitted on the current thread to complete.
extern void jitc_metal_sync(ThreadState *ts);

/// Pretty-print Metal device information at startup (debug helper)
extern void jitc_metal_dump_devices();

/// Return the human-readable name of the GPU
extern const char *jitc_metal_device_name(int device_id);

/// Stash a buffer pointer ↔ MTL::Buffer* mapping for later retrieval. Metal
/// kernels need access to the underlying ``MTLBuffer`` (rather than the raw
/// pointer) for ``useResource()`` calls, ``gpuAddress()``, etc.
extern void jitc_metal_register_buffer(void *ptr, void *mtl_buffer);

/// Look up the ``MTLBuffer*`` for a pointer that may lie at an offset
/// from a registered allocation's base. Returns the ``MTL::Buffer*`` and,
/// optionally, the byte offset from its start. Returns nullptr if no
/// registered allocation contains ``ptr``.
extern void *jitc_metal_find_buffer(void *ptr, size_t *offset_out);

/// Forget a previously registered pointer ↔ buffer mapping, returning
/// the ``MTL::Buffer*`` that was associated with it (or nullptr if the
/// pointer was not registered).
extern void *jitc_metal_unregister_buffer(void *ptr);

/// Retrieve a precompiled compute pipeline state from the utility kernel
/// library by kernel name (e.g. "block_reduce_add_f32_1024"). Returns
/// nullptr if the kernel is not found. Pipeline states are cached.
namespace MTL { class ComputePipelineState; class IntersectionFunctionTable; }
extern MTL::ComputePipelineState *
jitc_metal_get_pipeline(int device_id, const char *name);

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

/// Per-scene Metal ray-tracing state. One instance is allocated per call
/// to ``jitc_metal_configure_scene`` and wrapped in a JIT variable; its
/// lifetime is then driven by Dr.Jit's reference counting. When all
/// external references go away, Dr.Jit invokes a destruction callback
/// which releases the per-scene Metal objects.
///
/// All fields are owned by the MetalScene — releasing the scene releases
/// the IFT, the linked MTL::Library, and any cached compute pipelines /
/// per-entry buffers we hold strong references to.
struct MetalScene {
    /// MTL::AccelerationStructure* (TLAS). Bound at [[buffer(1)]] for
    /// every kernel that traces against this scene. Not retained — the
    /// caller owns the TLAS lifetime.
    void *tlas = nullptr;

    /// Resources referenced by the TLAS (BLAS handles, vertex/index
    /// buffers, etc.). useResource()'d at every launch. Not retained.
    std::vector<void *> resources;

    /// Optional MTL::Library* with custom intersection functions.
    /// Retained so the linker can find functions during pipeline
    /// creation.
    void *intersection_fn_library = nullptr;

    /// Per-IFT-entry MSL function names (heap-owned C strings).
    std::vector<std::string> intersection_fn_names;

    /// Per-IFT-entry data buffer (MTL::Buffer*, may be null). NOT
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
extern MTL::IntersectionFunctionTable *
jitc_metal_get_or_create_ift_for_scene(MetalScene *scene,
                                       MTL::ComputePipelineState *pso);

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

/// Return the active ``MTL::Device*`` for the current thread.
extern void *jitc_metal_context_impl();

/// Return the active ``MTL::CommandQueue*`` for the current thread.
extern void *jitc_metal_command_queue_impl();

/// Choose the per-target expansion factor for ``ReduceMode::Expand`` on
/// Metal. The scatter target gets allocated as ``factor`` consecutive
/// copies of ``size`` elements, and each GPU thread routes its update to
/// copy ``thread_position_in_grid % factor``. This trades a bounded
/// amount of extra memory (``factor * size * tsize`` bytes) for a roughly
/// ``factor``-fold reduction in atomic contention — which is what fixes
/// FP32 atomic_fetch_add precision loss in PRB backward.
///
/// Returns a power of 2 in [1, 1024]; 1 means "do not expand" (size×tsize
/// already exceeds the per-target memory budget). Must be deterministic
/// from (size, tsize): both ``jitc_var_expand`` and
/// ``jitc_var_reduce_expanded`` rely on agreeing on the same factor.
extern uint32_t jitc_metal_expand_factor(uint32_t size, uint32_t tsize);

/// Soft cap on the size of an array that ``ReduceMode::Auto`` will
/// promote to ``ReduceMode::Expand`` on Metal (mirrors
/// ``llvm_expand_threshold``).
extern size_t metal_expand_threshold;

#endif // defined(DRJIT_ENABLE_METAL)
