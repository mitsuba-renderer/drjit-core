/*
    src/metal_core.mm -- Metal device init, shutdown, compilation, and
    encoder management.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_METAL)

#include "metal.h"
#include "metal_ts.h"
#include "internal.h"
#include "malloc.h"
#include "log.h"
#include "io.h"
#include "var.h"
#include "trace.h"
#include "record_ts.h"
#include "drjit-core/metal.h"

// Suppress the obsolete Carbon <CarbonCore/Threads.h>, whose ThreadState collides with Dr.Jit
#define __THREADS__
#import <Metal/Metal.h>

#include <unordered_set>
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

// ============================================================================
// Buffer API
// ============================================================================
//
void *metal_buffer_new(void *device, size_t size, bool shared, void **ptr_out) {
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) device;
        MTLResourceOptions opts = shared ? MTLResourceStorageModeShared
                                         : MTLResourceStorageModePrivate;
        id<MTLBuffer> buf = [dev newBufferWithLength:size options:opts];
        *ptr_out = shared ? [buf contents]
                          : (void *) (uintptr_t) [buf gpuAddress];
        return (__bridge_retained void *) buf;
    }
}

void metal_buffer_free(void *buffer) {
    @autoreleasepool {
        (void) (__bridge_transfer id<MTLBuffer>) buffer; // release the +1
    }
}

void jitc_metal_cmdbuf_free_on_complete(void *cmdbuf, uint64_t info,
                                        void *ptr) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) cmdbuf;
        [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            jitc_malloc_release(info, ptr);
        }];
    }
}

// Lazily-sorted flat vector of (base address, id<MTLBuffer>, length) entries.
// Protected by ``state.lock``.
struct BufferEntry {
    uintptr_t base;
    void     *buf;    // id<MTLBuffer>
    size_t    length;
};

static std::vector<BufferEntry> metal_buffer_map;
static bool metal_buffer_map_sorted = true;

static void jitc_metal_ensure_sorted() {
    if (likely(metal_buffer_map_sorted))
        return;
    metal_buffer_map.erase(
        std::remove_if(metal_buffer_map.begin(), metal_buffer_map.end(),
                       [](const BufferEntry &e) { return e.buf == nullptr; }),
        metal_buffer_map.end());
    std::sort(metal_buffer_map.begin(), metal_buffer_map.end(),
              [](const BufferEntry &a, const BufferEntry &b) {
                  return a.base < b.base;
              });
    metal_buffer_map_sorted = true;
}

void jitc_metal_register_buffer(void *ptr, void *metal_buffer, size_t size) {
    metal_buffer_map.push_back({ (uintptr_t) ptr, metal_buffer, size });
    metal_buffer_map_sorted = false;
}

/// Look up a buffer that *contains* the given pointer and return the offset
void *jitc_metal_find_buffer(void *ptr, size_t *offset_out) {
    jitc_metal_ensure_sorted();

    uintptr_t addr = (uintptr_t) ptr;
    auto it = std::upper_bound(
        metal_buffer_map.begin(), metal_buffer_map.end(), addr,
        [](uintptr_t a, const BufferEntry &b) { return a < b.base; });

    if (it != metal_buffer_map.begin()) {
        --it;
        if (addr < it->base + it->length) {
            *offset_out = (size_t) (addr - it->base);
            return it->buf;
        }
    }

    *offset_out = 0;
    return nullptr;
}

void *jitc_metal_unregister_buffer(void *ptr) {
    jitc_metal_ensure_sorted();

    uintptr_t addr = (uintptr_t) ptr;
    auto it = std::lower_bound(
        metal_buffer_map.begin(), metal_buffer_map.end(), addr,
        [](const BufferEntry &e, uintptr_t a) { return e.base < a; });
    if (it == metal_buffer_map.end() || it->base != addr)
        return nullptr;
    void *buf = it->buf;
    // Turn the entry into a tombstone to postpone cleanup work
    it->buf = nullptr;
    return buf;
}

// ============================================================================
//  Utility kernel library
//
//  resources/metal_kernels.metal is compiled to a .metallib at build time (see
//  resources/embed_metal_kernels.cmake) and embedded as a byte array. At init time
//  each device instantiates the library from those bytes and eagerly creates a
//  compute pipeline state for every utility kernel, stored in MetalDevice and
//  indexed directly by MetalKernel (see state.metal_devices[...].pipelines).
// ============================================================================

/// Kernel function names, indexed by MetalKernel. The order must match the
/// MetalKernel enum in internal.h.
static const char *metal_kernel_names[(uint32_t) MetalKernel::Count] = {
    "reduce_all_init",
    "reduce_any_init",
    "reduce_all",
    "reduce_any",
    "compress_scatter",
    "mkperm_phase_1",
    "mkperm_phase_3",
    "mkperm_detect_offsets",
    "aggregate_kernel",
    "memset_u16",
    "memset_u32",
    "memset_u64",
    "convert_f32_f16"
};

/// Create a compute pipeline state for ``name`` from ``lib``. Returns an owned
/// (+1) handle, or nullptr if the function or pipeline could not be created.
static void *jitc_metal_create_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                        const char *name) {
    id<MTLFunction> func = [lib newFunctionWithName:@(name)];
    if (!func)
        return nullptr;
    NSError *err = nil;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&err];
    if (!pso) {
        const char *desc = err ? err.localizedDescription.UTF8String
                               : "<unknown>";
        jitc_log(Warn, "jitc_metal_create_pipeline(%s): pipeline creation "
                       "failed: %s",
                 name, desc);
        return nullptr;
    }
    return (__bridge_retained void *) pso;
}

// ============================================================================
//  Backend init / shutdown
// ============================================================================

bool jitc_metal_init() {
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (!devices || devices.count == 0) {
            jitc_log(Warn, "jit_metal_init(): no Metal-capable GPU was detected.");
            return false;
        }

        state.metal_devices.clear();

        for (id<MTLDevice> dev in devices) {
            if (![dev supportsFamily:MTLGPUFamilyMetal3]) {
                jitc_log(Warn,
                         "jit_metal_init(): skipping device \"%s\" because it "
                         "does not support Metal 3 (M1+ required).",
                         dev.name.UTF8String);
                continue;
            }

            MetalDevice md;
            md.device      = (__bridge_retained void *) dev;
            md.queue       = (__bridge_retained void *) [dev newCommandQueue];
            md.max_threads_per_threadgroup =
                (uint32_t) [dev maxThreadsPerThreadgroup].width;
            md.supports_ray_tracing = [dev supportsRaytracing];
            const char *name = dev.name.UTF8String;
            size_t len = std::strlen(name);
            md.name = (char *) std::malloc(len + 1);
            std::memcpy(md.name, name, len + 1);

            // Instantiate the precompiled utility kernel library
            NSError *err = nil;
            dispatch_data_t lib_data = dispatch_data_create(
                metal_kernels_metallib, metal_kernels_metallib_size, nullptr,
                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            id<MTLLibrary> lib = [dev newLibraryWithData:lib_data error:&err];
            if (!lib)
                jitc_fail("jit_metal_init(): could not instantiate the utility "
                          "kernel library for device \"%s\": %s",
                          md.name, err ? err.localizedDescription.UTF8String
                                       : "<unknown>");
            md.utility_lib = (__bridge_retained void *) lib;

            for (uint32_t i = 0; i < (uint32_t) MetalKernel::Count; ++i) {
                md.pipelines[i] =
                    jitc_metal_create_pipeline(dev, lib, metal_kernel_names[i]);
                if (!md.pipelines[i])
                    jitc_fail("jit_metal_init(): could not create pipeline "
                              "state \"%s\" for device \"%s\".",
                              metal_kernel_names[i], md.name);
            }

            // Query the SIMD execution width from a representative pipeline;
            // threadExecutionWidth is a pipeline property, not a device one.
            md.simd_width = (uint32_t)
                ((__bridge id<MTLComputePipelineState>)
                     md.pipelines[(uint32_t) MetalKernel::Aggregate])
                    .threadExecutionWidth;

            jitc_log(Info,
                     "jit_metal_init(): registered device \"%s\" "
                     "(simd=%u, max_threads=%u, rt=%s)",
                     md.name, md.simd_width, md.max_threads_per_threadgroup,
                     md.supports_ray_tracing ? "yes" : "no");

            state.metal_devices.push_back(md);
        }

        return !state.metal_devices.empty();
    }
}

void jitc_metal_shutdown() {
    @autoreleasepool {
        for (MetalDevice &d : state.metal_devices) {
            for (void *&pso : d.pipelines) {
                if (pso)
                    (void) (__bridge_transfer id<MTLComputePipelineState>) pso;
                pso = nullptr;
            }
            if (d.queue)
                (void) (__bridge_transfer id<MTLCommandQueue>) d.queue;
            if (d.utility_lib)
                (void) (__bridge_transfer id<MTLLibrary>) d.utility_lib;
            if (d.device)
                (void) (__bridge_transfer id<MTLDevice>) d.device;
            std::free(d.name);
        }
        state.metal_devices.clear();
        metal_buffer_map.clear();
    }
}

// ============================================================================
//  Kernel compilation
// ============================================================================

bool jitc_metal_kernel_compile(const char *source, size_t /*source_size*/,
                               const char *kernel_name, Kernel &kernel) {
    @autoreleasepool {
        if (state.metal_devices.empty())
            jitc_fail("jitc_metal_kernel_compile(): no Metal devices initialized.");

        // Compile against the device of the calling thread
        auto *ts = thread_state(JitBackend::Metal);
        id<MTLDevice> dev = (__bridge id<MTLDevice>) ts->metal_device;

        NSError *err = nil;
        NSString *src = @(source);

        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion3_2;

        // The relaxed/fast math modes are a little aggressive and break the Dr.Jit
        // test suite. We opt in on a per instruction basis by calling math functions
        // from the ``fast::`` namespace

        opts.mathMode = MTLMathModeSafe;
        opts.libraryType = MTLLibraryTypeExecutable;

        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];

        if (!lib) {
            const char *desc = err ? err.localizedDescription.UTF8String
                                   : "<unknown>";
            jitc_fail("jitc_metal_kernel_compile(): MSL compilation failed:\n%s\n\n"
                      "--- Source code ---\n%s",
                      desc, source);
        }

        id<MTLFunction> func = [lib newFunctionWithName:@(kernel_name)];
        if (!func)
            jitc_fail("jitc_metal_kernel_compile(): kernel function \"%s\" not found in "
                      "the compiled library.", kernel_name);

        // ---- Pipeline creation ---------------------------------------------
        // Link the union of custom intersection functions across every scene
        // registered with this kernel (collected by ``jitc_metal_assemble``'s
        // pre-walk into ``metal_kernel_scenes``). LinkedFunctions applies to
        // the entire PSO, not per-IFT, so scenes with disjoint function name
        // sets must all contribute. Per-scene IFTs are built lazily at launch
        // time in ``jitc_metal_get_or_create_ift_for_scene``.
        err = nil;
        id<MTLComputePipelineState> pso = nil;

        NSMutableArray<id<MTLFunction>> *linked_fns = [NSMutableArray array];
        std::vector<std::string> seen;
        for (MetalScene *scene : metal_kernel_scenes) {
            id<MTLLibrary> isect_lib =
                scene ? (__bridge id<MTLLibrary>) scene->intersection_fn_library
                      : nil;
            if (!isect_lib)
                continue;
            for (const std::string &name : scene->intersection_fn_names) {
                if (std::find(seen.begin(), seen.end(), name) != seen.end())
                    continue;
                seen.push_back(name);
                id<MTLFunction> f = [isect_lib newFunctionWithName:@(name.c_str())];
                if (!f)
                    jitc_fail("jitc_metal_kernel_compile(): intersection function "
                              "\"%s\" not found in user-supplied library.",
                              name.c_str());
                [linked_fns addObject:f];
            }
        }

        if (linked_fns.count > 0) {
            MTLLinkedFunctions *lf = [MTLLinkedFunctions new];
            lf.functions = linked_fns;

            MTLComputePipelineDescriptor *desc =
                [MTLComputePipelineDescriptor new];
            desc.computeFunction = func;
            desc.linkedFunctions = lf;

            pso = [dev newComputePipelineStateWithDescriptor:desc
                                                     options:MTLPipelineOptionNone
                                                  reflection:nil
                                                       error:&err];
        } else {
            pso = [dev newComputePipelineStateWithFunction:func error:&err];
        }

        if (!pso) {
            const char *desc = err ? err.localizedDescription.UTF8String
                                   : "<unknown>";
            jitc_fail("jitc_metal_kernel_compile(): pipeline creation failed: %s", desc);
        }

        kernel.metal.pipeline    = (__bridge_retained void *) pso;
        kernel.metal.library     = (__bridge_retained void *) lib;
        kernel.metal.scenes      = nullptr;
        kernel.metal.scene_count = 0;
        kernel.size = (uint32_t) std::strlen(source);
        return false;
    }
}

void jitc_metal_kernel_free(Kernel &kernel) {
    @autoreleasepool {
        if (kernel.metal.pipeline)
            (void) (__bridge_transfer id<MTLComputePipelineState>)
                kernel.metal.pipeline;
        if (kernel.metal.library)
            (void) (__bridge_transfer id<MTLLibrary>) kernel.metal.library;
        delete[] kernel.metal.scenes;
        kernel.metal.pipeline    = nullptr;
        kernel.metal.library     = nullptr;
        kernel.metal.scenes      = nullptr;
        kernel.metal.scene_count = 0;
    }
}

/// Resolve a Metal kernel-history entry's execution_time
float jitc_metal_finalize_kernel_history_entry(void *task_ptr) {
    @autoreleasepool {
        if (!task_ptr)
            return 0.f;
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>) task_ptr;
        [cb waitUntilCompleted];
        return (float) ((cb.GPUEndTime - cb.GPUStartTime) * 1000);
    }
}

/// Flush the thread's pending command buffer and wait for the GPU to finish
void jitc_metal_sync(ThreadState *ts) {
    // Unwrap recording/disabling wrappers (used by frozen functions)
    ((MetalThreadState *) ts->actual_state())->flush(/* wait = */ true);
}

// ============================================================================
//  Ray Tracing API
// ============================================================================

uint32_t jitc_metal_configure_scene(void *accel, void **resources,
                                    uint32_t n_resources,
                                    void *intersection_fn_library,
                                    uint32_t n_ift_entries,
                                    const char **ift_function_names,
                                    void **ift_buffers,
                                    const uint32_t *ift_buffer_slots,
                                    const uint64_t *ift_buffer_offsets,
                                    uint32_t geometry_types_mask) {
    jitc_log(InfoSym,
             "jitc_metal_configure_scene(accel=" DRJIT_PTR ", n_resources=%u, "
             "ift_lib=" DRJIT_PTR ", n_ift=%u, geom_mask=%u)",
             (uintptr_t) accel, n_resources,
             (uintptr_t) intersection_fn_library,
             n_ift_entries, geometry_types_mask);

    auto *scene = new MetalScene();
    scene->tlas = accel;
    scene->geometry_types_mask = geometry_types_mask;

    if (resources && n_resources > 0)
        scene->resources.assign(resources, resources + n_resources);

    if (intersection_fn_library) {
        // Retain a reference for the scene's lifetime.
        id<MTLLibrary> isect_lib =
            (__bridge id<MTLLibrary>) intersection_fn_library;
        scene->intersection_fn_library = (__bridge_retained void *) isect_lib;
    }

    if (n_ift_entries > 0) {
        scene->intersection_fn_names.reserve(n_ift_entries);
        for (uint32_t i = 0; i < n_ift_entries; ++i)
            scene->intersection_fn_names.emplace_back(ift_function_names[i]);

        if (ift_buffers) {
            scene->intersection_fn_buffers.assign(
                ift_buffers, ift_buffers + n_ift_entries);
        } else {
            scene->intersection_fn_buffers.assign(n_ift_entries, nullptr);
        }

        if (ift_buffer_slots) {
            scene->intersection_fn_buffer_slots.assign(
                ift_buffer_slots, ift_buffer_slots + n_ift_entries);
        } else {
            scene->intersection_fn_buffer_slots.assign(n_ift_entries, 0u);
        }

        if (ift_buffer_offsets) {
            scene->intersection_fn_buffer_offsets.assign(
                ift_buffer_offsets, ift_buffer_offsets + n_ift_entries);
        } else {
            scene->intersection_fn_buffer_offsets.assign(n_ift_entries, 0ull);
        }
    }

    uint32_t index =
        jitc_var_new_node_0(JitBackend::Metal, VarKind::Nop,
                            VarType::Void, 1, 0, (uintptr_t) scene);

    auto callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (!free)
            return;
        auto *s = (MetalScene *) ptr;
        jitc_log(InfoSym, "jit_metal_configure_scene(): freeing MetalScene "
                          "(ift_lib=" DRJIT_PTR ", n_ift=%zu, n_pso_cached=%zu)",
                 (uintptr_t) s->intersection_fn_library,
                 s->intersection_fn_names.size(),
                 s->ift_cache.size());
        for (auto &kv : s->ift_cache) {
            if (kv.second)
                (void) (__bridge_transfer id<MTLIntersectionFunctionTable>)
                    kv.second;
        }
        if (s->intersection_fn_library)
            (void) (__bridge_transfer id<MTLLibrary>) s->intersection_fn_library;
        jitc_metal_unregister_live_scene(s);
        delete s;
    };

    jitc_var_set_callback(index, callback, scene, true);
    jitc_metal_register_live_scene(scene);

    return index;
}

// ============================================================================
//  Live MetalScene registry
// ============================================================================
namespace {
    std::mutex g_live_scenes_mutex;
    std::unordered_set<void *> g_live_scenes;
    void *g_last_live_scene = nullptr;
}

void jitc_metal_register_live_scene(void *scene) {
    std::lock_guard<std::mutex> g(g_live_scenes_mutex);
    g_live_scenes.insert(scene);
    g_last_live_scene = scene;
}

void jitc_metal_unregister_live_scene(void *scene) {
    std::lock_guard<std::mutex> g(g_live_scenes_mutex);
    g_live_scenes.erase(scene);
    if (g_last_live_scene == scene) {
        g_last_live_scene =
            g_live_scenes.empty() ? nullptr : *g_live_scenes.begin();
    }
}

bool jitc_metal_is_live_scene(void *scene) {
    if (!scene) return false;
    std::lock_guard<std::mutex> g(g_live_scenes_mutex);
    return g_live_scenes.count(scene) != 0;
}

void *jitc_metal_last_live_scene() {
    std::lock_guard<std::mutex> g(g_live_scenes_mutex);
    return g_last_live_scene;
}

MetalScene *jitc_metal_get_scene(uint32_t scene_index) {
    if (!scene_index)
        return nullptr;
    Variable *v = jitc_var(scene_index);
    if (!v || (VarKind) v->kind != VarKind::Nop ||
        (VarType) v->type != VarType::Void)
        return nullptr;
    return (MetalScene *) v->literal;
}

void jit_metal_invalidate_scene_tlas(uint32_t scene_index) {
    MetalScene *scene = jitc_metal_get_scene(scene_index);
    if (scene)
        scene->tlas = nullptr;
}

MetalScene *jitc_metal_active_scene() {
    auto *ts = thread_state(JitBackend::Metal);
    return (MetalScene *) ts->metal_active_scene;
}

/// Lazily build (and cache) an IntersectionFunctionTable for the given scene
/// + compute pipeline. The function handles are derived from the pipeline so
/// each pipeline needs its own IFT instance. The cache owns the (+1) IFT; the
/// returned pointer is borrowed.
void *
jitc_metal_get_or_create_ift_for_scene(MetalScene *scene, void *pso_) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>) pso_;
    if (!scene || !pso || scene->intersection_fn_names.empty())
        return nullptr;

    // Cache hit: we already built an IFT for this pipeline.
    auto it = scene->ift_cache.find(pso_);
    if (it != scene->ift_cache.end())
        return it->second;

    id<MTLLibrary> isect_lib =
        (__bridge id<MTLLibrary>) scene->intersection_fn_library;
    if (!isect_lib)
        return nullptr;

    uint32_t n_ift = (uint32_t) scene->intersection_fn_names.size();

    // Resolve unique function objects (deduplicate so we look up each function
    // only once even if the IFT references it multiple times).
    NSMutableDictionary<NSString *, id<MTLFunction>> *unique_fns =
        [NSMutableDictionary dictionary];
    for (uint32_t i = 0; i < n_ift; ++i) {
        NSString *name = @(scene->intersection_fn_names[i].c_str());
        if (unique_fns[name])
            continue;
        id<MTLFunction> f = [isect_lib newFunctionWithName:name];
        if (!f)
            jitc_fail("jitc_metal_get_or_create_ift_for_scene(): intersection "
                      "function \"%s\" not found in user-supplied library.",
                      scene->intersection_fn_names[i].c_str());
        unique_fns[name] = f;
    }

    MTLIntersectionFunctionTableDescriptor *iftd =
        [MTLIntersectionFunctionTableDescriptor new];
    iftd.functionCount = n_ift;
    id<MTLIntersectionFunctionTable> ift =
        [pso newIntersectionFunctionTableWithDescriptor:iftd];

    for (uint32_t i = 0; i < n_ift; ++i) {
        NSString *name = @(scene->intersection_fn_names[i].c_str());
        id<MTLFunction> f = unique_fns[name];
        id<MTLFunctionHandle> handle = [pso functionHandleWithFunction:f];
        [ift setFunction:handle atIndex:i];

        if (i < scene->intersection_fn_buffers.size()) {
            id<MTLBuffer> buf =
                (__bridge id<MTLBuffer>) scene->intersection_fn_buffers[i];
            uint32_t slot = (i < scene->intersection_fn_buffer_slots.size())
                                ? scene->intersection_fn_buffer_slots[i] : 0u;
            uint64_t offset =
                (i < scene->intersection_fn_buffer_offsets.size())
                    ? scene->intersection_fn_buffer_offsets[i] : 0ull;
            if (buf)
                [ift setBuffer:buf offset:offset atIndex:slot];
        }
    }

    scene->ift_cache[pso_] = (__bridge_retained void *) ift;
    return scene->ift_cache[pso_];
}

void *jitc_metal_context_impl() {
    ThreadState *ts = thread_state(JitBackend::Metal);
    return ts->metal_device;
}

void *jitc_metal_command_queue_impl() {
    ThreadState *ts = thread_state(JitBackend::Metal);
    return ts->metal_queue;
}

void jitc_metal_ray_trace(uint32_t n_args, uint32_t *args,
                          uint32_t mask, uint32_t *out,
                          uint32_t n_out, uint32_t scene) {
    if (n_args != 8)
        jitc_raise("jit_metal_ray_trace(): expected 8 ray arguments, got %u.",
                   n_args);
    if (n_out != 7)
        jitc_raise("jit_metal_ray_trace(): expected 7 outputs, got %u.",
                   n_out);
    if (!scene)
        jitc_raise("jit_metal_ray_trace(): a valid scene_index "
                   "(returned by jit_metal_configure_scene) is required.");

    if (jitc_var_type(scene) != VarType::Void)
        jitc_raise("jit_metal_ray_trace(): type mismatch for scene argument!");

    uint32_t size = 0;
    bool symbolic = false;
    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *vi = jitc_var(args[i]);
        size = std::max(size, vi->size);
        symbolic |= (bool) vi->symbolic;
    }
    {
        const Variable *vm = jitc_var(mask);
        size = std::max(size, vm->size);
        symbolic |= (bool) vm->symbolic;
    }
    if (size == 0)
        size = 1;

    // Apply mask stack
    Ref valid = steal(jitc_var_mask_apply(mask, size));

    // Build TraceData with ray parameter indices
    TraceData *td = new TraceData();
    td->indices.reserve(n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        td->indices.push_back(args[i]);
        jitc_var_inc_ref(args[i]);
    }

    // Create TraceRay node with valid mask as dep[0] and scene as dep[1].
    Ref trace = steal(jitc_var_new_node_2(
        JitBackend::Metal, VarKind::TraceRay, VarType::Void, size,
        symbolic, valid, jitc_var(valid),
        scene, jitc_var(scene), (uintptr_t) td));

    // Register cleanup callback
    jitc_var_set_callback(
        trace,
        [](uint32_t, int free, void *ptr) {
            if (free)
                delete (TraceData *) ptr;
        },
        td, true);

    // Create Extract children for each output
    VarType out_types[7] = {
        VarType::Bool,    // valid
        VarType::Float32, // distance
        VarType::Float32, // bary_u
        VarType::Float32, // bary_v
        VarType::UInt32,  // instance_id
        VarType::UInt32,  // primitive_id
        VarType::UInt32   // geometry_id
    };

    for (uint32_t i = 0; i < 7; ++i) {
        out[i] = jitc_var_new_node_1(
            JitBackend::Metal, VarKind::Extract, out_types[i],
            size, symbolic, trace, jitc_var(trace), (uint64_t) i);
    }
}

#endif // defined(DRJIT_ENABLE_METAL)
