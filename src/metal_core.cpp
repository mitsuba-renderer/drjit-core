/*
    src/metal_core.cpp -- Metal device init, shutdown, compilation, and
    encoder management.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_METAL)

#include "metal.h"
#include "metal_api.h"
#include "metal_eval.h"
#include "metal_stage_buffer.h"
#include "internal.h"
#include "log.h"
#include "io.h"
#include "var.h"
#include "trace.h"
#include "record_ts.h"
#include "drjit-core/metal.h"  // public extern "C" declarations
#include "metal_kernels_src.h" // generated: drjit::metal_kernels_src

#include <Metal/Metal.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <atomic>
#include <algorithm>

// ============================================================================
//  Expand-mode tunables (mirror of ``llvm_expand_threshold`` for Metal)
// ============================================================================

// Soft cap on the size of an array that ``ReduceMode::Auto`` will promote to
// ``ReduceMode::Expand`` on Metal. ``jitc_var_infer_reduce_mode`` consults
// this when deciding whether to expand. Tuned to 1M entries (matches the
// LLVM default), since past this point each output already gets so many
// distinct atomics that contention isn't the dominant concern.
size_t metal_expand_threshold = 1024 * 1024;

uint32_t jitc_metal_expand_factor(uint32_t size, uint32_t tsize) {
    if (size == 0 || tsize == 0)
        return 1;

    // Cap the per-call Expand temp buffer at 4 MB / factor 64.
    //
    // The previous cap (64 MB / factor 1024) consistently produced ~40 ms
    // stalls per scatter on Metal at film sizes where it kicked in: the
    // reduce_expanded kernel's wall time is proportional to the temp
    // buffer size, and Apple's GPU + memory pipeline takes roughly 1 us
    // per element regardless of how parallel the kernel looks. With the
    // smaller cap each Expand call costs ~3-4 ms instead, while still
    // providing enough per-output expansion to avoid ULP loss in the
    // typical pixel-accumulation pattern (a few rays per pixel).
    //
    // For workloads with much higher per-output contention (e.g. 64+ spp
    // path tracing) callers can override by explicitly passing
    // ``ReduceMode::Expand`` with custom expand_factor.
    constexpr size_t MAX_BUFFER_BYTES = 4ull * 1024 * 1024;
    constexpr uint32_t MAX_FACTOR     = 64;

    size_t budget = MAX_BUFFER_BYTES / ((size_t) size * tsize);
    uint32_t factor = (uint32_t) std::min<size_t>(budget, MAX_FACTOR);
    if (factor < 2)
        return 1;

    // Round down to power of 2 so the slot computation in
    // ``jitc_var_infer_reduce_mode`` can use a bitwise AND instead of a
    // modulo.
    uint32_t pow2 = 1;
    while (pow2 * 2u <= factor)
        pow2 *= 2u;
    return pow2;
}

// ============================================================================
//  Pointer ↔ MTL::Buffer mapping
//
//  Dr.Jit treats device memory as raw void* but the Metal API requires
//  ``MTLBuffer`` references (for ``setBuffer``, ``useResource``, ``gpuAddress``,
//  etc.). We maintain a simple thread-safe hash map.
// ============================================================================

// Lazily-sorted flat vector of (base_addr, MTLBuffer*) entries.
using BufferEntry = std::pair<uintptr_t, void *>;
static std::vector<BufferEntry> metal_buffer_map;
static bool metal_buffer_map_sorted = true;

static void jitc_metal_ensure_sorted() {
    if (likely(metal_buffer_map_sorted))
        return;
    metal_buffer_map.erase(
        std::remove_if(metal_buffer_map.begin(), metal_buffer_map.end(),
                       [](const BufferEntry &e) { return e.second == nullptr; }),
        metal_buffer_map.end());
    std::sort(metal_buffer_map.begin(), metal_buffer_map.end());
    metal_buffer_map_sorted = true;
}

void jitc_metal_register_buffer(void *ptr, void *mtl_buffer) {
    metal_buffer_map.emplace_back((uintptr_t) ptr, mtl_buffer);
    metal_buffer_map_sorted = false;
}

/// Look up a buffer that *contains* the given pointer and return the offset
void *jitc_metal_find_buffer(void *ptr, size_t *offset_out) {
    jitc_metal_ensure_sorted();

    uintptr_t addr = (uintptr_t) ptr;
    auto it = std::upper_bound(
        metal_buffer_map.begin(), metal_buffer_map.end(), addr,
        [](uintptr_t a, const BufferEntry &b) { return a < b.first; });

    if (it != metal_buffer_map.begin()) {
        --it;
        uintptr_t base = it->first;
        MTL::Buffer *buf = (MTL::Buffer *) it->second;
        if (addr < base + buf->length()) {
            *offset_out = (size_t) (addr - base);
            return buf;
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
        [](const BufferEntry &e, uintptr_t a) { return e.first < a; });
    if (it == metal_buffer_map.end() || it->first != addr)
        return nullptr;
    void *buf = it->second;
    it->second = nullptr;
    return buf;
}

// ============================================================================
//  Encoder lifecycle helpers
//
//  At any given time a Metal command buffer hosts at most one open encoder.
//  Switching encoder type (compute → blit, blit → accel, ...) requires
//  closing the previous encoder and opening a fresh one.  We track which
//  type is currently open via a small enum stored alongside the command
//  buffer pointer in the ThreadState.
// ============================================================================

enum class MetalEncoderKind : uint32_t {
    None = 0,
    Compute,
    Blit,
    Acceleration
};

namespace {
    // Per-thread "currently active encoder" pointer + kind. We co-locate
    // these in a tiny struct rather than adding two new fields to
    // ``ThreadStateBase`` (which would require widening the file change set
    // beyond what the plan envisages).
    struct ActiveEncoder {
        void *encoder = nullptr;
        MetalEncoderKind kind = MetalEncoderKind::None;
    };
    thread_local ActiveEncoder g_active_encoder;
}

MTL::CommandBuffer *jitc_metal_acquire_cmdbuf(ThreadState *ts) {
    if (ts->metal_command_buffer)
        return (MTL::CommandBuffer *) ts->metal_command_buffer;

    auto *queue = (MTL::CommandQueue *) ts->metal_queue;
    auto *cb = queue->commandBuffer();
    cb->retain();
    ts->metal_command_buffer = cb;
    return cb;
}

void jitc_metal_close_encoder(ThreadState *) {
    if (g_active_encoder.kind == MetalEncoderKind::Compute) {
        auto *enc =
            (MTL::ComputeCommandEncoder *) g_active_encoder.encoder;
        enc->endEncoding();
        enc->release();
    } else if (g_active_encoder.kind == MetalEncoderKind::Blit) {
        auto *enc = (MTL::BlitCommandEncoder *) g_active_encoder.encoder;
        enc->endEncoding();
        enc->release();
    } else if (g_active_encoder.kind == MetalEncoderKind::Acceleration) {
        auto *enc = (MTL::AccelerationStructureCommandEncoder *)
            g_active_encoder.encoder;
        enc->endEncoding();
        enc->release();
    }
    g_active_encoder.encoder = nullptr;
    g_active_encoder.kind = MetalEncoderKind::None;
}

MTL::ComputeCommandEncoder *
jitc_metal_acquire_compute_encoder(ThreadState *ts) {
    if (g_active_encoder.kind == MetalEncoderKind::Compute)
        return (MTL::ComputeCommandEncoder *) g_active_encoder.encoder;
    jitc_metal_close_encoder(ts);
    auto *cb = jitc_metal_acquire_cmdbuf(ts);
    auto *enc = cb->computeCommandEncoder();
    enc->retain();
    g_active_encoder.encoder = enc;
    g_active_encoder.kind = MetalEncoderKind::Compute;
    return enc;
}

MTL::BlitCommandEncoder *jitc_metal_acquire_blit_encoder(ThreadState *ts) {
    if (g_active_encoder.kind == MetalEncoderKind::Blit)
        return (MTL::BlitCommandEncoder *) g_active_encoder.encoder;
    jitc_metal_close_encoder(ts);
    auto *cb = jitc_metal_acquire_cmdbuf(ts);
    auto *enc = cb->blitCommandEncoder();
    enc->retain();
    g_active_encoder.encoder = enc;
    g_active_encoder.kind = MetalEncoderKind::Blit;
    return enc;
}

// ============================================================================
//  Utility kernel library
//
//  The kernels in resources/metal_kernels.metal are compiled once per device
//  at init time and cached in the MetalDevice struct for the process lifetime.
//  Pipeline states are looked up by name via jitc_metal_get_pipeline().
// ============================================================================

/// The compiled utility library per device.  Stored in MetalDevice::binary_archive
/// (repurposed to hold the MTL::Library*).

/// Look up a precompiled pipeline state by kernel name. Returns nullptr if
/// the kernel was not found or not yet compiled.
static MTL::ComputePipelineState *
jitc_metal_get_pipeline_impl(MTL::Device *dev, MTL::Library *lib,
                             const char *name) {
    NS::String *fn_name =
        NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function *func = lib->newFunction(fn_name);
    if (!func)
        return nullptr;
    NS::Error *err = nullptr;
    MTL::ComputePipelineState *pso =
        dev->newComputePipelineState(func, &err);
    func->release();
    if (!pso) {
        const char *desc = err && err->localizedDescription()
                               ? err->localizedDescription()->utf8String()
                               : "<unknown>";
        jitc_log(Warn, "jitc_metal_get_pipeline(%s): pipeline creation "
                       "failed: %s",
                 name, desc);
    }
    return pso;
}

namespace {
    /// Cache: kernel name → pipeline state, per device.
    std::mutex g_pipeline_cache_mutex;
    std::unordered_map<std::string, void *> g_pipeline_cache;
}

MTL::ComputePipelineState *
jitc_metal_get_pipeline(int device_id, const char *name) {
    std::string key = std::to_string(device_id) + "/" + name;

    {
        std::lock_guard<std::mutex> g(g_pipeline_cache_mutex);
        auto it = g_pipeline_cache.find(key);
        if (it != g_pipeline_cache.end())
            return (MTL::ComputePipelineState *) it->second;
    }

    if (device_id < 0 || (size_t) device_id >= state.metal_devices.size())
        return nullptr;
    MetalDevice &md = state.metal_devices[device_id];
    auto *lib = (MTL::Library *) md.binary_archive;
    auto *dev = (MTL::Device *) md.device;
    if (!lib)
        return nullptr;

    auto *pso = jitc_metal_get_pipeline_impl(dev, lib, name);
    if (pso) {
        std::lock_guard<std::mutex> g(g_pipeline_cache_mutex);
        g_pipeline_cache[key] = pso;
    }
    return pso;
}

/// Compile the utility kernels for ``md`` from the embedded MSL source. The
/// source is generated at build time by ``cmake/embed_metal_kernels.cmake``
/// from ``resources/metal_kernels.metal`` and exposed as
/// ``drjit::metal_kernels_src`` — no runtime file I/O is needed.
static bool jitc_metal_load_utility_kernels(MetalDevice &md) {
    auto *dev = (MTL::Device *) md.device;
    NS::Error *err = nullptr;
    NS::String *src = NS::String::string(drjit::metal_kernels_src,
                                         NS::UTF8StringEncoding);

    MTL::CompileOptions *opts = MTL::CompileOptions::alloc()->init();
    opts->setLanguageVersion(MTL::LanguageVersion3_0);
    // Fast math is OFF: the DD (double-double) helpers used by Float64
    // reductions rely on Knuth-style TwoSum, which the compiler would
    // otherwise reassociate away ((a + b) - a == b becomes 0).
    opts->setFastMathEnabled(false);

    MTL::Library *lib = dev->newLibrary(src, opts, &err);
    opts->release();

    if (!lib) {
        const char *desc = err && err->localizedDescription()
                               ? err->localizedDescription()->utf8String()
                               : "<unknown>";
        jitc_log(Warn,
                 "jitc_metal_load_utility_kernels(): compilation failed: %s",
                 desc);
        return false;
    }

    // Store the library in the MetalDevice (repurposing binary_archive).
    md.binary_archive = lib;

    jitc_log(Info, "jit_metal_init(): compiled utility kernel library for "
                   "device \"%s\".",
             md.name);
    return true;
}

// ============================================================================
//  Backend init / shutdown
// ============================================================================

bool jitc_metal_init() {
    DRJIT_METAL_SCOPED_POOL;

    if (!jitc_metal_api_init()) {
        jitc_log(Warn, "jit_metal_init(): Metal API initialization failed.");
        return false;
    }

    NS::Array *devices = MTL::CopyAllDevices();
    if (!devices || devices->count() == 0) {
        jitc_log(Warn, "jit_metal_init(): no Metal-capable GPU was detected.");
        if (devices)
            devices->release();
        return false;
    }

    state.metal_devices.clear();

    for (NS::UInteger i = 0; i < devices->count(); ++i) {
        auto *dev = (MTL::Device *) devices->object(i);
        if (!dev->supportsFamily(MTL::GPUFamilyMetal3)) {
            jitc_log(Warn,
                     "jit_metal_init(): skipping device \"%s\" because it "
                     "does not support Metal 3 (M1+ required).",
                     dev->name()->utf8String());
            continue;
        }

        MetalDevice md;
        dev->retain();
        md.device = dev;
        md.queue  = dev->newCommandQueue();
        md.event  = dev->newSharedEvent();
        md.event_value = 0;

        md.max_threads_per_threadgroup =
            (uint32_t) dev->maxThreadsPerThreadgroup().width;
        md.max_threadgroup_memory =
            (uint32_t) dev->maxThreadgroupMemoryLength();
        md.simd_width = 32; // Apple Silicon
        md.supports_metal3 = true;
        md.supports_ray_tracing = dev->supportsRaytracing();
        md.supports_float_atomics =
            dev->supportsFamily(MTL::GPUFamilyApple7);
        const char *name = dev->name()->utf8String();
        size_t len = std::strlen(name);
        md.name = (char *) std::malloc(len + 1);
        std::memcpy(md.name, name, len + 1);

        state.metal_devices.push_back(md);

        jitc_log(Info,
                 "jit_metal_init(): registered device \"%s\" "
                 "(simd=%u, max_threads=%u, rt=%s, float_atomic=%s)",
                 name, md.simd_width, md.max_threads_per_threadgroup,
                 md.supports_ray_tracing ? "yes" : "no",
                 md.supports_float_atomics ? "yes" : "no");
    }

    devices->release();

    // Compile the utility kernel library for each device
    for (MetalDevice &md : state.metal_devices) {
        if (!jitc_metal_load_utility_kernels(md))
            jitc_log(Warn,
                     "jit_metal_init(): failed to compile utility kernels "
                     "for device \"%s\".",
                     md.name);
    }

    return !state.metal_devices.empty();
}

void jitc_metal_shutdown() {
    DRJIT_METAL_SCOPED_POOL;

    jitc_metal_stage_shutdown();

    // Release cached pipeline states
    {
        std::lock_guard<std::mutex> g(g_pipeline_cache_mutex);
        for (auto &kv : g_pipeline_cache)
            ((MTL::ComputePipelineState *) kv.second)->release();
        g_pipeline_cache.clear();
    }

    for (MetalDevice &d : state.metal_devices) {
        if (d.event)
            ((MTL::SharedEvent *) d.event)->release();
        if (d.queue)
            ((MTL::CommandQueue *) d.queue)->release();
        if (d.binary_archive)
            ((MTL::Library *) d.binary_archive)->release();
        if (d.device)
            ((MTL::Device *) d.device)->release();
        std::free(d.name);
    }
    state.metal_devices.clear();
    metal_buffer_map.clear();

    jitc_metal_api_shutdown();
}

const char *jitc_metal_device_name(int device_id) {
    if (device_id < 0 || (size_t) device_id >= state.metal_devices.size())
        return "<invalid>";
    return state.metal_devices[device_id].name;
}

void jitc_metal_dump_devices() {
    for (size_t i = 0; i < state.metal_devices.size(); ++i) {
        const MetalDevice &d = state.metal_devices[i];
        jitc_log(Info, "  [%zu] %s", i, d.name);
    }
}

// ============================================================================
//  Kernel compilation
// ============================================================================

bool jitc_metal_compile(const char *source, size_t /*source_size*/,
                        const char *kernel_name, Kernel &kernel) {
    DRJIT_METAL_SCOPED_POOL;

    if (state.metal_devices.empty())
        jitc_fail("jitc_metal_compile(): no Metal devices initialized.");

    // Compile against the device of the calling thread (defaults to device 0).
    auto *ts = thread_state(JitBackend::Metal);
    auto *dev = (MTL::Device *) ts->metal_device;

    NS::Error *err = nullptr;
    NS::String *src = NS::String::string(source, NS::UTF8StringEncoding);

    MTL::CompileOptions *opts = MTL::CompileOptions::alloc()->init();
    opts->setLanguageVersion(MTL::LanguageVersion3_2);
    opts->setFastMathEnabled(false);
    opts->setLibraryType(MTL::LibraryTypeExecutable);

    MTL::Library *lib = dev->newLibrary(src, opts, &err);
    opts->release();

    if (!lib) {
        const char *desc = err && err->localizedDescription()
                               ? err->localizedDescription()->utf8String()
                               : "<unknown>";
        jitc_fail("jitc_metal_compile(): MSL compilation failed:\n%s\n\n"
                  "--- Source code ---\n%s",
                  desc, source);
    }

    NS::String *fn_name =
        NS::String::string(kernel_name, NS::UTF8StringEncoding);
    MTL::Function *func = lib->newFunction(fn_name);
    if (!func)
        jitc_fail("jitc_metal_compile(): kernel function \"%s\" not found in "
                  "the compiled library.", kernel_name);

    // ---- Pipeline creation -------------------------------------------------
    // Link the union of custom intersection functions across every scene
    // registered with this kernel (collected by ``jitc_metal_assemble``'s
    // pre-walk into ``metal_kernel_scenes``). MTLLinkedFunctions applies
    // to the entire PSO, not per-IFT, so scenes with disjoint function
    // name sets must all contribute. Per-scene IFTs are built lazily at
    // launch time in ``jitc_metal_get_or_create_ift_for_scene``.
    //
    // Dedup by name only, consistent with how the link-identity comment
    // in the MSL source identifies scenes for cache-key purposes (see
    // ``jitc_metal_render`` TraceRay arm): names are assumed unique per
    // implementation within a process.
    err = nullptr;
    MTL::ComputePipelineState *pso = nullptr;

    std::vector<MTL::Function *> linked_fns;
    std::vector<std::string> seen;
    for (MetalScene *scene : metal_kernel_scenes) {
        auto *isect_lib = scene ? (MTL::Library *) scene->intersection_fn_library
                                : nullptr;
        if (!isect_lib)
            continue;
        for (const std::string &name : scene->intersection_fn_names) {
            if (std::find(seen.begin(), seen.end(), name) != seen.end())
                continue;
            seen.push_back(name);
            NS::String *nstr =
                NS::String::string(name.c_str(), NS::UTF8StringEncoding);
            MTL::Function *f = isect_lib->newFunction(nstr);
            if (!f) {
                for (auto *prev : linked_fns) prev->release();
                lib->release();
                jitc_fail("jitc_metal_compile(): intersection function \"%s\" "
                          "not found in user-supplied library.", name.c_str());
            }
            linked_fns.push_back(f);
        }
    }

    if (!linked_fns.empty()) {
        NS::Array *fn_array = NS::Array::array(
            (const NS::Object *const *) linked_fns.data(),
            linked_fns.size());

        MTL::LinkedFunctions *lf = MTL::LinkedFunctions::alloc()->init();
        lf->setFunctions(fn_array);

        MTL::ComputePipelineDescriptor *desc =
            MTL::ComputePipelineDescriptor::alloc()->init();
        desc->setComputeFunction(func);
        desc->setLinkedFunctions(lf);

        pso = dev->newComputePipelineState(desc, MTL::PipelineOptionNone,
                                            nullptr, &err);

        desc->release();
        lf->release();
        for (auto *f : linked_fns)
            f->release();
    } else {
        pso = dev->newComputePipelineState(func, &err);
    }

    func->release();

    if (!pso) {
        const char *desc = err && err->localizedDescription()
                               ? err->localizedDescription()->utf8String()
                               : "<unknown>";
        lib->release();
        jitc_fail("jitc_metal_compile(): pipeline creation failed: %s", desc);
    }

    kernel.metal.pipeline    = pso;
    kernel.metal.library     = lib;
    kernel.metal.scenes      = nullptr; // Set by eval.cpp after assemble
    kernel.metal.scene_count = 0;
    kernel.size = (uint32_t) std::strlen(source);

    return false; // No on-disk cache hit (Phase 5 will hook into BinaryArchive)
}

void jitc_metal_free(Kernel &kernel) {
    DRJIT_METAL_SCOPED_POOL;
    if (kernel.metal.pipeline)
        ((MTL::ComputePipelineState *) kernel.metal.pipeline)->release();
    if (kernel.metal.library)
        ((MTL::Library *) kernel.metal.library)->release();
    delete[] kernel.metal.scenes;
    kernel.metal.pipeline    = nullptr;
    kernel.metal.library     = nullptr;
    kernel.metal.scenes      = nullptr;
    kernel.metal.scene_count = 0;
}

/// Resolve a Metal kernel-history entry's execution_time by waiting on the
/// stashed MTL::CommandBuffer and reading its GPU times. Called from the
/// backend-agnostic `KernelHistory::get()` in init.cpp. Returns the
/// execution time in ms, or 0 if no buffer is attached.
float jitc_metal_finalize_kernel_history_entry(void *task_ptr) {
    DRJIT_METAL_SCOPED_POOL;
    if (!task_ptr)
        return 0.f;
    auto *cb = (MTL::CommandBuffer *) task_ptr;
    cb->waitUntilCompleted();
    float ms = (float) ((cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0);
    cb->release();
    return ms;
}

// ----------------------------------------------------------------------------
//  Sync instrumentation (Phase-1 perf investigation)
//
//  Gated by ``DRJIT_METAL_TRACE_SYNC``:
//    unset / 0   : no instrumentation, zero overhead.
//    "1"         : count + total wall time per thread, accessible via the
//                  jitc_metal_sync_stats_* functions below.
//    "verbose"   : in addition, log each sync to stderr with its `tag`.
//
//  jitc_metal_sync() funnels every commit+wait we issue, so timing here
//  captures every stall point. The optional ``tag`` is a static string
//  identifying the caller (e.g. "block_prefix_reduce.phase1"). Default is
//  the legacy untagged signature so existing call sites keep working.
// ----------------------------------------------------------------------------

namespace {
    // Process-wide so counters work whether the syncs happen on the Python
    // thread or on a Dr.Jit worker thread.
    std::atomic<uint64_t> g_sync_count{0};
    std::atomic<uint64_t> g_sync_total_us{0};
    int g_sync_trace_mode = -1;  // -1 = uninitialised, 0 off, 1 on, 2 verbose

    int sync_trace_mode() {
        if (g_sync_trace_mode < 0) {
            const char *e = std::getenv("DRJIT_METAL_TRACE_SYNC");
            if (!e || e[0] == '\0' || (e[0] == '0' && e[1] == '\0'))
                g_sync_trace_mode = 0;
            else if (std::strcmp(e, "verbose") == 0)
                g_sync_trace_mode = 2;
            else
                g_sync_trace_mode = 1;
        }
        return g_sync_trace_mode;
    }
}

/// Commit + waitUntilCompleted on a fresh ad-hoc command buffer. Counts
/// against the global sync stats just like jitc_metal_sync_tagged.
/// Use from sites that build their own command buffer (memcpy staging,
/// fillBuffer init, etc.) rather than going through jitc_metal_sync.
void jitc_metal_commit_and_wait_tagged(void *cb_ptr, const char *tag) {
    auto *cb = (MTL::CommandBuffer *) cb_ptr;
    int mode = sync_trace_mode();
    auto t0 = mode > 0 ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{};
    cb->commit();
    cb->waitUntilCompleted();
    if (mode > 0) {
        auto t1 = std::chrono::steady_clock::now();
        uint64_t us = (uint64_t) std::chrono::duration_cast<
            std::chrono::microseconds>(t1 - t0).count();
        g_sync_count.fetch_add(1, std::memory_order_relaxed);
        g_sync_total_us.fetch_add(us, std::memory_order_relaxed);
        if (mode == 2)
            fprintf(stderr, "[METAL SYNC %s] %llu us\n",
                    tag ? tag : "?", (unsigned long long) us);
    }
}

extern "C" JIT_EXPORT void
jitc_metal_sync_stats_get(uint64_t *count, uint64_t *total_us) {
    if (count) *count = g_sync_count.load(std::memory_order_relaxed);
    if (total_us) *total_us = g_sync_total_us.load(std::memory_order_relaxed);
}

extern "C" JIT_EXPORT void jitc_metal_sync_stats_reset() {
    g_sync_count.store(0, std::memory_order_relaxed);
    g_sync_total_us.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
//  Launch-path instrumentation. Same env-var gate as sync stats.
//  Process-wide atomics so worker-thread launches are captured.
// ---------------------------------------------------------------------------
namespace {
    std::atomic<uint64_t> g_launch_count{0};
    std::atomic<uint64_t> g_launch_total_us{0};
    std::atomic<uint64_t> g_launch_setup_us{0};
    std::atomic<uint64_t> g_launch_setbytes_us{0};
    std::atomic<uint64_t> g_launch_params_loop_us{0};
    std::atomic<uint64_t> g_launch_vcall_loop_us{0};
    std::atomic<uint64_t> g_launch_scene_loop_us{0};
    std::atomic<uint64_t> g_launch_dispatch_us{0};
    std::atomic<uint64_t> g_launch_n_params{0};
}

extern "C" JIT_EXPORT void jitc_metal_launch_stats_get(
    uint64_t *count, uint64_t *total_us, uint64_t *setup_us,
    uint64_t *setbytes_us, uint64_t *params_loop_us,
    uint64_t *vcall_loop_us, uint64_t *scene_loop_us,
    uint64_t *dispatch_us, uint64_t *n_params)
{
    if (count)          *count          = g_launch_count.load();
    if (total_us)       *total_us       = g_launch_total_us.load();
    if (setup_us)       *setup_us       = g_launch_setup_us.load();
    if (setbytes_us)    *setbytes_us    = g_launch_setbytes_us.load();
    if (params_loop_us) *params_loop_us = g_launch_params_loop_us.load();
    if (vcall_loop_us)  *vcall_loop_us  = g_launch_vcall_loop_us.load();
    if (scene_loop_us)  *scene_loop_us  = g_launch_scene_loop_us.load();
    if (dispatch_us)    *dispatch_us    = g_launch_dispatch_us.load();
    if (n_params)       *n_params       = g_launch_n_params.load();
}

extern "C" JIT_EXPORT void jitc_metal_launch_stats_reset() {
    g_launch_count.store(0); g_launch_total_us.store(0);
    g_launch_setup_us.store(0); g_launch_setbytes_us.store(0);
    g_launch_params_loop_us.store(0); g_launch_vcall_loop_us.store(0);
    g_launch_scene_loop_us.store(0); g_launch_dispatch_us.store(0);
    g_launch_n_params.store(0);
}

bool jitc_metal_launch_trace_enabled() { return sync_trace_mode() > 0; }

void jitc_metal_launch_stats_add(uint64_t total_us, uint64_t setup_us,
                                 uint64_t setbytes_us, uint64_t params_loop_us,
                                 uint64_t vcall_loop_us, uint64_t scene_loop_us,
                                 uint64_t dispatch_us, uint64_t n_params) {
    g_launch_count.fetch_add(1, std::memory_order_relaxed);
    g_launch_total_us.fetch_add(total_us, std::memory_order_relaxed);
    g_launch_setup_us.fetch_add(setup_us, std::memory_order_relaxed);
    g_launch_setbytes_us.fetch_add(setbytes_us, std::memory_order_relaxed);
    g_launch_params_loop_us.fetch_add(params_loop_us, std::memory_order_relaxed);
    g_launch_vcall_loop_us.fetch_add(vcall_loop_us, std::memory_order_relaxed);
    g_launch_scene_loop_us.fetch_add(scene_loop_us, std::memory_order_relaxed);
    g_launch_dispatch_us.fetch_add(dispatch_us, std::memory_order_relaxed);
    g_launch_n_params.fetch_add(n_params, std::memory_order_relaxed);
}

void jitc_metal_sync_tagged(ThreadState *ts, const char *tag) {
    DRJIT_METAL_SCOPED_POOL;
    int mode = sync_trace_mode();
    if (auto *rts = dynamic_cast<RecordThreadState *>(ts))
        ts = rts->m_internal;
    if (!ts->metal_command_buffer) {
        if (mode == 2)
            fprintf(stderr, "[METAL SYNC %s] noop (no pending CB)\n",
                    tag ? tag : "?");
        return;
    }
    auto *cb = (MTL::CommandBuffer *) ts->metal_command_buffer;
    jitc_metal_close_encoder(ts);

    auto t0 = mode > 0 ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{};

    cb->commit();
    cb->waitUntilCompleted();
    cb->release();
    ts->metal_command_buffer = nullptr;

    if (mode > 0) {
        auto t1 = std::chrono::steady_clock::now();
        uint64_t us = (uint64_t) std::chrono::duration_cast<
            std::chrono::microseconds>(t1 - t0).count();
        g_sync_count.fetch_add(1, std::memory_order_relaxed);
        g_sync_total_us.fetch_add(us, std::memory_order_relaxed);
        if (mode == 2)
            fprintf(stderr, "[METAL SYNC %s] %llu us\n",
                    tag ? tag : "?", (unsigned long long) us);
    }
}

void jitc_metal_sync(ThreadState *ts) {
    jitc_metal_sync_tagged(ts, "untagged");
}

// ============================================================================
//  Ray Tracing API
// ============================================================================

/// Build a MetalScene with the given configuration and wrap it in a JIT
/// variable. Returns the variable index; the caller is expected to hold
/// the reference for the scene's lifetime and dec_ref it on destruction.
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
        scene->intersection_fn_library = intersection_fn_library;
        ((MTL::Library *) intersection_fn_library)->retain();
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
                ((MTL::IntersectionFunctionTable *) kv.second)->release();
        }
        if (s->intersection_fn_library)
            ((MTL::Library *) s->intersection_fn_library)->release();
        jitc_metal_unregister_live_scene(s);
        delete s;
    };

    jitc_var_set_callback(index, callback, scene, true);
    jitc_metal_register_live_scene(scene);

    return index;
}

// ============================================================================
//  Live MetalScene registry
//
//  Frozen-function replay skips ``jitc_metal_assemble``, so it can't refresh
//  ``ts->metal_active_scene`` when a scene's TLAS is rebuilt mid-recording
//  (which dangles any cached scene pointer). At launch time we therefore
//  consult this registry to (a) detect that a saved scene pointer is stale
//  and (b) fall back to the most-recently-configured scene as a heuristic.
//  This works for the typical single-scene case; multi-scene workflows that
//  rebuild between recording and replay would need explicit scene-input
//  plumbing (out of scope here).
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

/// Look up the MetalScene attached to a JIT variable returned by
/// jitc_metal_configure_scene. Returns nullptr if the variable is not a
/// scene (e.g. has been destroyed already).
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

/// Read the active scene that the current ThreadState's kernel is being
/// assembled / launched against. Returns the MetalScene determined by the
/// schedule walk in jitc_metal_assemble (or nullptr if no TraceRay node is
/// present in this kernel).
MetalScene *jitc_metal_active_scene() {
    auto *ts = thread_state(JitBackend::Metal);
    return (MetalScene *) ts->metal_active_scene;
}

/// Lazily build (and cache) an IntersectionFunctionTable for the given
/// scene + compute pipeline. The function handles are derived from the
/// pipeline so each pipeline needs its own IFT instance. Returns nullptr
/// if the scene has no custom intersection functions configured.
MTL::IntersectionFunctionTable *
jitc_metal_get_or_create_ift_for_scene(MetalScene *scene,
                                       MTL::ComputePipelineState *pso) {
    if (!scene || !pso || scene->intersection_fn_names.empty())
        return nullptr;

    // Cache hit: we already built an IFT for this pipeline.
    auto it = scene->ift_cache.find(pso);
    if (it != scene->ift_cache.end())
        return (MTL::IntersectionFunctionTable *) it->second;

    auto *isect_lib = (MTL::Library *) scene->intersection_fn_library;
    if (!isect_lib)
        return nullptr;

    uint32_t n_ift = (uint32_t) scene->intersection_fn_names.size();

    // Resolve unique function objects (deduplicate so we look up each
    // function only once even if the IFT references it multiple times).
    std::unordered_map<std::string, MTL::Function *> unique_fns;
    for (uint32_t i = 0; i < n_ift; ++i) {
        const std::string &name = scene->intersection_fn_names[i];
        if (unique_fns.count(name))
            continue;
        NS::String *nstr =
            NS::String::string(name.c_str(), NS::UTF8StringEncoding);
        MTL::Function *f = isect_lib->newFunction(nstr);
        if (!f) {
            jitc_fail("jitc_metal_get_or_create_ift_for_scene(): intersection "
                      "function \"%s\" not found in user-supplied library.",
                      name.c_str());
        }
        unique_fns[name] = f;
    }

    MTL::IntersectionFunctionTableDescriptor *iftd =
        MTL::IntersectionFunctionTableDescriptor::alloc()->init();
    iftd->setFunctionCount(n_ift);
    MTL::IntersectionFunctionTable *ift =
        pso->newIntersectionFunctionTable(iftd);
    iftd->release();

    for (uint32_t i = 0; i < n_ift; ++i) {
        const std::string &name = scene->intersection_fn_names[i];
        MTL::Function *f = unique_fns[name];
        MTL::FunctionHandle *handle = pso->functionHandle(f);
        ift->setFunction(handle, i);

        if (i < scene->intersection_fn_buffers.size()) {
            auto *buf = (MTL::Buffer *) scene->intersection_fn_buffers[i];
            uint32_t slot = (i < scene->intersection_fn_buffer_slots.size())
                                ? scene->intersection_fn_buffer_slots[i] : 0u;
            uint64_t offset =
                (i < scene->intersection_fn_buffer_offsets.size())
                    ? scene->intersection_fn_buffer_offsets[i] : 0ull;
            if (buf)
                ift->setBuffer(buf, offset, slot);
        }
    }

    for (auto &kv : unique_fns)
        kv.second->release();

    scene->ift_cache[pso] = ift;
    return ift;
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
    // The scene dep keeps the MetalScene alive while the trace op is in
    // the IR, and lets codegen / launch resolve per-scene state.
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
