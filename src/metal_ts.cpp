/*
    src/metal_ts.cpp -- Implementation of MetalThreadState.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Metal memory model summary (relevant to this file):

      * ``StorageModePrivate`` buffers live in GPU-only VRAM. They are fast
        for compute but cannot be read or written by the CPU. The only way
        to move data in/out is via ``MTLBlitCommandEncoder::copyFromBuffer``.

      * ``StorageModeShared`` buffers are in unified memory and accessible
        to both CPU and GPU. They are used as staging areas for uploads and
        downloads, and also for ``HostPinned`` allocations.

      * ``gpuAddress()`` returns the GPU virtual address of a buffer. The
        kernel parameter buffer contains these addresses so the GPU can
        dereference them as ``device T*``. The CPU must never dereference
        a ``gpuAddress()`` pointer directly.

      * Every buffer accessed by a compute kernel must be declared via
        ``useResource()`` on the encoder so that Metal's residency system
        makes the pages available to the GPU.
*/

#if defined(DRJIT_ENABLE_METAL)

#include "metal_ts.h"
#include "metal.h"
#include "metal_api.h"
#include "metal_stage_buffer.h"
#include "log.h"
#include "var.h"
#include "malloc.h"
#include "io.h"
#include "util.h"

#include <Metal/Metal.hpp>
#include <cstdlib>

// Forward declaration of the MPS GEMM wrapper (in metal_mps.mm)
extern "C" void jitc_metal_mps_gemm(
    void *mtl_device, void *mtl_queue,
    void *a_buf, size_t a_offset,
    void *b_buf, size_t b_offset,
    void *c_buf, size_t c_offset,
    uint32_t M, uint32_t N, uint32_t K,
    bool At, bool Bt,
    uint32_t a_rows, uint32_t a_cols,
    uint32_t b_rows, uint32_t b_cols,
    uint32_t tsize, int mps_data_type,
    double alpha, double beta);

// Forward declarations of helpers in metal_core.cpp
extern MTL::CommandBuffer *jitc_metal_acquire_cmdbuf(ThreadState *ts);
extern MTL::ComputeCommandEncoder *
jitc_metal_acquire_compute_encoder(ThreadState *ts);
extern MTL::BlitCommandEncoder *
jitc_metal_acquire_blit_encoder(ThreadState *ts);
extern void jitc_metal_close_encoder(ThreadState *ts);
extern void jitc_metal_sync_tagged(ThreadState *ts, const char *tag);
extern void jitc_metal_commit_and_wait_tagged(void *cb_ptr, const char *tag);
extern bool jitc_metal_launch_trace_enabled();
extern void jitc_metal_launch_stats_add(
    uint64_t total_us, uint64_t setup_us, uint64_t setbytes_us,
    uint64_t params_loop_us, uint64_t vcall_loop_us,
    uint64_t scene_loop_us, uint64_t dispatch_us, uint64_t n_params);

#include <chrono>

MetalThreadState::~MetalThreadState() {
    DRJIT_METAL_SCOPED_POOL;

    if (metal_command_buffer) {
        jitc_metal_close_encoder(this);
        auto *cb = (MTL::CommandBuffer *) metal_command_buffer;
        cb->commit();
        cb->waitUntilCompleted();
        cb->release();
        metal_command_buffer = nullptr;
    }
}

void MetalThreadState::barrier() {
    // Metal queue submission order already serialises kernels, so the
    // semantic ``barrier`` (`subsequent kernels see results of earlier
    // kernels`) needs no GPU wait — only a ``commit`` so the GPU can
    // start chewing through the queued work while the CPU keeps
    // building the next command buffer.
    //
    // Previously this called commit+waitUntilCompleted, which stalled
    // the CPU on every jit_eval group boundary (~600 us per call) and
    // prevented any kernel pipelining. The full drain still happens via
    // ``jitc_metal_sync`` whenever the CPU actually needs results
    // (memcpy GPU→CPU, dr.sync_thread, destructor).
    DRJIT_METAL_SCOPED_POOL;
    if (!metal_command_buffer)
        return;
    jitc_metal_close_encoder(this);
    auto *cb = (MTL::CommandBuffer *) metal_command_buffer;
    cb->commit();              // submit asynchronously
    cb->release();
    metal_command_buffer = nullptr;
}

// ============================================================================
//  Kernel launch
// ============================================================================

Task *MetalThreadState::launch(Kernel kernel, KernelKey * /*key*/,
                               XXH128_hash_t /*hash*/, uint32_t size,
                               std::vector<void *> *kernel_params,
                               const std::vector<uint32_t> * /*kernel_param_ids*/,
                               KernelHistoryEntry *kernel_history_entry) {
    DRJIT_METAL_SCOPED_POOL;

    using clock = std::chrono::steady_clock;
    bool trace = jitc_metal_launch_trace_enabled();
    auto t_total_0 = trace ? clock::now() : clock::time_point{};
    uint64_t t_setup = 0, t_setbytes = 0, t_params_loop = 0;
    uint64_t t_vcall_loop = 0, t_scene_loop = 0, t_dispatch = 0;
    auto take = [&](clock::time_point &t0) -> uint64_t {
        auto t1 = clock::now();
        uint64_t us = (uint64_t) std::chrono::duration_cast<
            std::chrono::microseconds>(t1 - t0).count();
        t0 = t1;
        return us;
    };
    auto t_phase = trace ? clock::now() : clock::time_point{};

    auto *pso = (MTL::ComputePipelineState *) kernel.metal.pipeline;
    auto *enc = jitc_metal_acquire_compute_encoder(this);

    enc->setComputePipelineState(pso);
    if (trace) t_setup = take(t_phase);

    if (kernel_params && !kernel_params->empty()) {
        enc->setBytes(kernel_params->data(),
                      kernel_params->size() * sizeof(void *), 0);
        if (trace) t_setbytes = take(t_phase);

        // Mark every referenced MTLBuffer with useResource().
        // Entry 0 is the encoded launch size — skip it.
        for (size_t i = 1; i < kernel_params->size(); ++i) {
            void *ptr = (*kernel_params)[i];
            if (!ptr)
                continue;
            size_t offset = 0;
            auto *buf = (MTL::Buffer *)
                jitc_metal_find_buffer(ptr, &offset);
            if (buf)
                enc->useResource(buf, MTL::ResourceUsageRead |
                                          MTL::ResourceUsageWrite);
        }
        if (trace) t_params_loop = take(t_phase);
    }

    // Mark buffers referenced by vcall data sections
    for (void *ptr : metal_call_resources) {
        if (!ptr) continue;
        size_t off = 0;
        auto *buf = (MTL::Buffer *)
            jitc_metal_find_buffer(ptr, &off);
        if (buf)
            enc->useResource(buf, MTL::ResourceUsageRead |
                                      MTL::ResourceUsageWrite);
    }
    metal_call_resources.clear();
    if (trace) t_vcall_loop = take(t_phase);

    // Bind per-scene TLAS + IFT for every scene referenced by this kernel.
    // The kernel was generated with ``accel_<i> [[buffer(1+i)]]`` for each
    // scene at slot ``i`` in ``kernel.metal.scenes``, plus (optionally)
    // ``ift_<i>`` at packed slots in ``[1+N, 1+N+M)`` for scenes that
    // have an intersection function library.
    //
    // ``jitc_metal_assemble`` populates the kernel's scene list; frozen
    // replay skips re-assemble so we trust the persisted list. Stale
    // captured pointers are surfaced in the per-slot Warn below.
    auto pick_scene = [](void *cand) -> MetalScene * {
        if (!jitc_metal_is_live_scene(cand))
            return nullptr;
        // Reject scenes whose TLAS was invalidated by a prior
        // ``accel_release_metal`` (the MetalScene object may still be
        // alive due to a frozen recording's ref, but its TLAS handle
        // has been released).
        auto *s = (MetalScene *) cand;
        return s->tlas ? s : nullptr;
    };

    uint32_t n_scenes = kernel.metal.scene_count;
    // Two-phase binding: accels at slots [1, 1+N), then IFTs at the next
    // available slots — only for scenes that actually have an
    // ``intersection_fn_library``. Matches ``jitc_metal_finalize_scene_layout()``
    // in metal_eval.cpp, which assigns the same layout at signature-emit time.
    //
    // Pass 1: bind accels.
    std::vector<MetalScene *> resolved(n_scenes, nullptr);
    for (uint32_t i = 0; i < n_scenes; ++i) {
        auto *captured = (MetalScene *) kernel.metal.scenes[i];
        MetalScene *scene = pick_scene(captured);

        if (!scene) {
            // Captured scene has gone stale (TLAS released after the
            // accel was rebuilt, or the MetalScene was destroyed
            // between recording and replay).
            //
            // Slot 0 has historically had a fallback to the
            // ThreadState's active scene (single-scene workflows
            // sometimes recapture a fresh scene_index without
            // re-recording the kernel). Keep that for backward
            // compatibility. Slots >= 1 have no meaningful fallback
            // — we don't know which other scene the kernel intended
            // — so we surface a Warn and leave the slot unbound.
            // Any TraceRay routed to that slot will hit unbound
            // hardware state, which Metal handles by returning miss
            // (the intersector reports ``intersection_type::none``).
            if (i == 0) {
                scene = pick_scene(metal_active_scene);
                if (!scene)
                    scene = (MetalScene *) jitc_metal_last_live_scene();
            }
            if (!scene) {
                jitc_log(LogLevel::Warn,
                         "MetalThreadState::launch(): scene at slot %u "
                         "is stale (rebuilt or freed since this kernel "
                         "was compiled). Traces against this slot will "
                         "miss. Re-evaluate / rebuild to pick up the "
                         "new scene state.", i);
            }
        }
        resolved[i] = scene;
        if (!scene || !scene->tlas)
            continue;

        auto *tlas = (MTL::AccelerationStructure *) scene->tlas;
        enc->setAccelerationStructure(tlas, 1u + i);
        enc->useResource(tlas, MTL::ResourceUsageRead);

        for (void *res : scene->resources) {
            if (res)
                enc->useResource((MTL::Resource *) res,
                                 MTL::ResourceUsageRead);
        }
    }

    // Pass 2: bind IFTs at slots [1+N, 1+N+M), where M is the count of
    // scenes with ``intersection_fn_library``. The slot order *must*
    // match the order ``jitc_metal_finalize_scene_layout`` assigned —
    // which is "scenes with IFT, in scene-index order, packed".
    uint32_t ift_slot = 1u + n_scenes;
    for (uint32_t i = 0; i < n_scenes; ++i) {
        // Use the captured scene (not ``resolved``) to decide whether
        // the kernel signature reserved an IFT slot for index ``i``.
        // If the kernel was generated with an IFT for this scene, we
        // MUST consume that slot index now even if the resolved scene
        // is null/stale (otherwise subsequent IFTs land at the wrong
        // slots). When the resolved scene is good, bind its IFT;
        // otherwise just advance the slot counter.
        auto *captured = (MetalScene *) kernel.metal.scenes[i];
        bool kernel_reserved_ift =
            captured && captured->intersection_fn_library;
        if (!kernel_reserved_ift)
            continue;

        MetalScene *scene = resolved[i];
        if (scene && scene->intersection_fn_library) {
            auto *ift = jitc_metal_get_or_create_ift_for_scene(scene, pso);
            if (ift) {
                enc->setIntersectionFunctionTable(ift, ift_slot);
                enc->useResource(ift, MTL::ResourceUsageRead);
                for (void *bp : scene->intersection_fn_buffers) {
                    auto *buf = (MTL::Buffer *) bp;
                    if (buf)
                        enc->useResource(buf, MTL::ResourceUsageRead);
                }
            }
        }
        ift_slot++;
    }

    uint32_t threads_per_group = std::min<uint32_t>(
        (uint32_t) pso->maxTotalThreadsPerThreadgroup(), metal_max_threads);
    threads_per_group =
        (threads_per_group / metal_simd_width) * metal_simd_width;
    if (threads_per_group == 0)
        threads_per_group = metal_simd_width;

    if (trace) t_scene_loop = take(t_phase);

    MTL::Size grid     = MTL::Size::Make(size, 1, 1);
    MTL::Size group_sz = MTL::Size::Make(threads_per_group, 1, 1);
    enc->dispatchThreads(grid, group_sz);
    if (trace) t_dispatch = take(t_phase);

    if (trace) {
        uint64_t t_total = (uint64_t) std::chrono::duration_cast<
            std::chrono::microseconds>(clock::now() - t_total_0).count();
        uint64_t n_params = kernel_params ? kernel_params->size() : 0;
        jitc_metal_launch_stats_add(t_total, t_setup, t_setbytes,
                                    t_params_loop, t_vcall_loop,
                                    t_scene_loop, t_dispatch, n_params);
    }

    if (kernel_history_entry) {
        kernel_history_entry->size = size;

        // Stash an extra retain on the current command buffer so we can read
        // its GPU times once it has executed. With LaunchBlocking enabled
        // each kernel gets its own buffer (normal flush via jitc_metal_sync),
        // so the (GPUEndTime - GPUStartTime) we read in KernelHistory::get()
        // is the per-kernel time. Without LaunchBlocking, kernels may share
        // a buffer and all see the same overall time (best-effort).
        auto *cb = (MTL::CommandBuffer *) metal_command_buffer;
        if (cb) {
            cb->retain();
            kernel_history_entry->task = cb;
        }
        state.kernel_history.append(*kernel_history_entry);
    }

    return nullptr;
}

// ============================================================================
//  Memory operations
// ============================================================================

void MetalThreadState::memset_async(void *ptr, uint32_t size, uint32_t isize,
                                    const void *src) {
    DRJIT_METAL_SCOPED_POOL;
    size_t offset = 0;
    auto *buf = (MTL::Buffer *)
        jitc_metal_find_buffer(ptr, &offset);
    if (!buf)
        jitc_raise("MetalThreadState::memset_async(): unknown pointer.");

    if (isize == 1) {
        auto *enc = jitc_metal_acquire_blit_encoder(this);
        enc->fillBuffer(buf, NS::Range::Make(offset, (size_t) size),
                        *(const uint8_t *) src);
    } else {
        // Multi-byte pattern — build a small staging buffer with the
        // replicated pattern, then blit it.  For now, handle 4 and 8-byte
        // patterns which cover the common float/int/double cases.
        size_t total = (size_t) size * (size_t) isize;
        auto *dev = (MTL::Device *) metal_device;
        auto *staging =
            dev->newBuffer(total, MTL::ResourceStorageModeShared);
        uint8_t *dst = (uint8_t *) staging->contents();
        for (uint32_t i = 0; i < size; ++i)
            std::memcpy(dst + (size_t) i * isize, src, isize);

        auto *enc = jitc_metal_acquire_blit_encoder(this);
        enc->copyFromBuffer(staging, 0, buf, offset, total);

        // Release staging after GPU consumes it.
        auto *cb = (MTL::CommandBuffer *) metal_command_buffer;
        cb->addCompletedHandler(
            [staging](MTL::CommandBuffer *) { staging->release(); });
    }
}

/// Synchronous GPU→CPU or CPU→GPU copy.
///
/// The critical path is GPU→CPU (used by ``jitc_var_str`` and friends).
/// ``StorageModePrivate`` buffers require a blit to a shared staging buffer.
void MetalThreadState::memcpy(void *dst, const void *src, size_t size) {
    DRJIT_METAL_SCOPED_POOL;

    // Flush any pending GPU work — commit but don't wait. The GPU→CPU
    // staging blit below is queued onto a fresh CB; Metal queues are
    // FIFO so the new CB executes after the prior one completes, and
    // tracked-resource hazard tracking sequences buffer access across
    // CBs. The CPU only needs to wait for the staging blit itself
    // (handled by jitc_metal_commit_and_wait_tagged below).
    this->barrier();

    size_t src_offset = 0, dst_offset = 0;
    auto *src_buf = (MTL::Buffer *)
        jitc_metal_find_buffer((void *) src, &src_offset);
    auto *dst_buf = (MTL::Buffer *)
        jitc_metal_find_buffer(dst, &dst_offset);

    if (src_buf && !dst_buf) {
        // GPU → CPU readback (synchronous)
        void *staging_buf  = nullptr;
        size_t staging_off = 0;
        void *staging_tok  = nullptr;
        void *staging_ptr  = jitc_metal_stage_acquire_download(
            this, size, &staging_buf, &staging_off, &staging_tok);
        auto *queue = (MTL::CommandQueue *) metal_queue;
        auto *cb    = queue->commandBuffer();
        auto *enc   = cb->blitCommandEncoder();
        enc->copyFromBuffer(src_buf, src_offset,
                            (MTL::Buffer *) staging_buf, staging_off, size);
        enc->endEncoding();
        jitc_metal_commit_and_wait_tagged(cb, "memcpy.gpu2cpu");
        std::memcpy(dst, staging_ptr, size);
        jitc_metal_stage_release(staging_tok);
    } else if (!src_buf && dst_buf) {
        // CPU → GPU upload (synchronous)
        void *staging_buf  = nullptr;
        size_t staging_off = 0;
        void *staging_tok  = nullptr;
        void *staging_ptr  = jitc_metal_stage_acquire_upload(
            this, size, &staging_buf, &staging_off, &staging_tok);
        std::memcpy(staging_ptr, src, size);
        auto *queue = (MTL::CommandQueue *) metal_queue;
        auto *cb    = queue->commandBuffer();
        auto *enc   = cb->blitCommandEncoder();
        enc->copyFromBuffer((MTL::Buffer *) staging_buf, staging_off,
                            dst_buf, dst_offset, size);
        enc->endEncoding();
        jitc_metal_commit_and_wait_tagged(cb, "memcpy.cpu2gpu");
        jitc_metal_stage_release(staging_tok);
    } else if (!src_buf && !dst_buf) {
        std::memcpy(dst, src, size);
    } else {
        // GPU → GPU
        memcpy_async(dst, src, size);
        jitc_metal_sync_tagged(this, "memcpy.gpu2gpu");
    }
}

/// Asynchronous copy enqueued into the current command buffer.
void MetalThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    DRJIT_METAL_SCOPED_POOL;

    size_t src_offset = 0, dst_offset = 0;
    auto *src_buf = (MTL::Buffer *)
        jitc_metal_find_buffer((void *) src, &src_offset);
    auto *dst_buf = (MTL::Buffer *)
        jitc_metal_find_buffer(dst, &dst_offset);

    if (src_buf && dst_buf) {
        // GPU → GPU blit
        auto *enc = jitc_metal_acquire_blit_encoder(this);
        enc->copyFromBuffer(src_buf, src_offset, dst_buf, dst_offset, size);
    } else if (!src_buf && dst_buf) {
        // CPU → GPU: stage through a pooled shared buffer
        void *staging_buf  = nullptr;
        size_t staging_off = 0;
        void *staging_tok  = nullptr;
        void *staging_ptr  = jitc_metal_stage_acquire_upload(
            this, size, &staging_buf, &staging_off, &staging_tok);
        std::memcpy(staging_ptr, src, size);
        auto *enc = jitc_metal_acquire_blit_encoder(this);
        enc->copyFromBuffer((MTL::Buffer *) staging_buf, staging_off,
                            dst_buf, dst_offset, size);
        auto *cb = (MTL::CommandBuffer *) metal_command_buffer;
        cb->addCompletedHandler(
            [staging_tok](MTL::CommandBuffer *) {
                jitc_metal_stage_release(staging_tok);
            });
    } else if (src_buf && !dst_buf) {
        // GPU → CPU: stage through a pooled shared buffer (async —
        // completion handler copies from staging to dst, then releases)
        void *staging_buf  = nullptr;
        size_t staging_off = 0;
        void *staging_tok  = nullptr;
        void *staging_ptr  = jitc_metal_stage_acquire_download(
            this, size, &staging_buf, &staging_off, &staging_tok);
        auto *enc = jitc_metal_acquire_blit_encoder(this);
        enc->copyFromBuffer(src_buf, src_offset,
                            (MTL::Buffer *) staging_buf, staging_off, size);
        auto *cb = (MTL::CommandBuffer *) metal_command_buffer;
        cb->addCompletedHandler(
            [dst, staging_ptr, staging_tok, size](MTL::CommandBuffer *) {
                std::memcpy(dst, staging_ptr, size);
                jitc_metal_stage_release(staging_tok);
            });
    } else {
        std::memcpy(dst, src, size);
    }
}

void MetalThreadState::poke(void *dst, const void *src, uint32_t size) {
    DRJIT_METAL_SCOPED_POOL;
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);
    size_t offset = 0;
    auto *buf = (MTL::Buffer *)
        jitc_metal_find_buffer(dst, &offset);
    if (buf && buf->storageMode() == MTL::StorageModeShared) {
        std::memcpy((uint8_t *) buf->contents() + offset, src, size);
    } else {
        memcpy_async(dst, src, size);
    }
}

// ============================================================================
//  Stubs for Phase 3 operations
// ============================================================================

static VarType metal_make_unsigned(VarType vt) {
    switch (vt) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return vt;
    }
}

static const char *metal_type_suffix(VarType vt) {
    switch (vt) {
        case VarType::UInt8:   return "u8";
        case VarType::Float16: return "f16";
        case VarType::Float32: return "f32";
        case VarType::UInt32:  return "u32";
        case VarType::Int32:   return "i32";
        case VarType::UInt64:  return "u64";
        case VarType::Int64:   return "i64";
        default: return nullptr;
    }
}

static const char *metal_op_name(ReduceOp op) {
    switch (op) {
        case ReduceOp::Add: return "add";
        case ReduceOp::Mul: return "mul";
        case ReduceOp::Min: return "min";
        case ReduceOp::Max: return "max";
        case ReduceOp::Or:  return "or";
        case ReduceOp::And: return "and";
        default: return nullptr;
    }
}

void MetalThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                    uint32_t block_size, const void *in,
                                    void *out) {
    DRJIT_METAL_SCOPED_POOL;

    if (size == 0)
        return;
    if (block_size == 0 || block_size > size)
        jitc_raise("MetalThreadState::block_reduce(): invalid block size "
                   "(size=%u, block_size=%u)!", size, block_size);

    uint32_t tsize = type_size[(int) vt];
    if (block_size == 1) {
        memcpy_async(out, in, (size_t) size * tsize);
        return;
    }

    // Signed add/mul/and/or use the unsigned kernel (same bit pattern)
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And)
        vt = metal_make_unsigned(vt);

    const char *type_suffix = metal_type_suffix(vt);
    const char *op_name = metal_op_name(op);
    if (!type_suffix || !op_name)
        jitc_raise("MetalThreadState::block_reduce(): unsupported type=%s "
                   "or op=%s.", type_name[(int) vt], op_name ? op_name : "?");

    uint32_t block_count = (size + block_size - 1) / block_size;
    uint32_t chunk_size = round_pow2(std::min(block_size, 1024u));

    // Build kernel name: e.g. "block_reduce_add_f32_1024"
    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name),
             "block_reduce_%s_%s_%u", op_name, type_suffix, chunk_size);

    auto *pso = jitc_metal_get_pipeline(this->device, kernel_name);
    if (!pso)
        jitc_raise("MetalThreadState::block_reduce(): kernel \"%s\" not "
                   "found in the utility library.", kernel_name);

    // Prepare parameter struct — matches block_reduce_params in the MSL kernel
    struct {
        uint64_t in;
        uint64_t out;
        uint32_t size;
        uint32_t block_size;
        uint32_t chunk_count;
    } params;

    uint32_t chunks_per_block = (block_size + chunk_size - 1) / chunk_size;

    params.in = (uint64_t)(uintptr_t) in;
    params.size = size;
    params.block_size = block_size;
    params.chunk_count = block_count * chunks_per_block;

    bool need_recursive = (chunks_per_block > 1);
    void *temp_out = nullptr;

    if (need_recursive) {
        // Allocate a temporary buffer for the per-chunk outputs
        temp_out =
            jitc_malloc(JitBackend::Metal, (size_t) params.chunk_count * tsize);
        params.out = (uint64_t)(uintptr_t) temp_out;
    } else {
        // For HostPinned (shared) buffers, we need the GPU address
        size_t out_off = 0;
        auto *ob = (MTL::Buffer *)
            jitc_metal_find_buffer(out, &out_off);
        if (ob && ob->storageMode() == MTL::StorageModeShared)
            params.out = ob->gpuAddress() + out_off;
        else
            params.out = (uint64_t)(uintptr_t) out;
    }

    // Compute threadgroup configuration
    uint32_t threads_per_group =
        (chunk_size < 128) ? 128u : chunk_size;
    uint32_t chunks_per_tg =
        (chunk_size < 128) ? (threads_per_group / chunk_size) : 1u;
    uint32_t grid_size =
        ((params.chunk_count + chunks_per_tg - 1) / chunks_per_tg) *
        threads_per_group;

    // Shared memory: one value per SIMD group
    uint32_t simd_groups_per_tg = threads_per_group / 32;
    uint32_t smem_bytes = simd_groups_per_tg * tsize * chunks_per_tg;

    auto *enc = jitc_metal_acquire_compute_encoder(this);
    enc->setComputePipelineState(pso);
    enc->setBytes(&params, sizeof(params), 0);
    enc->setThreadgroupMemoryLength(smem_bytes, 0);

    // Mark input/output buffers as resident
    size_t offset = 0;
    auto *in_buf = (MTL::Buffer *)
        jitc_metal_find_buffer((void *) in, &offset);
    if (in_buf)
        enc->useResource(in_buf, MTL::ResourceUsageRead);

    auto *out_buf = (MTL::Buffer *)
        jitc_metal_find_buffer(
            need_recursive ? temp_out : out, &offset);
    if (out_buf)
        enc->useResource(out_buf, MTL::ResourceUsageWrite);

    MTL::Size grid     = MTL::Size::Make(grid_size, 1, 1);
    MTL::Size group_sz = MTL::Size::Make(threads_per_group, 1, 1);
    enc->dispatchThreads(grid, group_sz);

    jitc_log(Debug,
             "MetalThreadState::block_reduce(%s, %s, size=%u, block_size=%u) "
             "-> kernel \"%s\", grid=%u, tg=%u, smem=%u",
             type_name[(int) vt], op_name, size, block_size,
             kernel_name, grid_size, threads_per_group, smem_bytes);

    if (need_recursive) {
        // Close encoder and use a new one for the recursive call,
        // avoiding a full commit+wait round-trip
        jitc_metal_close_encoder(this);

        block_reduce(vt, op, params.chunk_count, chunks_per_block,
                     temp_out, out);
        jitc_free(temp_out);
    }
}

void MetalThreadState::block_prefix_reduce(VarType vt, ReduceOp op,
                                           uint32_t size, uint32_t block_size,
                                           bool exclusive, bool reverse,
                                           const void *in, void *out) {
    DRJIT_METAL_SCOPED_POOL;
    uint32_t tsize = type_size[(int) vt];

    if (size == 0) return;
    if (block_size == 0 || block_size > size)
        jitc_raise("MetalThreadState::block_prefix_reduce(): invalid "
                   "block_size=%u for size=%u.", block_size, size);

    if (block_size == 1) {
        uint64_t z = 0;
        if (exclusive)
            memset_async(out, size, tsize, &z);
        else if (in != out)
            memcpy_async(out, in, (size_t) size * tsize);
        return;
    }

    // Signed add/mul/and/or use the unsigned kernel
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And)
        vt = metal_make_unsigned(vt);

    const char *type_suffix = metal_type_suffix(vt);
    const char *op_name = metal_op_name(op);
    if (!type_suffix || !op_name)
        jitc_raise("MetalThreadState::block_prefix_reduce(): unsupported "
                   "type=%s or op=%s.", type_name[(int) vt],
                   op_name ? op_name : "?");

    uint32_t block_count = (size + block_size - 1) / block_size;
    uint32_t chunk_size = round_pow2(std::min(block_size, 1024u));

    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name),
             "block_prefix_%s_%s_%u", op_name, type_suffix, chunk_size);

    auto *pso = jitc_metal_get_pipeline(this->device, kernel_name);
    if (!pso)
        jitc_raise("MetalThreadState::block_prefix_reduce(): kernel \"%s\" "
                   "not found.", kernel_name);

    uint32_t chunks_per_block = (block_size + chunk_size - 1) / chunk_size;
    bool need_two_pass = (chunks_per_block > 1);

    void *offsets_ptr = nullptr;

    if (need_two_pass) {
        // For blocks > chunk_size elements, use a GPU-based three-pass approach:
        //  Pass 1: Run prefix kernel (inclusive, no offsets) → per-chunk local scans
        //  Pass 2: block_reduce to get per-chunk totals, prefix-sum per block
        //  Pass 3: Re-run prefix kernel with offsets → final result

        uint32_t total_chunks = block_count * chunks_per_block;

        // Pass 1: inclusive prefix scan per-chunk for the original block_size
        // — the kernel handles multi-chunk blocks via chunks_per_block
        // addressing. The per-chunk totals (the last element of each chunk)
        // are extracted on the CPU after the pass; they feed the
        // exclusive-prefix-of-totals computation that produces the inter-chunk
        // offsets passed back into the final scan.
        //
        // We *must* write pass 1's output to a scratch buffer rather than the
        // caller's `out`, because callers frequently pass ``in == out`` (e.g.
        // ``mkperm`` on the bucket histogram). Re-using `out` would clobber
        // the original input that the final scan still needs to read.
        void *p1_out = jitc_malloc(JitBackend::Metal, (size_t) size * tsize);
        {
            struct {
                uint64_t in_p, out_p, offsets_p;
                uint32_t size_p, block_size_p, exclusive_p, reverse_p, cpb_p;
            } p1;
            p1.in_p = (uint64_t)(uintptr_t) in;
            p1.out_p = (uint64_t)(uintptr_t) p1_out;
            p1.offsets_p = 0;
            p1.size_p = size;
            p1.block_size_p = block_size;
            p1.exclusive_p = 0;  // inclusive for totals extraction
            p1.reverse_p = reverse ? 1 : 0;
            p1.cpb_p = chunks_per_block;

            uint32_t tpg = (chunk_size < 128) ? 128u : chunk_size;
            uint32_t cptg = (chunk_size < 128) ? (tpg / chunk_size) : 1u;
            uint32_t gg = (total_chunks + cptg - 1) / cptg;

            auto *enc = jitc_metal_acquire_compute_encoder(this);
            enc->setComputePipelineState(pso);
            enc->setBytes(&p1, sizeof(p1), 0);
            enc->setThreadgroupMemoryLength(tpg * tsize, 0);

            size_t off = 0;
            auto *ib = (MTL::Buffer *) jitc_metal_find_buffer((void*) in, &off);
            auto *ob = (MTL::Buffer *) jitc_metal_find_buffer(p1_out, &off);
            if (ib) enc->useResource(ib, MTL::ResourceUsageRead);
            if (ob) enc->useResource(ob, MTL::ResourceUsageWrite);

            enc->dispatchThreadgroups(MTL::Size::Make(gg, 1, 1),
                                      MTL::Size::Make(tpg, 1, 1));
        }
        // No GPU sync here: the new ``extract_chunk_totals`` kernel below
        // is enqueued onto the same command buffer and reads the output
        // of the pass-1 kernel via the implicit dependency on p1_out.

        // GPU gather: one thread per chunk reads its inclusive-scan tail
        // and writes a flat (total_chunks,) device buffer. Replaces the
        // legacy CPU readback + index loop + re-upload that cost two
        // command-buffer roundtrips.
        void *chunk_totals_d =
            jitc_malloc(JitBackend::Metal, (size_t) total_chunks * tsize);
        {
            char gather_name[40];
            snprintf(gather_name, sizeof(gather_name),
                     "extract_chunk_totals_b%u", tsize);
            auto *gather_pso =
                jitc_metal_get_pipeline(this->device, gather_name);
            if (!gather_pso)
                jitc_raise("MetalThreadState::block_prefix_reduce(): "
                           "gather kernel \"%s\" not found.", gather_name);

            struct {
                uint64_t in_p, out_p;
                uint32_t size_p, block_size_p, chunk_size_p,
                         chunks_per_block_p, total_chunks_p, reverse_p;
            } gp;
            gp.in_p              = (uint64_t)(uintptr_t) p1_out;
            gp.out_p             = (uint64_t)(uintptr_t) chunk_totals_d;
            gp.size_p            = size;
            gp.block_size_p      = block_size;
            gp.chunk_size_p      = chunk_size;
            gp.chunks_per_block_p= chunks_per_block;
            gp.total_chunks_p    = total_chunks;
            gp.reverse_p         = reverse ? 1 : 0;

            uint32_t tpg = std::min(256u, round_pow2(total_chunks));
            if (tpg < 32) tpg = 32;
            uint32_t grid = total_chunks;

            auto *enc = jitc_metal_acquire_compute_encoder(this);
            enc->setComputePipelineState(gather_pso);
            enc->setBytes(&gp, sizeof(gp), 0);

            size_t off = 0;
            auto *ib = (MTL::Buffer *)
                jitc_metal_find_buffer(p1_out, &off);
            auto *ob = (MTL::Buffer *)
                jitc_metal_find_buffer(chunk_totals_d, &off);
            if (ib) enc->useResource(ib, MTL::ResourceUsageRead);
            if (ob) enc->useResource(ob, MTL::ResourceUsageWrite);

            enc->dispatchThreads(MTL::Size::Make(grid, 1, 1),
                                 MTL::Size::Make(tpg, 1, 1));
        }
        // p1_out is no longer needed past this point; the GPU still holds
        // a reference via the open command buffer, so jitc_free can drop
        // the host-side handle now.
        jitc_free(p1_out);

        // Exclusive prefix sum of chunk totals per block (always forward
        // order). The recursive call enqueues onto the same command
        // buffer; no sync needed here either.
        offsets_ptr =
            jitc_malloc(JitBackend::Metal, (size_t) total_chunks * tsize);
        block_prefix_reduce(vt, op, total_chunks, chunks_per_block,
                            true, false, chunk_totals_d, offsets_ptr);
        jitc_free(chunk_totals_d);
        // Fall through to re-run the prefix kernel with offsets
    }

    // Prepare parameter struct
    struct {
        uint64_t in, out, offsets;
        uint32_t size, block_size, exclusive, reverse, chunks_per_block;
    } params;
    params.in = (uint64_t)(uintptr_t) in;
    params.out = (uint64_t)(uintptr_t) out;
    params.offsets = offsets_ptr ? (uint64_t)(uintptr_t) offsets_ptr : 0;
    params.size = size;
    params.block_size = block_size;
    params.exclusive = exclusive ? 1 : 0;
    params.reverse = reverse ? 1 : 0;
    params.chunks_per_block = chunks_per_block;

    // Threadgroup configuration
    uint32_t threads_per_group =
        (chunk_size < 128) ? 128u : chunk_size;
    uint32_t chunks_per_tg =
        (chunk_size < 128) ? (threads_per_group / chunk_size) : 1u;
    uint32_t total_chunks = block_count * chunks_per_block;
    uint32_t grid_groups =
        (total_chunks + chunks_per_tg - 1) / chunks_per_tg;
    uint32_t smem_bytes = threads_per_group * tsize;

    auto *enc = jitc_metal_acquire_compute_encoder(this);
    enc->setComputePipelineState(pso);
    enc->setBytes(&params, sizeof(params), 0);
    enc->setThreadgroupMemoryLength(smem_bytes, 0);

    size_t off = 0;
    auto *in_buf = (MTL::Buffer *)
        jitc_metal_find_buffer((void *) in, &off);
    auto *out_buf = (MTL::Buffer *)
        jitc_metal_find_buffer(out, &off);
    if (in_buf) enc->useResource(in_buf, MTL::ResourceUsageRead);
    if (out_buf) enc->useResource(out_buf, MTL::ResourceUsageWrite);
    if (offsets_ptr) {
        auto *off_buf = (MTL::Buffer *)
            jitc_metal_find_buffer(offsets_ptr, &off);
        if (off_buf) enc->useResource(off_buf, MTL::ResourceUsageRead);
    }

    enc->dispatchThreadgroups(MTL::Size::Make(grid_groups, 1, 1),
                              MTL::Size::Make(threads_per_group, 1, 1));

    jitc_log(Debug,
             "MetalThreadState::block_prefix_reduce(%s, %s, size=%u, "
             "block_size=%u, exclusive=%d, reverse=%d) -> kernel \"%s\"",
             type_name[(int) vt], op_name, size, block_size,
             (int) exclusive, (int) reverse, kernel_name);

    if (need_two_pass)
        jitc_free(offsets_ptr);  // GPU lifetime tracked by the allocator
}

void MetalThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                  const void *ptr_2, uint32_t size,
                                  void *out) {
    DRJIT_METAL_SCOPED_POOL;
    if (size == 0) return;

    const char *type_suffix = metal_type_suffix(vt);
    if (!type_suffix)
        jitc_raise("MetalThreadState::reduce_dot(): unsupported type %s.",
                   type_name[(int) vt]);

    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name), "reduce_dot_%s", type_suffix);

    auto *pso = jitc_metal_get_pipeline(this->device, kernel_name);
    if (!pso)
        jitc_raise("MetalThreadState::reduce_dot(): kernel \"%s\" not found.",
                   kernel_name);

    uint32_t tsize = type_size[(int) vt];
    uint32_t thread_count = 1024;
    uint32_t block_count = (size + thread_count * 2 - 1) / (thread_count * 2);
    block_count = std::min(block_count, 128u);

    struct {
        uint64_t ptr_1, ptr_2;
        uint32_t size;
        uint64_t out;
    } params;
    params.ptr_1 = (uint64_t)(uintptr_t) ptr_1;
    params.ptr_2 = (uint64_t)(uintptr_t) ptr_2;
    params.size = size;

    void *temp = nullptr;
    if (block_count == 1) {
        params.out = (uint64_t)(uintptr_t) out;
    } else {
        temp = jitc_malloc(JitBackend::Metal, (size_t) block_count * tsize);
        params.out = (uint64_t)(uintptr_t) temp;
    }

    auto *enc = jitc_metal_acquire_compute_encoder(this);
    enc->setComputePipelineState(pso);
    enc->setBytes(&params, sizeof(params), 0);
    enc->setThreadgroupMemoryLength(thread_count * tsize, 0);

    size_t off = 0;
    auto *b1 = (MTL::Buffer *) jitc_metal_find_buffer((void *) ptr_1, &off);
    auto *b2 = (MTL::Buffer *) jitc_metal_find_buffer((void *) ptr_2, &off);
    auto *bo = (MTL::Buffer *) jitc_metal_find_buffer(
        block_count == 1 ? out : temp, &off);
    if (b1) enc->useResource(b1, MTL::ResourceUsageRead);
    if (b2) enc->useResource(b2, MTL::ResourceUsageRead);
    if (bo) enc->useResource(bo, MTL::ResourceUsageWrite);

    enc->dispatchThreadgroups(MTL::Size::Make(block_count, 1, 1),
                              MTL::Size::Make(thread_count, 1, 1));

    if (block_count > 1) {
        // No sync needed: block_reduce appends a kernel to the same
        // command buffer; Metal hazard tracking serialises the per-
        // block dot kernel and the reduce kernel on the shared `temp`.
        block_reduce(vt, ReduceOp::Add, block_count, block_count, temp, out);
        jitc_free(temp);
    }
}

void MetalThreadState::batched_gemm(VarType vt, bool At, bool Bt,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    const GemmBatch *batch,
                                    const void *A, const void *B, void *C) {
    DRJIT_METAL_SCOPED_POOL;

    bool type_ok = (vt == VarType::Float16 || vt == VarType::Float32);
    if (!type_ok) {
        if (vt == VarType::Int32 || vt == VarType::UInt32)
            jitc_raise("jit_batched_gemm(): integer GEMM unsupported on the "
                       "Metal backend (only Float16 and Float32 are "
                       "supported).");
        jitc_raise("MetalThreadState::batched_gemm(): unsupported type '%s'.",
                   type_name[(int) vt]);
    }

    if (At && Bt)
        jitc_raise("MetalThreadState::batched_gemm(): At=Bt=True should have "
                   "been rewritten by the caller.");

    uint32_t grid_count, reduce_count;
    if (!jitc_gemm_batch_counts(batch, grid_count, reduce_count))
        return;

    uint32_t tsize = type_size[(int) vt];

    // For the non-batched case (and grid_count==1, reduce_count==1 batched
    // case), use MPSMatrixMultiplication for hardware-accelerated GEMM.
    // For complex batched cases with reduce dims, fall back to a CPU loop
    // that dispatches MPS per batch slice.

    // MPS data type mapping
    // MPS uses column-major by default, but we can encode row-major by
    // swapping A↔B and M↔N: C = A@B (row-major) ≡ C^T = B^T @ A^T (col-major)
    // Since MPS expects column-major, we compute: C^T = B^T @ A^T
    // Which means: swap(A,B), swap(M,N), swap(At,Bt), negate transposes

    // After the swap for MPS column-major convention:
    // MPS sees: result(N×M) = B_mps(N×K) @ A_mps(K×M)
    // With transposes adjusted for the swap.

    auto *dev = (MTL::Device *) metal_device;

    // Commit any pending work without waiting — MPS opens its own
    // MPSCommandBuffer from the same queue, so FIFO submission order
    // already serialises it after our committed CB.
    this->barrier();

    size_t a_buf_offset = 0, b_buf_offset = 0, c_buf_offset = 0;
    auto *a_buf = (MTL::Buffer *)
        jitc_metal_find_buffer((void *) A, &a_buf_offset);
    auto *b_buf = (MTL::Buffer *)
        jitc_metal_find_buffer((void *) B, &b_buf_offset);
    auto *c_buf = (MTL::Buffer *)
        jitc_metal_find_buffer(C, &c_buf_offset);

    if (!a_buf || !b_buf || !c_buf)
        jitc_raise("MetalThreadState::batched_gemm(): could not resolve "
                   "buffer pointers.");

    GemmBatch batch_eff = batch ? *batch : GemmBatch{};

    // Zero the output buffer
    {
        size_t c_size = (size_t) grid_count * M * N * tsize;
        auto *queue = (MTL::CommandQueue *) metal_queue;
        auto *cb = queue->commandBuffer();
        auto *blit = cb->blitCommandEncoder();
        blit->fillBuffer(c_buf, NS::Range::Make(c_buf_offset, c_size), 0);
        blit->endEncoding();
        jitc_metal_commit_and_wait_tagged(cb, "batched_gemm.zero_init");
    }

    // Iterate over grid and reduce dimensions
    for (uint32_t g = 0; g < grid_count; ++g) {
        for (uint32_t r = 0; r < reduce_count; ++r) {
            // Compute batch offsets for A and B
            uint32_t a_offset_elems = 0, b_offset_elems = 0;
            uint32_t combined = g * reduce_count + r;
            uint32_t total_bdims = batch_eff.n_bdims + batch_eff.n_rdims;

            uint32_t idx = combined;
            for (int d = (int) total_bdims - 1; d >= 0; --d) {
                uint32_t ext = batch_eff.extent[d];
                uint32_t pos = idx % ext;
                idx /= ext;
                a_offset_elems += pos * batch_eff.a_stride[d];
                b_offset_elems += pos * batch_eff.b_stride[d];
            }

            size_t a_byte_off = a_buf_offset + (size_t) a_offset_elems * tsize;
            size_t b_byte_off = b_buf_offset + (size_t) b_offset_elems * tsize;
            size_t c_byte_off = c_buf_offset + (size_t) g * M * N * tsize;

            // Row strides (in bytes) for MPS
            // A is stored as (rows_a × cols_a) row-major
            uint32_t a_rows = At ? K : M, a_cols = At ? M : K;
            uint32_t b_rows = Bt ? N : K, b_cols = Bt ? K : N;

            // MPS data type (type_ok above restricts this to Float16 / Float32)
            int mps_type = (vt == VarType::Float16) ? (0x10000000 | 16)   // MPSDataTypeFloat16
                                                    : (0x10000000 | 32);  // MPSDataTypeFloat32

            double alpha = 1.0, beta = (r > 0) ? 1.0 : 0.0;

            jitc_metal_mps_gemm(
                dev, metal_queue,
                a_buf, a_byte_off,
                b_buf, b_byte_off,
                c_buf, c_byte_off,
                M, N, K, At, Bt,
                a_rows, a_cols, b_rows, b_cols,
                tsize, mps_type, alpha, beta);
        }
    }

    jitc_log(Debug,
             "MetalThreadState::batched_gemm(type=%s, At=%d, Bt=%d, "
             "M=%u, N=%u, K=%u, grid=%u, reduce=%u) via MPS",
             type_name[(int) vt], (int) At, (int) Bt,
             M, N, K, grid_count, reduce_count);
}

uint32_t MetalThreadState::compress(const uint8_t *in, uint32_t size,
                                   uint32_t *out) {
    DRJIT_METAL_SCOPED_POOL;
    if (size == 0) return 0;

    uint32_t tsize_count = sizeof(uint32_t);

    // Allocate a small buffer for the count result (shared mode for CPU readback)
    auto *dev_mtl = (MTL::Device *) metal_device;
    auto *count_buf =
        dev_mtl->newBuffer(tsize_count, MTL::ResourceStorageModeShared);

    if (size <= 4096) {
        // Small path: single threadgroup
        uint32_t thread_count = round_pow2((size + 3) / 4);
        uint32_t smem = thread_count * 2 * tsize_count;

        auto *pso = jitc_metal_get_pipeline(this->device, "compress_small");
        if (!pso)
            jitc_raise("MetalThreadState::compress(): compress_small kernel "
                       "not found.");

        struct {
            uint64_t in, out, scratch;
            uint32_t size, count_offset;
        } params;
        params.in = (uint64_t)(uintptr_t) in;
        params.out = (uint64_t)(uintptr_t) out;
        params.scratch = (uint64_t) count_buf->gpuAddress();
        params.size = size;
        params.count_offset = 0;

        auto *enc = jitc_metal_acquire_compute_encoder(this);
        enc->setComputePipelineState(pso);
        enc->setBytes(&params, sizeof(params), 0);
        enc->setThreadgroupMemoryLength(smem, 0);

        size_t off = 0;
        auto *in_buf = (MTL::Buffer *)
            jitc_metal_find_buffer((void *) in, &off);
        auto *out_buf = (MTL::Buffer *)
            jitc_metal_find_buffer((void *) out, &off);
        if (in_buf) enc->useResource(in_buf, MTL::ResourceUsageRead);
        if (out_buf) enc->useResource(out_buf, MTL::ResourceUsageWrite);
        enc->useResource(count_buf, MTL::ResourceUsageWrite);

        enc->dispatchThreads(MTL::Size::Make(thread_count, 1, 1),
                             MTL::Size::Make(thread_count, 1, 1));
    } else {
        // Large path: count → prefix sum → write
        uint32_t thread_count = 128;
        uint32_t items_per_thread = 16;
        uint32_t items_per_block = thread_count * items_per_thread;
        uint32_t block_count = (size + items_per_block - 1) / items_per_block;

        auto *count_pso = jitc_metal_get_pipeline(this->device, "compress_count");
        auto *write_pso = jitc_metal_get_pipeline(this->device, "compress_write");
        if (!count_pso || !write_pso)
            jitc_raise("MetalThreadState::compress(): kernels not found.");

        // Scratch for per-block counts
        void *scratch_ptr =
            jitc_malloc(JitBackend::Metal, (block_count + 1) * tsize_count);

        struct {
            uint64_t in, out, scratch;
            uint32_t size, count_offset;
        } params;
        params.in = (uint64_t)(uintptr_t) in;
        params.out = (uint64_t)(uintptr_t) out;
        params.scratch = (uint64_t)(uintptr_t) scratch_ptr;
        params.size = size;
        params.count_offset = 0;

        // Pass 1: count
        {
            auto *enc = jitc_metal_acquire_compute_encoder(this);
            enc->setComputePipelineState(count_pso);
            enc->setBytes(&params, sizeof(params), 0);
            enc->setThreadgroupMemoryLength(128, 0); // small shared for SIMD reduction

            size_t off = 0;
            auto *ib = (MTL::Buffer *)
                jitc_metal_find_buffer((void *) in, &off);
            auto *sb = (MTL::Buffer *)
                jitc_metal_find_buffer(scratch_ptr, &off);
            if (ib) enc->useResource(ib, MTL::ResourceUsageRead);
            if (sb) enc->useResource(sb, MTL::ResourceUsageWrite);

            enc->dispatchThreadgroups(MTL::Size::Make(block_count, 1, 1),
                                     MTL::Size::Make(thread_count, 1, 1));
        }

        // Save the per-block counts before overwriting with the prefix
        // sum. Stays on the same command buffer — block_prefix_reduce
        // and the compress_total kernel both consume it without any
        // explicit GPU sync.
        void *counts_copy = jitc_malloc(JitBackend::Metal, block_count * tsize_count);
        memcpy_async(counts_copy, scratch_ptr, block_count * tsize_count);

        // Exclusive prefix sum of per-block counts (in-place over scratch).
        block_prefix_reduce(VarType::UInt32, ReduceOp::Add, block_count,
                            block_count, true, false, scratch_ptr, scratch_ptr);

        // GPU kernel: total = exclusive[last] + counts[last] → count_buf.
        // Replaces the two memcpy GPU→CPU readbacks + a CPU add (3 syncs).
        {
            auto *total_pso =
                jitc_metal_get_pipeline(this->device, "compress_total_kernel");
            if (!total_pso)
                jitc_raise("MetalThreadState::compress(): "
                           "compress_total_kernel not found.");

            struct {
                uint64_t prefix, counts, out;
                uint32_t block_count;
            } tp;
            tp.prefix      = (uint64_t)(uintptr_t) scratch_ptr;
            tp.counts      = (uint64_t)(uintptr_t) counts_copy;
            tp.out         = (uint64_t) count_buf->gpuAddress();
            tp.block_count = block_count;

            auto *enc = jitc_metal_acquire_compute_encoder(this);
            enc->setComputePipelineState(total_pso);
            enc->setBytes(&tp, sizeof(tp), 0);

            size_t off = 0;
            auto *pb = (MTL::Buffer *)
                jitc_metal_find_buffer(scratch_ptr, &off);
            auto *cb_buf = (MTL::Buffer *)
                jitc_metal_find_buffer(counts_copy, &off);
            if (pb) enc->useResource(pb, MTL::ResourceUsageRead);
            if (cb_buf) enc->useResource(cb_buf, MTL::ResourceUsageRead);
            enc->useResource(count_buf, MTL::ResourceUsageWrite);

            enc->dispatchThreads(MTL::Size::Make(1, 1, 1),
                                 MTL::Size::Make(1, 1, 1));
        }
        jitc_free(counts_copy);

        // Pass 2: write indices (no sync — read_count below flushes once)
        {
            auto *enc = jitc_metal_acquire_compute_encoder(this);
            enc->setComputePipelineState(write_pso);
            enc->setBytes(&params, sizeof(params), 0);
            // Hillis-Steele exclusive scan needs 2 * thread_count uint slots
            enc->setThreadgroupMemoryLength(thread_count * 2 * tsize_count, 0);

            size_t off = 0;
            auto *ib = (MTL::Buffer *)
                jitc_metal_find_buffer((void *) in, &off);
            auto *ob = (MTL::Buffer *)
                jitc_metal_find_buffer((void *) out, &off);
            auto *sb = (MTL::Buffer *)
                jitc_metal_find_buffer(scratch_ptr, &off);
            if (ib) enc->useResource(ib, MTL::ResourceUsageRead);
            if (ob) enc->useResource(ob, MTL::ResourceUsageWrite);
            if (sb) enc->useResource(sb, MTL::ResourceUsageRead);

            enc->dispatchThreadgroups(MTL::Size::Make(block_count, 1, 1),
                                     MTL::Size::Make(thread_count, 1, 1));
        }

        jitc_free(scratch_ptr);
    }

    // Sync and read count
    jitc_metal_sync_tagged(this, "compress.read_count");
    uint32_t result = *(uint32_t *) count_buf->contents();
    count_buf->release();
    return result;
}

uint32_t MetalThreadState::block_mkperm(const uint32_t *values, uint32_t size,
                                        uint32_t block_size,
                                        uint32_t bucket_count, uint32_t *perm,
                                        uint32_t *offsets) {
    DRJIT_METAL_SCOPED_POOL;
    if (size == 0) return 0;
    if (bucket_count == 0)
        jitc_fail("MetalThreadState::block_mkperm(): bucket_count cannot be zero!");
    if (block_size != size)
        jitc_raise("MetalThreadState::block_mkperm(): per-block permutations "
                   "(block_size != size) are not yet implemented on Metal.");

    // Allocate and zero-initialize histogram
    uint32_t bucket_bytes = bucket_count * sizeof(uint32_t);
    void *buckets = jitc_malloc(JitBackend::Metal, bucket_bytes);

    // Zero the histogram
    {
        size_t off = 0;
        auto *bb = (MTL::Buffer *)
            jitc_metal_find_buffer(buckets, &off);
        auto *enc = jitc_metal_acquire_blit_encoder(this);
        enc->fillBuffer(bb, NS::Range::Make(off, bucket_bytes), 0);
    }

    auto *phase1_pso = jitc_metal_get_pipeline(this->device, "mkperm_phase_1");
    auto *phase3_pso = jitc_metal_get_pipeline(this->device, "mkperm_phase_3");
    if (!phase1_pso || !phase3_pso)
        jitc_raise("MetalThreadState::mkperm(): kernels not found.");

    struct {
        uint64_t values, buckets, perm;
        uint32_t size, bucket_count;
    } params;
    params.values = (uint64_t)(uintptr_t) values;
    params.buckets = (uint64_t)(uintptr_t) buckets;
    params.perm = (uint64_t)(uintptr_t) perm;
    params.size = size;
    params.bucket_count = bucket_count;

    // Phase 1: Build histogram via global atomics
    {
        auto *enc = jitc_metal_acquire_compute_encoder(this);
        enc->setComputePipelineState(phase1_pso);
        enc->setBytes(&params, sizeof(params), 0);

        size_t off = 0;
        auto *vb = (MTL::Buffer *)
            jitc_metal_find_buffer((void *) values, &off);
        auto *bb = (MTL::Buffer *)
            jitc_metal_find_buffer(buckets, &off);
        if (vb) enc->useResource(vb, MTL::ResourceUsageRead);
        if (bb) enc->useResource(bb, MTL::ResourceUsageRead |
                                          MTL::ResourceUsageWrite);

        uint32_t tg = std::min(1024u, round_pow2(size));
        enc->dispatchThreads(MTL::Size::Make(size, 1, 1),
                             MTL::Size::Make(tg, 1, 1));
    }

    // No sync between phases — block_prefix_reduce and the detect
    // kernel both append to the current command buffer; Metal hazard
    // tracking serialises access to `buckets` automatically.

    // Phase 2: Exclusive prefix sum over histogram
    block_prefix_reduce(VarType::UInt32, ReduceOp::Add, bucket_count,
                        bucket_count, true, false, buckets, buckets);

    // Phase 2.5 (optional): Detect non-empty buckets
    uint32_t unique_count = 0;
    if (offsets) {
        auto *detect_pso =
            jitc_metal_get_pipeline(this->device, "mkperm_detect_offsets");
        if (!detect_pso)
            jitc_raise("MetalThreadState::mkperm(): mkperm_detect_offsets "
                       "kernel not found.");

        auto *dev_mtl = (MTL::Device *) metal_device;
        auto *counter_buf =
            dev_mtl->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        *(uint32_t *) counter_buf->contents() = 0;

        // The kernel needs to write to ``offsets`` from the GPU. If the
        // caller passed a HostPinned (= Shared) buffer, we can hand its GPU
        // address directly. If they passed a plain Host pointer (CPU-only
        // memory, which the GPU cannot dereference), stage through a fresh
        // Shared buffer and copy the results back after the kernel runs.
        size_t off_off = 0;
        auto *ob_buf = (MTL::Buffer *)
            jitc_metal_find_buffer((void *) offsets, &off_off);
        size_t offsets_bytes = (size_t) bucket_count * 4u * sizeof(uint32_t);
        MTL::Buffer *staging_buf = nullptr;
        uint64_t offsets_gpu_addr;
        if (ob_buf) {
            offsets_gpu_addr = ob_buf->gpuAddress() + off_off;
        } else {
            staging_buf = dev_mtl->newBuffer(
                offsets_bytes, MTL::ResourceStorageModeShared);
            offsets_gpu_addr = staging_buf->gpuAddress();
        }

        struct {
            uint64_t buckets, offsets, counter;
            uint32_t bucket_count, perm_size;
        } oparams;
        oparams.buckets = (uint64_t)(uintptr_t) buckets;
        oparams.offsets = offsets_gpu_addr;
        oparams.counter = (uint64_t) counter_buf->gpuAddress();
        oparams.bucket_count = bucket_count;
        oparams.perm_size = size;

        auto *enc = jitc_metal_acquire_compute_encoder(this);
        enc->setComputePipelineState(detect_pso);
        enc->setBytes(&oparams, sizeof(oparams), 0);

        size_t off = 0;
        auto *bb = (MTL::Buffer *)
            jitc_metal_find_buffer(buckets, &off);
        if (bb) enc->useResource(bb, MTL::ResourceUsageRead);
        if (ob_buf)
            enc->useResource(ob_buf, MTL::ResourceUsageWrite);
        if (staging_buf)
            enc->useResource(staging_buf, MTL::ResourceUsageWrite);
        enc->useResource(counter_buf, MTL::ResourceUsageRead |
                                           MTL::ResourceUsageWrite);

        uint32_t tg = std::min(1024u, round_pow2(bucket_count));
        enc->dispatchThreads(MTL::Size::Make(bucket_count, 1, 1),
                             MTL::Size::Make(tg, 1, 1));

        jitc_metal_sync_tagged(this, "mkperm.detect");
        unique_count = *(uint32_t *) counter_buf->contents();

        if (staging_buf) {
            // Only the first `unique_count` entries were written; copy them
            // (each entry = uint4 = 16 bytes) back to the Host pointer.
            std::memcpy(offsets, staging_buf->contents(),
                        (size_t) unique_count * 4u * sizeof(uint32_t));
            staging_buf->release();
        }

        // Write unique_count into offsets[4 * bucket_count]
        offsets[4 * bucket_count] = unique_count;

        counter_buf->release();
    }

    // Phase 3: Scatter indices using atomics on the prefix-summed histogram
    //
    // ``perm`` may be a Host-only pointer (CPU memory, not GPU-accessible);
    // in that case stage through a Shared buffer and copy back afterwards.
    auto *dev_mtl_perm = (MTL::Device *) metal_device;
    size_t perm_off = 0;
    auto *pb_caller = (MTL::Buffer *)
        jitc_metal_find_buffer((void *) perm, &perm_off);
    MTL::Buffer *perm_staging = nullptr;
    uint64_t perm_gpu_addr;
    if (pb_caller) {
        perm_gpu_addr = pb_caller->gpuAddress() + perm_off;
    } else {
        perm_staging = dev_mtl_perm->newBuffer(
            (size_t) size * sizeof(uint32_t),
            MTL::ResourceStorageModeShared);
        perm_gpu_addr = perm_staging->gpuAddress();
    }
    {
        // Patch params.perm with the address the kernel will actually write
        // to (caller buffer or staging Shared buffer).
        params.perm = perm_gpu_addr;

        auto *enc = jitc_metal_acquire_compute_encoder(this);
        enc->setComputePipelineState(phase3_pso);
        enc->setBytes(&params, sizeof(params), 0);

        size_t off = 0;
        auto *vb = (MTL::Buffer *)
            jitc_metal_find_buffer((void *) values, &off);
        auto *bb = (MTL::Buffer *)
            jitc_metal_find_buffer(buckets, &off);
        if (vb) enc->useResource(vb, MTL::ResourceUsageRead);
        if (bb) enc->useResource(bb, MTL::ResourceUsageRead |
                                          MTL::ResourceUsageWrite);
        if (pb_caller)
            enc->useResource(pb_caller, MTL::ResourceUsageWrite);
        if (perm_staging)
            enc->useResource(perm_staging, MTL::ResourceUsageWrite);

        uint32_t tg = std::min(1024u, round_pow2(size));
        enc->dispatchThreads(MTL::Size::Make(size, 1, 1),
                             MTL::Size::Make(tg, 1, 1));
    }

    jitc_metal_sync_tagged(this, "mkperm.tail");
    if (perm_staging) {
        std::memcpy((void *) perm, perm_staging->contents(),
                    (size_t) size * sizeof(uint32_t));
        perm_staging->release();
    }
    jitc_free(buckets);
    return unique_count;
}

void MetalThreadState::aggregate(void *dst, AggregationEntry *agg,
                                 uint32_t size) {
    DRJIT_METAL_SCOPED_POOL;
    // ``agg`` is allocated via ``jit_malloc(Metal, shared=true)`` by both
    // the call infrastructure (call.cpp) and the freeze-replay machinery
    // (record_ts.cpp); it must be released with ``jit_free`` (NOT
    // ``std::free``) — they may track the pointer in their own bookkeeping.
    if (size == 0) { jitc_free(agg); return; }

    // Resolve the destination MTLBuffer (allocated by the caller via
    // jit_malloc(Metal) → registered in the buffer map).
    size_t dst_off = 0;
    auto *dst_buf = (MTL::Buffer *)
        jitc_metal_find_buffer(dst, &dst_off);
    if (!dst_buf)
        jitc_fail("MetalThreadState::aggregate(): unknown dst pointer.");

    // Stage entries in a Shared (CPU+GPU) buffer — no upload required.
    // The host-side AggregationEntry layout (16 bytes) matches the MSL
    // ``AggregationEntry`` declared in metal_kernels.metal.
    auto *dev = (MTL::Device *) metal_device;
    size_t entries_bytes = sizeof(AggregationEntry) * (size_t) size;
    auto *entries_buf = dev->newBuffer(agg, entries_bytes,
                                       MTL::ResourceStorageModeShared);

    // Look up the precompiled aggregate kernel from the per-device library.
    auto *pso = jitc_metal_get_pipeline(this->device, "aggregate_kernel");
    if (!pso) {
        entries_buf->release();
        jitc_fail("MetalThreadState::aggregate(): aggregate_kernel "
                  "pipeline missing.");
    }

    auto *enc = jitc_metal_acquire_compute_encoder(this);
    enc->setComputePipelineState(pso);
    enc->setBuffer(dst_buf, dst_off, 0);
    enc->setBuffer(entries_buf, 0, 1);
    enc->setBytes(&size, sizeof(uint32_t), 2);
    enc->useResource(dst_buf, MTL::ResourceUsageWrite);

    // Mark each src buffer (negative-size entries: src is a device pointer)
    // as resident so the GPU can dereference it.
    auto *entries_ptr = (const AggregationEntry *) entries_buf->contents();
    for (uint32_t i = 0; i < size; ++i) {
        const AggregationEntry &e = entries_ptr[i];
        if (e.size < 0 && e.src) {
            size_t off = 0;
            auto *src_buf = (MTL::Buffer *)
                jitc_metal_find_buffer((void *) e.src, &off);
            if (src_buf)
                enc->useResource(src_buf, MTL::ResourceUsageRead);
        }
    }

    // One thread per entry. Sizes are typically small (≤ a few hundred), so
    // a single threadgroup wide enough to cover everything is sufficient.
    uint32_t threads_per_group = std::min<uint32_t>(
        (uint32_t) pso->maxTotalThreadsPerThreadgroup(), size);
    if (threads_per_group == 0)
        threads_per_group = 1;
    enc->dispatchThreads(MTL::Size::Make(size, 1, 1),
                         MTL::Size::Make(threads_per_group, 1, 1));

    // Release the staging buffer once the GPU is finished with it. Attach
    // to the *current* command buffer (same pattern as memset_async).
    auto *cb = (MTL::CommandBuffer *) metal_command_buffer;
    cb->addCompletedHandler([entries_buf](MTL::CommandBuffer *) {
        entries_buf->release();
    });
}

void MetalThreadState::enqueue_host_func(void (*callback)(void *),
                                         void *payload) {
    DRJIT_METAL_SCOPED_POOL;
    auto *cb = jitc_metal_acquire_cmdbuf(this);
    cb->addCompletedHandler(
        [callback, payload](MTL::CommandBuffer *) { callback(payload); });
}

#endif // defined(DRJIT_ENABLE_METAL)
