#if defined(DRJIT_ENABLE_METAL)

#include "metal_ts.h"
#include "metal.h"
#include "log.h"
#include "var.h"
#include "malloc.h"
#include "io.h"
#include "util.h"

#include <cstdlib>

// Carbon (deprecated) defines a ThreadState type that conflicts with Dr.Jit.
// The following definition suppresses this.
#define __THREADS__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

MetalThreadState::~MetalThreadState() {
    // Drain queued work before the thread state goes away. flush() is a no-op
    // if no command buffer is open (e.g. a thread state that never recorded
    // work, or one deleted on a jitc_init_thread_state() error path).
    flush(/* wait = */ true);
}

// ============================================================================
//  Command buffer / encoder lifecycle
// ============================================================================

enum class MetalEncoderKind : uint32_t {
    None = 0,
    Compute,
    Blit,
    Acceleration
};

void *MetalThreadState::ensure_cmdbuf() {
    if (!metal_command_buffer) {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) metal_queue;
        metal_command_buffer = (__bridge_retained void *) [queue commandBuffer];
    }
    return metal_command_buffer;
}

void MetalThreadState::close_encoder() {
    if (!metal_encoder)
        return;
    id<MTLCommandEncoder> enc =
        (__bridge_transfer id<MTLCommandEncoder>) metal_encoder;
    [enc endEncoding];
    metal_encoder = nullptr;
    metal_encoder_kind = (uint32_t) MetalEncoderKind::None;
}

void *MetalThreadState::ensure_compute_encoder() {
    if ((MetalEncoderKind) metal_encoder_kind == MetalEncoderKind::Compute)
        return metal_encoder;
    close_encoder();
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ensure_cmdbuf();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    metal_encoder = (__bridge_retained void *) enc;
    metal_encoder_kind = (uint32_t) MetalEncoderKind::Compute;
    return metal_encoder;
}

void *MetalThreadState::ensure_blit_encoder() {
    if ((MetalEncoderKind) metal_encoder_kind == MetalEncoderKind::Blit)
        return metal_encoder;
    close_encoder();
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ensure_cmdbuf();
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    metal_encoder = (__bridge_retained void *) enc;
    metal_encoder_kind = (uint32_t) MetalEncoderKind::Blit;
    return metal_encoder;
}

void MetalThreadState::flush(bool wait) {
    @autoreleasepool {
        if (!metal_command_buffer)
            return;
        close_encoder();
        id<MTLCommandBuffer> cb =
            (__bridge_transfer id<MTLCommandBuffer>) metal_command_buffer;
        metal_command_buffer = nullptr;
        [cb commit];
        if (wait)
            [cb waitUntilCompleted];
    }
}

void MetalThreadState::barrier() {
    // barrier() is called after enqueueing kernels in jitc_eval(). The Metal
    // backend interprets it as a hint to submit the command buffer to the GPU
    // to improve overlapping of GPU execution with subsequent tracing steps.
    flush(/* wait = */ false);
}

// ============================================================================
//  Kernel launch
// ============================================================================

Task *MetalThreadState::launch(Kernel kernel, KernelKey & /*key*/,
                               XXH128_hash_t /*hash*/, uint32_t size,
                               std::vector<void *> &kernel_params,
                               const std::vector<uint32_t> & /*kernel_param_ids*/,
                               KernelHistoryEntry *kernel_history_entry) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>) kernel.metal.pipeline;
        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>)
                ensure_compute_encoder();

        [enc setComputePipelineState:pso];

        // Upload kernel arguments within the command buffer
        [enc setBytes:kernel_params.data()
               length:kernel_params.size() * sizeof(void *)
              atIndex:0];

        // Indicate Metal buffers used by this kernel
        for (size_t i = 1; i < kernel_params.size(); ++i) {
            size_t offset = 0;
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>) jitc_metal_find_buffer(
                kernel_params[i], &offset);
            [enc useResource:buf
                       usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }

        // Mark buffers referenced by vcall data sections
        for (void *ptr : metal_call_resources) {
            size_t off = 0;
            id<MTLBuffer> buf =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer(ptr, &off);
            [enc useResource:buf usage:MTLResourceUsageRead];
        }
        metal_call_resources.clear();

        // Bind per-scene TLAS + IFT for every scene referenced by this kernel.
        // The kernel was generated with ``accel_<i> [[buffer(1+i)]]`` for each
        // scene at slot ``i`` in ``kernel.metal.scenes``, plus (optionally)
        // ``ift_<i>`` at packed slots in ``[1+N, 1+N+M)`` for scenes that have an
        // intersection function library.
        auto pick_scene = [](void *cand) -> MetalScene * {
            if (!jitc_metal_is_live_scene(cand))
                return nullptr;
            auto *s = (MetalScene *) cand;
            return s->tlas ? s : nullptr;
        };

        uint32_t n_scenes = kernel.metal.scene_count;
        // Pass 1: bind accels at slots [1, 1+N).
        std::vector<MetalScene *> resolved(n_scenes, nullptr);
        for (uint32_t i = 0; i < n_scenes; ++i) {
            auto *captured = (MetalScene *) kernel.metal.scenes[i];
            MetalScene *scene = pick_scene(captured);

            if (!scene) {
                // Captured scene has gone stale. Slot 0 has a fallback to the
                // ThreadState's active scene; slots >= 1 have no meaningful
                // fallback — surface a Warn and leave the slot unbound.
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

            id<MTLAccelerationStructure> tlas =
                (__bridge id<MTLAccelerationStructure>) scene->tlas;
            [enc setAccelerationStructure:tlas atBufferIndex:1u + i];
            [enc useResource:tlas usage:MTLResourceUsageRead];

            for (void *res : scene->resources) {
                if (res)
                    [enc useResource:(__bridge id<MTLResource>) res
                               usage:MTLResourceUsageRead];
            }
        }

        // Pass 2: bind IFTs at slots [1+N, 1+N+M).
        uint32_t ift_slot = 1u + n_scenes;
        for (uint32_t i = 0; i < n_scenes; ++i) {
            auto *captured = (MetalScene *) kernel.metal.scenes[i];
            bool kernel_reserved_ift =
                captured && captured->intersection_fn_library;
            if (!kernel_reserved_ift)
                continue;

            MetalScene *scene = resolved[i];
            if (scene && scene->intersection_fn_library) {
                id<MTLIntersectionFunctionTable> ift =
                    (__bridge id<MTLIntersectionFunctionTable>)
                        jitc_metal_get_or_create_ift_for_scene(
                            scene, (__bridge void *) pso);
                if (ift) {
                    [enc setIntersectionFunctionTable:ift atBufferIndex:ift_slot];
                    [enc useResource:ift usage:MTLResourceUsageRead];
                    for (void *bp : scene->intersection_fn_buffers) {
                        id<MTLBuffer> buf = (__bridge id<MTLBuffer>) bp;
                        if (buf)
                            [enc useResource:buf usage:MTLResourceUsageRead];
                    }
                }
            }
            ift_slot++;
        }

        uint32_t threads_per_group = std::min<uint32_t>(
            (uint32_t) pso.maxTotalThreadsPerThreadgroup, metal_max_threads);
        threads_per_group =
            (threads_per_group / metal_simd_width) * metal_simd_width;
        if (threads_per_group == 0)
            threads_per_group = metal_simd_width;

        [enc dispatchThreads:MTLSizeMake(size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

        if (kernel_history_entry) {
            kernel_history_entry->size = size;

            // Stash an extra retain on the current command buffer so we can read
            // its GPU times once it has executed (released in
            // jitc_metal_finalize_kernel_history_entry).
            if (metal_command_buffer) {
                id<MTLCommandBuffer> cb =
                    (__bridge id<MTLCommandBuffer>) metal_command_buffer;
                kernel_history_entry->task = (__bridge_retained void *) cb;
            }
            state.kernel_history.append(*kernel_history_entry);
        }

        return nullptr;
    }
}

// ============================================================================
//  Memory operations
// ============================================================================

void MetalThreadState::memset_async(void *ptr, uint32_t size, uint32_t isize,
                                    const void *src) {
    @autoreleasepool {
        size_t offset = 0;
        id<MTLBuffer> buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(ptr, &offset);
        if (!buf)
            jitc_raise("MetalThreadState::memset_async(): unknown pointer.");

        if (isize == 1) {
            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>)
                    ensure_blit_encoder();
            [enc fillBuffer:buf
                      range:NSMakeRange(offset, (size_t) size)
                      value:*(const uint8_t *) src];
        } else {
            // Multi-byte pattern — build a small staging buffer with the
            // replicated pattern, then blit it. The blit encoder retains the
            // staging buffer until the command buffer completes, so we can let
            // ARC drop our reference at scope exit.
            size_t total = (size_t) size * (size_t) isize;
            id<MTLDevice> dev = (__bridge id<MTLDevice>) metal_device;
            id<MTLBuffer> staging =
                [dev newBufferWithLength:total options:MTLResourceStorageModeShared];
            uint8_t *dst = (uint8_t *) [staging contents];
            for (uint32_t i = 0; i < size; ++i)
                std::memcpy(dst + (size_t) i * isize, src, isize);

            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>)
                    ensure_blit_encoder();
            [enc copyFromBuffer:staging
                   sourceOffset:0
                       toBuffer:buf
              destinationOffset:offset
                           size:total];
        }
    }
}

void MetalThreadState::memcpy(void *dst, const void *src, size_t size) {
    memcpy_async(dst, src, size);
    flush(/* wait = */ true);
}

/// Asynchronous copy enqueued into the current command buffer.
void MetalThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    @autoreleasepool {
        size_t src_offset = 0, dst_offset = 0;
        id<MTLBuffer> src_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) src, &src_offset);
        id<MTLBuffer> dst_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(dst, &dst_offset);

        if (src_buf && dst_buf) {
            // GPU -> GPU blit
            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>)
                    ensure_blit_encoder();
            [enc copyFromBuffer:src_buf
                   sourceOffset:src_offset
                       toBuffer:dst_buf
              destinationOffset:dst_offset
                           size:size];
        } else if (!src_buf && dst_buf) {
            // CPU -> GPU upload via staging buffer
            void *staging = jitc_malloc(JitBackend::Metal, size, /*shared=*/true);
            std::memcpy(staging, src, size);
            size_t staging_off = 0;
            id<MTLBuffer> staging_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(staging, &staging_off);
            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>)
                    ensure_blit_encoder();
            [enc copyFromBuffer:staging_buf
                   sourceOffset:staging_off
                       toBuffer:dst_buf
              destinationOffset:dst_offset
                           size:size];
            jitc_free(staging);
        } else if (src_buf && !dst_buf) {
            // GPU → CPU readback: a completion handler copies staging → dst
            void *staging = jitc_malloc(JitBackend::Metal, size, /*shared=*/true);
            size_t staging_off = 0;
            id<MTLBuffer> staging_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(staging, &staging_off);
            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>)
                    ensure_blit_encoder();
            [enc copyFromBuffer:src_buf
                   sourceOffset:src_offset
                       toBuffer:staging_buf
              destinationOffset:staging_off
                           size:size];
            id<MTLCommandBuffer> cb =
                (__bridge id<MTLCommandBuffer>) metal_command_buffer;
            [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                std::memcpy(dst, staging, size);
            }];
            jitc_free(staging);
        } else {
            std::memcpy(dst, src, size);
        }
    }
}

void MetalThreadState::poke(void *dst, const void *src, uint32_t size) {
    @autoreleasepool {
        jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);
        size_t offset = 0;
        id<MTLBuffer> buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(dst, &offset);

        if (buf && [buf storageMode] == MTLStorageModeShared)
            std::memcpy((uint8_t *) [buf contents] + offset, src, size);
        else
            memcpy_async(dst, src, size);
    }
}

// ============================================================================
//  Reductions / scans / compaction
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

/// Resolve a Dr.Jit Metal pointer and return both its id<MTLBuffer> pointer
/// and the associated GPU memory address
static id<MTLBuffer> metal_resolve(const void *ptr, uint64_t *addr_out) {
    size_t off = 0;
    id<MTLBuffer> buf =
        (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) ptr, &off);
    *addr_out = (uint64_t) [buf gpuAddress] + off;
    return buf;
}

void MetalThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                    uint32_t block_size, const void *in,
                                    void *out) {
    @autoreleasepool {
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

        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)
                jitc_metal_get_pipeline(this->device, kernel_name);
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

        id<MTLBuffer> in_buf = metal_resolve(in, &params.in);
        params.size = size;
        params.block_size = block_size;
        params.chunk_count = block_count * chunks_per_block;

        bool need_recursive = (chunks_per_block > 1);
        void *temp_out = nullptr;

        if (need_recursive) {
            // Allocate a temporary buffer for the per-chunk outputs
            temp_out =
                jitc_malloc(JitBackend::Metal, (size_t) params.chunk_count * tsize);
        }
        id<MTLBuffer> out_buf =
            metal_resolve(need_recursive ? temp_out : out, &params.out);

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

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>)
                ensure_compute_encoder();
        [enc setComputePipelineState:pso];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        // Mark input/output buffers as resident (resolved above)
        if (in_buf)
            [enc useResource:in_buf usage:MTLResourceUsageRead];
        if (out_buf)
            [enc useResource:out_buf usage:MTLResourceUsageWrite];

        [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

        jitc_log(Debug,
                 "MetalThreadState::block_reduce(%s, %s, size=%u, block_size=%u) "
                 "-> kernel \"%s\", grid=%u, tg=%u, smem=%u",
                 type_name[(int) vt], op_name, size, block_size,
                 kernel_name, grid_size, threads_per_group, smem_bytes);

        if (need_recursive) {
            // Close encoder and use a new one for the recursive call, avoiding a
            // full commit+wait round-trip
            close_encoder();

            block_reduce(vt, op, params.chunk_count, chunks_per_block,
                         temp_out, out);
            jitc_free(temp_out);
        }
    }
}

void MetalThreadState::block_prefix_reduce(VarType vt, ReduceOp op,
                                           uint32_t size, uint32_t block_size,
                                           bool exclusive, bool reverse,
                                           const void *in, void *out) {
    @autoreleasepool {
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

        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)
                jitc_metal_get_pipeline(this->device, kernel_name);
        if (!pso)
            jitc_raise("MetalThreadState::block_prefix_reduce(): kernel \"%s\" "
                       "not found.", kernel_name);

        uint32_t chunks_per_block = (block_size + chunk_size - 1) / chunk_size;
        bool need_two_pass = (chunks_per_block > 1);

        void *offsets_ptr = nullptr;

        if (need_two_pass) {
            uint32_t total_chunks = block_count * chunks_per_block;

            // Pass 1: inclusive prefix scan per-chunk. Must write to a scratch
            // buffer rather than the caller's `out` (callers often pass in==out).
            void *p1_out = jitc_malloc(JitBackend::Metal, (size_t) size * tsize);
            {
                struct {
                    uint64_t in_p, out_p, offsets_p;
                    uint32_t size_p, block_size_p, exclusive_p, reverse_p, cpb_p;
                } p1;
                id<MTLBuffer> ib = metal_resolve(in, &p1.in_p);
                id<MTLBuffer> ob = metal_resolve(p1_out, &p1.out_p);
                p1.offsets_p = 0;
                p1.size_p = size;
                p1.block_size_p = block_size;
                p1.exclusive_p = 0;  // inclusive for totals extraction
                p1.reverse_p = reverse ? 1 : 0;
                p1.cpb_p = chunks_per_block;

                uint32_t tpg = (chunk_size < 128) ? 128u : chunk_size;
                uint32_t cptg = (chunk_size < 128) ? (tpg / chunk_size) : 1u;
                uint32_t gg = (total_chunks + cptg - 1) / cptg;

                id<MTLComputeCommandEncoder> enc =
                    (__bridge id<MTLComputeCommandEncoder>)
                        ensure_compute_encoder();
                [enc setComputePipelineState:pso];
                [enc setBytes:&p1 length:sizeof(p1) atIndex:0];
                [enc setThreadgroupMemoryLength:tpg * tsize atIndex:0];

                if (ib) [enc useResource:ib usage:MTLResourceUsageRead];
                if (ob) [enc useResource:ob usage:MTLResourceUsageWrite];

                [enc dispatchThreadgroups:MTLSizeMake(gg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            }

            // GPU gather: one thread per chunk reads its inclusive-scan tail and
            // writes a flat (total_chunks,) device buffer.
            void *chunk_totals_d =
                jitc_malloc(JitBackend::Metal, (size_t) total_chunks * tsize);
            {
                char gather_name[40];
                snprintf(gather_name, sizeof(gather_name),
                         "extract_chunk_totals_b%u", tsize);
                id<MTLComputePipelineState> gather_pso =
                    (__bridge id<MTLComputePipelineState>)
                        jitc_metal_get_pipeline(this->device, gather_name);
                if (!gather_pso)
                    jitc_raise("MetalThreadState::block_prefix_reduce(): "
                               "gather kernel \"%s\" not found.", gather_name);

                struct {
                    uint64_t in_p, out_p;
                    uint32_t size_p, block_size_p, chunk_size_p,
                             chunks_per_block_p, total_chunks_p, reverse_p;
                } gp;
                id<MTLBuffer> ib = metal_resolve(p1_out, &gp.in_p);
                id<MTLBuffer> ob = metal_resolve(chunk_totals_d, &gp.out_p);
                gp.size_p            = size;
                gp.block_size_p      = block_size;
                gp.chunk_size_p      = chunk_size;
                gp.chunks_per_block_p= chunks_per_block;
                gp.total_chunks_p    = total_chunks;
                gp.reverse_p         = reverse ? 1 : 0;

                uint32_t tpg = std::min(256u, round_pow2(total_chunks));
                if (tpg < 32) tpg = 32;
                uint32_t grid = total_chunks;

                id<MTLComputeCommandEncoder> enc =
                    (__bridge id<MTLComputeCommandEncoder>)
                        ensure_compute_encoder();
                [enc setComputePipelineState:gather_pso];
                [enc setBytes:&gp length:sizeof(gp) atIndex:0];

                if (ib) [enc useResource:ib usage:MTLResourceUsageRead];
                if (ob) [enc useResource:ob usage:MTLResourceUsageWrite];

                [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            }
            jitc_free(p1_out);

            // Exclusive prefix sum of chunk totals per block (always forward
            // order). The recursive call enqueues onto the same command buffer.
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
        id<MTLBuffer> in_buf = metal_resolve(in, &params.in);
        id<MTLBuffer> out_buf = metal_resolve(out, &params.out);
        params.offsets = 0;
        id<MTLBuffer> off_buf = nil;
        if (offsets_ptr)
            off_buf = metal_resolve(offsets_ptr, &params.offsets);
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

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>)
                ensure_compute_encoder();
        [enc setComputePipelineState:pso];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        if (in_buf) [enc useResource:in_buf usage:MTLResourceUsageRead];
        if (out_buf) [enc useResource:out_buf usage:MTLResourceUsageWrite];
        if (off_buf) [enc useResource:off_buf usage:MTLResourceUsageRead];

        [enc dispatchThreadgroups:MTLSizeMake(grid_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

        jitc_log(Debug,
                 "MetalThreadState::block_prefix_reduce(%s, %s, size=%u, "
                 "block_size=%u, exclusive=%d, reverse=%d) -> kernel \"%s\"",
                 type_name[(int) vt], op_name, size, block_size,
                 (int) exclusive, (int) reverse, kernel_name);

        if (need_two_pass)
            jitc_free(offsets_ptr);  // GPU lifetime tracked by the allocator
    }
}

void MetalThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                  const void *ptr_2, uint32_t size,
                                  void *out) {
    @autoreleasepool {
        if (size == 0) return;

        const char *type_suffix = metal_type_suffix(vt);
        if (!type_suffix)
            jitc_raise("MetalThreadState::reduce_dot(): unsupported type %s.",
                       type_name[(int) vt]);

        char kernel_name[64];
        snprintf(kernel_name, sizeof(kernel_name), "reduce_dot_%s", type_suffix);

        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)
                jitc_metal_get_pipeline(this->device, kernel_name);
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
        id<MTLBuffer> b1 = metal_resolve(ptr_1, &params.ptr_1);
        id<MTLBuffer> b2 = metal_resolve(ptr_2, &params.ptr_2);
        params.size = size;

        void *temp = nullptr;
        if (block_count != 1)
            temp = jitc_malloc(JitBackend::Metal, (size_t) block_count * tsize);
        id<MTLBuffer> bo = metal_resolve(block_count == 1 ? out : temp, &params.out);

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>)
                ensure_compute_encoder();
        [enc setComputePipelineState:pso];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        [enc setThreadgroupMemoryLength:thread_count * tsize atIndex:0];

        if (b1) [enc useResource:b1 usage:MTLResourceUsageRead];
        if (b2) [enc useResource:b2 usage:MTLResourceUsageRead];
        if (bo) [enc useResource:bo usage:MTLResourceUsageWrite];

        [enc dispatchThreadgroups:MTLSizeMake(block_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];

        if (block_count > 1) {
            block_reduce(vt, ReduceOp::Add, block_count, block_count, temp, out);
            jitc_free(temp);
        }
    }
}

void MetalThreadState::batched_gemm(VarType vt, bool At, bool Bt,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    const GemmBatch *batch,
                                    const void *A, const void *B, void *C) {
    @autoreleasepool {
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

        size_t a_buf_offset = 0, b_buf_offset = 0, c_buf_offset = 0;
        id<MTLBuffer> a_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) A, &a_buf_offset);
        id<MTLBuffer> b_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) B, &b_buf_offset);
        id<MTLBuffer> c_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(C, &c_buf_offset);

        if (!a_buf || !b_buf || !c_buf)
            jitc_raise("MetalThreadState::batched_gemm(): could not resolve buffers.");

        GemmBatch batch_eff = batch ? *batch : GemmBatch{};
        size_t a_mat_elems = (size_t) M * K, b_mat_elems = (size_t) K * N;

        // A batched GEMM is a sequence of individual matrix multiplies. This helper
        // steps through them: it turns a flat counter ``idx`` into the element
        // offsets ``A_off`` and ``B_off`` of the operand sub-matrices, walking the
        // ``ndims`` batch dimensions that start at ``base`` (the two callers pass
        // either the grid range or the reduce range). Dimension ``base`` varies
        // fastest. A zero stride marks a broadcast operand, whose offset stays put.
        auto decode = [&](uint32_t idx, uint32_t base, uint32_t ndims,
                          size_t &A_off, size_t &B_off) {
            for (uint32_t d = 0; d < ndims; ++d) {
                uint32_t dim = base + d, pos = idx % batch_eff.extent[dim];
                idx /= batch_eff.extent[dim];
                A_off += (size_t) pos * batch_eff.a_stride[dim];
                B_off += (size_t) pos * batch_eff.b_stride[dim];
            }
        };

        // MPS can multiply a whole stack of matrices in one dispatch, but only if
        // they are evenly spaced by a fixed stride. That holds when the operand is
        // one densely packed stack in grid-major order with no broadcasts:
        // ``expect`` is the dense stride for each grid dim (the running product of
        // inner extents times ``mat_elems``). Any mismatch — usually a broadcast
        // (stride 0) — forces the fallback of one MPS multiply per output tile.
        auto grid_dense = [&](const uint32_t *stride, size_t mat_elems) {
            size_t expect = mat_elems;
            for (uint32_t d = 0; d < batch_eff.n_bdims; ++d) {
                if (batch_eff.extent[d] == 1)
                    continue;
                if (stride[d] != expect)
                    return false;
                expect *= batch_eff.extent[d];
            }
            return true;
        };

        bool uniform = grid_dense(batch_eff.a_stride, a_mat_elems) &&
                       grid_dense(batch_eff.b_stride, b_mat_elems);

        // Matrices stacked per dispatch: the whole grid when densely packed,
        // otherwise one matrix per (grid, reduce) tile.
        uint32_t matrices = uniform ? grid_count : 1;

        // Physical (stored) shapes; MPS applies the transposes logically.
        uint32_t a_rows = At ? K : M, a_cols = At ? M : K;
        uint32_t b_rows = Bt ? N : K, b_cols = Bt ? K : N;
        MPSDataType dt = (vt == VarType::Float16) ? MPSDataTypeFloat16
                                                  : MPSDataTypeFloat32;

        MPSMatrixDescriptor *desc_a = [MPSMatrixDescriptor
            matrixDescriptorWithRows:a_rows columns:a_cols matrices:matrices
            rowBytes:(NSUInteger) a_cols * tsize
            matrixBytes:(NSUInteger) a_rows * a_cols * tsize dataType:dt];
        MPSMatrixDescriptor *desc_b = [MPSMatrixDescriptor
            matrixDescriptorWithRows:b_rows columns:b_cols matrices:matrices
            rowBytes:(NSUInteger) b_cols * tsize
            matrixBytes:(NSUInteger) b_rows * b_cols * tsize dataType:dt];
        MPSMatrixDescriptor *desc_c = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N matrices:matrices
            rowBytes:(NSUInteger) N * tsize
            matrixBytes:(NSUInteger) M * N * tsize dataType:dt];

        // Close any open compute/blit encoder first; MPS encodes directly onto the
        // command buffer.
        close_encoder();
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ensure_cmdbuf();
        id<MTLDevice> dev = (__bridge id<MTLDevice>) metal_device;

        // Encode one ``C = A @ B`` at the given byte offsets. ``accumulate`` picks
        // beta: the first reduce step overwrites (beta=0), later steps add (beta=1).
        // MPS bakes alpha/beta into the object, so cache one per beta; the MPSMatrix
        // views are cheap to recreate per call.
        MPSMatrixMultiplication *gemm_overwrite = nil, *gemm_accumulate = nil;
        auto encode = [&](size_t a_off, size_t b_off, size_t c_off, bool accumulate) {
            MPSMatrixMultiplication *gemm =
                accumulate ? gemm_accumulate : gemm_overwrite;
            if (!gemm) {
                gemm = [[MPSMatrixMultiplication alloc]
                    initWithDevice:dev transposeLeft:At transposeRight:Bt
                    resultRows:M resultColumns:N interiorColumns:K
                    alpha:1.0 beta:(accumulate ? 1.0 : 0.0)];
                (accumulate ? gemm_accumulate : gemm_overwrite) = gemm;
            }
            MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:a_buf
                offset:a_off descriptor:desc_a];
            MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:b_buf
                offset:b_off descriptor:desc_b];
            MPSMatrix *mc = [[MPSMatrix alloc] initWithBuffer:c_buf
                offset:c_off descriptor:desc_c];
            [gemm encodeToCommandBuffer:cb leftMatrix:ma
                rightMatrix:mb resultMatrix:mc];
        };

        if (uniform) {
            // One dispatch per reduce step, each spanning the full grid; reduce
            // steps accumulate into the same output batch.
            for (uint32_t r = 0; r < reduce_count; ++r) {
                size_t A_off = 0, B_off = 0;
                decode(r, batch_eff.n_bdims, batch_eff.n_rdims, A_off, B_off);
                encode(a_buf_offset + A_off * tsize, b_buf_offset + B_off * tsize,
                       c_buf_offset, r > 0);
            }
        } else {
            // One dispatch per (grid, reduce) point; reduce steps fold into the tile.
            for (uint32_t g = 0; g < grid_count; ++g) {
                size_t A_off_grid = 0, B_off_grid = 0;
                decode(g, 0, batch_eff.n_bdims, A_off_grid, B_off_grid);
                size_t c_off = c_buf_offset + (size_t) g * M * N * tsize;
                for (uint32_t r = 0; r < reduce_count; ++r) {
                    size_t A_off = A_off_grid, B_off = B_off_grid;
                    decode(r, batch_eff.n_bdims, batch_eff.n_rdims, A_off, B_off);
                    encode(a_buf_offset + A_off * tsize, b_buf_offset + B_off * tsize,
                           c_off, r > 0);
                }
            }
        }
    }
}

uint32_t MetalThreadState::compress(const uint8_t *in, uint32_t size,
                                   uint32_t *out) {
    @autoreleasepool {
        if (size == 0) return 0;

        uint32_t tsize_count = sizeof(uint32_t);

        // Allocate a small buffer for the count result (shared mode for CPU readback)
        id<MTLDevice> dev_mtl = (__bridge id<MTLDevice>) metal_device;
        id<MTLBuffer> count_buf =
            [dev_mtl newBufferWithLength:tsize_count
                                 options:MTLResourceStorageModeShared];

        if (size <= 4096) {
            // Small path: single threadgroup
            uint32_t thread_count = round_pow2((size + 3) / 4);
            uint32_t smem = thread_count * 2 * tsize_count;

            id<MTLComputePipelineState> pso =
                (__bridge id<MTLComputePipelineState>)
                    jitc_metal_get_pipeline(this->device, "compress_small");
            if (!pso)
                jitc_raise("MetalThreadState::compress(): compress_small kernel "
                           "not found.");

            struct {
                uint64_t in, out, scratch;
                uint32_t size, count_offset;
            } params;
            params.in = (uint64_t)(uintptr_t) in;
            params.out = (uint64_t)(uintptr_t) out;
            params.scratch = (uint64_t) [count_buf gpuAddress];
            params.size = size;
            params.count_offset = 0;

            id<MTLComputeCommandEncoder> enc =
                (__bridge id<MTLComputeCommandEncoder>)
                    ensure_compute_encoder();
            [enc setComputePipelineState:pso];
            [enc setBytes:&params length:sizeof(params) atIndex:0];
            [enc setThreadgroupMemoryLength:smem atIndex:0];

            size_t off = 0;
            id<MTLBuffer> in_buf =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) in, &off);
            id<MTLBuffer> out_buf =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) out, &off);
            if (in_buf) [enc useResource:in_buf usage:MTLResourceUsageRead];
            if (out_buf) [enc useResource:out_buf usage:MTLResourceUsageWrite];
            [enc useResource:count_buf usage:MTLResourceUsageWrite];

            [enc dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        } else {
            // Large path: count → prefix sum → write
            uint32_t thread_count = 128;
            uint32_t items_per_thread = 16;
            uint32_t items_per_block = thread_count * items_per_thread;
            uint32_t block_count = (size + items_per_block - 1) / items_per_block;

            id<MTLComputePipelineState> count_pso =
                (__bridge id<MTLComputePipelineState>)
                    jitc_metal_get_pipeline(this->device, "compress_count");
            id<MTLComputePipelineState> write_pso =
                (__bridge id<MTLComputePipelineState>)
                    jitc_metal_get_pipeline(this->device, "compress_write");
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
                id<MTLComputeCommandEncoder> enc =
                    (__bridge id<MTLComputeCommandEncoder>)
                        ensure_compute_encoder();
                [enc setComputePipelineState:count_pso];
                [enc setBytes:&params length:sizeof(params) atIndex:0];
                [enc setThreadgroupMemoryLength:128 atIndex:0]; // SIMD reduction

                size_t off = 0;
                id<MTLBuffer> ib =
                    (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) in, &off);
                id<MTLBuffer> sb =
                    (__bridge id<MTLBuffer>) jitc_metal_find_buffer(scratch_ptr, &off);
                if (ib) [enc useResource:ib usage:MTLResourceUsageRead];
                if (sb) [enc useResource:sb usage:MTLResourceUsageWrite];

                [enc dispatchThreadgroups:MTLSizeMake(block_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
            }

            // Save the per-block counts before overwriting with the prefix sum.
            void *counts_copy = jitc_malloc(JitBackend::Metal, block_count * tsize_count);
            memcpy_async(counts_copy, scratch_ptr, block_count * tsize_count);

            // Exclusive prefix sum of per-block counts (in-place over scratch).
            block_prefix_reduce(VarType::UInt32, ReduceOp::Add, block_count,
                                block_count, true, false, scratch_ptr, scratch_ptr);

            // GPU kernel: total = exclusive[last] + counts[last] → count_buf.
            {
                id<MTLComputePipelineState> total_pso =
                    (__bridge id<MTLComputePipelineState>)
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
                tp.out         = (uint64_t) [count_buf gpuAddress];
                tp.block_count = block_count;

                id<MTLComputeCommandEncoder> enc =
                    (__bridge id<MTLComputeCommandEncoder>)
                        ensure_compute_encoder();
                [enc setComputePipelineState:total_pso];
                [enc setBytes:&tp length:sizeof(tp) atIndex:0];

                size_t off = 0;
                id<MTLBuffer> pb =
                    (__bridge id<MTLBuffer>) jitc_metal_find_buffer(scratch_ptr, &off);
                id<MTLBuffer> cb_buf =
                    (__bridge id<MTLBuffer>) jitc_metal_find_buffer(counts_copy, &off);
                if (pb) [enc useResource:pb usage:MTLResourceUsageRead];
                if (cb_buf) [enc useResource:cb_buf usage:MTLResourceUsageRead];
                [enc useResource:count_buf usage:MTLResourceUsageWrite];

                [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            }
            jitc_free(counts_copy);

            // Pass 2: write indices (no sync — read_count below flushes once)
            {
                id<MTLComputeCommandEncoder> enc =
                    (__bridge id<MTLComputeCommandEncoder>)
                        ensure_compute_encoder();
                [enc setComputePipelineState:write_pso];
                [enc setBytes:&params length:sizeof(params) atIndex:0];
                // Hillis-Steele exclusive scan needs 2 * thread_count uint slots
                [enc setThreadgroupMemoryLength:thread_count * 2 * tsize_count atIndex:0];

                size_t off = 0;
                id<MTLBuffer> ib =
                    (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) in, &off);
                id<MTLBuffer> ob =
                    (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) out, &off);
                id<MTLBuffer> sb =
                    (__bridge id<MTLBuffer>) jitc_metal_find_buffer(scratch_ptr, &off);
                if (ib) [enc useResource:ib usage:MTLResourceUsageRead];
                if (ob) [enc useResource:ob usage:MTLResourceUsageWrite];
                if (sb) [enc useResource:sb usage:MTLResourceUsageRead];

                [enc dispatchThreadgroups:MTLSizeMake(block_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
            }

            jitc_free(scratch_ptr);
        }

        // Sync and read count
        flush(/* wait = */ true);
        uint32_t result = *(uint32_t *) [count_buf contents];
        return result;
    }
}

uint32_t MetalThreadState::block_mkperm(const uint32_t *values, uint32_t size,
                                        uint32_t block_size,
                                        uint32_t bucket_count, uint32_t *perm,
                                        uint32_t *offsets) {
    @autoreleasepool {
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
            id<MTLBuffer> bb =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer(buckets, &off);
            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>)
                    ensure_blit_encoder();
            [enc fillBuffer:bb range:NSMakeRange(off, bucket_bytes) value:0];
        }

        id<MTLComputePipelineState> phase1_pso =
            (__bridge id<MTLComputePipelineState>)
                jitc_metal_get_pipeline(this->device, "mkperm_phase_1");
        id<MTLComputePipelineState> phase3_pso =
            (__bridge id<MTLComputePipelineState>)
                jitc_metal_get_pipeline(this->device, "mkperm_phase_3");
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
            id<MTLComputeCommandEncoder> enc =
                (__bridge id<MTLComputeCommandEncoder>)
                    ensure_compute_encoder();
            [enc setComputePipelineState:phase1_pso];
            [enc setBytes:&params length:sizeof(params) atIndex:0];

            size_t off = 0;
            id<MTLBuffer> vb =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) values, &off);
            id<MTLBuffer> bb =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer(buckets, &off);
            if (vb) [enc useResource:vb usage:MTLResourceUsageRead];
            if (bb) [enc useResource:bb usage:MTLResourceUsageRead |
                                              MTLResourceUsageWrite];

            uint32_t tg = std::min(1024u, round_pow2(size));
            [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // Phase 2: Exclusive prefix sum over histogram (same command buffer).
        block_prefix_reduce(VarType::UInt32, ReduceOp::Add, bucket_count,
                            bucket_count, true, false, buckets, buckets);

        // Phase 2.5 (optional): Detect non-empty buckets
        uint32_t unique_count = 0;
        if (offsets) {
            id<MTLComputePipelineState> detect_pso =
                (__bridge id<MTLComputePipelineState>)
                    jitc_metal_get_pipeline(this->device, "mkperm_detect_offsets");
            if (!detect_pso)
                jitc_raise("MetalThreadState::mkperm(): mkperm_detect_offsets "
                           "kernel not found.");

            id<MTLDevice> dev_mtl = (__bridge id<MTLDevice>) metal_device;
            id<MTLBuffer> counter_buf =
                [dev_mtl newBufferWithLength:sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
            *(uint32_t *) [counter_buf contents] = 0;

            // The kernel writes to ``offsets`` from the GPU. If the caller passed
            // a HostPinned (= Shared) buffer, hand its GPU address directly.
            // Otherwise stage through a fresh Shared buffer and copy back.
            size_t off_off = 0;
            id<MTLBuffer> ob_buf =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) offsets, &off_off);
            size_t offsets_bytes = (size_t) bucket_count * 4u * sizeof(uint32_t);
            id<MTLBuffer> staging_buf = nil;
            uint64_t offsets_gpu_addr;
            if (ob_buf) {
                offsets_gpu_addr = [ob_buf gpuAddress] + off_off;
            } else {
                staging_buf = [dev_mtl newBufferWithLength:offsets_bytes
                                                   options:MTLResourceStorageModeShared];
                offsets_gpu_addr = [staging_buf gpuAddress];
            }

            struct {
                uint64_t buckets, offsets, counter;
                uint32_t bucket_count, perm_size;
            } oparams;
            oparams.buckets = (uint64_t)(uintptr_t) buckets;
            oparams.offsets = offsets_gpu_addr;
            oparams.counter = (uint64_t) [counter_buf gpuAddress];
            oparams.bucket_count = bucket_count;
            oparams.perm_size = size;

            id<MTLComputeCommandEncoder> enc =
                (__bridge id<MTLComputeCommandEncoder>)
                    ensure_compute_encoder();
            [enc setComputePipelineState:detect_pso];
            [enc setBytes:&oparams length:sizeof(oparams) atIndex:0];

            size_t off = 0;
            id<MTLBuffer> bb =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer(buckets, &off);
            if (bb) [enc useResource:bb usage:MTLResourceUsageRead];
            if (ob_buf)
                [enc useResource:ob_buf usage:MTLResourceUsageWrite];
            if (staging_buf)
                [enc useResource:staging_buf usage:MTLResourceUsageWrite];
            [enc useResource:counter_buf usage:MTLResourceUsageRead |
                                               MTLResourceUsageWrite];

            uint32_t tg = std::min(1024u, round_pow2(bucket_count));
            [enc dispatchThreads:MTLSizeMake(bucket_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

            flush(/* wait = */ true);
            unique_count = *(uint32_t *) [counter_buf contents];

            if (staging_buf) {
                // Only the first `unique_count` entries were written; copy them
                // (each entry = uint4 = 16 bytes) back to the Host pointer.
                std::memcpy(offsets, [staging_buf contents],
                            (size_t) unique_count * 4u * sizeof(uint32_t));
            }

            // Write unique_count into offsets[4 * bucket_count]
            offsets[4 * bucket_count] = unique_count;
        }

        // Phase 3: Scatter indices using atomics on the prefix-summed histogram.
        // ``perm`` may be a Host-only pointer; stage through a Shared buffer if so.
        id<MTLDevice> dev_mtl_perm = (__bridge id<MTLDevice>) metal_device;
        size_t perm_off = 0;
        id<MTLBuffer> pb_caller =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) perm, &perm_off);
        id<MTLBuffer> perm_staging = nil;
        uint64_t perm_gpu_addr;
        if (pb_caller) {
            perm_gpu_addr = [pb_caller gpuAddress] + perm_off;
        } else {
            perm_staging = [dev_mtl_perm
                newBufferWithLength:(size_t) size * sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
            perm_gpu_addr = [perm_staging gpuAddress];
        }
        {
            // Patch params.perm with the address the kernel will actually write to.
            params.perm = perm_gpu_addr;

            id<MTLComputeCommandEncoder> enc =
                (__bridge id<MTLComputeCommandEncoder>)
                    ensure_compute_encoder();
            [enc setComputePipelineState:phase3_pso];
            [enc setBytes:&params length:sizeof(params) atIndex:0];

            size_t off = 0;
            id<MTLBuffer> vb =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) values, &off);
            id<MTLBuffer> bb =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer(buckets, &off);
            if (vb) [enc useResource:vb usage:MTLResourceUsageRead];
            if (bb) [enc useResource:bb usage:MTLResourceUsageRead |
                                              MTLResourceUsageWrite];
            if (pb_caller)
                [enc useResource:pb_caller usage:MTLResourceUsageWrite];
            if (perm_staging)
                [enc useResource:perm_staging usage:MTLResourceUsageWrite];

            uint32_t tg = std::min(1024u, round_pow2(size));
            [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        flush(/* wait = */ true);
        if (perm_staging) {
            std::memcpy((void *) perm, [perm_staging contents],
                        (size_t) size * sizeof(uint32_t));
        }
        jitc_free(buckets);
        return unique_count;
    }
}

void MetalThreadState::aggregate(void *dst, AggregationEntry *agg,
                                 uint32_t size) {
    @autoreleasepool {
        if (size == 0)
            return;

        // Resolve the destination MTLBuffer.
        size_t dst_off = 0;
        id<MTLBuffer> dst_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(dst, &dst_off);
        if (!dst_buf)
            jitc_fail("MetalThreadState::aggregate(): unknown dst pointer.");

        // Stage entries in a Shared (CPU+GPU) buffer (copies the host bytes). The
        // compute encoder retains it until the command buffer completes, so we can
        // let ARC drop our reference at scope exit.
        id<MTLDevice> dev = (__bridge id<MTLDevice>) metal_device;
        size_t entries_bytes = sizeof(AggregationEntry) * (size_t) size;
        id<MTLBuffer> entries_buf =
            [dev newBufferWithBytes:agg
                             length:entries_bytes
                            options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)
                jitc_metal_get_pipeline(this->device, "aggregate_kernel");
        if (!pso)
            jitc_fail("MetalThreadState::aggregate(): aggregate_kernel "
                      "pipeline missing.");

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>)
                ensure_compute_encoder();
        [enc setComputePipelineState:pso];
        [enc setBuffer:dst_buf offset:dst_off atIndex:0];
        [enc setBuffer:entries_buf offset:0 atIndex:1];
        [enc setBytes:&size length:sizeof(uint32_t) atIndex:2];
        [enc useResource:dst_buf usage:MTLResourceUsageWrite];

        // Mark each src buffer (negative-size entries: src is a device pointer)
        // as resident so the GPU can dereference it.
        auto *entries_ptr = (const AggregationEntry *) [entries_buf contents];
        for (uint32_t i = 0; i < size; ++i) {
            const AggregationEntry &e = entries_ptr[i];
            if (e.size < 0 && e.src) {
                size_t off = 0;
                id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)
                    jitc_metal_find_buffer((void *) e.src, &off);
                if (src_buf)
                    [enc useResource:src_buf usage:MTLResourceUsageRead];
            }
        }

        // One thread per entry.
        uint32_t threads_per_group = std::min<uint32_t>(
            (uint32_t) pso.maxTotalThreadsPerThreadgroup, size);
        if (threads_per_group == 0)
            threads_per_group = 1;
        [enc dispatchThreads:MTLSizeMake(size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    }
}

void MetalThreadState::enqueue_host_func(void (*callback)(void *),
                                         void *payload) {
      @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ensure_cmdbuf();
        [cb addCompletedHandler:^(id<MTLCommandBuffer>) { callback(payload); }];
    }
}

#endif // defined(DRJIT_ENABLE_METAL)
