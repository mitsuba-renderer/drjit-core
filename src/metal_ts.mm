#include "metal_ts.h"
#include "metal.h"
#include "metal_tex.h"
#include "log.h"
#include "var.h"
#include "malloc.h"
#include "io.h"
#include "util.h"

#include <cstdlib>
#include <cstring>

// Carbon (deprecated) defines a ThreadState type that conflicts with Dr.Jit.
// The following definition suppresses this.
#define __THREADS__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_launch.h"

MetalThreadState::~MetalThreadState() {
    // Wait for any queued computation to finish
    flush(/* wait = */ true);
}

// ============================================================================
//  Command buffer / encoder lifecycle
//  ``ensure_*/close_encoder`` must be called from within an @autoreleasepool.
// ============================================================================

void *MetalThreadState::ensure_cmdbuf() {
    if (!metal_cb) {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) metal_queue;
        metal_cb = (__bridge_retained void *) [queue commandBuffer];
    }
    return metal_cb;
}

void MetalThreadState::close_encoder() {
    if (!metal_encoder)
        return;
    id<MTLCommandEncoder> enc =
        (__bridge_transfer id<MTLCommandEncoder>) metal_encoder;
    [enc endEncoding];
    metal_encoder = nullptr;
    metal_encoder_kind = MetalEncoderKind::None;
}

void *MetalThreadState::ensure_compute_encoder() {
    if (metal_encoder_kind == MetalEncoderKind::Compute)
        return metal_encoder;
    close_encoder();
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ensure_cmdbuf();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    metal_encoder = (__bridge_retained void *) enc;
    metal_encoder_kind = MetalEncoderKind::Compute;
    return metal_encoder;
}

void *MetalThreadState::ensure_blit_encoder() {
    if (metal_encoder_kind == MetalEncoderKind::Blit)
        return metal_encoder;
    close_encoder();
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ensure_cmdbuf();
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    metal_encoder = (__bridge_retained void *) enc;
    metal_encoder_kind = MetalEncoderKind::Blit;
    return metal_encoder;
}

void MetalThreadState::flush(bool wait) {
    @autoreleasepool {
        if (metal_cb) {
            // Commit the current command buffer
            close_encoder();
            id<MTLCommandBuffer> cb =
                (__bridge_transfer id<MTLCommandBuffer>) metal_cb;
            metal_cb = nullptr;
            [cb commit];

            // Keep track of the last submitted command buffer
            if (metal_last_cb)
                (void) (__bridge_transfer id<MTLCommandBuffer>) metal_last_cb;
            metal_last_cb = (__bridge_retained void *) cb;
        }

        // If requested, wait and release metal_last_cb
        if (wait && metal_last_cb) {
            id<MTLCommandBuffer> last =
                (__bridge_transfer id<MTLCommandBuffer>) metal_last_cb;
            metal_last_cb = nullptr;
            [last waitUntilCompleted];
        }
    }
}

void MetalThreadState::barrier() {
    // barrier() is called after enqueueing kernels in jitc_eval(). The Metal
    // backend interprets it as a hint to submit the command buffer to the GPU
    // to overlap GPU execution with subsequent tracing steps.
    flush(/* wait = */ false);
}

// ============================================================================
//  Kernel-history timing
//
//  Metal exposes GPU timing per command buffer. To correctly track individual
//  backend operations, we therefore package each in its own command buffer.
//  The RAII guard ``MetalHistoryScope`` below automates this process.
// ============================================================================

struct MetalHistoryScope {
    MetalThreadState *ts;
    KernelType type;
    uint32_t size;
    bool enabled, outermost, recorded = false;

    MetalHistoryScope(MetalThreadState *ts, KernelType type, uint32_t size)
        : ts(ts), type(type), size(size),
          enabled(jit_flags() & (uint32_t) JitFlag::KernelHistory),
          outermost(false) {
        if (!enabled)
            return;
        outermost = (ts->metal_history_depth++ == 0);
        if (outermost)
            ts->flush(/* wait = */ false);
    }

    /// Append the history entry for this operation's command buffer. Must
    /// be called before an operation synchronizes internally.
    void record() {
        if (!enabled || !outermost || recorded || !ts->metal_cb)
            return;
        recorded = true;

        KernelHistoryEntry entry = {};
        entry.backend = JitBackend::Metal;
        entry.type = type;
        entry.recording_mode = ts->recording_mode;
        entry.size = size;
        entry.input_count = 1;
        entry.output_count = 1;

        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ts->metal_cb;
        entry.task = (__bridge_retained void *) cb;
        state.kernel_history.append(entry);
    }

    ~MetalHistoryScope() {
        if (!enabled)
            return;
        ts->metal_history_depth--;
        if (outermost) {
            record();
            ts->flush(/* wait = */ false);
        }
    }
};

// ============================================================================
//  Shared launch helpers for the precompiled utility kernels
// ============================================================================

static const char *metal_op_name(ReduceOp op) {
    switch (op) {
        case ReduceOp::Add: return "add";
        case ReduceOp::Mul: return "mul";
        case ReduceOp::Min: return "min";
        case ReduceOp::Max: return "max";
        case ReduceOp::Or:  return "or";
        case ReduceOp::And: return "and";
        default: return "<unsupported>";
    }
}

static VarType make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

/// Resolve a result pointer that a kernel writes to: device allocations
/// resolve directly; host memory is staged through a fresh Shared allocation
/// (returned via ``*staging_out``) that the caller must copy back after
/// synchronizing and then release.
static id<MTLBuffer> metal_resolve_or_stage(void *ptr, size_t bytes,
                                            uint64_t *addr_out,
                                            void **staging_out) {
    size_t off = 0;
    id<MTLBuffer> buf =
        (__bridge id<MTLBuffer>) jitc_metal_find_buffer(ptr, &off);
    if (buf) {
        *addr_out = (uint64_t) [buf gpuAddress] + off;
        return buf;
    }
    *staging_out = jitc_malloc(JitBackend::Metal, bytes, /*shared=*/true);
    return metal_resolve(*staging_out, addr_out);
}

/// Lazily created block (prefix) reduction pipeline, or nil if the
/// (kind, op, type) combination is unsupported
static id<MTLComputePipelineState>
metal_reduce_pipeline(int device, MetalReduceKind kind, ReduceOp op,
                      VarType vt) {
    return (__bridge id<MTLComputePipelineState>)
        jitc_metal_block_reduce_pipeline(device, kind, op, vt);
}

/// Contiguous elements each thread of a chunked (prefix) reduction kernel
/// processes per loop iteration
static constexpr uint32_t MetalGrain = 4;

/// Elements covered by one full threadgroup (1024 threads at MetalGrain);
/// reduction blocks larger than this are split into chunks of this size
static constexpr uint32_t MetalChunkSize = 4096;

/// Threadgroup width that covers ``chunk_size`` elements at MetalGrain,
/// rounded up to a whole simdgroup (32) and clamped to the pipeline limit
static uint32_t metal_chunk_tg_width(id<MTLComputePipelineState> pso,
                                     uint32_t chunk_size) {
    return std::min(ceil_div(chunk_size, MetalGrain * 32u) * 32u,
                    (uint32_t) pso.maxTotalThreadsPerThreadgroup);
}

// Host-side mirrors of the kernel parameter structs in
// resources/metal_kernels.metal — the layouts must match exactly.

struct BlockReduceParams {
    uint64_t in, out;
    uint32_t size, block_size, chunk_size, chunks_per_block;
};

struct BlockPrefixReduceParams {
    uint64_t in, out, prefixes;
    uint32_t size, block_size, chunk_size, chunks_per_block;
    uint32_t exclusive, reverse;
};

struct ReduceDotParams {
    uint64_t in1, in2, out;
    uint32_t size, chunk_size;
};

struct CompressScatterParams {
    uint64_t in, prefix, out;
};

struct MkpermParams {
    uint64_t values, buckets, perm;
};

struct MkpermOffsetsParams {
    uint64_t buckets, offsets, counter;
    uint32_t bucket_count, perm_size, row_stride;
};

struct MkpermTinyParams {
    uint64_t values, buckets, perm;
    uint32_t size, size_per_block, bucket_count, block_size, rows_per_group;
};

struct MemsetParams {
    uint64_t dst, value;
};

struct ConvertParams {
    uint64_t src, dst;
};

// ============================================================================
//  Kernel launch
// ============================================================================

Task *MetalThreadState::launch(Kernel kernel, KernelKey & /*key*/,
                               XXH128_hash_t /*hash*/, uint32_t size,
                               std::vector<void *> &kernel_params,
                               const std::vector<uint32_t> & /*kernel_param_ids*/,
                               KernelHistoryEntry *kernel_history_entry) {
    @autoreleasepool {
        // Separate from prior launches if kernel history tracking is enabled
        if (kernel_history_entry)
            flush(/* wait = */ false);

        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>) kernel.metal.pipeline;
        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>)
                ensure_compute_encoder();

        [enc setComputePipelineState:pso];

        // Resolve and collect kernel-parameter resources. A buffer slot stays
        // a plain pointer; every other kind is replaced by the owner's live
        // gpuResourceID.
        const KernelParamInfo *info = kernel.param_info;
        for (uint32_t i = 1; i < kernel_params.size(); ++i) {
            ResourceKind pk = (ResourceKind) info[i].kind;
            void *owner = kernel_params[i];

            switch (pk) {
                case ResourceKind::Buffer: {
                    // Preserve read/write information for Metal's hazard tracker
                    metal_call_resources.push_back(
                        { owner, ResourceKind::Buffer, info[i].write != 0 });
                    break;
                }

                case ResourceKind::IFT: {
                    // The residency pass below resolves the table again via
                    // the scene's per-PSO cache (a cheap hit). It cannot move
                    // here: scenes queued by aggregate() reach only that pass.
                    id<MTLIntersectionFunctionTable> ift =
                        (__bridge id<MTLIntersectionFunctionTable>)
                            jitc_metal_get_or_create_ift_for_scene(
                                (MetalScene *) owner, (__bridge void *) pso);
                    kernel_params[i] = ift ? (void *) (uintptr_t)
                        memcpy_cast<uint64_t>(ift.gpuResourceID) : nullptr;
                    break;
                }

                case ResourceKind::Texture: {
                    bool write = ((MetalTexResource *) owner)->parent->writable;
                    metal_call_resources.push_back(
                        { owner, ResourceKind::Texture, write });
                    void *rid = nullptr;
                    jitc_metal_resource_id(owner, pk, &rid);
                    kernel_params[i] = rid;
                    break;
                }

                case ResourceKind::Accel:
                    metal_call_resources.push_back({ owner, pk, /*write=*/false });
                    [[fallthrough]];
                case ResourceKind::Sampler: {
                    void *rid = nullptr;
                    jitc_metal_resource_id(owner, pk, &rid);
                    kernel_params[i] = rid;
                    break;
                }
            }
        }

        // Append the kernel's visible function table as a trailing bindless
        // ``params.args[]`` slot when it performs indirect calls.
        if (id<MTLVisibleFunctionTable> vft =
                (__bridge id<MTLVisibleFunctionTable>) kernel.metal.call_table_vft) {
            kernel_params.push_back((void *) (uintptr_t)
                memcpy_cast<uint64_t>(vft.gpuResourceID));
            [enc useResource:vft usage:MTLResourceUsageRead];
        }

        // Prefer to include the parameters directly in the command buffer if
        // <= 4KiB, otherwise stage them in a buffer.
        size_t params_bytes = kernel_params.size() * sizeof(void *);
        if (params_bytes <= 4096) {
            [enc setBytes:kernel_params.data() length:params_bytes atIndex:0];
        } else {
            void *staging =
                jitc_malloc(JitBackend::Metal, params_bytes, /*shared=*/true);
            std::memcpy(staging, kernel_params.data(), params_bytes);
            size_t staging_off = 0;
            id<MTLBuffer> staging_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(staging, &staging_off);
            [enc setBuffer:staging_buf offset:staging_off atIndex:0];
            jitc_free(staging);
        }

        // Make every resource this kernel touches resident on the encoder
        for (const CallResource &res : metal_call_resources) {
            switch (res.kind) {
                case ResourceKind::Buffer: {
                    size_t off = 0;
                    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)
                        jitc_metal_find_buffer(res.ptr, &off);
                    [enc useResource:buf
                               usage:MTLResourceUsageRead |
                                     (res.write ? MTLResourceUsageWrite : 0)];
                    break;
                }
                case ResourceKind::Accel:
                case ResourceKind::IFT: {
                    // Make the scene's TLAS, its BLAS / vertex / index buffers,
                    // and (for custom-primitive scenes) its IFT + per-entry
                    // buffers resident.
                    auto *scene = (MetalScene *) res.ptr;
                    [enc useResource:(__bridge id<MTLAccelerationStructure>) scene->tlas
                               usage:MTLResourceUsageRead];
                    for (void *r : scene->resources)
                        if (r)
                            [enc useResource:(__bridge id<MTLResource>) r
                                       usage:MTLResourceUsageRead];
                    if (!scene->intersection_fn_library)
                        break;

                    id<MTLIntersectionFunctionTable> ift =
                        (__bridge id<MTLIntersectionFunctionTable>)
                            jitc_metal_get_or_create_ift_for_scene(
                                scene, (__bridge void *) pso);
                    if (!ift)
                        break;

                    [enc useResource:ift usage:MTLResourceUsageRead];
                    for (const IFTEntry &e : scene->intersection_fns)
                        if (id<MTLBuffer> buf = (__bridge id<MTLBuffer>) e.buffer)
                            [enc useResource:buf usage:MTLResourceUsageRead];
                    break;
                }

                case ResourceKind::Texture: {
                    id<MTLTexture> t = (__bridge id<MTLTexture>)
                        ((MetalTexResource *) res.ptr)->object;
                    [enc useResource:t
                               usage:MTLResourceUsageRead |
                                     (res.write ? MTLResourceUsageWrite : 0)];
                    break;
                }

                case ResourceKind::Sampler:
                    break; // Not a memory resource, skip.
            }
        }
        metal_call_resources.clear();

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

            if (metal_cb) {
                id<MTLCommandBuffer> cb =
                    (__bridge id<MTLCommandBuffer>) metal_cb;
                kernel_history_entry->task = (__bridge_retained void *) cb;
            }
            state.kernel_history.append(*kernel_history_entry);

            // Commit this kernel's command buffer so it is timed in isolation.
            flush(/* wait = */ false);
        }

        return nullptr;
    }
}

// ============================================================================
//  Memory operations
// ============================================================================

void MetalThreadState::memset_async(void *ptr, uint32_t size, uint32_t isize,
                                    const void *src) {
    if (size == 0)
        return;

    @autoreleasepool {
        MetalHistoryScope hist(this, KernelType::Memset, size);

        jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
                   (uintptr_t) ptr, isize, size);

        size_t offset = 0;
        id<MTLBuffer> buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(ptr, &offset);

        if (isize == 1) {
            // Single-byte pattern: a blit fill is the most direct option.
            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>)
                    ensure_blit_encoder();
            [enc fillBuffer:buf
                      range:NSMakeRange(offset, (size_t) size)
                      value:*(const uint8_t *) src];
        } else {
            // Multi-byte pattern: write one element per thread via a dedicated
            // memset kernel for the element width (2, 4, or 8 bytes).
            MetalKernel kernel = isize == 2 ? MetalKernel::MemsetU16
                               : isize == 4 ? MetalKernel::MemsetU32
                                            : MetalKernel::MemsetU64;

            MemsetParams params {};
            std::memcpy(&params.value, src, isize); // low ``isize`` bytes
            params.dst = (uint64_t) [buf gpuAddress] + offset;

            metal_dispatch_threads(this, metal_pipeline(this, kernel), params,
                                   {{ buf, MTLResourceUsageWrite }}, size);
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
        MetalHistoryScope hist(this, KernelType::Memcpy, (uint32_t) size);

        size_t src_off = 0, dst_off = 0;
        id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)
            jitc_metal_find_buffer((void *) src, &src_off);
        id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)
            jitc_metal_find_buffer(dst, &dst_off);

        if (!src_buf && !dst_buf) {
            std::memcpy(dst, src, size);
            return;
        }

        // Host-side endpoints are routed through a Shared staging buffer:
        // uploads fill it now, readbacks copy from it in a completion handler.
        void *staging = nullptr;
        bool readback = false;
        if (!src_buf) {
            // CPU -> GPU upload
            staging = jitc_malloc(JitBackend::Metal, size, /*shared=*/true);
            std::memcpy(staging, src, size);
            src_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(staging, &src_off);
        } else if (!dst_buf) {
            // GPU -> CPU readback
            readback = true;
            staging = jitc_malloc(JitBackend::Metal, size, /*shared=*/true);
            dst_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(staging, &dst_off);
        }

        id<MTLBlitCommandEncoder> enc =
            (__bridge id<MTLBlitCommandEncoder>) ensure_blit_encoder();
        [enc copyFromBuffer:src_buf
               sourceOffset:src_off
                   toBuffer:dst_buf
          destinationOffset:dst_off
                       size:size];

        if (readback) {
            id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) metal_cb;
            [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                std::memcpy(dst, staging, size);
            }];
        }

        // Deferred: the allocation stays live until the command buffer retires
        if (staging)
            jitc_free(staging);
    }
}

void MetalThreadState::poke(void *dst, const void *src, uint32_t size) {
    @autoreleasepool {
        MetalHistoryScope hist(this, KernelType::Poke, size);
        jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);
        memcpy_async(dst, src, size);
    }
}

// ============================================================================
//  Reductions / scans / compaction
// ============================================================================

void MetalThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                    uint32_t block_size, const void *in,
                                    void *out) {
    // See the documentation of the associated kernels in
    // 'resources/metal_kernels.metal' for the overall strategy.
    @autoreleasepool {
        if (size == 0)
            return;
        if (block_size == 0 || block_size > size)
            jitc_raise("jit_block_reduce(): invalid block size "
                       "(size=%u, block_size=%u)!", size, block_size);

        uint32_t tsize = type_size[(int) vt];
        if (block_size == 1) {
            memcpy_async(out, in, (size_t) size * tsize);
            return;
        }

        // Signed sum/product/bitwise reductions can use the unsigned kernels
        if (op == ReduceOp::Add || op == ReduceOp::Mul ||
            op == ReduceOp::Or || op == ReduceOp::And)
            vt = make_int_type_unsigned(vt);

        MetalHistoryScope hist(this, KernelType::BlockReduce, size);

        uint32_t block_count = ceil_div(size, block_size);

        // Below this block size, one thread serially reduces an entire block;
        // larger blocks are split into chunks that map to one threadgroup each
        constexpr uint32_t SmallThreshold = 256;
        bool small = block_size < SmallThreshold;

        id<MTLComputePipelineState> pso = metal_reduce_pipeline(
            device, small ? MetalReduceKind::Small : MetalReduceKind::Chunk,
            op, vt);
        if (!pso)
            jitc_raise("jit_block_reduce(): unsupported type '%s' for op '%s' "
                       "on the Metal backend.",
                       type_name[(int) vt], metal_op_name(op));

        uint32_t chunk_size = 0, chunks_per_block = 1,
                 chunk_count = block_count, tg_width = 0;
        if (!small) {
            chunk_size = std::min(block_size, MetalChunkSize);
            chunks_per_block = ceil_div(block_size, chunk_size);
            chunk_count = block_count * chunks_per_block;
            tg_width = metal_chunk_tg_width(pso, chunk_size);
        }

        // Multi-chunk blocks produce per-chunk partial results that a
        // recursive invocation reduces to the final output
        void *dst = out;
        if (chunks_per_block > 1)
            dst = jitc_malloc(JitBackend::Metal, (size_t) chunk_count * tsize);

        BlockReduceParams params;
        id<MTLBuffer> in_buf = metal_resolve(in, &params.in);
        id<MTLBuffer> out_buf = metal_resolve(dst, &params.out);
        params.size = size;
        params.block_size = block_size;
        params.chunk_size = chunk_size;
        params.chunks_per_block = chunks_per_block;

        jitc_log(Debug,
                 "jit_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
                 ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u, "
                 "chunk_size=%u, chunks_per_block=%u, tg_width=%u)",
                 (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
                 metal_op_name(op), size, block_size, block_count, chunk_size,
                 chunks_per_block, tg_width);

        if (small) {
            // One thread per block; the grid is sized to the data
            metal_dispatch_threads(this, pso, params,
                                   {{ in_buf, MTLResourceUsageRead },
                                    { out_buf, MTLResourceUsageWrite }},
                                   block_count);
        } else {
            // One threadgroup per (chunk, block) grid cell
            metal_dispatch_groups(this, pso, params,
                                  {{ in_buf, MTLResourceUsageRead },
                                   { out_buf, MTLResourceUsageWrite }},
                                  MTLSizeMake(chunks_per_block, block_count, 1),
                                  tg_width);
        }

        if (chunks_per_block > 1) {
            // Reduce the per-chunk partial results (recursion depth <= 2)
            block_reduce(vt, op, chunk_count, chunks_per_block, dst, out);
            jitc_free(dst);
        }
    }
}

void MetalThreadState::narrow_f32_to_f16(void *dst, const void *src,
                                         uint32_t size) {
    @autoreleasepool {
        if (size == 0)
            return;

        // No MetalHistoryScope here: this is an internal helper of the
        // float16 scatter-reduction path (it has no public entry point or
        // KernelType); its runtime is attributed to the enclosing operation.
        jitc_log(Debug, "jit_narrow_f32_to_f16(size=%u)", size);

        ConvertParams params;
        id<MTLBuffer> src_buf = metal_resolve(src, &params.src);
        id<MTLBuffer> dst_buf = metal_resolve(dst, &params.dst);
        if (!src_buf || !dst_buf)
            jitc_raise("jit_narrow_f32_to_f16(): buffer lookup failed.");

        metal_dispatch_threads(this,
                               metal_pipeline(this, MetalKernel::ConvertF32F16),
                               params,
                               {{ src_buf, MTLResourceUsageRead },
                                { dst_buf, MTLResourceUsageWrite }},
                               size);
    }
}

/// Shared implementation of the *reduce-then-scan* prefix-reduction
/// decomposition, used by ``block_prefix_reduce()`` and ``compress()``.
/// ``vt`` is the storage type (already mapped to the unsigned kernel set
/// where applicable) and ``vta`` the accumulator/chunk-prefix type, which
/// may be wider: f16 sums/products carry f32 chunk prefixes (matching the
/// CUDA backend's intermediate precision), and compress() widens its u8
/// mask scan to u32 counts.
static void metal_block_prefix_reduce(MetalThreadState *ts, VarType vt,
                                      VarType vta, ReduceOp op, uint32_t size,
                                      uint32_t block_size, bool exclusive,
                                      bool reverse, const void *in,
                                      void *out) {
    id<MTLComputePipelineState> scan_pso =
        metal_reduce_pipeline(ts->device, MetalReduceKind::Scan, op, vt);
    if (!scan_pso)
        jitc_raise("jit_block_prefix_reduce(): unsupported type '%s' for "
                   "op '%s' on the Metal backend.",
                   type_name[(int) vt], metal_op_name(op));

    // Unlike the chunked reduction kernel, the scan kernel does not loop:
    // each thread covers exactly MetalGrain slots, so a chunk must fit
    // within a single threadgroup of the scan pipeline.
    uint32_t block_count = ceil_div(size, block_size),
             chunk_size = std::min(
                 { block_size, MetalChunkSize,
                   (uint32_t) scan_pso.maxTotalThreadsPerThreadgroup *
                       MetalGrain }),
             chunks_per_block = ceil_div(block_size, chunk_size),
             chunk_count = block_count * chunks_per_block;

    uint32_t tg_width = metal_chunk_tg_width(scan_pso, chunk_size);

    jitc_log(Debug,
             "jit_block_prefix_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, exclusive=%i, "
             "reverse=%i, block_count=%u, chunk_size=%u, "
             "chunks_per_block=%u, tg_width=%u)",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             metal_op_name(op), size, block_size, (int) exclusive,
             (int) reverse, block_count, chunk_size, chunks_per_block,
             tg_width);

    // Blocks spanning multiple chunks need each chunk's exclusive prefix:
    // compute per-chunk totals with the block-reduction kernel (using the
    // identical chunk decomposition), then prefix-reduce them in place via a
    // recursive invocation. The totals/prefixes are kept in the accumulator
    // type ``vta``.
    void *temp = nullptr;
    if (chunks_per_block > 1) {
        id<MTLComputePipelineState> red_pso = metal_reduce_pipeline(
            ts->device,
            vta != vt ? MetalReduceKind::WideChunk : MetalReduceKind::Chunk,
            op, vt);
        if (!red_pso)
            jitc_raise("jit_block_prefix_reduce(): unsupported type '%s' "
                       "for op '%s' on the Metal backend.",
                       type_name[(int) vt], metal_op_name(op));

        temp = jitc_malloc(JitBackend::Metal,
                           (size_t) chunk_count * type_size[(int) vta]);

        BlockReduceParams rparams;
        id<MTLBuffer> in_buf = metal_resolve(in, &rparams.in);
        id<MTLBuffer> temp_buf = metal_resolve(temp, &rparams.out);
        rparams.size = size;
        rparams.block_size = block_size;
        rparams.chunk_size = chunk_size;
        rparams.chunks_per_block = chunks_per_block;

        // The reduction kernel loops as needed, so any threadgroup width
        // within its pipeline limit works
        metal_dispatch_groups(
            ts, red_pso, rparams,
            {{ in_buf, MTLResourceUsageRead },
             { temp_buf, MTLResourceUsageWrite }},
            MTLSizeMake(chunks_per_block, block_count, 1),
            metal_chunk_tg_width(red_pso, chunk_size));

        // In-place exclusive prefix reduction over the chunk totals; the
        // 'reverse' flag carries over so that each chunk receives the
        // combined total of its predecessors in scan order.
        ts->block_prefix_reduce(vta, op, chunk_count, chunks_per_block,
                                /* exclusive = */ true, reverse, temp, temp);
    }

    BlockPrefixReduceParams params;
    id<MTLBuffer> in_buf = metal_resolve(in, &params.in);
    id<MTLBuffer> out_buf = metal_resolve(out, &params.out);
    id<MTLBuffer> temp_buf = nil;
    params.prefixes = 0;
    if (temp)
        temp_buf = metal_resolve(temp, &params.prefixes);
    params.size = size;
    params.block_size = block_size;
    params.chunk_size = chunk_size;
    params.chunks_per_block = chunks_per_block;
    params.exclusive = exclusive ? 1u : 0u;
    params.reverse = reverse ? 1u : 0u;

    metal_dispatch_groups(ts, scan_pso, params,
                          {{ in_buf, MTLResourceUsageRead },
                           { out_buf, MTLResourceUsageWrite },
                           { temp_buf, MTLResourceUsageRead }},
                          MTLSizeMake(chunks_per_block, block_count, 1),
                          tg_width);

    if (temp)
        jitc_free(temp);
}

void MetalThreadState::block_prefix_reduce(VarType vt, ReduceOp op,
                                           uint32_t size, uint32_t block_size,
                                           bool exclusive, bool reverse,
                                           const void *in, void *out) {
    @autoreleasepool {
        if (size == 0)
            return;
        if (block_size == 0 || block_size > size)
            jitc_raise("jit_block_prefix_reduce(): invalid "
                       "block_size=%u for size=%u.", block_size, size);

        uint32_t tsize = type_size[(int) vt];
        if (block_size == 1) {
            if (exclusive) {
                uint64_t ident = jitc_reduce_identity(vt, op);
                memset_async(out, size, tsize, &ident);
            } else if (in != out) {
                memcpy_async(out, in, (size_t) size * tsize);
            }
            return;
        }

        // Signed sum/product/bitwise reductions can use the unsigned kernels
        if (op == ReduceOp::Add || op == ReduceOp::Mul ||
            op == ReduceOp::Or || op == ReduceOp::And)
            vt = make_int_type_unsigned(vt);

        MetalHistoryScope hist(this, KernelType::BlockPrefixReduce, size);

        // f16 sums/products accumulate in f32
        VarType vta = vt;
        if (vt == VarType::Float16 &&
            (op == ReduceOp::Add || op == ReduceOp::Mul))
            vta = VarType::Float32;

        metal_block_prefix_reduce(this, vt, vta, op, size, block_size,
                                  exclusive, reverse, in, out);
    }
}

void MetalThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                  const void *ptr_2, uint32_t size,
                                  void *out) {
    @autoreleasepool {
        if (size == 0) return;

        id<MTLComputePipelineState> pso = metal_reduce_pipeline(
            device, MetalReduceKind::Dot, ReduceOp::Add, vt);
        if (!pso)
            jitc_raise("jit_reduce_dot(): unsupported type %s.",
                       type_name[(int) vt]);

        MetalHistoryScope hist(this, KernelType::Dot, size);

        // One threadgroup produces a partial dot product per chunk; a regular
        // sum block reduction then combines the partial results.
        uint32_t chunk_size = std::min(size, MetalChunkSize),
                 chunk_count = ceil_div(size, chunk_size);

        jitc_log(Debug, "jit_reduce_dot(" DRJIT_PTR ", " DRJIT_PTR
                 ", type=%s, size=%u, chunk_count=%u)",
                 (uintptr_t) ptr_1, (uintptr_t) ptr_2,
                 type_name[(int) vt], size, chunk_count);

        void *dst = out;
        if (chunk_count > 1)
            dst = jitc_malloc(JitBackend::Metal,
                              (size_t) chunk_count * type_size[(int) vt]);

        ReduceDotParams params;
        id<MTLBuffer> in1_buf = metal_resolve(ptr_1, &params.in1);
        id<MTLBuffer> in2_buf = metal_resolve(ptr_2, &params.in2);
        id<MTLBuffer> out_buf = metal_resolve(dst, &params.out);
        params.size = size;
        params.chunk_size = chunk_size;

        metal_dispatch_groups(this, pso, params,
                              {{ in1_buf, MTLResourceUsageRead },
                               { in2_buf, MTLResourceUsageRead },
                               { out_buf, MTLResourceUsageWrite }},
                              MTLSizeMake(chunk_count, 1, 1),
                              metal_chunk_tg_width(pso, chunk_size));

        if (chunk_count > 1) {
            block_reduce(vt, ReduceOp::Add, chunk_count, chunk_count, dst, out);
            jitc_free(dst);
        }
    }
}

void MetalThreadState::batched_gemm(VarType vt, bool At, bool Bt,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    const GemmBatch *batch,
                                    const void *A, const void *B, void *C) {
    @autoreleasepool {
        // Float16 / Float32 / Float64 are validated centrally in
        // ``jitc_batched_gemm()``. Apple GPUs have no FP64, so the Metal
        // backend additionally rejects Float64.
        if (vt == VarType::Float64)
            jitc_raise("jit_batched_gemm(): the Metal backend does not support "
                       "double precision (Float64) matrix multiplication. Use "
                       "Float16 or Float32 instead.");

        if (At && Bt)
            jitc_raise("jit_batched_gemm(): At=Bt=True should have "
                       "been rewritten by the caller.");

        uint32_t grid_count, reduce_count;
        if (!jitc_gemm_batch_counts(batch, grid_count, reduce_count))
            return;

        // Match the CUDA backend: report the total output element count.
        MetalHistoryScope hist(this, KernelType::BatchedGemm,
                               grid_count * M * N);

        jitc_log(Debug,
                 "jit_batched_gemm(" DRJIT_PTR ", " DRJIT_PTR " -> " DRJIT_PTR
                 ", type=%s, At=%i, Bt=%i, M=%u, N=%u, K=%u, grid=%u, "
                 "reduce=%u).", (uintptr_t) A, (uintptr_t) B, (uintptr_t) C,
                 type_name[(int) vt], (int) At, (int) Bt, M, N, K,
                 grid_count, reduce_count);

        uint32_t tsize = type_size[(int) vt];

        size_t a_buf_offset = 0, b_buf_offset = 0, c_buf_offset = 0;
        id<MTLBuffer> a_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) A, &a_buf_offset);
        id<MTLBuffer> b_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) B, &b_buf_offset);
        id<MTLBuffer> c_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(C, &c_buf_offset);

        if (!a_buf || !b_buf || !c_buf)
            jitc_raise("jit_batched_gemm(): could not resolve buffers.");

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
        // ``expect`` is the dense stride for each grid dimension. A mismatches
        // reuquires falling back to one MPS multiply per output tile.
        auto is_dense_grid = [&](const uint32_t *stride, size_t mat_elems) {
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

        bool uniform = is_dense_grid(batch_eff.a_stride, a_mat_elems) &&
                       is_dense_grid(batch_eff.b_stride, b_mat_elems);

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

        // Recorded via the internal flush(wait=true) below (compress syncs to
        // read back the total count).
        MetalHistoryScope hist(this, KernelType::Compress, size);

        jitc_log(Debug, "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR ", size=%u)",
                 (uintptr_t) in, (uintptr_t) out, size);

        // Step 1: inclusive uint32 prefix sum of the 0/1 mask bytes over a
        // single block spanning the entire array, using the widened u8 -> u32
        // kernel variants. A trivial scatter kernel then writes the compacted
        // indices: out[prefix[i] - 1] = i where mask[i] != 0, and the last
        // prefix entry is the total count.
        void *prefix = jitc_malloc(JitBackend::Metal,
                                   (size_t) size * sizeof(uint32_t));
        void *count = jitc_malloc(JitBackend::Metal, sizeof(uint32_t),
                                  /*shared=*/true);

        metal_block_prefix_reduce(this, VarType::UInt8, VarType::UInt32,
                                  ReduceOp::Add, size, /* block_size = */ size,
                                  /* exclusive = */ false,
                                  /* reverse = */ false, in, prefix);

        // Step 2: scatter the compacted indices
        CompressScatterParams params;
        id<MTLBuffer> in_buf = metal_resolve(in, &params.in);
        id<MTLBuffer> pre_buf = metal_resolve(prefix, &params.prefix);
        id<MTLBuffer> out_buf = metal_resolve(out, &params.out);

        metal_dispatch_threads(
            this, metal_pipeline(this, MetalKernel::CompressScatter), params,
            {{ in_buf, MTLResourceUsageRead },
             { pre_buf, MTLResourceUsageRead },
             { out_buf, MTLResourceUsageWrite }},
            size);

        // The total count is the last entry of the inclusive prefix sum;
        // stage it in the shared 'count' allocation for host readback
        memcpy_async(count, (uint8_t *) prefix + (size_t) (size - 1) * 4, 4);

        hist.record(); // capture timing before the synchronizing flush below
        flush(/* wait = */ true);
        uint32_t result = *(uint32_t *) count;
        jitc_free(prefix);
        jitc_free(count);
        return result;
    }
}

/// mkperm phase 2.5 (optional): scan the prefix-summed histogram for
/// non-empty buckets, appending (bucket, offset, count) records to
/// ``offsets`` and counting them in a fresh Shared allocation (returned via
/// ``*counter_out`` for post-flush readback). ``row_stride`` is the distance
/// between consecutive bucket base entries; ``offsets`` may live in host
/// memory, in which case a staging buffer is returned via ``*staging_out``.
static void metal_mkperm_detect_offsets(MetalThreadState *ts, void *buckets,
                                        uint32_t bucket_count, uint32_t size,
                                        uint32_t row_stride, uint32_t *offsets,
                                        void **counter_out,
                                        void **staging_out) {
    void *counter = jitc_malloc(JitBackend::Metal, sizeof(uint32_t),
                                /*shared=*/true);
    *(uint32_t *) counter = 0;
    *counter_out = counter;

    MkpermOffsetsParams params;
    id<MTLBuffer> buckets_buf = metal_resolve(buckets, &params.buckets);
    id<MTLBuffer> counter_buf = metal_resolve(counter, &params.counter);
    id<MTLBuffer> offsets_buf = metal_resolve_or_stage(
        offsets, (size_t) bucket_count * 4u * sizeof(uint32_t),
        &params.offsets, staging_out);
    params.bucket_count = bucket_count;
    params.perm_size = size;
    params.row_stride = row_stride;

    metal_dispatch_threads(
        ts, metal_pipeline(ts, MetalKernel::MkpermDetectOffsets), params,
        {{ buckets_buf, MTLResourceUsageRead },
         { offsets_buf, MTLResourceUsageWrite },
         { counter_buf, MTLResourceUsageRead | MTLResourceUsageWrite }},
        bucket_count);
}

/// Shared mkperm epilogue: release the histogram, record kernel history,
/// flush (waiting only when results must be read back on the host), and copy
/// out the optional offsets/permutation stagings. Returns the number of
/// non-empty buckets (0 if ``offsets`` was not requested).
static uint32_t metal_mkperm_finish(MetalThreadState *ts,
                                    MetalHistoryScope &hist, void *buckets,
                                    uint32_t size, uint32_t bucket_count,
                                    uint32_t *perm, void *perm_staging,
                                    uint32_t *offsets, void *counter,
                                    void *offsets_staging) {
    // 'buckets' frees on completion of the command buffer just encoded
    // (its last reader), so block only when results go back to the host.
    jitc_free(buckets);
    hist.record();
    ts->flush(/* wait = */ offsets || perm_staging);

    uint32_t unique_count = 0;
    if (offsets) {
        unique_count = *(uint32_t *) counter;
        if (offsets_staging)
            std::memcpy(offsets, offsets_staging,
                        (size_t) unique_count * 4u * sizeof(uint32_t));
        offsets[4 * bucket_count] = unique_count;
    }
    if (perm_staging)
        std::memcpy(perm, perm_staging, (size_t) size * sizeof(uint32_t));

    jitc_free(counter);
    jitc_free(offsets_staging);
    jitc_free(perm_staging);
    return unique_count;
}

uint32_t MetalThreadState::block_mkperm(const uint32_t *values, uint32_t size,
                                        uint32_t block_size,
                                        uint32_t bucket_count, uint32_t *perm,
                                        uint32_t *offsets) {
    @autoreleasepool {
        if (size == 0) return 0;
        if (bucket_count == 0)
            jitc_fail("jit_block_mkperm(): bucket_count cannot be zero!");
        const uint32_t warp = metal_simd_width;

        // ---- Stable "tiny" path: per-SIMD-group threadgroup histograms ------
        // As many warps as fit (per-warp histograms cost warp_count *
        // bucket_count * 4 bytes), else fall back to the global-atomic path.
        MetalDevice &md = state.metal_devices[this->device];
        id<MTLComputePipelineState> p1_tiny =
            metal_pipeline(this, MetalKernel::MkpermPhase1Tiny);

        uint32_t max_warps_pso =
            std::max(1u, (uint32_t) p1_tiny.maxTotalThreadsPerThreadgroup / warp);
        uint32_t max_warps_mem = md.threadgroup_memory_bytes / (bucket_count * 4u);
        uint32_t warp_count =
            std::min(std::min(8u, std::max(1u, metal_max_threads / warp)),
                     std::min(max_warps_pso, max_warps_mem));

        if (warp_count >= 1) {
            uint32_t thread_count = warp_count * warp;
            uint32_t n_blocks     = ceil_div(size, block_size);

            // Sub-blocks per block. Smaller threadgroups (memory pressure) yield
            // more sub-blocks, recovering parallelism.
            const uint32_t ELEMS_PER_THREAD = 16;
            uint32_t elems_per_blk = thread_count * ELEMS_PER_THREAD;
            uint32_t gpu_blocks_per_group =
                std::max(1u, ceil_div(block_size, elems_per_blk));

            // The merge buffer holds one histogram count per (block, sub-block,
            // warp, bucket) and the prefix sum scans them all. Bound that count
            // so the merge stays a minority of element traffic -- a quarter of
            // the input -- while still parallelizing a single large block.
            uint32_t max_cells = std::max(1u, size / 4u);
            uint64_t denom = (uint64_t) n_blocks * warp_count * bucket_count;
            uint32_t max_sub = (uint32_t) std::max<uint64_t>(1, max_cells / denom);
            gpu_blocks_per_group = std::min(gpu_blocks_per_group, max_sub);

            uint32_t size_per_block =
                ceil_div(ceil_div(block_size, gpu_blocks_per_group), warp) * warp;
            uint32_t rows_per_group = gpu_blocks_per_group * warp_count;
            uint32_t seg            = rows_per_group * bucket_count;
            uint32_t total_cells    = n_blocks * seg;
            uint32_t shared_bytes   = bucket_count * warp_count * 4u;

            if (offsets && n_blocks != 1)
                jitc_raise("jit_block_mkperm(): offset extraction requires "
                           "block_size == size.");

            jitc_log(Debug,
                     "jit_block_mkperm(" DRJIT_PTR ", size=%u, block_size=%u, "
                     "bucket_count=%u, variant=tiny, blocks=%u, sub/block=%u, "
                     "warps=%u, size_per_block=%u)",
                     (uintptr_t) values, size, block_size, bucket_count, n_blocks,
                     gpu_blocks_per_group, warp_count, size_per_block);

            MetalHistoryScope hist(this, KernelType::MkPerm, size);

            void *buckets =
                jitc_malloc(JitBackend::Metal, (size_t) total_cells * 4u);

            MkpermTinyParams tp;
            id<MTLBuffer> values_buf = metal_resolve(values, &tp.values);
            id<MTLBuffer> buckets_buf = metal_resolve(buckets, &tp.buckets);
            tp.perm           = 0; // patched before phase 4
            tp.size           = size;
            tp.size_per_block = size_per_block;
            tp.bucket_count   = bucket_count;
            tp.block_size     = block_size;
            tp.rows_per_group = rows_per_group;

            MTLSize grid = MTLSizeMake(gpu_blocks_per_group, n_blocks, 1);

            // Phase 1: per-SIMD-group histograms -> bucket-major global layout
            metal_dispatch_groups(
                this, p1_tiny, tp,
                {{ values_buf, MTLResourceUsageRead },
                 { buckets_buf, MTLResourceUsageRead | MTLResourceUsageWrite }},
                grid, thread_count, shared_bytes);

            // Phase 2: per-group exclusive prefix sum (segment length = seg)
            block_prefix_reduce(VarType::UInt32, ReduceOp::Add, total_cells, seg,
                                /*exclusive=*/true, /*reverse=*/false, buckets,
                                buckets);

            // Phase 2.5 (optional, single block): detect non-empty buckets.
            // The bucket-major layout puts bucket b's base at buckets[b*rows],
            // so the detector strides by rows_per_group.
            void *counter = nullptr, *offsets_staging = nullptr;
            if (offsets)
                metal_mkperm_detect_offsets(this, buckets, bucket_count, size,
                                            rows_per_group, offsets, &counter,
                                            &offsets_staging);

            // Phase 4: scatter the permutation (stage perm if host-only)
            void *perm_staging = nullptr;
            id<MTLBuffer> perm_buf = metal_resolve_or_stage(
                perm, (size_t) size * sizeof(uint32_t), &tp.perm,
                &perm_staging);

            metal_dispatch_groups(
                this, metal_pipeline(this, MetalKernel::MkpermPhase4Tiny), tp,
                {{ values_buf, MTLResourceUsageRead },
                 { buckets_buf, MTLResourceUsageRead },
                 { perm_buf, MTLResourceUsageWrite }},
                grid, thread_count, shared_bytes);

            return metal_mkperm_finish(this, hist, buckets, size, bucket_count,
                                       perm, perm_staging, offsets, counter,
                                       offsets_staging);
        }

        // ---- Fallback: global-atomic path (single block only) ------
        if (block_size != size)
            jitc_raise("jit_block_mkperm(): bucket_count=%u exceeds the stable "
                       "threadgroup-memory path, and the global-atomic fallback "
                       "does not support per-block permutations (block_size != "
                       "size) on Metal.", bucket_count);

        // Suppress kernel-history recording for the nested block_prefix_reduce
        // (phase 2): block_mkperm synchronizes internally (flush+wait to read
        // back counts), so it cannot itself be timed via the command-buffer
        // mechanism, and we do not want a spurious entry for its internal scan.
        MetalHistoryScope hist(this, KernelType::MkPerm, size);

        jitc_log(Debug, "jit_block_mkperm(" DRJIT_PTR ", size=%u, block_size=%u, "
                 "bucket_count=%u, variant=global)", (uintptr_t) values, size,
                 block_size, bucket_count);

        // Allocate the histogram and zero it via a blit fill
        uint32_t bucket_bytes = bucket_count * sizeof(uint32_t);
        void *buckets = jitc_malloc(JitBackend::Metal, bucket_bytes);
        {
            size_t off = 0;
            id<MTLBuffer> bb =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer(buckets, &off);
            id<MTLBlitCommandEncoder> enc =
                (__bridge id<MTLBlitCommandEncoder>) ensure_blit_encoder();
            [enc fillBuffer:bb range:NSMakeRange(off, bucket_bytes) value:0];
        }

        MkpermParams params;
        id<MTLBuffer> values_buf = metal_resolve(values, &params.values);
        id<MTLBuffer> buckets_buf = metal_resolve(buckets, &params.buckets);
        params.perm = 0; // patched before phase 3

        // Phase 1: Build histogram via global atomics
        metal_dispatch_threads(
            this, metal_pipeline(this, MetalKernel::MkpermPhase1), params,
            {{ values_buf, MTLResourceUsageRead },
             { buckets_buf, MTLResourceUsageRead | MTLResourceUsageWrite }},
            size);

        // Phase 2: Exclusive prefix sum over histogram (same command buffer).
        block_prefix_reduce(VarType::UInt32, ReduceOp::Add, bucket_count,
                            bucket_count, true, false, buckets, buckets);

        // Phase 2.5 (optional): Detect non-empty buckets
        void *counter = nullptr, *offsets_staging = nullptr;
        if (offsets)
            metal_mkperm_detect_offsets(this, buckets, bucket_count, size,
                                        /* row_stride = */ 1, offsets,
                                        &counter, &offsets_staging);

        // Phase 3: Scatter indices using atomics on the prefix-summed
        // histogram (stage perm if host-only)
        void *perm_staging = nullptr;
        id<MTLBuffer> perm_buf = metal_resolve_or_stage(
            perm, (size_t) size * sizeof(uint32_t), &params.perm,
            &perm_staging);

        metal_dispatch_threads(
            this, metal_pipeline(this, MetalKernel::MkpermPhase3), params,
            {{ values_buf, MTLResourceUsageRead },
             { buckets_buf, MTLResourceUsageRead | MTLResourceUsageWrite },
             { perm_buf, MTLResourceUsageWrite }},
            size);

        return metal_mkperm_finish(this, hist, buckets, size, bucket_count,
                                   perm, perm_staging, offsets, counter,
                                   offsets_staging);
    }
}

void MetalThreadState::aggregate(void *dst, AggregationEntry *agg,
                                 uint32_t size) {
    @autoreleasepool {
        if (size == 0)
            return;

        MetalHistoryScope hist(this, KernelType::Aggregate, size);

        jitc_log(InfoSym, "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u)", (uintptr_t) agg, (uintptr_t) dst, size);

        // Replace each opaque-resource handle with its owner's live id (resolved
        // now, so frozen replay picks up the current handle) and queue the owner
        // for residency. Shares jitc_metal_resource_id() with the launch path.
        for (uint32_t i = 0; i < size; ++i) {
            AggregationEntry &e = agg[i];
            void *id;
            if (e.size == 8) {
                ResourceKind kind = (ResourceKind) e.resource_kind;
                void *owner = (void *) e.src;
                if (jitc_metal_resource_id(owner, kind, &id)) {
                    e.src = id;
                    metal_call_resources.push_back({ owner, kind, /*write=*/false });
                } else if (kind == ResourceKind::Buffer && owner) {
                    metal_call_resources.push_back(
                        { owner, ResourceKind::Buffer, /*write=*/true });
                }
            }
        }

        // Resolve the destination MTLBuffer.
        size_t dst_off = 0;
        id<MTLBuffer> dst_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(dst, &dst_off);

        // Stage entries through a shared buffer
        id<MTLDevice> dev = (__bridge id<MTLDevice>) metal_device;
        size_t entries_bytes = sizeof(AggregationEntry) * (size_t) size;
        id<MTLBuffer> entries_buf =
            [dev newBufferWithBytes:agg
                             length:entries_bytes
                            options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pso =
            metal_pipeline(this, MetalKernel::Aggregate);

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>)
                ensure_compute_encoder();
        [enc setComputePipelineState:pso];
        [enc setBuffer:dst_buf offset:dst_off atIndex:0];
        [enc setBuffer:entries_buf offset:0 atIndex:1];
        [enc useResource:dst_buf usage:MTLResourceUsageWrite];

        // Mark each src buffer (negative-size entries: src is a device pointer)
        // as resident so the GPU can dereference it. Read from the host ``agg``
        // array directly (``entries_buf`` is just a copy of it).
        for (uint32_t i = 0; i < size; ++i) {
            const AggregationEntry &e = agg[i];
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
