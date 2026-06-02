#include "metal_ts.h"
#include "metal.h"
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
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

MetalThreadState::~MetalThreadState() {
    // Wait for any queued computation to finish
    flush(/* wait = */ true);

    // Release the MPSGraph objects
    @autoreleasepool {
        for (const auto &kv : metal_graph_cache) {
            const MetalGraph &g = kv.second;
            if (g.graph) (void) (__bridge_transfer id) g.graph;
            for (void *p : g.in)  if (p) (void) (__bridge_transfer id) p;
            for (void *p : g.out) if (p) (void) (__bridge_transfer id) p;
        }
    }
}

// ============================================================================
//  Command buffer / encoder lifecycle
//  ``ensure_*/close_encoder`` must be called from within an @autoreleasepool.
// ============================================================================

enum class MetalEncoderKind : uint32_t {
    None = 0,
    Compute,
    Blit
};

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
//  Metal exposes GPU timing per command buffer (GPUEndTime - GPUStartTime), not
//  per kernel. To attribute a runtime to a single operation we therefore give it
//  its own command buffer: commit any prior pending work before the operation
//  (so it starts in a fresh buffer), and commit the operation's own work after.
//  This is done for both the built-in operations (RAII wrapper below) and for
//  individual kernel launches (see launch()), so that several kernels emitted by
//  one jitc_eval() are timed separately rather than reporting the combined
//  command-buffer time. The committing ``flush(false)`` is only issued while
//  kernel history is enabled; each call site already knows that state (the scope
//  caches it in ``enabled``; launch() has a non-null ``kernel_history_entry``),
//  so we avoid re-reading the thread-local jit flags here.
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
            ts->flush(/* wait = */ false); // commit prior work -> fresh buffer
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
            ts->flush(/* wait = */ false); // commit this operation's buffer
        }
    }
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

        // Indicate Metal buffers used by this kernel
        for (size_t i = 1; i < kernel_params.size(); ++i) {
            size_t offset = 0;
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>) jitc_metal_find_buffer(
                kernel_params[i], &offset);
            [enc useResource:buf
                       usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }

        // Indicate additional resources detected during code generation
        for (const CallResource &res : metal_extra_resources) {
            size_t off = 0;
            id<MTLBuffer> buf =
                (__bridge id<MTLBuffer>) jitc_metal_find_buffer(res.ptr, &off);
            [enc useResource:buf
                       usage:res.write ? (MTLResourceUsageRead |
                                          MTLResourceUsageWrite)
                                       : MTLResourceUsageRead];
        }
        metal_extra_resources.clear();

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
            id<MTLComputePipelineState> pso =
                (__bridge id<MTLComputePipelineState>)
                    state.metal_devices[this->device].pipelines[(uint32_t) kernel];
            if (!pso)
                jitc_raise("jit_memset_async(): memset pipeline missing.");

            struct {
                uint64_t dst, value;
            } params {};
            std::memcpy(&params.value, src, isize); // low ``isize`` bytes
            params.dst = (uint64_t) [buf gpuAddress] + offset;

            id<MTLComputeCommandEncoder> enc =
                (__bridge id<MTLComputeCommandEncoder>) ensure_compute_encoder();
            [enc setComputePipelineState:pso];
            [enc setBytes:&params length:sizeof(params) atIndex:0];
            [enc useResource:buf usage:MTLResourceUsageWrite];

            uint32_t tg = std::min((uint32_t) pso.maxTotalThreadsPerThreadgroup,
                                   round_pow2(size));
            [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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
                (__bridge id<MTLCommandBuffer>) metal_cb;
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
        MetalHistoryScope hist(this, KernelType::Poke, size);
        jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);
        memcpy_async(dst, src, size);
    }
}

// ============================================================================
//  Reductions / scans / compaction
// ============================================================================

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

// ============================================================================
//  MPSGraph infrastructure
//
//  The operations block_reduce, block_prefix_reduce, reduce_dot, and compress
//  are all expressed as MPSGraphs. Building (and compiling) a graph has a
//  nontrivial cost, so we cache and reuse them.
// ============================================================================

namespace {
    enum class MetalGraphKind : uint32_t { Scan, Reduce, Dot, Compress };

    /// One input or output binding for metal_run_graph: a graph tensor paired
    /// with the Dr.Jit device buffer that feeds or receives it.
    struct MetalBind {
        void *tensor;                  // MPSGraphTensor*
        void *ptr;                     // Dr.Jit device pointer
        NSArray<NSNumber *> *shape;    // logical shape of the buffer view
        MPSDataType dt;
        size_t bytes;                  // byte size of the view (for staging)
    };
}

/// Map a Dr.Jit type to the MPSDataType used for reductions
static bool metal_mps_dtype(VarType vt, MPSDataType &dt) {
    switch (vt) {
        case VarType::Float16: dt = MPSDataTypeFloat16; return true;
        case VarType::Float32: dt = MPSDataTypeFloat32; return true;
        case VarType::Int32:   dt = MPSDataTypeInt32;   return true;
        case VarType::UInt32:  dt = MPSDataTypeUInt32;  return true;
        case VarType::Int64:   dt = MPSDataTypeInt64;   return true;
        case VarType::UInt64:  dt = MPSDataTypeUInt64;  return true;
        default: return false;
    }
}

/// Look up a cached graph by key, or build and cache it via ``build``
template <typename Build>
static MetalGraph metal_graph_cached(MetalThreadState *ts, uint64_t key,
                                     Build &&build) {
    auto &cache = ts->metal_graph_cache;
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;
    MetalGraph g = build();
    cache[key] = g;
    return g;
}

/// Execute a previously recorded metal graph with the given buffer bindings
static void metal_run_graph(MetalThreadState *ts, const MetalGraph &g,
                            std::initializer_list<MetalBind> inputs,
                            std::initializer_list<MetalBind> outputs) {
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
        [NSMutableDictionary dictionary];

    auto bind = [&](const MetalBind &b,
                    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *dict) {
        size_t off = 0;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>) jitc_metal_find_buffer(b.ptr, &off);
        if (unlikely(off != 0))
            jitc_fail("metal_run_graph(): operand at non-zero buffer offset; "
                      "MPSGraph reductions assume base-aligned operands.");
        MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:buf shape:b.shape dataType:b.dt];
        [dict setObject:data forKeyedSubscript:(__bridge MPSGraphTensor *) b.tensor];
    };

    for (const MetalBind &b : inputs)  bind(b, feeds);
    for (const MetalBind &b : outputs) bind(b, results);

    ts->close_encoder();
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) ts->ensure_cmdbuf();
    MPSCommandBuffer *mcb = [MPSCommandBuffer commandBufferWithCommandBuffer:cb];

    [(__bridge MPSGraph *) g.graph encodeToCommandBuffer:mcb
                                                   feeds:feeds
                                        targetOperations:nil
                                       resultsDictionary:results
                                     executionDescriptor:nil];

    // MPSGraph _may_ commit the command buffer, in which case we continue using the new one
    id<MTLCommandBuffer> root = mcb.rootCommandBuffer;
    if ((__bridge void *) root != ts->metal_cb) {
        (void) (__bridge_transfer id<MTLCommandBuffer>) ts->metal_cb;
        ts->metal_cb = (__bridge_retained void *) root;
    }
}

/// Pad a buffer to exactly ``block_count * block_size`` elements for a scan/reduce
static const void *metal_pad_input(MetalThreadState *ts, ReduceOp op, VarType vt,
                                   uint32_t size, uint32_t padded,
                                   const void *in, void **scratch) {
    if (padded == size) {
        *scratch = nullptr;
        return in;
    }
    uint32_t tsize = type_size[(int) vt];
    void *p = jitc_malloc(JitBackend::Metal, (size_t) padded * tsize);
    ts->memcpy_async(p, in, (size_t) size * tsize);
    uint64_t ident = jitc_reduce_identity(vt, op);
    ts->memset_async((uint8_t *) p + (size_t) size * tsize,
                     padded - size, tsize, &ident);
    *scratch = p;
    return p;
}

void MetalThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                    uint32_t block_size, const void *in,
                                    void *out) {
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

        MetalHistoryScope hist(this, KernelType::BlockReduce, size);

        uint32_t block_count = (size + block_size - 1) / block_size;

        // Bitwise Or/And block reductions are unsupported on Metal. (Boolean
        // any()/all() is handled separately by block_reduce_bool().)
        if (op == ReduceOp::Or || op == ReduceOp::And)
            jitc_raise("jit_block_reduce(): 'Or'/'And' reductions are not "
                       "supported on the Metal backend.");

        // Sum/product/minimum/maximum go through MPSGraph.
        MPSDataType dt;
        if (metal_mps_dtype(vt, dt)) {
            uint32_t padded = block_count * block_size;
            void *scratch = nullptr;
            const void *in_use =
                metal_pad_input(this, op, vt, size, padded, in, &scratch);

            uint64_t key = ((uint64_t) MetalGraphKind::Reduce << 56) |
                           ((uint64_t) (uint32_t) device << 40) |
                           ((uint64_t) (int) op << 16) |
                           ((uint64_t) (int) vt << 4);

            MetalGraph g = metal_graph_cached(this, key, [&]() -> MetalGraph {
                MPSGraph *graph = [MPSGraph new];
                MPSGraphTensor *in = [graph placeholderWithShape:@[ @(-1), @(-1) ] dataType:dt name:nil];
                MPSGraphTensor *r = nil;
                switch (op) {
                    case ReduceOp::Add: r = [graph reductionSumWithTensor:in     axes:@[ @1 ] name:nil]; break;
                    case ReduceOp::Mul: r = [graph reductionProductWithTensor:in axes:@[ @1 ] name:nil]; break;
                    case ReduceOp::Min: r = [graph reductionMinimumWithTensor:in axes:@[ @1 ] name:nil]; break;
                    case ReduceOp::Max: r = [graph reductionMaximumWithTensor:in axes:@[ @1 ] name:nil]; break;
                    default: return MetalGraph{};
                }
                r = [graph reshapeTensor:r withShape:@[ @(-1) ] name:nil];
                return MetalGraph{ (__bridge_retained void *) graph,
                                 { (__bridge_retained void *) in, nullptr },
                                 { (__bridge_retained void *) r, nullptr } };
            });
            metal_run_graph(this, g,
                { { g.in[0], (void *) in_use, @[ @(block_count), @(block_size) ],
                    dt, (size_t) padded * tsize } },
                { { g.out[0], out, @[ @(block_count) ],
                    dt, (size_t) block_count * tsize } });

            if (scratch)
                jitc_free(scratch);

            jitc_log(Debug,
                     "jit_block_reduce(type=%s, op=%s, size=%u, "
                     "block_size=%u) -> MPSGraph",
                     type_name[(int) vt], metal_op_name(op), size, block_size);
            return;
        }

        // The MPSGraph path handles every supported (type, op) combination and
        // returns above; reaching here means the element type is not reducible
        // on Metal (e.g. a 64-bit integer sum, which MPSGraph does not provide).
        jitc_raise("jit_block_reduce(): unsupported type '%s' for op '%s' on "
                   "the Metal backend.", type_name[(int) vt], metal_op_name(op));
    }
}

void MetalThreadState::narrow_f32_to_f16(void *dst, const void *src,
                                         uint32_t size) {
    @autoreleasepool {
        if (size == 0)
            return;

        jitc_log(Debug, "jit_narrow_f32_to_f16(size=%u)", size);

        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)
                state.metal_devices[this->device]
                    .pipelines[(uint32_t) MetalKernel::ConvertF32F16];
        if (!pso)
            jitc_raise("jit_narrow_f32_to_f16(): convert pipeline missing.");

        size_t src_off = 0, dst_off = 0;
        id<MTLBuffer> src_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) src, &src_off);
        id<MTLBuffer> dst_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(dst, &dst_off);
        if (!src_buf || !dst_buf)
            jitc_raise("jit_narrow_f32_to_f16(): buffer lookup failed.");

        struct { uint64_t src, dst; } params;
        params.src = (uint64_t) [src_buf gpuAddress] + src_off;
        params.dst = (uint64_t) [dst_buf gpuAddress] + dst_off;

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>) ensure_compute_encoder();
        [enc setComputePipelineState:pso];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        [enc useResource:src_buf usage:MTLResourceUsageRead];
        [enc useResource:dst_buf usage:MTLResourceUsageWrite];

        uint32_t tg = std::min((uint32_t) pso.maxTotalThreadsPerThreadgroup,
                               round_pow2(size));
        [enc dispatchThreads:MTLSizeMake(size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void MetalThreadState::block_reduce_bool(uint8_t *values, uint32_t size,
                                         uint8_t *out, ReduceOp op) {
    @autoreleasepool {
        bool is_all = (op == ReduceOp::And);
        if (!is_all && op != ReduceOp::Or)
            jitc_raise("jit_block_reduce_bool(): expected 'Or' or 'And'.");

        jitc_log(Debug, "jit_%s(" DRJIT_PTR ", size=%u)",
                 is_all ? "all" : "any", (uintptr_t) values, size);

        MetalHistoryScope hist(this, KernelType::BlockReduce, size);

        MetalKernel init_kernel = is_all ? MetalKernel::ReduceAllInit
                                         : MetalKernel::ReduceAnyInit;
        MetalKernel kernel      = is_all ? MetalKernel::ReduceAll
                                         : MetalKernel::ReduceAny;
        id<MTLComputePipelineState> init_pso =
            (__bridge id<MTLComputePipelineState>)
                state.metal_devices[this->device].pipelines[(uint32_t) init_kernel];
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)
                state.metal_devices[this->device].pipelines[(uint32_t) kernel];
        if (!init_pso || !pso)
            jitc_raise("jit_block_reduce_bool(): reduction kernel not found.");

        size_t out_off = 0;
        id<MTLBuffer> out_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(out, &out_off);

        struct { uint64_t in, out; uint32_t size; } params;
        id<MTLBuffer> in_buf = metal_resolve(values, &params.in);
        metal_resolve(out, &params.out);
        params.size = size;

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>) ensure_compute_encoder();
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        if (in_buf)  [enc useResource:in_buf  usage:MTLResourceUsageRead];
        if (out_buf) [enc useResource:out_buf usage:MTLResourceUsageRead |
                                                    MTLResourceUsageWrite];

        // 1. Seed the result with the reduction identity (one thread).
        [enc setComputePipelineState:init_pso];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

        // 2. Reduce: one thread per coalesced 16-byte packet (thread 0 also
        // handles the < 16 tail bytes). The grid is sized to the data and the
        // hardware schedules occupancy; no grid-size heuristic is needed.
        uint32_t tg = (uint32_t) pso.maxTotalThreadsPerThreadgroup;
        uint32_t grid = std::max(size >> 4, 1u);

        [enc setComputePipelineState:pso];
        [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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
            jitc_raise("jit_block_prefix_reduce(): invalid "
                       "block_size=%u for size=%u.", block_size, size);

        if (block_size == 1) {
            if (exclusive) {
                uint64_t ident = jitc_reduce_identity(vt, op);
                memset_async(out, size, tsize, &ident);
            } else if (in != out) {
                memcpy_async(out, in, (size_t) size * tsize);
            }
            return;
        }

        if (op == ReduceOp::Or || op == ReduceOp::And)
            jitc_raise(
                "jit_block_prefix_reduce(): 'Or' and 'And' "
                "prefix reductions are not supported by Metal's MPSGraph API.");

        MPSDataType dt;
        if (!metal_mps_dtype(vt, dt))
            jitc_raise("jit_block_prefix_reduce(): unsupported "
                       "type '%s'.", type_name[(int) vt]);

        MetalHistoryScope hist(this, KernelType::BlockPrefixReduce, size);

        // Cumulative scan along axis 1 of a [block_count, block_size] tensor.
        uint64_t key = ((uint64_t) MetalGraphKind::Scan << 56) |
                       ((uint64_t) (uint32_t) device << 40) |
                       ((uint64_t) (int) op << 16) | ((uint64_t) (int) vt << 4) |
                       (exclusive ? 2u : 0u) | (reverse ? 1u : 0u);
        MetalGraph g = metal_graph_cached(this, key, [&]() -> MetalGraph {
            MPSGraph *graph = [MPSGraph new];
            MPSGraphTensor *in = [graph placeholderWithShape:@[ @(-1), @(-1) ] dataType:dt name:nil];
            MPSGraphTensor *r = nil;
            switch (op) {
                case ReduceOp::Add: r = [graph cumulativeSumWithTensor:in     axis:1 exclusive:exclusive reverse:reverse name:nil]; break;
                case ReduceOp::Mul: r = [graph cumulativeProductWithTensor:in axis:1 exclusive:exclusive reverse:reverse name:nil]; break;
                case ReduceOp::Min: r = [graph cumulativeMinimumWithTensor:in axis:1 exclusive:exclusive reverse:reverse name:nil]; break;
                case ReduceOp::Max: r = [graph cumulativeMaximumWithTensor:in axis:1 exclusive:exclusive reverse:reverse name:nil]; break;
                default: return MetalGraph{};
            }
            return MetalGraph{ (__bridge_retained void *) graph,
                               { (__bridge_retained void *) in, nullptr },
                               { (__bridge_retained void *) r, nullptr } };
        });

        uint32_t block_count = (size + block_size - 1) / block_size;
        uint32_t padded = block_count * block_size;
        size_t padded_bytes = (size_t) padded * tsize;

        // Pad a ragged final block with the reduction identity.
        void *scratch_in = nullptr;
        const void *in_use =
            metal_pad_input(this, op, vt, size, padded, in, &scratch_in);

        // MPSGraph must not alias input and output: a padded run already targets
        // a fresh buffer; otherwise stage when the caller passed in == out.
        void *scratch_out = nullptr, *out_use = out;
        if (padded != size || in == out) {
            scratch_out = jitc_malloc(JitBackend::Metal, padded_bytes);
            out_use = scratch_out;
        }

        MPSShape *shape = @[ @(block_count), @(block_size) ];
        metal_run_graph(this, g,
            { { g.in[0],  (void *) in_use, shape, dt, padded_bytes } },
            { { g.out[0], out_use,         shape, dt, padded_bytes } });

        // Copy the logical-size result out of any scratch output.
        if (out_use != out)
            memcpy_async(out, out_use, (size_t) size * tsize);

        if (scratch_in)  jitc_free(scratch_in);
        if (scratch_out) jitc_free(scratch_out);

        jitc_log(Debug,
                 "jit_block_prefix_reduce(type=%s, op=%s, size=%u, "
                 "block_size=%u, exclusive=%d, reverse=%d) -> MPSGraph",
                 type_name[(int) vt], metal_op_name(op), size, block_size,
                 (int) exclusive, (int) reverse);
    }
}

void MetalThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                  const void *ptr_2, uint32_t size,
                                  void *out) {
    @autoreleasepool {
        if (size == 0) return;

        MPSDataType dt;
        if (!metal_mps_dtype(vt, dt))
            jitc_raise("jit_reduce_dot(): unsupported type %s.",
                       type_name[(int) vt]);

        MetalHistoryScope hist(this, KernelType::Dot, size);

        jitc_log(Debug, "jit_reduce_dot(" DRJIT_PTR ", " DRJIT_PTR
                 ", type=%s, size=%u)", (uintptr_t) ptr_1, (uintptr_t) ptr_2,
                 type_name[(int) vt], size);

        uint32_t tsize = type_size[(int) vt];

        // Dot product: out[0] = sum(a * b) over two [size] tensors.
        uint64_t key = ((uint64_t) MetalGraphKind::Dot << 56) |
                       ((uint64_t) (uint32_t) device << 40) |
                       ((uint64_t) (int) vt << 4);
        MetalGraph g = metal_graph_cached(this, key, [&]() -> MetalGraph {
            MPSGraph *graph = [MPSGraph new];
            MPSGraphTensor *a = [graph placeholderWithShape:@[ @(-1) ] dataType:dt name:nil];
            MPSGraphTensor *b = [graph placeholderWithShape:@[ @(-1) ] dataType:dt name:nil];
            MPSGraphTensor *p = [graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
            MPSGraphTensor *r = [graph reductionSumWithTensor:p axes:@[ @0 ] name:nil];
            return MetalGraph{ (__bridge_retained void *) graph,
                               { (__bridge_retained void *) a, (__bridge_retained void *) b },
                               { (__bridge_retained void *) r, nullptr } };
        });
        metal_run_graph(this, g,
            { { g.in[0], (void *) ptr_1, @[ @(size) ], dt, (size_t) size * tsize },
              { g.in[1], (void *) ptr_2, @[ @(size) ], dt, (size_t) size * tsize } },
            { { g.out[0], out, @[ @1 ], dt, tsize } });
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
            jitc_raise("jit_batched_gemm(): unsupported type '%s'.",
                       type_name[(int) vt]);
        }

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

        // One MPSGraph produces the inclusive uint32 prefix sum of the
        // (cast-to-bool) mask plus the total count; a trivial scatter kernel then
        // writes the compacted indices: out[prefix[i] - 1] = i where mask[i] != 0.
        // ``count`` is a Shared allocation so the host can read the total back.
        void *prefix = jitc_malloc(JitBackend::Metal, (size_t) size * sizeof(uint32_t));
        void *count  = jitc_malloc(JitBackend::Metal, sizeof(uint32_t), /*shared=*/true);

        // From a uint8 mask, produce the inclusive uint32 prefix sum of its
        // non-zero indicator (out[0]) and the total count (out[1]).
        uint64_t key = ((uint64_t) MetalGraphKind::Compress << 56) |
                       ((uint64_t) (uint32_t) device << 40);
        MetalGraph g = metal_graph_cached(this, key, [&]() -> MetalGraph {
            MPSGraph *graph = [MPSGraph new];
            MPSGraphTensor *mask = [graph placeholderWithShape:@[ @(-1) ] dataType:MPSDataTypeUInt8 name:nil];
            MPSGraphTensor *bits = [graph castTensor:mask toType:MPSDataTypeBool  name:nil];
            MPSGraphTensor *ind  = [graph castTensor:bits toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor *pre  = [graph cumulativeSumWithTensor:ind axis:0 exclusive:NO reverse:NO name:nil];
            MPSGraphTensor *tot  = [graph reductionSumWithTensor:ind axes:@[ @0 ] name:nil];
            return MetalGraph{ (__bridge_retained void *) graph,
                               { (__bridge_retained void *) mask, nullptr },
                               { (__bridge_retained void *) pre, (__bridge_retained void *) tot } };
        });
        metal_run_graph(this, g,
            { { g.in[0], (void *) in, @[ @(size) ], MPSDataTypeUInt8, (size_t) size } },
            { { g.out[0], prefix, @[ @(size) ], MPSDataTypeInt32,
                (size_t) size * sizeof(uint32_t) },
              { g.out[1], count, @[ @1 ], MPSDataTypeInt32, sizeof(uint32_t) } });

        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)
                state.metal_devices[this->device].pipelines[(uint32_t) MetalKernel::CompressScatter];
        if (!pso)
            jitc_raise("jit_compress(): compress_scatter kernel "
                       "not found.");

        struct { uint64_t in, prefix, out; uint32_t size; } params;
        params.in     = (uint64_t)(uintptr_t) in;
        params.prefix = (uint64_t)(uintptr_t) prefix;
        params.out    = (uint64_t)(uintptr_t) out;
        params.size   = size;

        size_t off = 0;
        id<MTLBuffer> in_buf  = (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) in, &off);
        id<MTLBuffer> pre_buf = (__bridge id<MTLBuffer>) jitc_metal_find_buffer(prefix, &off);
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) out, &off);

        id<MTLComputeCommandEncoder> enc =
            (__bridge id<MTLComputeCommandEncoder>) ensure_compute_encoder();
        [enc setComputePipelineState:pso];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        if (in_buf)  [enc useResource:in_buf  usage:MTLResourceUsageRead];
        if (pre_buf) [enc useResource:pre_buf usage:MTLResourceUsageRead];
        if (out_buf) [enc useResource:out_buf usage:MTLResourceUsageWrite];

        uint32_t tg = std::min((uint32_t) pso.maxTotalThreadsPerThreadgroup,
                               round_pow2(size));
        [enc dispatchThreads:MTLSizeMake(size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        hist.record(); // capture timing before the synchronizing flush below
        flush(/* wait = */ true);
        uint32_t result = *(uint32_t *) count;
        jitc_free(prefix);
        jitc_free(count);
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
            jitc_fail("jit_block_mkperm(): bucket_count cannot be zero!");
        if (block_size != size)
            jitc_raise("jit_block_mkperm(): per-block permutations "
                       "(block_size != size) are not yet implemented on Metal.");

        // Suppress kernel-history recording for the nested block_prefix_reduce
        // (phase 2): block_mkperm synchronizes internally (flush+wait to read
        // back counts), so it cannot itself be timed via the command-buffer
        // mechanism, and we do not want a spurious entry for its internal scan.
        MetalHistoryScope hist(this, KernelType::MkPerm, size);

        jitc_log(Debug, "jit_block_mkperm(" DRJIT_PTR ", size=%u, block_size=%u, "
                 "bucket_count=%u)", (uintptr_t) values, size, block_size,
                 bucket_count);

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
                state.metal_devices[this->device].pipelines[(uint32_t) MetalKernel::MkpermPhase1];
        id<MTLComputePipelineState> phase3_pso =
            (__bridge id<MTLComputePipelineState>)
                state.metal_devices[this->device].pipelines[(uint32_t) MetalKernel::MkpermPhase3];
        if (!phase1_pso || !phase3_pso)
            jitc_raise("jit_block_mkperm(): kernels not found.");

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

            uint32_t tg = std::min(
                (uint32_t) phase1_pso.maxTotalThreadsPerThreadgroup,
                round_pow2(size));
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
                    state.metal_devices[this->device].pipelines[(uint32_t) MetalKernel::MkpermDetectOffsets];
            if (!detect_pso)
                jitc_raise("jit_block_mkperm(): mkperm_detect_offsets "
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

            uint32_t tg = std::min(
                (uint32_t) detect_pso.maxTotalThreadsPerThreadgroup,
                round_pow2(bucket_count));
            [enc dispatchThreads:MTLSizeMake(bucket_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

            hist.record(); // capture timing before the synchronizing flush below
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

            uint32_t tg = std::min(
                (uint32_t) phase3_pso.maxTotalThreadsPerThreadgroup,
                round_pow2(size));
            [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        hist.record(); // capture timing before the synchronizing flush below
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

        MetalHistoryScope hist(this, KernelType::Aggregate, size);

        jitc_log(InfoSym, "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u)", (uintptr_t) agg, (uintptr_t) dst, size);

        // Resolve the destination MTLBuffer.
        size_t dst_off = 0;
        id<MTLBuffer> dst_buf =
            (__bridge id<MTLBuffer>) jitc_metal_find_buffer(dst, &dst_off);

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
                state.metal_devices[this->device].pipelines[(uint32_t) MetalKernel::Aggregate];
        if (!pso)
            jitc_fail("jit_aggregate(): aggregate_kernel "
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
