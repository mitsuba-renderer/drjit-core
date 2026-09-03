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
#include "metal_tex.h"
#include "internal.h"
#include "eval.h"
#include "malloc.h"
#include "log.h"
#include "io.h"
#include "var.h"
#include "util.h"
#include "trace.h"
#include "record_ts.h"
#include "drjit-core/metal.h"
#include "resources/metal_kernels.h"

// Suppress the obsolete Carbon <CarbonCore/Threads.h>, whose ThreadState collides with Dr.Jit
#define __THREADS__
#import <Metal/Metal.h>
#import <objc/message.h>

#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <unistd.h>

// Metal 4 symbols (MTLGPUFamilyMetal4, MTLLanguageVersion4_0) only exist in the
// macOS 26 SDK. When present they are compiled in but gated at runtime via
// ``@available``, so the resulting binary still runs on older macOS releases.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
#  define DRJIT_SUPPORTS_METAL4 1
#endif

// ============================================================================
//  Opaque-resource handle resolution
// ============================================================================

bool jitc_metal_resource_id(void *owner, ResourceKind kind, void **value_out) {
    uint64_t v64;
    MTLResourceID rid;
    switch (kind) {
        case ResourceKind::Accel: {
            MetalScene *scene = (MetalScene *) owner;
            if (unlikely(scene->tlas_rid_for != scene->tlas)) {
                rid = ((__bridge id<MTLAccelerationStructure>) scene->tlas).gpuResourceID;
                std::memcpy(&scene->tlas_rid, &rid, sizeof(uint64_t));
                scene->tlas_rid_for = scene->tlas;
            }
            v64 = scene->tlas_rid;
            break;
        }

        case ResourceKind::Texture:
        case ResourceKind::Sampler: {
            MetalTexResource *res = (MetalTexResource *) owner;
            if (unlikely(res->rid_for != res->object)) {
                if (kind == ResourceKind::Texture)
                    rid = ((__bridge id<MTLTexture>) res->object).gpuResourceID;
                else
                    rid = ((__bridge id<MTLSamplerState>) res->object).gpuResourceID;
                std::memcpy(&res->rid, &rid, sizeof(uint64_t));
                res->rid_for = res->object;
            }
            v64 = res->rid;
            break;
        }

        default:
            // A buffer is an ordinary pointer; an IFT is PSO-dependent and is
            // refreshed at launch. Neither resolves to a gpuResourceID here.
            return false;
    }

    *value_out = (void *) (uintptr_t) v64;
    return true;
}

// ============================================================================
// Buffer API
// ============================================================================
//
void *metal_buffer_new(void *dev, size_t size, bool shared, void **ptr_out) {
    @autoreleasepool {
        id<MTLDevice> mtl_dev = (__bridge id<MTLDevice>) dev;
        MTLResourceOptions opts = shared ? MTLResourceStorageModeShared
                                         : MTLResourceStorageModePrivate;
        id<MTLBuffer> buf = [mtl_dev newBufferWithLength:size options:opts];
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


// Lazily-sorted flat vector of (base address, id<MTLBuffer>, length) entries.
// Protected by ``state.lock``.
struct BufferEntry {
    uintptr_t base;
    void     *buf;    // id<MTLBuffer>
    size_t    length;
};

static std::vector<BufferEntry> metal_buffer_map;
static bool metal_buffer_map_sorted = true;

/// Hash table mapping pointer addresses to id<MTLBuffer>
static tsl::robin_map<uintptr_t, BufferEntry, UInt64Hasher> metal_buffer_lut;

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
    metal_buffer_lut.insert_or_assign(
        (uintptr_t) ptr, BufferEntry { (uintptr_t) ptr, metal_buffer, size });
}

/// Look up a buffer that *contains* the given pointer and return the offset
void *jitc_metal_find_buffer(void *ptr, size_t *offset_out) {
    uintptr_t addr = (uintptr_t) ptr;

    // Fast path: the pointer references the start of an allocation
    auto it_lut = metal_buffer_lut.find(addr);
    if (likely(it_lut != metal_buffer_lut.end())) {
        *offset_out = 0;
        return it_lut->second.buf;
    }

    jitc_metal_ensure_sorted();

    auto it = std::upper_bound(
        metal_buffer_map.begin(), metal_buffer_map.end(), addr,
        [](uintptr_t a, const BufferEntry &b) { return a < b.base; });

    if (it != metal_buffer_map.begin()) {
        --it;
        if (it->buf && addr < it->base + it->length) {
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
    metal_buffer_lut.erase(addr);

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
// ============================================================================

/// Kernel function names, indexed by MetalKernel. The order must match the
/// MetalKernel enum in internal.h.
static const char *metal_kernel_names[(uint32_t) MetalKernel::Count] = {
    "compress_scatter",
    "mkperm_phase_1",
    "mkperm_phase_3",
    "mkperm_detect_offsets",
    "mkperm_phase_1_tiny",
    "mkperm_phase_4_tiny",
    "aggregate_kernel",
    "memset_u16",
    "memset_u32",
    "memset_u64",
    "convert_f32_f16",
    "deinterleave_u8",
    "deinterleave_u16",
    "deinterleave_u32",
    "interleave_u8",
    "interleave_u16",
    "interleave_u32"
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

/// Map a Dr.Jit type to the suffix used by the block-reduction kernel names
static const char *metal_reduce_type_name(VarType vt) {
    switch (vt) {
        case VarType::Bool:
        case VarType::UInt8:   return "u8";
        case VarType::Float16: return "f16";
        case VarType::Float32: return "f32";
        case VarType::UInt32:  return "u32";
        case VarType::Int32:   return "i32";
        case VarType::UInt64:  return "u64";
        case VarType::Int64:   return "i64";
        default:               return nullptr;
    }
}

/// Return the block (prefix) reduction pipeline for the requested kernel
/// family, type, and reduction, creating it on first use. There are too many
/// (op, type) combinations to create them all eagerly in jitc_metal_init()
/// (each pipeline costs a few milliseconds). Returns nullptr if the
/// combination is unsupported.
void *jitc_metal_block_reduce_pipeline(int device, MetalReduceKind kind,
                                       ReduceOp op, VarType vt) {
    MetalDevice &md = state.metal_devices[device];
    void *&slot = md.reduce_pipelines[(int) kind][(int) op][(int) vt];
    if (slot)
        return slot;

    const char *tname = metal_reduce_type_name(vt);
    if (!tname)
        return nullptr;

    const char *prefix = nullptr;
    switch (kind) {
        case MetalReduceKind::Small:     prefix = "block_reduce_small"; break;
        case MetalReduceKind::Chunk:     prefix = "block_reduce"; break;
        case MetalReduceKind::WideChunk: prefix = "block_reduce_wide"; break;
        case MetalReduceKind::Scan:      prefix = "block_prefix_reduce"; break;
        case MetalReduceKind::Dot:       prefix = "reduce_dot"; break;
        default: return nullptr;
    }

    char name[64];
    if (kind == MetalReduceKind::Dot) // the reduction op is implicit here
        snprintf(name, sizeof(name), "%s_%s", prefix, tname);
    else
        snprintf(name, sizeof(name), "%s_%s_%s",
                 prefix, red_name[(int) op], tname);

    @autoreleasepool {
        slot = jitc_metal_create_pipeline(
            (__bridge id<MTLDevice>) md.device,
            (__bridge id<MTLLibrary>) md.utility_lib, name);
    }
    return slot;
}

// ============================================================================
//  Backend init / shutdown
// ============================================================================

bool jitc_metal_init() {
    @autoreleasepool {
        // The backend emits Metal Shading Language 3.2, which requires macOS 15+
        if (@available(macOS 15.0, *)) {
        } else {
            jitc_log(Warn, "jit_metal_init(): the Metal backend requires macOS "
                           "15.0 or newer.");
            return false;
        }

        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (!devices || devices.count == 0) {
            jitc_log(Warn, "jit_metal_init(): no Metal-capable GPU was detected.");
            return false;
        }

        state.metal_devices.clear();

        // Decompress the precompiled utility kernel library once. The
        // dispatch object copies the buffer.
        char *metallib = jitc_lz4_inflate(metal_kernels,
                                          metal_kernels_size_compressed,
                                          metal_kernels_size_uncompressed,
                                          "utility kernel library");
        dispatch_data_t lib_data = dispatch_data_create(
            metallib, metal_kernels_size_uncompressed, nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        free(metallib);

        for (id<MTLDevice> dev in devices) {
            if (![dev supportsFamily:MTLGPUFamilyMetal3]) {
                jitc_log(Warn,
                         "jit_metal_init(): skipping device \"%s\" because it "
                         "does not support Metal 3 (M1+ required).",
                         dev.name.UTF8String);
                continue;
            }

            // Lift the default cap of 2 concurrent shader-compilation tasks
            if ([dev respondsToSelector:@selector(setShouldMaximizeConcurrentCompilation:)])
                dev.shouldMaximizeConcurrentCompilation = YES;

            MetalDevice md {};
            md.device = (__bridge_retained void *) dev;
            md.queue  = (__bridge_retained void *) [dev newCommandQueue];
            md.max_threads_per_threadgroup =
                (uint32_t) [dev maxThreadsPerThreadgroup].width;
            md.threadgroup_memory_bytes =
                (uint32_t) [dev maxThreadgroupMemoryLength];
            md.supports_ray_tracing = [dev supportsRaytracing];
            md.supports_metal4 = false;
#if defined(DRJIT_SUPPORTS_METAL4)
            if (@available(macOS 26.0, *))
                md.supports_metal4 = [dev supportsFamily:MTLGPUFamilyMetal4];
#endif
            const char *name = dev.name.UTF8String;
            size_t len = std::strlen(name);
            md.name = (char *) std::malloc(len + 1);
            std::memcpy(md.name, name, len + 1);

            // Instantiate the precompiled utility kernel library
            NSError *err = nil;
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
                     "(simd=%u, max_threads=%u, rt=%s, metal4=%s)",
                     md.name, md.simd_width, md.max_threads_per_threadgroup,
                     md.supports_ray_tracing ? "yes" : "no",
                     md.supports_metal4 ? "yes" : "no");

            state.metal_devices.push_back(md);
        }

        return !state.metal_devices.empty();
    }
}

void jitc_metal_shutdown() {
    jitc_unit_cache_flush((int) JitBackend::Metal);
    @autoreleasepool {
        for (MetalDevice &d : state.metal_devices) {
            for (void *&pso : d.pipelines) {
                if (pso)
                    (void) (__bridge_transfer id<MTLComputePipelineState>) pso;
                pso = nullptr;
            }
            for (auto &by_op : d.reduce_pipelines) {
                for (auto &by_type : by_op) {
                    for (void *&pso : by_type) {
                        if (pso)
                            (void) (__bridge_transfer id<MTLComputePipelineState>) pso;
                        pso = nullptr;
                    }
                }
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
        metal_buffer_lut.clear();
    }
}

// ============================================================================
//  Kernel compilation
// ============================================================================

/// Metal unit artifacts store a retained id<MTLLibrary> in ptr[0] and
/// id<MTLFunction> in ptr[1]. The device index serves as the cache salt.
static void jitc_metal_unit_release(UnitArtifact &a) {
    @autoreleasepool {
        (void) (__bridge_transfer id<MTLLibrary>) a.ptr[0];
        (void) (__bridge_transfer id<MTLFunction>) a.ptr[1];
    }
}

// ----------------------------------------------------------------------------
//  Disk cache
// ----------------------------------------------------------------------------
//
// Metal caches compiled shaders in ``$DARWIN_USER_CACHE_DIR/com.apple.metal``,
// but that cache saturates at around a gigabyte and then drops its largest
// entries after a handful of unrelated insertions. Dr.Jit therefore maintains
// its own cache in ``~/.drjit``, at three granularities:
//
//  - ``MetalLibrary`` (``.air.metallib``): a library image produced by the
//    MSL front end, per unit
//  - ``MetalFunction`` (``.func.metallib``): a binary archive holding one
//    callable's device-specialized machine code (the back-end result), per unit
//  - ``MetalPipeline`` (``.pso.metallib``): a binary archive holding a
//    kernel's pipeline machine code

/// Cleared once the library image cannot be exported, which disables AIR
/// caching for the remainder of the process (archives still work).
static bool metal_library_export_available = true;

/// Extract the compiled image of ``lib`` via private API. Returns false when
/// the accessor is unavailable or misbehaves, which disables AIR caching.
static bool jitc_metal_library_data(id<MTLLibrary> lib,
                                    std::vector<uint8_t> &out) {
    static SEL selector = nullptr;
    static bool initialized = false, reported = false;
    static std::mutex export_mutex;
    std::lock_guard<std::mutex> guard(export_mutex);

    auto disable = [](const char *reason) {
        jitc_log(Info, "jit_metal_kernel_compile(): %s, the Metal AIR cache "
                       "is disabled.", reason);
        metal_library_export_available = false;
        return false;
    };

    if (!initialized) {
        initialized = true;
        const char *env = getenv("DRJIT_DISABLE_AIR_CACHE");
        if (env && *env && strcmp(env, "0") != 0)
            return disable("DRJIT_DISABLE_AIR_CACHE is set");
        selector = NSSelectorFromString(@"libraryDataContents");
    }

    if (!metal_library_export_available)
        return false;

    if (!selector || ![lib respondsToSelector:selector])
        return disable("-[MTLLibrary libraryDataContents] is unavailable");

    // An explicit cast keeps the compiler from inferring a wrong signature
    id result = ((id (*)(id, SEL)) objc_msgSend)(lib, selector);

    if (![result isKindOfClass:[NSData class]] || ((NSData *) result).length == 0)
        return disable("-[MTLLibrary libraryDataContents] returned an "
                       "unexpected value");

    NSData *data = (NSData *) result;
    out.resize(data.length);
    std::memcpy(out.data(), data.bytes, data.length);

    if (!reported) {
        reported = true;
        jitc_log(Info, "jit_metal_kernel_compile(): caching compiled kernels via "
                       "-[MTLLibrary libraryDataContents].");
    }

    return true;
}

/// Load a ``MTLLibrary`` or return nil upon failure
static id<MTLLibrary> jitc_metal_library_from_data(id<MTLDevice> dev,
                                                   const std::vector<uint8_t> &data) {
    NSError *err = nil;
    dispatch_data_t dd = dispatch_data_create(data.data(), data.size(), nullptr,
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> lib = [dev newLibraryWithData:dd error:&err];

    if (!lib)
        jitc_log(Debug, "jit_metal_kernel_compile(): could not instantiate a "
                        "cached library image: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");

    return lib;
}

/// Removes a scratch file when it goes out of scope
struct ScopedFile {
    NSURL *url = nil;
    ~ScopedFile() {
        if (url)
            [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
    }
};

/// Metal reads and writes binary archives through a URL, so both directions
/// detour via a scratch file in the temporary directory.
static NSURL *jitc_metal_scratch_url() {
    static std::atomic<uint32_t> counter { 0 };
    NSString *name = [NSString stringWithFormat:@"drjit-%u-%u.metalar",
                                                (unsigned) getpid(),
                                                counter.fetch_add(1)];
    return [NSURL fileURLWithPath:[NSTemporaryDirectory()
                                      stringByAppendingPathComponent:name]];
}

/// Load a ``MTLBinaryArchive`` or return nil upon failure
static id<MTLBinaryArchive>
jitc_metal_archive_from_data(id<MTLDevice> dev,
                             const std::vector<uint8_t> &data) {
    ScopedFile scratch;
    scratch.url = jitc_metal_scratch_url();

    NSData *nsdata = [NSData dataWithBytesNoCopy:(void *) data.data()
                                          length:data.size()
                                    freeWhenDone:NO];

    NSError *err = nil;
    id<MTLBinaryArchive> archive = nil;

    if ([nsdata writeToURL:scratch.url options:0 error:&err]) {
        MTLBinaryArchiveDescriptor *bad = [MTLBinaryArchiveDescriptor new];
        bad.url = scratch.url;
        archive = [dev newBinaryArchiveWithDescriptor:bad error:&err];
    }

    if (!archive)
        jitc_log(Debug, "jit_metal_kernel_compile(): could not load a cached "
                        "binary archive: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");

    return archive;
}

/// Serialize ``archive`` into ``out``. Returns false on failure.
static bool jitc_metal_archive_data(id<MTLBinaryArchive> archive,
                                    std::vector<uint8_t> &out) {
    ScopedFile scratch;
    scratch.url = jitc_metal_scratch_url();

    NSError *err = nil;
    if (![archive serializeToURL:scratch.url error:&err]) {
        jitc_log(Debug, "jit_metal_kernel_compile(): could not serialize a "
                        "binary archive: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");
        return false;
    }

    NSData *data = [NSData dataWithContentsOfURL:scratch.url
                                         options:0
                                           error:&err];
    if (!data || data.length == 0) {
        jitc_log(Debug, "jit_metal_kernel_compile(): could not read back a "
                        "serialized binary archive: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");
        return false;
    }

    out.resize(data.length);
    std::memcpy(out.data(), data.bytes, data.length);

    return true;
}

/// Serialize ``archive`` and store it in the kernel cache on a pool worker.
/// The caller does not wait for the store, which overlaps with whatever
/// follows the kernel compilation.
static void jitc_metal_archive_store_async(id<MTLBinaryArchive> archive,
                                           CacheKind kind, XXH128_hash_t hash) {
    struct Payload {
        id<MTLBinaryArchive> archive;
        CacheKind kind;
        XXH128_hash_t hash;
    };

    Task *task = task_submit_dep(
        nullptr, nullptr, 0, 1,
        [](uint32_t, void *p) {
            @autoreleasepool {
                Payload *payload = (Payload *) p;
                std::vector<uint8_t> data;
                if (jitc_metal_archive_data(payload->archive, data))
                    jitc_cache_blob_store(payload->kind, payload->hash,
                                          data.data(), data.size());
            }
        },
        new Payload { archive, kind, hash }, 0,
        [](void *p) { delete (Payload *) p; }, /* always_async = */ 1);

    task_release(task);
}

/// Fold device identity, OS build, and cache version into a disk-cache key.
static XXH128_hash_t jitc_metal_disk_key(id<MTLDevice> dev,
                                         XXH128_hash_t hash) {
    // A seeded reset consults the previous state, so it must be initialized
    XXH3_state_t xs;
    XXH3_INITSTATE(&xs);
    XXH3_128bits_reset_withSeed(&xs, DRJIT_CACHE_VERSION);
    XXH3_128bits_update(&xs, &hash, sizeof(XXH128_hash_t));

    uint64_t registry_id = dev.registryID;
    XXH3_128bits_update(&xs, &registry_id, sizeof(uint64_t));

    const char *name = dev.name.UTF8String;
    XXH3_128bits_update(&xs, name, std::strlen(name));

    const char *os =
        NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String;
    XXH3_128bits_update(&xs, os, std::strlen(os));

    return XXH3_128bits_digest(&xs);
}

/// State of one per-unit compilation (see jitc_metal_kernel_compile())
struct MetalCompileJob : UnitCompileJob {
    XXH128_hash_t disk_key;        // device/OS-salted unit hash
    std::vector<uint8_t> bfn_data; // serialized binary-function archive
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    NSError *error = nil;
    bool hit = false;              // served by the in-memory unit cache
    bool from_disk = false;        // library came from a disk entry
    bool bfn_hit = false;          // binary function came from a disk archive
};

/// Instantiate the function described by ``fd`` with machine code required
/// to come from ``ar`` (``FailOnBinaryArchiveMiss`` makes a hit provable)
static id<MTLFunction> jitc_metal_function_from_archive(id<MTLLibrary> lib,
                                                        MTLFunctionDescriptor *fd,
                                                        id<MTLBinaryArchive> ar,
                                                        NSError **e) {
    MTLFunctionDescriptor *fd2 = [fd copy];
    fd2.binaryArchives = @[ ar ];
    if (@available(macOS 15.0, *))
        fd2.options = MTLFunctionOptionCompileToBinary |
                      MTLFunctionOptionFailOnBinaryArchiveMiss;
    return [lib newFunctionWithDescriptor:fd2 error:e];
}

/// Produce the device-specialized binary function of one callable unit from
/// its (compiled or cached) library. Prefers a previously serialized archive;
/// otherwise compiles once into a fresh archive so the result can be stored
/// without a second back-end pass. Runs concurrently on the nanothread pool.
static void jitc_metal_build_binary_function(id<MTLDevice> dev,
                                             MetalCompileJob *jp) {
    id<MTLLibrary> lib = jp->library;

    MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
    fd.name = @(jp->symbol);
    fd.options = MTLFunctionOptionCompileToBinary;

    // 1. Reuse a previously serialized binary-function archive
    if (!jp->bfn_data.empty()) {
        id<MTLBinaryArchive> ar = jitc_metal_archive_from_data(dev, jp->bfn_data);
        if (ar) {
            NSError *e = nil;
            id<MTLFunction> f = jitc_metal_function_from_archive(lib, fd, ar, &e);
            if (f) {
                jp->function = f;
                jp->bfn_hit = true;
                return;
            }
            jitc_log(Debug, "jit_metal_kernel_compile(): binary-function "
                            "archive did not apply to \"%s\": %s",
                     jp->symbol,
                     e ? e.localizedDescription.UTF8String : "<unknown>");
        }
    }

    // 2. Compile into a fresh archive, then instantiate from it (one back-end
    //    pass total), and persist the archive
    if (jitc_cache_writable()) {
        NSError *e = nil;
        id<MTLBinaryArchive> ar = [dev
            newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
                                     error:&e];
        if (ar && [ar addFunctionWithDescriptor:fd library:lib error:&e]) {
            id<MTLFunction> f = jitc_metal_function_from_archive(lib, fd, ar, &e);
            if (f) {
                jp->function = f;
                std::vector<uint8_t> data;
                if (jitc_metal_archive_data(ar, data))
                    jitc_cache_blob_store_async(CacheKind::MetalFunction,
                                                jp->disk_key, std::move(data));
                return;
            }
        }
        jitc_log(Debug, "jit_metal_kernel_compile(): could not archive the "
                        "binary function \"%s\" (%s), compiling directly.",
                 jp->symbol, e ? e.localizedDescription.UTF8String : "<unknown>");
    }

    // 3. Plain compile (nothing persisted)
    NSError *e = nil;
    jp->function = [lib newFunctionWithDescriptor:fd error:&e];
    if (!jp->function)
        jp->error = e;
}

/// Compile a compilation unit (if not loaded from disk) and persist its AIR
/// image, then fetch its entry point or build a device-specialized binary
/// function. Runs concurrently on the nanothread pool.
static void jitc_metal_compile_unit(id<MTLDevice> dev, MTLCompileOptions *opts,
                                    MetalCompileJob *jp, bool entry) {
    @autoreleasepool {
        if (!jp->from_disk) {
            NSString *src =
                [[NSString alloc] initWithBytes:jp->source
                                         length:jp->source_size
                                       encoding:NSUTF8StringEncoding];
            NSError *e = nil;
            id<MTLLibrary> l = [dev newLibraryWithSource:src
                                                 options:opts
                                                   error:&e];
            if (!l) {
                jp->error = e;
                return;
            }
            jp->library = l;

            if (jitc_cache_writable()) {
                std::vector<uint8_t> data;
                if (jitc_metal_library_data(l, data))
                    jitc_cache_blob_store_async(CacheKind::MetalLibrary,
                                                jp->disk_key, std::move(data));
            }
        }

        if (entry)
            jp->function = [jp->library newFunctionWithName:@(jp->symbol)];
        else
            jitc_metal_build_binary_function(dev, jp);
    }
}

bool jitc_metal_kernel_compile(ThreadState *ts, Kernel &kernel) {
    @autoreleasepool {
        if (state.metal_devices.empty())
            jitc_fail("jitc_metal_kernel_compile(): no Metal devices initialized.");

        id<MTLDevice> dev = (__bridge id<MTLDevice>) ts->metal_device;
        int device_id = ts->device;

        XXH128_hash_t khash = kernel_hash;

        size_t n_call = callable_units.size(),
               n_units = 1 + n_call,
               source_bytes = 0,
               n_misses = 0;

        std::vector<MetalCompileJob> jobs(n_units);

        for (size_t i = 0; i < n_units; ++i) {
            MetalCompileJob &job = jobs[i];
            jitc_unit_job_init(i, job);
            source_bytes += job.source_size;

            UnitArtifact artifact;
            if (jitc_unit_cache_lookup(JitBackend::Metal, job.unit_hash,
                                       (uint64_t) device_id, artifact)) {
                job.library = (__bridge id<MTLLibrary>) artifact.ptr[0];
                job.function = (__bridge id<MTLFunction>) artifact.ptr[1];
                job.hit = true;
                continue;
            }

            job.disk_key = jitc_metal_disk_key(dev, job.unit_hash);
            n_misses++;
        }

        // Resolve the union of custom intersection functions across every
        // scene registered with this kernel
        NSMutableArray<id<MTLFunction>> *isect_fns = [NSMutableArray array];
        std::vector<std::string> seen;
        for (MetalScene *scene : metal_kernel_scenes) {
            id<MTLLibrary> isect_lib =
                scene ? (__bridge id<MTLLibrary>) scene->intersection_fn_library
                      : nil;
            if (!isect_lib)
                continue;
            for (const std::string &name : scene->intersection_fns) {
                if (std::find(seen.begin(), seen.end(), name) != seen.end())
                    continue;
                seen.push_back(name);
                id<MTLFunction> f = [isect_lib newFunctionWithName:@(name.c_str())];
                if (!f)
                    jitc_fail("jitc_metal_kernel_compile(): intersection function "
                              "\"%s\" not found in user-supplied library.",
                              name.c_str());
                [isect_fns addObject:f];
            }
        }

        bool has_call_table = metal_vft_arg_index >= 0;

        MTLCompileOptions *opts = [MTLCompileOptions new];

        // macOS 15+ is guaranteed by jitc_metal_init(); Metal 4 needs macOS 26
        if (@available(macOS 15.0, *)) {
            opts.languageVersion = MTLLanguageVersion3_2;
#if defined(DRJIT_SUPPORTS_METAL4)
            if (uses_metal4) {
                if (@available(macOS 26.0, *))
                    opts.languageVersion = MTLLanguageVersion4_0;
            }
#endif

            // The relaxed/fast math modes are a little aggressive and break the
            // Dr.Jit test suite. We opt in on a per instruction basis by calling
            // math functions from the ``fast::`` namespace
            opts.mathMode = MTLMathModeSafe;
        }

        opts.libraryType = MTLLibraryTypeExecutable;

        id<MTLComputePipelineState> pso = nil;
        id<MTLVisibleFunctionTable> vft = nil;
        bool pso_from_archive = false;
        NSMutableArray<id<MTLFunction>> *callable_fns =
            [NSMutableArray arrayWithCapacity:n_call];

        // Release the lock while compiling and linking. The job sources stay
        // valid throughout (see UnitCompileJob in unit.h)
        {
            unlock_guard guard(state.lock);

            // Compile the unit-cache misses concurrently
            if (n_misses > 0) {
                // Disk cache probes: per-unit AIR images, and the binary-
                // function archives of the callables
                for (size_t i = 0; i < n_units; ++i) {
                    MetalCompileJob &job = jobs[i];
                    if (job.hit)
                        continue;

                    std::vector<uint8_t> data;
                    if (jitc_cache_blob_load(CacheKind::MetalLibrary,
                                             job.disk_key, data)) {
                        id<MTLLibrary> l = jitc_metal_library_from_data(dev, data);
                        if (l && (i != 0 ||
                                  (job.function =
                                       [l newFunctionWithName:@(job.symbol)]))) {
                            job.library = l;
                            job.from_disk = true;
                        }
                    }

                    if (i != 0)
                        jitc_cache_blob_load(CacheKind::MetalFunction,
                                             job.disk_key, job.bfn_data);
                }

                // The entry unit only needs its plain function (the PSO
                // compute function); callables continue into a binary-
                // function build even when the library came from disk.
                std::vector<uint32_t> pending;
                for (size_t i = 0; i < n_units; ++i) {
                    const MetalCompileJob &job = jobs[i];
                    if (job.hit || (i == 0 && job.from_disk))
                        continue;
                    pending.push_back((uint32_t) i);
                }

                if (!pending.empty()) {
                    MetalCompileJob *jobs_p = jobs.data();
                    jitc_unit_compile_parallel(
                        pending,
                        [&](uint32_t i) { return jobs_p[i].source_size; },
                        [&](uint32_t i) {
                            jitc_metal_compile_unit(dev, opts, &jobs_p[i],
                                                    i == 0);
                        });
                }

                for (size_t i = 0; i < n_units; ++i) {
                    MetalCompileJob &job = jobs[i];
                    if (job.hit || job.function)
                        continue;
                    jitc_fail("jitc_metal_kernel_compile(): compilation of "
                              "%s\"%s\" failed: %s\n\n--- Source code ---\n%s",
                              i == 0 ? "" : "callable ", job.symbol,
                              job.error
                                  ? job.error.localizedDescription.UTF8String
                                  : "<unknown>",
                              job.source);
                }

                // Publish the new artifacts; the cache pins them (+1 each)
                for (MetalCompileJob &job : jobs) {
                    if (job.hit)
                        continue;
                    UnitArtifact artifact {
                        { (__bridge_retained void *) job.library,
                          (__bridge_retained void *) job.function }, 0, 0 };
                    jitc_unit_cache_insert(JitBackend::Metal, job.unit_hash,
                                           (uint64_t) device_id, artifact,
                                           jitc_metal_unit_release);
                }
            }

            // ---- Link: pipeline state + visible function table -------------

            for (size_t i = 1; i < n_units; ++i)
                [callable_fns addObject:jobs[i].function];

            MTLComputePipelineDescriptor *desc =
                [MTLComputePipelineDescriptor new];
            desc.computeFunction = jobs[0].function;

            if (isect_fns.count > 0 || callable_fns.count > 0) {
                MTLLinkedFunctions *lf = [MTLLinkedFunctions new];
                if (isect_fns.count > 0)
                    lf.functions = isect_fns;
                if (callable_fns.count > 0)
                    lf.binaryFunctions = callable_fns;
                desc.linkedFunctions = lf;
            }

            // Pipeline machine code is cached in a per-kernel binary archive.
            // ``FailOnBinaryArchiveMiss`` ensures that pipeline creation
            // reuses the archived code or fails instead of recompiling.
            XXH128_hash_t pso_key = jitc_metal_disk_key(dev, khash);

            {
                std::vector<uint8_t> data;
                if (jitc_cache_blob_load(CacheKind::MetalPipeline, pso_key,
                                         data)) {
                    id<MTLBinaryArchive> ar =
                        jitc_metal_archive_from_data(dev, data);
                    if (ar) {
                        MTLComputePipelineDescriptor *desc_a = [desc copy];
                        desc_a.binaryArchives = @[ ar ];
                        NSError *e = nil;
                        pso = [dev newComputePipelineStateWithDescriptor:desc_a
                                       options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                    reflection:nil
                                         error:&e];
                        pso_from_archive = pso != nil;
                        if (!pso)
                            jitc_log(Debug,
                                     "jit_metal_kernel_compile(): the cached "
                                     "binary archive does not contain this "
                                     "pipeline: %s",
                                     e ? e.localizedDescription.UTF8String
                                       : "<unknown>");
                    }
                }
            }

            if (!pso && jitc_cache_writable()) {
                NSError *e = nil;
                id<MTLBinaryArchive> ar = [dev
                    newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
                                             error:&e];
                if (ar &&
                    [ar addComputePipelineFunctionsWithDescriptor:desc error:&e]) {
                    MTLComputePipelineDescriptor *desc_a = [desc copy];
                    desc_a.binaryArchives = @[ ar ];
                    pso = [dev newComputePipelineStateWithDescriptor:desc_a
                                   options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                reflection:nil
                                     error:&e];
                    if (pso)
                        jitc_metal_archive_store_async(
                            ar, CacheKind::MetalPipeline, pso_key);
                }
                if (!pso)
                    jitc_log(Debug, "jit_metal_kernel_compile(): could not "
                                    "archive the pipeline (%s), compiling "
                                    "directly.",
                             e ? e.localizedDescription.UTF8String
                               : "<unknown>");
            }

            if (!pso) {
                NSError *err = nil;
                pso = [dev newComputePipelineStateWithDescriptor:desc
                                                         options:MTLPipelineOptionNone
                                                      reflection:nil
                                                           error:&err];
                if (!pso)
                    jitc_fail("jitc_metal_kernel_compile(): pipeline creation "
                              "failed: %s",
                              err ? err.localizedDescription.UTF8String
                                  : "<unknown>");
            }

            // Build the visible function table used to dispatch indirect
            // calls. Function handles derive from the pipeline, which makes
            // the table PSO-specific.
            if (callable_fns.count > 0) {
                MTLVisibleFunctionTableDescriptor *vftd =
                    [MTLVisibleFunctionTableDescriptor new];
                vftd.functionCount = callable_fns.count;
                vft = [pso newVisibleFunctionTableWithDescriptor:vftd];
                for (NSUInteger i = 0; i < callable_fns.count; ++i) {
                    id<MTLFunctionHandle> h =
                        [pso functionHandleWithFunction:callable_fns[i]];
                    if (!h)
                        jitc_fail("jitc_metal_kernel_compile(): could not "
                                  "obtain a function handle for callable %u.",
                                  (uint32_t) i);
                    [vft setFunction:h atIndex:i];
                }
            }
        }

        kernel.metal.pipeline       = (__bridge_retained void *) pso;
        kernel.metal.library        = (__bridge_retained void *) jobs[0].library;
        kernel.metal.call_table_vft = vft ? (__bridge_retained void *) vft
                                          : nullptr;
        // Check if kernels must be launched with a call table slot
        kernel.metal.has_call_table = has_call_table;
        kernel.size = (uint32_t) source_bytes;

        // Report a soft miss when every unit and the pipeline were
        // reconstructed from the in-memory or disk caches
        bool all_cached = pso_from_archive &&
                          (jobs[0].hit || jobs[0].from_disk);
        for (size_t i = 1; i < n_units; ++i)
            all_cached &= jobs[i].hit ||
                          (jobs[i].from_disk && jobs[i].bfn_hit);
        return all_cached;
    }
}

void jitc_metal_kernel_free(Kernel &kernel) {
    @autoreleasepool {
        if (kernel.metal.pipeline)
            (void) (__bridge_transfer id<MTLComputePipelineState>)
                kernel.metal.pipeline;
        if (kernel.metal.library)
            (void) (__bridge_transfer id<MTLLibrary>) kernel.metal.library;
        if (kernel.metal.call_table_vft)
            (void) (__bridge_transfer id<MTLVisibleFunctionTable>)
                kernel.metal.call_table_vft;
        kernel.metal.pipeline       = nullptr;
        kernel.metal.library        = nullptr;
        kernel.metal.call_table_vft = nullptr;
    }
}

/// Retain an extra reference to a kernel-history command buffer
void jitc_metal_history_retain(void *cb_ptr) {
    (void) (__bridge_retained void *) (__bridge id<MTLCommandBuffer>) cb_ptr;
}

/// Wait for a kernel-history command buffer and return its GPU time (ms)
float jitc_metal_history_wait(void *cb_ptr) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) cb_ptr;
        [cb waitUntilCompleted];
        return (float) ((cb.GPUEndTime - cb.GPUStartTime) * 1000);
    }
}

/// Release a kernel-history command buffer without waiting for it
void jitc_metal_history_release(void *cb_ptr) {
    @autoreleasepool {
        if (cb_ptr)
            (void) (__bridge_transfer id<MTLCommandBuffer>) cb_ptr;
    }
}

/// The Metal backend generates unformatted MSL since indentation tracking is
/// costly during code generation. This function appends an indented copy of
/// the MSL in ``src`` to ``out``.
///
/// The function relies on three properties of the emitted MSL:
///   1. Every ``{``/``}`` is related to control flow (i.e. braces don't occur
///      in strings, initializer lists, etc.)
///   2. The only comments are ``//`` line comments
///   3. No statement is split across lines by a blank or comment line, which
///      would reset the continuation state mid-statement.
///
/// These are currently satisfied. Violating them degrades indentation quality
/// but never changes semantics.
void jitc_metal_format(const char *src, size_t n, StringBuffer &out) {
    int depth = 0;
    bool stmt_open = false; // previous code line left a statement unterminated

    for (size_t i = 0; i < n; ) {
        // Carve out one line and trim its horizontal whitespace.
        size_t b = i;
        while (i < n && src[i] != '\n')
            i++;
        size_t e = i;
        if (i < n)
            i++; // consume '\n'
        while (b < e && (src[b] == ' ' || src[b] == '\t'))
            b++;
        while (e > b && (src[e - 1] == ' ' || src[e - 1] == '\t'))
            e--;

        if (b == e) { // blank line
            out.put('\n');
            stmt_open = false;
            continue;
        }

        bool is_comment = src[b] == '/' && b + 1 < e && src[b + 1] == '/';
        bool is_preproc = src[b] == '#';

        // Scan for brace deltas and the last significant character, stopping at
        // a trailing ``//`` comment.
        int opens = 0, closes = 0, leading_close = 0;
        bool seen = false;
        char last = 0;
        for (size_t k = b; k < e; k++) {
            char c = src[k];
            if (c == '/' && k + 1 < e && src[k + 1] == '/')
                break;
            if (c == '{') {
                opens++;
                seen = true;
            } else if (c == '}') {
                closes++;
                if (!seen)
                    leading_close++;
            } else if (c != ' ' && c != '\t') {
                seen = true;
            }
            if (c != ' ' && c != '\t')
                last = c;
        }

        int indent = depth - leading_close;
        if (indent < 0)
            indent = 0;
        if (stmt_open && !is_preproc && !is_comment)
            indent++; // hanging indent for continuation lines

        if (indent > 0)
            out.put(' ', (size_t) indent * 4);
        out.put(src + b, e - b);
        out.put('\n');

        depth += opens - closes;
        if (depth < 0)
            depth = 0;

        stmt_open = !(is_comment || is_preproc || last == ';' || last == '{' ||
                      last == '}');
    }

    jitc_assert(depth == 0, "jitc_metal_format(): mismatched braces!");
}

/// Flush the thread's pending command buffer and wait for the GPU to finish
void jitc_metal_sync(ThreadState *ts) {
    ((MetalThreadState *) ts->actual_state())->flush(/* wait = */ true);
}

/// Submit the thread's pending command buffer to the GPU without waiting
void jitc_metal_flush(ThreadState *ts) {
    ((MetalThreadState *) ts->actual_state())->flush(/* wait = */ false);
}

// ============================================================================
//  Ray Tracing API
// ============================================================================

uint32_t jitc_metal_configure_scene(void *accel, void **resources,
                                    uint32_t n_resources,
                                    void *intersection_fn_library,
                                    uint32_t n_ift_entries,
                                    const char **ift_function_names,
                                    uint32_t n_ift_buffers,
                                    void **ift_buffers,
                                    const uint32_t *ift_buffer_slots,
                                    uint32_t geometry_types_mask) {
    jitc_log(InfoSym,
             "jit_metal_configure_scene(accel=" DRJIT_PTR ", "
             "n_resources=%u, ift_lib=" DRJIT_PTR ", n_ift=%u, geom_mask=%u)",
             (uintptr_t) accel, n_resources,
             (uintptr_t) intersection_fn_library,
             n_ift_entries, geometry_types_mask);

    // A fresh scene per configuration; a geometry edit just registers another.
    MetalScene *scene = new MetalScene();

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

    scene->intersection_fns.reserve(n_ift_entries);
    for (uint32_t i = 0; i < n_ift_entries; ++i)
        scene->intersection_fns.emplace_back(ift_function_names[i]);

    scene->ift_bindings.reserve(n_ift_buffers);
    for (uint32_t i = 0; i < n_ift_buffers; ++i)
        scene->ift_bindings.push_back({ ift_buffer_slots[i], ift_buffers[i] });

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
                 s->intersection_fns.size(),
                 s->ift_cache.size());
        // Release the cached TLAS/IFT resource handles
        if (s->accel_handle)
            jitc_var_dec_ref(s->accel_handle);
        if (s->ift_handle)
            jitc_var_dec_ref(s->ift_handle);
        for (auto &kv : s->ift_cache) {
            if (kv.second)
                (void) (__bridge_transfer id<MTLIntersectionFunctionTable>)
                    kv.second;
        }
        if (s->intersection_fn_library)
            (void) (__bridge_transfer id<MTLLibrary>) s->intersection_fn_library;
        void (*cleanup)(void *) = s->cleanup;
        void *cleanup_payload = s->cleanup_payload;
        delete s;
        // Release the application-owned Metal objects (TLAS/BLAS/buffers)
        if (cleanup)
            cleanup(cleanup_payload);
    };

    jitc_var_set_callback(index, callback, scene, true);

    return index;
}

MetalScene *jitc_metal_get_scene(uint32_t scene_index) {
    Variable *v = scene_index ? jitc_var(scene_index) : nullptr;
    if (!v || (VarKind) v->kind != VarKind::Nop ||
        (VarType) v->type != VarType::Void)
        jitc_fail("jitc_metal_get_scene(): r%u does not wrap a Metal scene.",
                  scene_index);
    return (MetalScene *) v->literal;
}

uint32_t jitc_metal_make_resource_handle(void *ptr, ResourceKind kind) {
    if (!ptr)
        return 0;
    uint32_t backing = jitc_var_mem_map(JitBackend::Metal, VarType::UInt64,
                                        ptr, 1, /*free=*/0);
    uint32_t handle = jitc_var_resource_pointer(backing, kind);
    jitc_var_dec_ref(backing);
    return handle;
}

uint32_t jitc_metal_scene_resource_handle(MetalScene *scene, ResourceKind kind) {
    if (!scene)
        return 0;
    uint32_t &slot = (kind == ResourceKind::IFT) ? scene->ift_handle
                                                 : scene->accel_handle;
    if (!slot)
        slot = jitc_metal_make_resource_handle(scene, kind);
    jitc_var_inc_ref(slot);
    return slot;
}

uint32_t jitc_metal_scene_owner_handle(uint32_t scene_index) {
    MetalScene *scene = jitc_metal_get_scene(scene_index);
    /* The data pointer is the MetalScene owner -- the same pointer carried by
       jit_metal_ray_trace's Accel/IFT parameters (their dep[3] backing maps
       this MetalScene*), so the recorder keys both to one input slot. */
    return jitc_var_mem_map(JitBackend::Metal, VarType::UInt64,
                            (void *) scene, 1, /*free=*/0);
}

/// Lazily build (and cache) an IntersectionFunctionTable for the given scene
/// + compute pipeline. The function handles are derived from the pipeline so
/// each pipeline needs its own IFT instance. The cache owns the (+1) IFT; the
/// returned pointer is borrowed.
void *
jitc_metal_get_or_create_ift_for_scene(MetalScene *scene, void *pso_) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>) pso_;
    if (!scene || !pso || scene->intersection_fns.empty())
        return nullptr;

    // Cache hit: we already built an IFT for this pipeline.
    for (const auto &kv : scene->ift_cache)
        if (kv.first == pso_)
            return kv.second;

    id<MTLLibrary> isect_lib =
        (__bridge id<MTLLibrary>) scene->intersection_fn_library;
    if (!isect_lib)
        return nullptr;

    uint32_t n_ift = (uint32_t) scene->intersection_fns.size();

    // Resolve unique function objects (deduplicate so we look up each function
    // only once even if the IFT references it multiple times).
    NSMutableDictionary<NSString *, id<MTLFunction>> *unique_fns =
        [NSMutableDictionary dictionary];
    for (const std::string &fn_name : scene->intersection_fns) {
        NSString *name = @(fn_name.c_str());
        if (unique_fns[name])
            continue;
        id<MTLFunction> f = [isect_lib newFunctionWithName:name];
        if (!f)
            jitc_fail("jitc_metal_get_or_create_ift_for_scene(): intersection "
                      "function \"%s\" not found in user-supplied library.",
                      fn_name.c_str());
        unique_fns[name] = f;
    }

    MTLIntersectionFunctionTableDescriptor *iftd =
        [MTLIntersectionFunctionTableDescriptor new];
    iftd.functionCount = n_ift;
    id<MTLIntersectionFunctionTable> ift =
        [pso newIntersectionFunctionTableWithDescriptor:iftd];

    for (uint32_t i = 0; i < n_ift; ++i) {
        id<MTLFunction> f = unique_fns[@(scene->intersection_fns[i].c_str())];
        id<MTLFunctionHandle> handle = [pso functionHandleWithFunction:f];
        [ift setFunction:handle atIndex:i];
    }

    // The table's argument table is shared by all entries; bind each slot once.
    for (const IFTBinding &b : scene->ift_bindings)
        if (id<MTLBuffer> buf = (__bridge id<MTLBuffer>) b.buffer)
            [ift setBuffer:buf offset:0 atIndex:b.slot];

    scene->ift_cache.push_back({ pso_, (__bridge_retained void *) ift });
    return scene->ift_cache.back().second;
}

void *jitc_metal_context_impl() {
    return thread_state(JitBackend::Metal)->metal_device;
}

void *jitc_metal_command_queue_impl() {
    return thread_state(JitBackend::Metal)->metal_queue;
}

// ============================================================================
// GPU profile capture
// ============================================================================

static id<MTLCaptureScope> metal_capture_scope = nil;
static bool metal_capture_active = false;

void jitc_metal_profile_start() {
    if (metal_capture_active) {
        jitc_log(Warn, "jit_profile_start(): a Metal capture is already active.");
        return;
    }

    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>) thread_state(JitBackend::Metal)->metal_queue;

    MTLCaptureManager *mgr = [MTLCaptureManager sharedCaptureManager];

    metal_capture_scope = [mgr newCaptureScopeWithCommandQueue:queue];
    metal_capture_scope.label = @"Dr.Jit";
    mgr.defaultCaptureScope = metal_capture_scope;

    // Attempt a programmatic capture so a .gputrace can be produced directly
    // from the command line.
    MTLCaptureDescriptor *desc = [MTLCaptureDescriptor new];
    desc.captureObject = metal_capture_scope;

    if ([mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        const char *path_env = getenv("DRJIT_METAL_CAPTURE_PATH");
        NSString *path = path_env ? @(path_env) : @"drjit.gputrace";
        // A pre-existing document at the destination makes startCapture fail.
        [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
        desc.destination = MTLCaptureDestinationGPUTraceDocument;
        desc.outputURL = [NSURL fileURLWithPath:path];
    }

    NSError *error = nil;
    if (![mgr startCaptureWithDescriptor:desc error:&error]) {
        jitc_log(Debug,
                 "jit_profile_start(): could not start a Metal GPU capture (%s). "
                 "Set MTL_CAPTURE_ENABLED=1 for command-line capture, or trigger "
                 "one from Xcode.",
                 error ? error.localizedDescription.UTF8String : "unknown error");
    }

    [metal_capture_scope beginScope];
    metal_capture_active = true;
}

void jitc_metal_profile_stop() {
    if (!metal_capture_active)
        return;

    [metal_capture_scope endScope];

    MTLCaptureManager *mgr = [MTLCaptureManager sharedCaptureManager];
    if (mgr.isCapturing)
        [mgr stopCapture];
    if (mgr.defaultCaptureScope == metal_capture_scope)
        mgr.defaultCaptureScope = nil;

    metal_capture_scope = nil;
    metal_capture_active = false;
}

void jitc_metal_ray_trace(uint32_t n_args, uint32_t *args,
                          uint32_t mask, uint32_t *out,
                          uint32_t n_out, uint32_t scene, int shadow) {
    if (n_args != 9 && n_args != 10)
        jitc_raise("jit_metal_ray_trace(): expected 9 or 10 ray arguments, "
                   "got %u.", n_args);
    if (n_args == 10 && jitc_var_type(args[9]) != VarType::UInt32)
        jitc_raise("jit_metal_ray_trace(): the ray visibility mask "
                   "(argument 10) must be of type UInt32.");
    if (n_out != 8)
        jitc_raise("jit_metal_ray_trace(): expected 8 outputs, got %u.",
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
    td->shadow = shadow != 0;
    td->indices.reserve(n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        td->indices.push_back(args[i]);
        jitc_var_inc_ref(args[i]);
    }

    // Build acceleration-structure and intersection-function-table handles
    // The scene reference in dep[1] keeps MetalScene alive.
    MetalScene *scene_obj = jitc_metal_get_scene(scene);
    Ref accel_h = steal(jitc_metal_scene_resource_handle(scene_obj,
                                                         ResourceKind::Accel));
    Ref ift_h = steal(jitc_metal_scene_resource_handle(
        scene_obj->intersection_fn_library ? scene_obj : nullptr,
        ResourceKind::IFT));

    // dep[0]=valid, dep[1]=scene, dep[2]=accel handle, dep[3]=IFT.
    Ref trace;
    if (ift_h)
        trace = steal(jitc_var_new_node_4(
            JitBackend::Metal, VarKind::TraceRay, VarType::Void, size, symbolic,
            valid, jitc_var(valid), scene, jitc_var(scene),
            accel_h, jitc_var(accel_h), ift_h, jitc_var(ift_h),
            (uintptr_t) td));
    else
        trace = steal(jitc_var_new_node_3(
            JitBackend::Metal, VarKind::TraceRay, VarType::Void, size, symbolic,
            valid, jitc_var(valid), scene, jitc_var(scene),
            accel_h, jitc_var(accel_h), (uintptr_t) td));

    // Register cleanup callback
    jitc_var_set_callback(
        trace,
        [](uint32_t, int free, void *ptr) {
            if (free)
                delete (TraceData *) ptr;
        },
        td, true);

    // Create Extract children for each output
    VarType out_types[8] = {
        VarType::Bool,    // valid
        VarType::Float32, // distance
        VarType::Float32, // bary_u
        VarType::Float32, // bary_v
        VarType::UInt32,  // instance_id (raw TLAS instance index)
        VarType::UInt32,  // primitive_id
        VarType::UInt32,  // geometry_id
        VarType::UInt32   // user-provided instance ID
    };

    for (uint32_t i = 0; i < (td->shadow ? 1u : 8u); ++i)
        out[i] = jitc_var_new_node_1(
            JitBackend::Metal, VarKind::Extract, out_types[i],
            size, symbolic, trace, jitc_var(trace), (uint64_t) i);
}

#endif // defined(DRJIT_ENABLE_METAL)
