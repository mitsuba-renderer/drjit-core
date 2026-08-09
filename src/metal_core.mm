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

// Suppress the obsolete Carbon <CarbonCore/Threads.h>, whose ThreadState collides with Dr.Jit
#define __THREADS__
#import <Metal/Metal.h>
#import <objc/message.h>

#include <vector>
#include <mutex>
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
    MTLResourceID rid;
    switch (kind) {
        case ResourceKind::Accel:
            rid = ((__bridge id<MTLAccelerationStructure>)
                       ((MetalScene *) owner)->tlas).gpuResourceID;
            break;

        case ResourceKind::Texture:
            rid = ((__bridge id<MTLTexture>)
                       ((MetalTexResource *) owner)->object).gpuResourceID;
            break;

        case ResourceKind::Sampler:
            rid = ((__bridge id<MTLSamplerState>)
                       ((MetalTexResource *) owner)->object).gpuResourceID;
            break;

        default:
            // A buffer is an ordinary pointer; an IFT is PSO-dependent and is
            // refreshed at launch. Neither resolves to a gpuResourceID here.
            return false;
    }

    uint64_t v64;
    std::memcpy(&v64, &rid, sizeof(v64));
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

        for (id<MTLDevice> dev in devices) {
            if (![dev supportsFamily:MTLGPUFamilyMetal3]) {
                jitc_log(Warn,
                         "jit_metal_init(): skipping device \"%s\" because it "
                         "does not support Metal 3 (M1+ required).",
                         dev.name.UTF8String);
                continue;
            }

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
    }
}

// ============================================================================
//  Kernel compilation
// ============================================================================

/// Compile MSL source into an executable library. Failures are fatal. The
/// compiler diagnostics are returned in ``err_out`` even on success.
static id<MTLLibrary> jitc_metal_compile_source(id<MTLDevice> dev,
                                                const char *source,
                                                NSError **err_out) {
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

    id<MTLLibrary> lib = [dev newLibraryWithSource:@(source)
                                           options:opts
                                             error:err_out];

    if (!lib) {
        NSError *err = *err_out;
        jitc_fail("jitc_metal_kernel_compile(): MSL compilation failed:\n%s\n\n"
                  "--- Source code ---\n%s",
                  err ? err.localizedDescription.UTF8String : "<unknown>",
                  source);
    }

    return lib;
}

/// Resolve the kernel entry point along with every ``[[visible]]`` callable
/// that code generation registered in ``metal_kernel_callables``, appending the
/// latter to ``callables`` in callable-index order. Returns nil when any of
/// these functions is missing.
static id<MTLFunction>
jitc_metal_resolve_functions(id<MTLLibrary> lib, const char *kernel_name,
                             NSMutableArray<id<MTLFunction>> *callables) {
    id<MTLFunction> func = [lib newFunctionWithName:@(kernel_name)];
    if (!func)
        return nil;

    for (XXH128_hash_t hash : metal_kernel_callables) {
        char name[64];
        snprintf(name, sizeof(name), "func_%016llx%016llx",
                 (unsigned long long) hash.high64,
                 (unsigned long long) hash.low64);
        id<MTLFunction> f = [lib newFunctionWithName:@(name)];
        if (!f) {
            [callables removeAllObjects];
            return nil;
        }
        [callables addObject:f];
    }

    return func;
}

// ----------------------------------------------------------------------------
//  Disk cache
// ----------------------------------------------------------------------------
//
// Metal caches compiled shaders in ``$DARWIN_USER_CACHE_DIR/com.apple.metal``,
// but that cache saturates at around a gigabyte and then drops its largest
// entries after a handful of unrelated insertions. A Dr.Jit megakernel is
// precisely the entry it refuses to keep, so both halves of the compile are
// serialized into ``~/.drjit`` instead: the library image produced by the front
// end (MSL to AIR) and an ``MTLBinaryArchive`` holding the machine code
// produced by the back end (AIR to GPU ISA). Our own cache has the size bound
// that the system one lacks.

/// Cleared once the library image cannot be exported, which disables the disk
/// cache for the remainder of the process.
static bool metal_library_export_available = true;

/// Extract the compiled image of ``lib`` via private API. Returns false when
/// the accessor is unavailable or misbehaves, which disables the cache but
/// never breaks compilation.
static bool jitc_metal_library_data(id<MTLLibrary> lib,
                                    std::vector<uint8_t> &out) {
    static SEL selector = nullptr;
    static bool initialized = false, reported = false;

    auto disable = [](const char *reason) {
        jitc_log(Info, "jit_metal_kernel_compile(): %s, the Metal kernel cache "
                       "is disabled.", reason);
        metal_library_export_available = false;
        return false;
    };

    if (!initialized) {
        initialized = true;
        const char *env = getenv("DRJIT_METAL_NO_LIBRARY_EXPORT");
        if (env && *env && strcmp(env, "0") != 0)
            return disable("DRJIT_METAL_NO_LIBRARY_EXPORT is set");
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

/// Instantiate an ``MTLLibrary`` from a previously exported image. Returns nil
/// when Metal does not accept it, which merely costs us a cache hit.
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
    static uint32_t counter = 0;
    NSString *name = [NSString stringWithFormat:@"drjit-%u-%u.metalar",
                                                (unsigned) getpid(), counter++];
    return [NSURL fileURLWithPath:[NSTemporaryDirectory()
                                      stringByAppendingPathComponent:name]];
}

/// Compile ``desc`` into a fresh binary archive. This performs the back-end
/// half of the compilation and is correspondingly expensive. Returns nil when
/// the archive could not be created, in which case nothing is cached.
static id<MTLBinaryArchive>
jitc_metal_archive_new(id<MTLDevice> dev, MTLComputePipelineDescriptor *desc) {
    NSError *err = nil;
    id<MTLBinaryArchive> archive =
        [dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
                                      error:&err];

    if (archive &&
        ![archive addComputePipelineFunctionsWithDescriptor:desc error:&err])
        archive = nil;

    // DIAGNOSTIC: fatal so that a suite run cannot hide this path
    if (!archive)
        jitc_fail("jit_metal_kernel_compile(): could not build a binary "
                  "archive (%s), the back end will now run a second time.",
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
        jitc_log(Debug, "jit_metal_kernel_compile(): could not serialize the "
                        "binary archive: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");
        return false;
    }

    NSData *data = [NSData dataWithContentsOfURL:scratch.url
                                         options:0
                                           error:&err];
    if (!data || data.length == 0) {
        jitc_log(Debug, "jit_metal_kernel_compile(): could not read back the "
                        "serialized binary archive: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");
        return false;
    }

    out.resize(data.length);
    std::memcpy(out.data(), data.bytes, data.length);

    return true;
}

/// Create a compute pipeline from ``archive``. ``FailOnBinaryArchiveMiss``
/// makes a hit provable: creation either reuses the archived machine code or
/// fails, it never silently recompiles. Returns nil in the latter case.
static id<MTLComputePipelineState>
jitc_metal_pipeline_from_archive(id<MTLDevice> dev,
                                 MTLComputePipelineDescriptor *desc,
                                 id<MTLBinaryArchive> archive) {
    MTLComputePipelineDescriptor *desc_a = [desc copy];
    desc_a.binaryArchives = @[ archive ];

    NSError *err = nil;
    id<MTLComputePipelineState> pso = [dev
        newComputePipelineStateWithDescriptor:desc_a
                                      options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                   reflection:nil
                                        error:&err];

    // DIAGNOSTIC: Warn rather than Debug, only the load path reaches this
    if (!pso)
        jitc_log(Warn, "jit_metal_kernel_compile(): the binary archive does "
                       "not contain this pipeline: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");

    return pso;
}

/// Recreate a pipeline from a previously serialized binary archive. Returns nil
/// when the archive cannot be read or does not cover this pipeline, which
/// merely costs us the back-end half of a cache hit.
static id<MTLComputePipelineState>
jitc_metal_pipeline_from_data(id<MTLDevice> dev,
                              MTLComputePipelineDescriptor *desc,
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

    if (!archive) {
        jitc_log(Debug, "jit_metal_kernel_compile(): could not load a cached "
                        "binary archive: %s",
                 err ? err.localizedDescription.UTF8String : "<unknown>");
        return nil;
    }

    return jitc_metal_pipeline_from_archive(dev, desc, archive);
}

/// Round-trip an exported library image before it is committed to disk, so that
/// a future change of Apple's format disables the cache instead of poisoning
/// it. This costs ~1 ms against a multi-second compilation.
static bool jitc_metal_validate_library(id<MTLDevice> dev,
                                        const std::vector<uint8_t> &data,
                                        const char *kernel_name) {
    id<MTLLibrary> lib = jitc_metal_library_from_data(dev, data);
    NSMutableArray<id<MTLFunction>> *callables = [NSMutableArray array];

    if (lib && jitc_metal_resolve_functions(lib, kernel_name, callables))
        return true;

    jitc_log(Warn, "jit_metal_kernel_compile(): the exported library image did "
                   "not validate, the Metal kernel cache is disabled.");
    metal_library_export_available = false;

    return false;
}

/// Cache key of a Metal kernel. Unlike the library image, the binary archive
/// stored alongside it is specific to the GPU and its driver, so the device
/// identity and the OS build are folded into the kernel hash.
static XXH128_hash_t jitc_metal_cache_hash(id<MTLDevice> dev,
                                           XXH128_hash_t hash) {
    XXH3_state_t xxh_state;
    XXH3_128bits_reset(&xxh_state);
    XXH3_128bits_update(&xxh_state, &hash, sizeof(XXH128_hash_t));

    uint64_t registry_id = dev.registryID;
    XXH3_128bits_update(&xxh_state, &registry_id, sizeof(uint64_t));

    const char *name = dev.name.UTF8String;
    XXH3_128bits_update(&xxh_state, name, std::strlen(name));

    const char *os =
        NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String;
    XXH3_128bits_update(&xxh_state, os, std::strlen(os));

    // Selects the language version, which the source does not spell out
    uint8_t metal4 = uses_metal4 ? 1 : 0;
    XXH3_128bits_update(&xxh_state, &metal4, 1);

    return XXH3_128bits_digest(&xxh_state);
}

bool jitc_metal_kernel_compile(const char *source, size_t source_size,
                               const char *kernel_name, XXH128_hash_t hash,
                               Kernel &kernel) {
    @autoreleasepool {
        if (state.metal_devices.empty())
            jitc_fail("jitc_metal_kernel_compile(): no Metal devices initialized.");

        // Compile against the device of the calling thread
        auto *ts = thread_state(JitBackend::Metal);
        id<MTLDevice> dev = (__bridge id<MTLDevice>) ts->metal_device;
        XXH128_hash_t cache_hash = jitc_metal_cache_hash(dev, hash);

        // ---- Front end: MSL source to AIR ----------------------------------

        std::vector<uint8_t> library_data, archive_data;
        NSMutableArray<id<MTLFunction>> *callable_fns =
            [NSMutableArray arrayWithCapacity:metal_kernel_callables.size()];
        id<MTLLibrary> lib = nil;
        id<MTLFunction> func = nil;

        if (jitc_kernel_load_metal(source, (uint32_t) source_size, cache_hash,
                                   library_data, archive_data)) {
            lib = jitc_metal_library_from_data(dev, library_data);
            if (lib)
                func = jitc_metal_resolve_functions(lib, kernel_name, callable_fns);
            if (!func) {
                lib = nil;
                library_data.clear();
                archive_data.clear();
            }
        }

        bool cache_hit = func != nil;

        if (!cache_hit) {
            NSError *err = nil;
            {
                // Drop the central Dr.Jit lock during the expensive MSL compilation
                unlock_guard guard(state.lock);
                lib = jitc_metal_compile_source(dev, source, &err);
            }
            func = jitc_metal_resolve_functions(lib, kernel_name, callable_fns);

            if (!func) {
                // Metal sometimes returns a (non-null) library that omits the
                // entry point when the source has diagnostics. Generate useful
                // errors also in this case.
                NSMutableString *names = [NSMutableString string];
                for (NSString *fn in lib.functionNames)
                    [names appendFormat:@"%s%@", names.length ? ", " : "", fn];
                jitc_fail("jitc_metal_kernel_compile(): kernel function \"%s\" or "
                          "one of its callables was not found in the compiled "
                          "library.\nCompiler diagnostics: %s\n"
                          "Functions in library: [%s]\n\n--- Source code ---\n%s",
                          kernel_name,
                          err ? err.localizedDescription.UTF8String : "<none>",
                          names.UTF8String, source);
            }
        }

        // ---- Pipeline setup -------------------------------------------------
        // Link the union of custom intersection functions across every scene
        // registered with this kernel (collected into ``metal_kernel_scenes``
        // during code generation). LinkedFunctions applies to the entire PSO,
        // not per-IFT, so scenes with disjoint function name sets must all
        // contribute. Per-scene IFTs are built lazily at launch time in
        // ``jitc_metal_get_or_create_ift_for_scene``.
        NSMutableArray<id<MTLFunction>> *linked_fns = [NSMutableArray array];
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
                [linked_fns addObject:f];
            }
        }

        // The kernel's own [[visible]] callables must be linked into the
        // pipeline as well so they can be reached through a function table.
        [linked_fns addObjectsFromArray:callable_fns];

        MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
        desc.computeFunction = func;
        if (linked_fns.count > 0) {
            MTLLinkedFunctions *lf = [MTLLinkedFunctions new];
            lf.functions = linked_fns;
            desc.linkedFunctions = lf;
        }

        // ---- Back end: AIR to GPU ISA ---------------------------------------

        id<MTLComputePipelineState> pso = nil;
        bool archive_hit = false, store = false;

        {
            // Drop the central Dr.Jit lock during the expensive back end steps
            unlock_guard guard(state.lock);

            if (!archive_data.empty()) {
                pso = jitc_metal_pipeline_from_data(dev, desc, archive_data);
                archive_hit = pso != nil;
            }

            if (!pso) {
                // Caching the pipeline requires the library image next to it,
                // and exporting that needs private API. Skip the work when it
                // is unavailable, or when there is nowhere to put the result.
                bool can_store = jitc_cache_writable() &&
                                 (!library_data.empty() ||
                                  jitc_metal_library_data(lib, library_data));

                // Populating an archive costs a full back-end compilation, so
                // the pipeline is created from it rather than compiled twice.
                id<MTLBinaryArchive> archive =
                    can_store ? jitc_metal_archive_new(dev, desc) : nil;

                if (archive) {
                    pso = jitc_metal_pipeline_from_archive(dev, desc, archive);
                    // DIAGNOSTIC: the archive was just populated with this
                    // exact pipeline, so a miss means the back end runs twice
                    if (!pso)
                        jitc_fail("jit_metal_kernel_compile(): the pipeline is "
                                  "missing from a binary archive that was just "
                                  "populated with it.");
                    store = jitc_metal_archive_data(archive, archive_data);
                }

                if (!pso) {
                    NSError *err = nil;
                    pso = [dev newComputePipelineStateWithDescriptor:desc
                                                             options:MTLPipelineOptionNone
                                                          reflection:nil
                                                               error:&err];
                    if (!pso)
                        jitc_fail("jitc_metal_kernel_compile(): pipeline "
                                  "creation failed: %s",
                                  err ? err.localizedDescription.UTF8String
                                      : "<unknown>");
                }

                // Without a usable archive the library image alone still saves
                // the front end. Don't rewrite an entry that already has it.
                if (!store) {
                    archive_data.clear();
                    store = can_store && !cache_hit;
                }
            }

            if (store &&
                (cache_hit ||
                 jitc_metal_validate_library(dev, library_data, kernel_name)))
                jitc_kernel_write_metal(source, (uint32_t) source_size,
                                        cache_hash, library_data, archive_data);
        }

        // Build the visible function table used to dispatch indirect calls.
        // Function handles are derived from the pipeline, so the table is
        // PSO-specific; it is built once here and reused for every launch.
        id<MTLVisibleFunctionTable> vft = nil;
        if (callable_fns.count > 0) {
            MTLVisibleFunctionTableDescriptor *vftd =
                [MTLVisibleFunctionTableDescriptor new];
            vftd.functionCount = callable_fns.count;
            vft = [pso newVisibleFunctionTableWithDescriptor:vftd];
            for (NSUInteger i = 0; i < callable_fns.count; ++i) {
                id<MTLFunctionHandle> h =
                    [pso functionHandleWithFunction:callable_fns[i]];
                [vft setFunction:h atIndex:i];
            }
        }

        kernel.metal.pipeline       = (__bridge_retained void *) pso;
        kernel.metal.library        = (__bridge_retained void *) lib;
        kernel.metal.call_table_vft = vft ? (__bridge_retained void *) vft
                                          : nullptr;
        // Check if kernels must be launched with a call table slot
        kernel.metal.has_call_table = metal_vft_arg_index >= 0;
        kernel.size = (uint32_t) source_size;

        return cache_hit && archive_hit;
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

/// Resolve a Metal kernel-history entry's execution_time
float jitc_metal_finalize_kernel_history_entry(void *task_ptr) {
    @autoreleasepool {
        if (!task_ptr)
            return 0.f;
        id<MTLCommandBuffer> cb =
            (__bridge_transfer id<MTLCommandBuffer>) task_ptr;
        [cb waitUntilCompleted];
        return (float) ((cb.GPUEndTime - cb.GPUStartTime) * 1000);
    }
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
    if (n_args != 8)
        jitc_raise("jit_metal_ray_trace(): expected 8 ray arguments, got %u.",
                   n_args);
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
