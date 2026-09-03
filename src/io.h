/*
    src/io.h -- Disk cache for compiled kernel artifacts

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "hash.h"
#include <vector>

using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, uint32_t thread_id, void **ptr);
#if defined(DRJIT_ENABLE_CUDA)
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
#endif
#if defined(DRJIT_ENABLE_OPTIX)
using OptixModule = void*;
using OptixProgramGroup = void*;
using OptixPipeline = void*;
#endif
enum class JitBackend: uint32_t;

/// Per kernel-parameter-slot metadata, indexed identically to the launch
/// ``kernel_params`` vector. Built during code generation and persisted onto
/// the ``Kernel``.
struct KernelParamInfo {
    /// 1 if the kernel writes this buffer (output or scatter target), else 0.
    uint8_t write;
    /// ``ResourceKind`` (raw, since the enum is defined later). Only Metal uses
    /// non-``Buffer`` kinds.
    uint8_t kind;
};

/// Represents a compiled kernel for the different backends
struct Kernel {
    uint32_t size;
    uint32_t operation_count;

    /// Generated source code and its length
    char *src;
    size_t src_size;

    /// Per-slot parameter metadata, parallel to the launch ``kernel_params``
    /// vector (see KernelParamInfo).
    KernelParamInfo *param_info;

    union {
        /// 1. LLVM
        struct {
            /// Entry points of the kernel's units: the kernel entry at index
            /// 0, followed by the indirect callables in callable-index order.
            void **reloc;

#if defined(DRJIT_ENABLE_ITTNOTIFY)
            void *itt;
#endif
        } llvm;

#if defined(DRJIT_ENABLE_CUDA)
        /// 2. CUDA
        struct {
            /// Compiled CUmodule
            CUmodule mod;

            /// Main kernel entry point
            CUfunction func;

            // Preferred block size to maximize occupancy
            uint32_t block_size;
        } cuda;
#endif

#if defined(DRJIT_ENABLE_OPTIX)
        /// 3. OptiX
        struct {
            /// Program groups referencing the kernel's unit modules (which
            /// are owned by the OptiX unit cache)
            OptixProgramGroup *pg;
            OptixPipeline pipeline;
            uint8_t *sbt_record;
            uint32_t pg_count;
        } optix;
#endif

#if defined(DRJIT_ENABLE_METAL)
        /// 4. Metal
        struct {
            /// id<MTLComputePipelineState>
            void *pipeline;

            /// id<MTLLibrary>
            void *library;

            /// id<MTLVisibleFunctionTable> for indirect-call dispatch, or null
            /// if the kernel performs no multi-target calls.
            void *call_table_vft;

            /// Whether codegen reserved a trailing ``params.args[]`` call-table
            /// slot (even if call_table_vft is potentially NULL).
            bool has_call_table;
        } metal;
#endif
    };
};

/// Version number of the cache format and dictionaries
#define DRJIT_CACHE_VERSION 1

/// Locate and prepare the kernel cache directory. Never fails, never throws.
extern void jitc_cache_init();

/// Forget the cache directory
extern void jitc_cache_shutdown();

/// Return the cache directory, or NULL when the on-disk cache is unavailable
extern const char *jitc_cache_dir();

/// Can new entries be added to the kernel cache?
extern bool jitc_cache_writable();

/// Artifact types held by the kernel cache. The kind determines the file
/// name suffix and the LZ4 dictionary.
enum class CacheKind : uint32_t {
    /// Relocatable object file of the LLVM backend
    Object,

    /// Metal library image produced by the MSL front end
    MetalLibrary,

    /// Metal binary archive holding one callable's device-specialized code
    MetalFunction,

    /// Metal binary archive holding a kernel's pipeline
    MetalPipeline
};

/// Typed cache entries holding a single binary artifact. A store leaves a
/// pre-existing entry in place. These functions may be called concurrently
/// from worker threads.
extern bool jitc_cache_blob_load(CacheKind kind, XXH128_hash_t hash,
                                 std::vector<uint8_t> &data);

extern bool jitc_cache_blob_store(CacheKind kind, XXH128_hash_t hash,
                                  const uint8_t *data, size_t size);

/// Compress and store 'data' on a pool worker without waiting for the
/// result. The store is best effort: tasks still queued at shutdown are
/// dropped, and the next process simply recompiles the artifact.
extern void jitc_cache_blob_store_async(CacheKind kind, XXH128_hash_t hash,
                                        std::vector<uint8_t> &&data);

/// Occasionally prune the kernel cache on a detached background thread
extern void jitc_cache_sweep();

/// Decompress an embedded LZ4 blob (e.g. the builtin kernels) into a
/// malloc()-ed buffer of 'size' bytes plus a terminating zero. 'what'
/// names the blob in the error message should decompression fail.
extern char *jitc_lz4_inflate(const char *compressed, size_t compressed_size,
                              size_t size, const char *what);

extern void jitc_kernel_free(int device_id, const Kernel &kernel);

extern void jitc_flush_kernel_cache();
