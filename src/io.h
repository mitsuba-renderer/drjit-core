/*
    src/io.h -- Disk cache for LLVM/CUDA kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "hash.h"
#if defined(DRJIT_ENABLE_AMD)
#  include "amd_api.h"
#endif

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
    void *data;
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
            /// Relocation table, the first element is the kernel entry point
            void **reloc;

            /// Length of the 'reloc' table
            uint32_t n_reloc;

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
            OptixModule mod;
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

#if defined(DRJIT_ENABLE_AMD)
        /// 5. AMD/HIP
        struct {
            /// Compiled HIP module
            hipModule_t mod;

            /// Main kernel entry point
            hipFunction_t func;

            /// Preferred block size
            uint32_t block_size;

            /// Module ownership remains with HIPRT's compiler cache
            bool hiprt_owned;
        } amd;
#endif
    };
};

// LZ4 compression dictionary
static const int jitc_lz4_dict_size = 65536;
extern char jitc_lz4_dict[];

/// Initialize dictionary
extern void jitc_lz4_init();

extern bool jitc_kernel_load(const char *source, uint32_t source_size,
                             JitBackend backend, XXH128_hash_t hash,
                             Kernel &kernel);

extern bool jitc_kernel_write(const char *source, uint32_t source_size,
                              JitBackend backend, XXH128_hash_t hash,
                              const Kernel &kernel);

extern void jitc_kernel_free(int device_id, const Kernel &kernel);

extern void jitc_flush_kernel_cache();
