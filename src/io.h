/*
    src/io.h -- Disk cache for LLVM/CUDA kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "hash.h"

using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, uint32_t thread_id, void **ptr);
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using OptixModule = void*;
using OptixProgramGroup = void*;
using OptixPipeline = void*;
enum class JitBackend: uint32_t;

/// Represents a compiled kernel for the three different backends
struct Kernel {
    void *data;
    uint32_t size;
    uint32_t operation_count;
    union {
        /// 1. CUDA
        struct {
            /// Compiled CUmodule
            CUmodule mod;

            /// Main kernel entry point
            CUfunction func;

            // Preferred block size to maximize occupancy
            uint32_t block_size;
        } cuda;

        /// 2. LLVM
        struct {
            /// Relocation table, the first element is the kernel entry point
            void **reloc;

            /// Length of the 'reloc' table
            uint32_t n_reloc;

#if defined(DRJIT_ENABLE_ITTNOTIFY)
            void *itt;
#endif
        } llvm;

#if defined(DRJIT_ENABLE_OPTIX)
        struct {
            OptixModule mod;
            OptixProgramGroup *pg;
            OptixPipeline pipeline;
            uint8_t *sbt_record;
            uint32_t pg_count;
        } optix;
#endif

#if defined(DRJIT_ENABLE_METAL)
        struct {
            /// Compiled MTL::ComputePipelineState* (kept opaque)
            void *pipeline;

            /// Owning MTL::Library* (released at kernel teardown)
            void *library;

            /// Recommended threadgroup size, queried via
            /// ``threadExecutionWidth`` and capped at 1024
            uint32_t block_size;

            /// Ordered list of ``MetalScene*`` used at codegen time, captured
            /// here so a frozen function replay (which skips re-assemble) can
            /// still bind the correct TLASes / IFTs at launch. Slot index
            /// ``i`` ↔ MSL kernel argument ``accel_<i>``. The pointers are
            /// owned externally (refcounted via the scene_index JIT variables
            /// held by the recording's input set), so this kernel struct is a
            /// non-owning reference array. ``scenes == nullptr`` for kernels
            /// that do not perform ray tracing. The array is heap-allocated
            /// via ``new[]`` and released in ``jitc_metal_kernel_free``.
            void **scenes;
            uint32_t scene_count;
        } metal;
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
