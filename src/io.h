/*
    src/io.h -- Disk cache for LLVM/CUDA kernels

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>
#include <stdlib.h>

using LLVMKernelFunction = void (*)(uint32_t start, uint32_t end, void **ptr);
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using OptixModule = void*;
using OptixProgramGroup = void*;
using OptixPipeline = void*;

/// A kernel and its preferred lauch configuration
struct Kernel {
    void *data;
    uint32_t size;
    union {
        struct {
            CUmodule mod;
            CUfunction func;
            uint32_t block_size;
        } cuda;

        struct {
            LLVMKernelFunction func;
#if defined(ENOKI_JIT_ENABLE_ITTNOTIFY)
            void *itt;
#endif
        } llvm;

#if defined(ENOKI_JIT_ENABLE_OPTIX)
        struct {
            OptixModule mod;
            OptixProgramGroup *pg;
            OptixPipeline pipeline;
            uint8_t *sbt_record;
            uint32_t pg_count;
            uint32_t sbt_count;
        } optix;
#endif
    };
};

// LZ4 compression dictionary
static const int jit_lz4_dict_size = 65536;
extern char jit_lz4_dict[];

/// Initialize dictionary
extern void jit_lz4_init();

extern bool jit_kernel_load(const char *source, uint32_t source_size,
                            bool cuda, size_t hash, Kernel &kernel);

extern bool jit_kernel_write(const char *source, uint32_t source_size,
                             bool cuda, size_t hash, const Kernel &kernel);

extern void jit_kernel_free(int device_id, const Kernel kernel);

