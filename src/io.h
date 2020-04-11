/*
    src/io.h -- Disk cache for LLVM/CUDA kernels

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>
#include <stdlib.h>

using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, void **ptr);
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;

/// A kernel and its preferred lauch configuration
struct Kernel {
    void *data;
    uint32_t size;
    union {
        struct {
            CUmodule cu_module;
            CUfunction cu_func;
            uint32_t block_size;
        } cuda;

        struct {
            LLVMKernelFunction func;
            LLVMKernelFunction func_scalar;
        } llvm;
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

