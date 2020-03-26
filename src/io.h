#pragma once

#include <stdint.h>
#include <stdlib.h>

using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, void **ptr);
using CUmodule = void *;
using CUfunction = void *;

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
        } llvm;
    };
};

extern bool jit_kernel_load(const char *source, uint32_t source_size,
                            bool llvm, size_t hash, Kernel &kernel);

extern bool jit_kernel_write(const char *source, uint32_t source_size,
                             bool llvm, size_t hash, const Kernel &kernel);

extern void jit_kernel_free(int device_id, const Kernel kernel);
