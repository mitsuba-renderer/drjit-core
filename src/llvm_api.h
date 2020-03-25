#pragma once

#include <enoki/jit.h>
#include "cuda_api.h"

using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, void **ptr);

/// A kernel and its preferred lauch configuration
struct Kernel {
    union {
        struct {
            /// CUDA kernel variables
            CUmodule cu_module;
            CUfunction cu_func;
            int min_grid_size;
            int block_size;
        } cuda;

        struct {
            /// LLVM kernel variables
            void *buffer;
            size_t size;
            LLVMKernelFunction func;
        } llvm;
    };
};


/// Target CPU string used by the LLVM backend
extern char *jit_llvm_target_cpu;

/// Target feature string used by the LLVM backend
extern char *jit_llvm_target_features;

/// Vector width used by the LLVM backend
extern int jit_llvm_vector_width;

/// Try to load the LLVM backend
extern bool jit_llvm_init();

/// Compile an IR string
extern Kernel jit_llvm_compile(const char *str, size_t size, uint32_t hash,
                               bool &cache_hit);

/// Release a compiled function
extern void jit_llvm_free(Kernel kernel);

/// Fully unload LLVM
extern void jit_llvm_shutdown();

/// Override the target architecture
extern void jit_llvm_set_target(const char *target_cpu,
                                const char *target_features,
                                int vector_width);
