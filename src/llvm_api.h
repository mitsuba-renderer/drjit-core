#pragma once

#include <enoki/jit.h>
#include "cuda_api.h"

using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, void **ptr);

/// A kernel and its preferred lauch configuration
struct Kernel {
    struct {
        /// CUDA kernel variables
        CUmodule cu_module = nullptr;
        CUfunction cu_func = nullptr;
        int thread_count = 0;
        int block_count = 0;
    } cuda;

    struct {
        /// LLVM kernel variables
        LLVMKernelFunction func;
        size_t size;
    } llvm;
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
extern Kernel jit_llvm_compile(const char *str, size_t size);

/// Release a compiled function
extern void jit_llvm_free(Kernel kernel);

/// Fully unload LLVM
extern void jit_llvm_shutdown();
