/*
    src/cuda.h -- CUDA backend functionality

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "cuda_api.h"
#include <utility>

/// Major version of the detected CUDA version
extern int jitc_cuda_version_major;

/// Minor version of the detected CUDA version
extern int jitc_cuda_version_minor;

/// Maximum number of pointer arguments that can be directly passed to a CUDA kernel
extern uint32_t jitc_cuda_arg_limit;

/// Return value of the call to cuInit()
extern CUresult jitc_cuda_cuinit_result;

/// Attempt to dynamically load CUDA into the process
extern bool jitc_llvm_api_init();

/// Free any resources allocated by jitc_cuda_api_init()
extern void jitc_llvm_api_shutdown();

/// Try to load CUDA and initialize Dr.Jit-specific kernels
extern bool jitc_cuda_init();

/// Free any resources allocated by jitc_cuda_init()
extern void jitc_cuda_shutdown();

/// Dynamically look up a CUDA symbol by name
extern void *jitc_cuda_lookup(const char *name);

struct Kernel;

/// Compile an PTX string. Returns the resulting module and a cache hit true/false flag
extern std::pair<CUmodule, bool>
jitc_cuda_compile(const char *str, bool release_state_lock = true);

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
extern void cuda_check_impl(CUresult errval, const char *file, const int line);

/// RAII wrapper to set the CUDA context within a scope
struct scoped_set_context {
    scoped_set_context(CUcontext ctx) {
        cuda_check(cuCtxPushCurrent(ctx));
    }
    ~scoped_set_context() {
        cuda_check(cuCtxPopCurrent(nullptr));
    }
};

/// Like `scoped_set_context`, but only trigger a CUDA call if `ctx != nullptr`
struct scoped_set_context_maybe {
    scoped_set_context_maybe(CUcontext ctx) : active(ctx != nullptr) {
        if (active)
            cuda_check(cuCtxPushCurrent(ctx));
    }
    ~scoped_set_context_maybe() {
        if (active)
            cuda_check(cuCtxPopCurrent(nullptr));
    }
    bool active;
};

// Dr.Jit kernel functions
extern CUfunction *jitc_cuda_fill_64;
extern CUfunction *jitc_cuda_mkperm_phase_1_tiny;
extern CUfunction *jitc_cuda_mkperm_phase_1_small;
extern CUfunction *jitc_cuda_mkperm_phase_1_large;
extern CUfunction *jitc_cuda_mkperm_phase_3;
extern CUfunction *jitc_cuda_mkperm_phase_4_tiny;
extern CUfunction *jitc_cuda_mkperm_phase_4_small;
extern CUfunction *jitc_cuda_mkperm_phase_4_large;
extern CUfunction *jitc_cuda_transpose;
extern CUfunction *jitc_cuda_compress_small;
extern CUfunction *jitc_cuda_compress_large;
extern CUfunction *jitc_cuda_compress_large_init;
extern CUfunction *jitc_cuda_poke[(int) VarType::Count];
extern CUfunction *jitc_cuda_block_reduce[(int) ReduceOp::Count]
                                         [(int) VarType::Count][10];
extern CUfunction *jitc_cuda_block_reduce_vec[(int) ReduceOp::Count]
                                             [(int) VarType::Count];
extern CUfunction *jitc_cuda_block_prefix_reduce[(int) ReduceOp::Count]
                                                [(int) VarType::Count][10];
extern CUfunction *jitc_cuda_reduce_dot[(int) VarType::Count];
extern CUfunction *jitc_cuda_aggregate;
