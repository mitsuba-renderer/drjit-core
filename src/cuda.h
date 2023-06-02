/*
    src/cuda.h -- CUDA backend functionality

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

/// Major version of the detected CUDA version
extern int jitc_cuda_version_major;

/// Minor version of the detected CUDA version
extern int jitc_cuda_version_minor;

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

/// Compile an IR string
extern void jitc_cuda_compile(const char *str, size_t size, Kernel &kernel);

/// Set the currently active device & stream
extern void jitc_cuda_set_device(int device);

/// Return a pointer to the CUDA stream associated with the currently active device
extern void* jitc_cuda_stream();

/// Return a pointer to the CUDA context associated with the currently active device
extern void* jitc_cuda_context();

/// Push a new CUDA context to the currently active device
extern void jitc_cuda_push_context(void *);

/// Pop the current CUDA context and return it
extern void* jitc_cuda_pop_context();

/// Initialize a per-thread state (requires that the backend is initialized)
extern ThreadState *jitc_cuda_thread_state_new();

/// Free a compiled CUDA kernel
extern void jitc_cuda_free(int device_id, bool shared, void *ptr);

/// Release a memory allocation made by the CUDA backend
extern void jitc_cuda_kernel_free(int device_id, const Kernel &kernel);

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
extern CUfunction *jitc_cuda_scan_small_u32;
extern CUfunction *jitc_cuda_scan_large_u32;
extern CUfunction *jitc_cuda_scan_large_u32_init;
extern CUfunction *jitc_cuda_compress_small;
extern CUfunction *jitc_cuda_compress_large;
extern CUfunction *jitc_cuda_block_copy[(int) VarType::Count];
extern CUfunction *jitc_cuda_block_sum [(int) VarType::Count];
extern CUfunction *jitc_cuda_reductions[(int) ReduceOp::Count]
                                       [(int) VarType::Count];
extern CUfunction *jitc_cuda_vcall_prepare;

/// Can't pass more than 4096 bytes of parameter data to a CUDA kernel
#define DRJIT_CUDA_ARG_LIMIT 512
