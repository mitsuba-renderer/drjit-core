/*
    src/optix_api.h -- Low-level interface to OptiX

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "cuda_api.h"

using OptixDeviceContext = void *;
struct ThreadState;

/// Create an OptiX device context on the current ThreadState
extern OptixDeviceContext jit_optix_context();

/// Destroy an OptiX device context
extern void jit_optix_context_destroy(ThreadState *ts);

/// Look up an OptiX function by name
extern void *jit_optix_lookup(const char *name);

/// Unload the OptiX library
extern void jit_optix_shutdown();

/// Inform Enoki about a partially created OptiX pipeline
extern void jit_optix_configure(const OptixPipelineCompileOptions *pco,
                                const OptixShaderBindingTable *sbt,
                                const OptixProgramGroup *pg,
                                uint32_t pg_count);

/// Insert a function call to optixTrace into the program
extern void jit_optix_trace(uint32_t nargs, uint32_t *args);

/// Compile an OptiX kernel
extern void jit_optix_compile(ThreadState *ts, const char *buffer,
                              size_t buffer_size, Kernel &kernel,
                              uint64_t kernel_hash);

/// Free a compiled OptiX kernel
extern void jit_optix_free(const Kernel &kernel);

/// Perform an OptiX kernel launch
extern void jit_optix_launch(ThreadState *ts, const Kernel &kernel, uint32_t size,
                             const void *args, uint32_t args_size);
