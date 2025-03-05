/*
    src/optix.h -- Low-level interface to OptiX

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "cuda.h"

using OptixDeviceContext = void *;
using OptixProgramGroup = void*;
using OptixModule = void*;
using OptixResult = int;
struct OptixPipelineCompileOptions;
struct OptixShaderBindingTable;
struct ThreadState;
struct OptixPipelineData;

/// Create an OptiX device context on the current ThreadState
extern OptixDeviceContext jitc_optix_context();

/// Destroy an OptiX device context
struct Device;
extern void jitc_optix_context_destroy(Device &d);

/// Look up an OptiX function by name
extern void *jitc_optix_lookup(const char *name);

/// Inform Dr.Jit about a partially created OptiX pipeline
extern uint32_t jitc_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                                              OptixModule module,
                                              const OptixProgramGroup *pg,
                                              uint32_t pg_count);

/// Inform Dr.Jit about an OptiX Shader Binding Table
extern uint32_t jitc_optix_configure_sbt(const OptixShaderBindingTable *sbt,
                                         uint32_t pipeline);

/// Overwrite existing OptiX Shader Binding Table given an index
extern void jitc_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt);

enum class OptixHitObjectField: uint32_t;

/// Insert a function call to optixTrace into the program
extern void jitc_optix_ray_trace(uint32_t n_args, uint32_t *args,
                                 uint32_t n_hit_object_field,
                                 OptixHitObjectField *hit_object_fields,
                                 uint32_t *hit_object_out, int invoke,
                                 uint32_t mask, uint32_t pipeline,
                                 uint32_t sbt);

// Read data from the SBT data buffer
extern JIT_EXPORT uint32_t jitc_optix_sbt_data_load(uint32_t sbt_data_ptr,
                                                    VarType type,
                                                    uint32_t offset,
                                                    uint32_t mask);

/// Compile an OptiX kernel
extern bool jitc_optix_compile(ThreadState *ts, const char *buffer,
                               size_t buffer_size, const char *kernel_name,
                               Kernel &kernel);

/// Free a compiled OptiX kernel
extern void jitc_optix_free(const Kernel &kernel);

/// Perform an OptiX kernel launch
extern void jitc_optix_launch(ThreadState *ts, const Kernel &kernel,
                              uint32_t size, const void *args, uint32_t n_args);

/// Optional: set the desired launch size
extern void jitc_optix_set_launch_size(uint32_t width, uint32_t height, uint32_t samples);

/// Convert a Dr.Jit variable type into an OptiX CoopVec variable type
extern uint32_t jitc_optix_coop_vec_type_id(VarType vt);

/// Convert a Dr.Jit matrix layout type into an OptiX CoopVec matrix layout type
extern uint32_t jitc_optix_coop_vec_layout_id(MatrixLayout ml);

#define jitc_optix_check(err) jitc_optix_check_impl((err), __FILE__, __LINE__)
extern void jitc_optix_check_impl(OptixResult errval, const char *file, const int line);

// Is the cooperative vector ABI available?
extern bool jitc_optix_has_abi_105;

#if defined(DRJIT_ENABLE_OPTIX)
/// Maximum cooperative vector size in a program
extern uint32_t jitc_optix_max_coopvec_size;

/// Should OptiX use continuation callables (vs. direct callables?)
inline bool jitc_optix_use_continuation_callables() {
    // Prefer continuation callables when compiling programs that use coperative vectors
    // (This gives slightly better performance)
    return jitc_optix_max_coopvec_size > 0;
}
#else
inline bool jitc_optix_use_continuation_callables() { return false; }
#endif
