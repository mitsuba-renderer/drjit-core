/*
    drjit-core/optix.h -- JIT-compilation of kernels that use OptiX ray tracing

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef void *OptixDeviceContext;
typedef void *OptixProgramGroup;
struct OptixPipelineCompileOptions;
struct OptixShaderBindingTable;

/// Return the OptiX device context associated with the currently active device
extern JIT_EXPORT OptixDeviceContext jit_optix_context();

/// Look up an OptiX function by name
extern JIT_EXPORT void *jit_optix_lookup(const char *name);

/**
 * \brief Check the return value of an OptiX function and terminate the
 * application with a helpful error message upon failure
 */
#define jit_optix_check(err) jit_optix_check_impl((err), __FILE__, __LINE__)
extern JIT_EXPORT void jit_optix_check_impl(int errval, const char *file,
                                              const int line);

/// Inform Dr.Jit about a partially created OptiX pipeline
extern JIT_EXPORT void
jit_optix_configure(const OptixPipelineCompileOptions *pco,
                    const OptixShaderBindingTable *sbt,
                    const OptixProgramGroup *pg,
                    uint32_t pg_count);

/**
  * \brief Insert a function call to optixTrace into the program
  *
  * The \c args list should contain a list of variable indices corresponding to
  * the 15 required function arguments <tt>handle, ox, oy, oz, dx, dy, dz, tmin,
  * tmax, time, mask, flags, sbt_offset, sbt_stride, miss_sbt_index</tt>.
  *
  * Up to 32 payload values can optionally be provided by setting \c n_args to a
  * value greater than 15. In this case, the corresponding elements will be
  * overwritten with the new variable indices with external reference count 1
  * containing the final payload value.
  */
extern JIT_EXPORT void
jit_optix_ray_trace(uint32_t n_args, uint32_t *args, uint32_t mask);

/// Mark a variable as an expression requiring compilation via OptiX
extern JIT_EXPORT void jit_optix_mark(uint32_t index);

#if defined(__cplusplus)
}
#endif
