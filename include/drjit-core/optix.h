/*
    drjit-core/optix.h -- JIT-compilation of kernels that use OptiX ray tracing

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef void *OptixDeviceContext;
typedef void *OptixProgramGroup;
typedef void *OptixModule;
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

/**
 * \brief Inform Dr.Jit about a partially created OptiX pipeline
 *
 * This function creates a JIT variable responsible for the lifetime management
 * of the OptiX pipeline and returns its corresponding index. Once the reference
 * count of this variable reaches zero, the OptiX resources related to this
 * pipeline will be freed.
 *
 * The returned index should be passed as argument to subsequent calls to
 * \c jit_optix_ray_trace in order to use this pipeline for the ray tracing
 * operations. See the docstring of \c jit_optix_ray_trace for a small example
 * of how those functions relate to each other.
 */
extern JIT_EXPORT uint32_t
jit_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                             OptixModule module,
                             const OptixProgramGroup *pg,
                             uint32_t pg_count);

/**
 * \brief Inform Dr.Jit about an OptiX Shader Binding Table
 *
 * This function creates a JIT variable responsible for the lifetime management
 * of the OptiX Shader Binding Table and returns its corresponding index. Once
 * the reference count of this variable reaches zero, the OptiX resources
 * related to this Shader Binding Table will be freed.
 *
 * The returned index should be passed as argument to subsequent calls to
 * \c jit_optix_ray_trace in order to use this Shader Binding Table for the ray
 * tracing operations. See the docstring of \c jit_optix_ray_trace for a small
 * example of how those functions relate to each other.
 */
extern JIT_EXPORT uint32_t
jit_optix_configure_sbt(const OptixShaderBindingTable *sbt, uint32_t pipeline);

/**
 * \brief  Update existing OptiX Shader Binding Table data
 *
 * This function updates the Shader Binding Table data held by the JIT
 * variable \c index previously created using \c jit_optix_configure_sbt. This
 * update is necessary when adding more geometry to an existing scene or when
 * sharing the OptiX pipeline and SBT across multiple scenes (e.g. ray tracing
 * against different scenes within the same megakernel).
 */
extern JIT_EXPORT void
jit_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt);

/* \brief Fields that can be queried from OptiX's generic ``HitObject``.
 *
 * The ``HitObject`` construct is the result of a AS traversal. See
 * \c jit_optix_ray_trace for how to use these. The field type is documented in
 * brackets for each member.
 */
#if defined(__cplusplus)
enum class OptixHitObjectField: uint32_t {
    /// Whether the HitObject is for a hit or not (miss or nop) [UInt32]
    IsHit,

    /// Instance ID of the hit object if it is an instanced object [UInt32]
    InstanceId,

    /// Index of the primitive that was hit, i.e triangle ID [UInt32]
    PrimitiveIndex,

    /// Pointer to the SBT data buffer of the hit object [Pointer]
    SBTDataPointer,

    /// Ray's distance to the object what was hit [Float32]
    RayTMax,

    /// The HitObject attribute at index 0 [UInt32]
    Attribute0,

    /// The HitObject attribute at index 1 [UInt32]
    Attribute1,

    /// The HitObject attribute at index 2 [UInt32]
    Attribute2,

    /// The HitObject attribute at index 3 [UInt32]
    Attribute3,

    /// The HitObject attribute at index 4 [UInt32]
    Attribute4,

    /// The HitObject attribute at index 5 [UInt32]
    Attribute5,

    /// The HitObject attribute at index 6 [UInt32]
    Attribute6,

    /// The HitObject attribute at index 7 [UInt32]
    Attribute7,

    /// Denotes the number of different field types
    Count
};
#else
enum OptixHitObjectField {
    OptixHitObjectFieldIsHit,
    OptixHitObjectFieldInstanceId,
    OptixHitObjectFieldPrimitiveIndex,
    OptixHitObjectFieldSBTDataPointer,
    OptixHitObjectFieldRayTMax,
    OptixHitObjectFieldAttribute0,
    OptixHitObjectFieldAttribute1,
    OptixHitObjectFieldAttribute2,
    OptixHitObjectFieldAttribute3,
    OptixHitObjectFieldAttribute4,
    OptixHitObjectFieldAttribute5,
    OptixHitObjectFieldAttribute6,
    OptixHitObjectFieldAttribute7,
    OptixHitObjectFieldCount
};
#endif

/**
  * \brief Insert an OptiX ray tracing call into the program
  *
  * The \c args list should contain a list of variable indices corresponding to
  * the 15 required function arguments <tt>handle, ox, oy, oz, dx, dy, dz, tmin,
  * tmax, time, mask, flags, sbt_offset, sbt_stride, miss_sbt_index</tt> of a
  * ``optixTraverse`` call.
  *
  * Up to 32 payload values can optionally be provided by setting \c n_args to a
  * value greater than 15. In this case, the corresponding elements will be
  * overwritten with the new variable indices with external reference count 1
  * containing the final payload value.
  *
  * The outgoing hit object produced by the ``optixTraverse`` call can be
  * queried by specifying a list of requested fields with \c n_hit_object_field
  * and \c hit_object_fields. The results will be stored in new variables whose
  * indices are written to \c hit_object_out.
  *
  * Shader execution reordering can be requested by using the \c reorder flag.
  * When the flag is set, the reordering will use the interesected shape's ID
  * as a sorting key. In addtion, an extra reordering hint can be passed in the
  * \c reorder_hit argument of which only the the last \c reorder_hint_num_bits
  * will be used (starting from the lowest signifcant bit). The hint will serve
  * as an extra sorting level for threads that intersected the same shape. The
  * hint is optional, it can be discared by setting \c reorder_hint_num_bits to
  * 0. If you wish to completely ignore the intersected shape's ID for the
  * reordering, \ref jit_reorder is more appropriate. Finally, if some threads
  * are masked with \c mask, they will still take part in the reordering and
  * will be grouped together. Note that if \c JitFlag::ShaderExecutionReordering
  * is not set, the \c reorder flag will be ignored.
  *
  * The \c invoke flag determines whether the closest hit and miss programs are
  * executed or not.
  *
  * The \c pipeline JIT variable index specifies the OptiX pipeline to be used
  * in the kernel executing this ray tracing operation. This index should be
  * computed using the \c jit_optix_configure_pipeline function.
  *
  * The \c sbt JIT variable index specifies the OptiX Shader Binding Table to be
  * used in the kernel executing this ray tracing operation. This index should
  * be computed using the \c jit_optix_configure_sbt function.
  *
  * Here is a small example of how to use those functions together:
  * <tt>
  *   OptixPipelineCompileOptions pco = ...;
  *   OptixModule mod = ...;
  *   OptixProgramGroup pgs = ...;
  *   uint32_t pg_count = ...;
  *   uint32_t pipeline_idx = jit_optix_configure_pipeline(pco, mod, pgs, pg_count);
  *
  *   OptixShaderBindingTable sbt = ...;
  *   uint32_t sbt_idx = jit_optix_configure_sbt(&sbt, pipeline_idx);
  *
  *   OptixHitObjectField fields[] {
  *         OptixHitObjectField::IsHit,
  *         OptixHitObjectField::RayTMax
  *   };
  *   uint32 hit_object_out[2];
  *
  *   active_idx = ...;
  *   trace_args = ...;
  *   jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args,
  *                       2, hit_object_fields, hit_object_out,
  *                       active_idx, pipeline_idx, sbt_idx);
  * </tt>
  */
extern JIT_EXPORT void jit_optix_ray_trace(
    uint32_t n_args, uint32_t *args, uint32_t n_hit_object_field,
    OptixHitObjectField *hit_object_fields, uint32_t *hit_object_out,
    int reorder, uint32_t reorder_hint, uint32_t reorder_hint_num_bits,
    int invoke, uint32_t mask, uint32_t pipeline, uint32_t sbt);

/**
 * \brief Read data from \c OptixHitObjectField::SBTDataPointer
 *
 * When querying the OptixHitObjectField::SBTDataPointer field during a
 * \c jit_optix_ray_trace, a pointer is returned to each thread. This function
 * can be used to read data in each lane from their respective pointer. The read
 * can be offset by \c offset bytes and will return exactly one new variable of
 * type \c type.
 */
extern JIT_EXPORT uint32_t jit_optix_sbt_data_load(uint32_t sbt_data_ptr,
                                                   VarType type,
                                                   uint32_t offset,
                                                   uint32_t mask);
#if defined(__cplusplus)
}
#endif
