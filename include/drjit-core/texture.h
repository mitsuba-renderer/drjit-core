/*
    drjit-core/texture.h -- creating and querying of 1D/2D/3D hardware textures

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

// ---------------------------------------------------------------------------
//  Hardware textures. CUDA and Metal both expose hardware-filtered textures
//  through this single set of entry points; the ``backend`` argument selects
//  the implementation. Other backends raise an error.
// ---------------------------------------------------------------------------

/**
 * \brief Allocate hardware texture memory
 *
 * Allocates memory for a texture of size \c ndim with a total of
 * <tt>shape[0] x ... x shape[ndim - 1]</tt> texels/voxels, where each
 * voxel is furthermore composed of \c n_channels color components.
 * The value of the \c n_channels argument must be greater or equal than 1.
 * The function returns an opaque texture handle.
 *
 * The \c format parameter is a \ref VarType code and must be either
 * <tt>VarType::Float32</tt> (single precision) or <tt>VarType::Float16</tt>
 * (half precision).
 *
 * The \c filter_mode parameter supports the following options:
 *
 * <ul>
 * <li><tt>filter_mode == 0</tt>: Nearest-neighbor sampling</li>
 * <li><tt>filter_mode == 1</tt>: Linear/bilinear/trilinear interpolation</li>
 * <ul>
 *
 * The \c wrap_mode parameter supports the following options:
 *
 * <ul>
 * <li><tt>wrap_mode == 0</tt>: Repeat</li>
 * <li><tt>wrap_mode == 1</tt>: Clamp</li>
 * <li><tt>wrap_mode == 2</tt>: Mirror</li>
 * <ul>
 *
 * Further modes (e.g. MIP-mapping) may be added in the future.
 */
extern JIT_EXPORT void *jit_tex_create(JitBackend backend, size_t ndim,
                                       const size_t *shape, size_t n_channels,
                                       int format, int filter_mode JIT_DEF(1),
                                       int wrap_mode JIT_DEF(0));

/**
 * \brief Retrieves the shape (including channels) of an existing texture
 *
 * \param shape
 *     Pointer to an array of size <tt>ndim + 1<\tt>, to which will be written
 *     the texture shape. The number of channels of the texture will be written
 *     at index \c ndim.
 */
extern JIT_EXPORT void jit_tex_get_shape(JitBackend backend, size_t ndim,
                                         const void *texture_handle,
                                         size_t *shape);

/**
 * \brief Retrieves the JIT indices of the underlying texture objects. This can
 * be used to traverse a texture for frozen function recording.
 *
 * \param indices
 *     Pointer to an array to which the JIT variable indices of the underlying
 *     texture objects are written. The required size is
 *     <tt>n_textures = 1 + ((n_channels - 1) / 4)</tt> on CUDA; on Metal it is
 *     one larger (a separate sampler handle is appended).
 */
extern JIT_EXPORT void jit_tex_get_indices(JitBackend backend,
                                           const void *texture_handle,
                                           uint32_t *indices);

/**
 * \brief Copy from device to texture memory
 *
 * Fills the texture with data from device memory at \c src_ptr. The other
 * arguments are analogous to \ref jit_tex_create(). The operation runs
 * asynchronously.
 */
extern JIT_EXPORT void jit_tex_memcpy_d2t(JitBackend backend, size_t ndim,
                                          const size_t *shape,
                                          const void *src_ptr,
                                          void *dst_texture_handle);

/**
 * \brief Copy from texture to device memory
 *
 * Implements the reverse of \ref jit_tex_memcpy_d2t
 */
extern JIT_EXPORT void jit_tex_memcpy_t2d(JitBackend backend, size_t ndim,
                                          const size_t *shape,
                                          const void *src_texture_handle,
                                          void *dst_ptr);

/**
 * \brief Performs a hardware texture lookup
 *
 * \param pos
 *     Pointer to a list of \c ndim float32 variable indices encoding the
 *     position of the texture lookup
 *
 * \param active
 *     A mask value that specified whether the texture fetch should be
 *     performed. When the associated predicate is false, the operation
 *     returns zero.
 *
 * \param out
 *     Pointer to an array of size equal to the number of channels in the
 *     texture, which will receive the lookup result.
 */
extern JIT_EXPORT void jit_tex_lookup(JitBackend backend, size_t ndim,
                                      const void *texture_handle,
                                      const uint32_t *pos, uint32_t active,
                                      uint32_t *out);

/**
 * \brief Fetches the four texels that would be referenced in a texture lookup
 * with bilinear interpolation without actually performing this interpolation.
 *
 * This function exclusively operates on two-dimensional textures. A lower or
 * higher number of dimensions will raise an error.
 *
 * \param pos
 *     Pointer to an array of two float32 variable indices encoding the position
 *     of the texture lookup
 *
 * \param active
 *     A mask value that specified whether the texture fetch should be
 *     performed. When the associated predicate is false, the operation
 *     returns zero.
 *
 * \param out
 *     Pointer to an array of size <tt>4 * n_channels<\tt>, which will receive
 *     the texel indices. Starting at the lower left corner, the texels are
 *     written to the array in counter-clockwise order.
 */
extern JIT_EXPORT void jit_tex_bilerp_fetch(JitBackend backend, size_t ndim,
                                            const void *texture_handle,
                                            const uint32_t *pos, uint32_t active,
                                            uint32_t *out);

/// Destroys the provided texture handle
extern JIT_EXPORT void jit_tex_destroy(JitBackend backend,
                                       void *texture_handle);

#if defined(__cplusplus)
}
#endif
