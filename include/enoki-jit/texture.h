/*
    enoki-jit/texture.h -- creating and querying of 1D/2D/3D textures on CUDA

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \brief Allocate CUDA texture memory
 *
 * Allocates memory for a texture of size \c ndim with a total of
 * <tt>shape[0] x ... x shape[ndim - 1]</tt> texels/voxels, where each
 * voxel is furthermore composed of \c n_channels color components.
 * The value of the \c n_channels argument must equal 1, 2, or 4.
 * The function returns an opaque texture handle.
 */
extern JIT_EXPORT void *jit_cuda_tex_create(size_t ndim, const size_t *shape,
                                            size_t n_channels);

/**
 * \brief Copy from device to texture memory
 *
 * Fills the texture with data from device memory at \c src_ptr. The other
 * arguments are analogous to \ref jit_cuda_tex_create(). The operation runs
 * asynchronously.
 */
extern JIT_EXPORT void jit_cuda_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                                               size_t n_channels,
                                               const void *src_ptr,
                                               void *dst_texture);

/**
 * \brief Copy from texture to device memory
 *
 * Implements the reverse of \ref jit_cuda_tex_memcpy_d2t
 */
extern JIT_EXPORT void jit_cuda_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                                               size_t n_channels,
                                               const void *src_texture,
                                               void *dst_ptr);

/**
 * \brief Performs a CUDA texture lookup
 *
 * \param ndim
 *     Dimensionality of the texture
 *
 * \param texture_id
 *     Index of a 64 bit variable encapsulating the texture handle value
 *
 * \param pos
 *     Pointer to a list of <tt>ndim - 1 </tt> float32 variable indices
 *     encoding the position of the texture lookup
 *
 * \param out
 *     Pointer to an array of size 4, which will receive the lookup result.
 *     Note that 4 values are always returned even if the texture does not have
 *     4 channels---they should be ignored in this case (though note that the
 *     reference counts must still be decreased to avoid a variable leak).
 */
extern JIT_EXPORT void jit_cuda_tex_lookup(size_t ndim, uint32_t texture_id,
                                           const uint32_t *pos, uint32_t mask,
                                           uint32_t *out);

/// Destroys the provided texture handle
extern JIT_EXPORT void jit_cuda_tex_destroy(void *texture);

#if defined(__cplusplus)
}
#endif
