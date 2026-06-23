/*
    drjit-core/texture.h -- API for hardware textures access

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
//  Common API for 1/2/3D hardware textures on the CUDA and Metal backends
// ---------------------------------------------------------------------------

/**
 * \brief Allocate a hardware-accelerated texture object
 *
 * Allocates a texture of <tt>shape[0] x ... x shape[ndim - 1]</tt> texels with
 * \c n_channels components each, and returns an opaque texture handle. Further
 * modes (e.g. MIP-mapping) may be added in the future.
 *
 * \param ndim
 *     Texture dimensionality (1, 2, or 3).
 *
 * \param shape
 *     The \c ndim texel counts, one per dimension.
 *
 * \param n_channels
 *     Components per texel (>= 1).
 *
 * \param format
 *     Per-channel \ref VarType: <tt>Float32</tt>, <tt>Float16</tt>, or
 *     <tt>UInt8</tt>. An <tt>UInt8</tt> texture stores normalized 8-bit values
 *     and returns floats in <tt>[0, 1]</tt> when sampled.
 *
 * \param filter_mode
 *     0 = nearest-neighbor, 1 = linear/bi-/tri-linear interpolation.
 *
 * \param wrap_mode
 *     Out-of-bounds behavior: 0 = repeat, 1 = clamp, 2 = mirror.
 *
 * \param writable
 *     If nonzero, kernels may also store into the texture via
 *     \ref jit_tex_write().
 *
 * \param srgb
 *     If nonzero (<tt>UInt8</tt> textures only), samples are decoded from sRGB
 *     to linear by the hardware.
 */
extern JIT_EXPORT void *jit_tex_create(JitBackend backend,
                                       size_t ndim,
                                       const size_t *shape,
                                       size_t n_channels,
                                       int format,
                                       int filter_mode JIT_DEF(1),
                                       int wrap_mode JIT_DEF(0),
                                       int writable JIT_DEF(0),
                                       int srgb JIT_DEF(0));

/**
 * \brief Wrap an existing native texture object as a Dr.Jit texture
 *
 * Wraps an externally-owned native texture in a Dr.Jit handle rather than
 * allocating new storage. Dr.Jit infers the shape and channel count from
 * \c handle, and validates \c ndim and \c format against it, raising an
 * exception on a mismatch.
 *
 * The two backends manage the native texture's lifetime differently. On Metal,
 * Dr.Jit retains the ``MTLTexture`` and releases it in \ref jit_tex_destroy().
 * On CUDA, the user has to ensure that OpenGL textures are kept alive while
 * being accessed through the wrapper.
 *
 * Call this function once to obtain the Dr.Jit handle, then reuse it for the
 * texture's lifetime. On CUDA, each round of use must be bracketed by \ref
 * jit_tex_map() / \ref jit_tex_unmap() (a no-op on Metal):
 *
 * \code
 * void *tex = jit_tex_wrap(backend, gl_handle, 2, VarType::Float32, 0);
 * // per frame:
 * jit_tex_map(tex);
 * jit_tex_lookup(tex, pos, active, out);
 * jit_tex_unmap(tex);
 * // once done:
 * jit_tex_destroy(tex);
 * \endcode
 *
 * \param handle
 *     The native texture: an ``id<MTLTexture>`` pointer on Metal, or an OpenGL
 *     texture ID on CUDA (registered for interop).
 *
 * \param ndim / format
 *     The expected dimensionality and per-channel \ref VarType.
 *
 * \param writable
 *     If nonzero, kernels may store into the texture via \ref jit_tex_write().
 *     On Metal, the ``MTLTexture`` must have been created with
 *     ``MTLTextureUsageShaderWrite``.
 */
extern JIT_EXPORT void *jit_tex_wrap(JitBackend backend,
                                     uintptr_t handle,
                                     size_t ndim,
                                     int format,
                                     int writable,
                                     int filter_mode JIT_DEF(1),
                                     int wrap_mode JIT_DEF(0),
                                     int srgb JIT_DEF(0));

/**
 * \brief Make a wrapped texture's storage available to Dr.Jit kernels
 *
 * On CUDA, an OpenGL-wrapped texture belongs to OpenGL between uses; this call
 * hands it to CUDA. ``cuGraphicsMapResources()`` synchronizes against pending
 * OpenGL work and exposes the storage, after which Dr.Jit (re)builds the
 * sampling/surface object that kernels access. The matching \ref
 * jit_tex_unmap() returns the texture to OpenGL. On Metal, this is a no-op.
 */
extern JIT_EXPORT void jit_tex_map(void *handle);

/// Release a mapping established by \ref jit_tex_map()
extern JIT_EXPORT void jit_tex_unmap(void *handle);

/**
 * \brief Retrieve the shape and channel count of an existing texture
 *
 * \param handle
 *     The texture handle to query.
 *
 * \param shape
 *     Output array. The function writes the \c ndim per-dimension texel counts
 *     followed by the channel count, so it must hold at least <tt>ndim + 1</tt>
 *     entries (where \c ndim is the texture's dimensionality).
 */
extern JIT_EXPORT void jit_tex_get_shape(const void *handle,
                                         size_t *shape);

/**
 * \brief Retrieve JIT indices of the underlying texture object(s)
 *
 * Used to traverse a texture for frozen function recording.
 *
 * \param handle
 *     The texture handle to query.
 *
 * \param indices
 *     Output array receiving the JIT variable indices. It must hold at least
 *     <tt>n_textures = 1 + ((n_channels - 1) / 4)</tt> entries on CUDA, and one
 *     more on Metal for the sampler object.
 */
extern JIT_EXPORT void jit_tex_get_indices(const void *handle,
                                           uint32_t *indices);

/**
 * \brief Copy from linear device memory into a texture
 *
 * Uploads the densely packed, row-major buffer at \c src_ptr into the texture's
 * (potentially swizzled/tiled) native layout. The texel count and channels are
 * taken from the texture. Runs asynchronously.
 *
 * \param src_ptr
 *     Source device buffer matching the texture's texel and channel counts.
 *
 * \param dst_handle
 *     Destination texture handle from \ref jit_tex_create().
 */
extern JIT_EXPORT void jit_tex_memcpy_d2t(const void *src_ptr,
                                          void *dst_handle);

/**
 * \brief Copy from a texture into linear device memory
 *
 * The reverse of \ref jit_tex_memcpy_d2t(): reads the texture back into a
 * densely packed, row-major buffer. Runs asynchronously.
 *
 * \param src_handle
 *     Source texture handle from \ref jit_tex_create().
 *
 * \param dst_ptr
 *     Destination device buffer matching the texture's texel and channel
 *     counts.
 */
extern JIT_EXPORT void jit_tex_memcpy_t2d(const void *src_handle,
                                          void *dst_ptr);

/**
 * \brief Perform a hardware texture lookup
 *
 * Samples the texture at the normalized coordinate \c pos using the
 * \c filter_mode and \c wrap_mode chosen in \ref jit_tex_create() (nearest or
 * linear filtering, with hardware interpolation when linear).
 *
 * \param pos
 *     One float32 variable index per texture dimension, giving the lookup
 *     coordinate.
 *
 * \param active
 *     Mask variable. Lookups in lanes where it is false return zero.
 *
 * \param out
 *     Receives one float32 variable index per texture channel.
 */
extern JIT_EXPORT void jit_tex_lookup(const void *handle,
                                      const uint32_t *pos,
                                      uint32_t active,
                                      uint32_t *out);

/**
 * \brief Fetches the four texels that would be referenced in a texture lookup
 * with bilinear interpolation without actually performing this interpolation.
 *
 * The texture must be 2D; any other dimensionality raises an error.
 *
 * \param pos
 *     The two float32 variable indices giving the lookup coordinate.
 *
 * \param active
 *     Mask variable. Fetches in lanes where it is false return zero.
 *
 * \param out
 *     Receives <tt>4 * n_channels</tt> variable indices: the four texels,
 *     starting at the lower-left corner in counter-clockwise order.
 */
extern JIT_EXPORT void jit_tex_bilerp_fetch(const void *handle,
                                            const uint32_t *pos,
                                            uint32_t active,
                                            uint32_t *out);

/**
 * \brief Write to a texture object
 *
 * For each lane, write the per-channel values in \c value to the texel
 * position \c pos (in integer coordinates). The texture must
 * have been created with the \c writable flag of \ref jit_tex_create().
 *
 * \param pos
 *     One uint32 variable index per texture dimension.
 *
 * \param values
 *     One float32 variable index per texture channel.
 *
 * \param active
 *     A mask value selecting which lanes perform the write.
 */
extern JIT_EXPORT void jit_tex_write(void *handle,
                                     const uint32_t *pos,
                                     const uint32_t *values,
                                     uint32_t active);

/**
 * \brief Return the native handle of a texture
 *
 * Returns, as an integer, the handle a GUI of the backend would display: an
 * ``id<MTLTexture>`` pointer on Metal, or the wrapped OpenGL texture id on CUDA.
 * On CUDA a Dr.Jit-allocated texture has no OpenGL identity and returns zero.
 *
 * \param handle
 *     The texture handle to query.
 *
 * \param sub_index
 *     A logical texture with more than four channels is split into
 *     sub-textures; selects which one's native handle is returned.
 */
extern JIT_EXPORT uintptr_t jit_tex_native_handle(const void *handle,
                                                  size_t sub_index JIT_DEF(0));

/**
 * \brief Destroy a texture handle
 *
 * \param handle
 *     The texture handle to destroy. A wrapped (borrowed) native texture is
 *     released without freeing the underlying storage.
 */
extern JIT_EXPORT void jit_tex_destroy(void *handle);

#if defined(__cplusplus)
}
#endif
