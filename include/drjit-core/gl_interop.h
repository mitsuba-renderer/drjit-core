#pragma once
#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \brief Register a GL buffer as a CUDA graphics resource.
 *
 * The created ``CUgraphicsResource`` (pointer) is returned as an opaque pointer.
 * The resource must be unregistered with \ref jit_unregister_cuda_resource() when done.
 *
 * \param gl_buffer
 *     Integer identifying the GL buffer.
 * \return
 *     An opaque pointer to the CUDA graphics resource.
 */
extern JIT_EXPORT
void *jit_register_gl_buffer(/* GLuint */ unsigned int gl_buffer);

/**
 * \brief Register a GL texture as a CUDA graphics resource.
 *
 * The created ``CUgraphicsResource`` (pointer) is returned as an opaque pointer.
 * The resource must be unregistered with \ref jit_unregister_cuda_resource() when done.
 *
 * \param gl_texture
 *     Integer identifying the GL texture.
 * \return
 *     An opaque pointer to the CUDA graphics resource.
 */
extern JIT_EXPORT
void *jit_register_gl_texture(/* GLuint */ unsigned int gl_texture);

/**
 * Unregister a CUDA graphics resource that was previously registered
 * with \ref jit_register_gl_buffer() or \ref jit_register_gl_texture().
 * This frees the associated CUDA graphics resource.
 *
 * \param cuda_resource
 *     The CUDA graphics resource to unregister.
 */
extern JIT_EXPORT
void jit_unregister_cuda_resource(void *cuda_resource);

/**
 * Map a CUDA graphics resource and return a device pointer to the mapped resource.
 * The resource must be unmapped with \ref jit_unmap_graphics_resource() when done.
 *
 * \param cuda_resource
 *     The CUDA graphics resource to map
 * \param n_bytes
 *     Output parameter that receives the size in bytes of the mapped resource
 * \return Device pointer to the mapped resource
 */
extern JIT_EXPORT
void *jit_map_graphics_resource_ptr(void *cuda_resource, size_t *n_bytes);

/**
 * Map a CUDA graphics resource and return a CUDA array handle to the specified
 * sub-resource. The resource must be unmapped with \ref jit_unmap_graphics_resource() when done.
 *
 * \param cuda_resource
 *     The CUDA graphics resource to map
 * \param mip_level
 *     The mip level of the sub-resource (default: 0)
 * \return CUDA array handle to the mapped sub-resource
 */
extern JIT_EXPORT
void *jit_map_graphics_resource_array(void *cuda_resource, uint32_t mip_level = 0);

/**
 * Unmap a CUDA graphics resource that was previously mapped with
 * \ref jit_map_graphics_resource_ptr() or \ref jit_map_graphics_resource_array().
 *
 * \param cuda_resource
 *     The CUDA graphics resource to unmap
 */
extern JIT_EXPORT
void jit_unmap_graphics_resource(void *cuda_resource);

/**
 * Perform an asynchronous 2D memory copy from a source buffer to a CUDA array.
 *
 * \param dst
 *     Destination CUDA array
 * \param src
 *     Source buffer (host or device memory)
 * \param src_pitch
 *     Pitch (bytes per row) of the source buffer
 * \param height
 *     Height of the region to copy (in elements)
 * \param from_host
 *     True if copying from host memory, false for device memory
 */
extern JIT_EXPORT void jit_memcpy_2d_to_array_async(void *dst, const void *src,
                                                    size_t src_pitch,
                                                    size_t height,
                                                    bool from_host);

#if defined(__cplusplus)
}
#endif
