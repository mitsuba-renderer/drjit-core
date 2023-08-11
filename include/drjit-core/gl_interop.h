#pragma once
#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

extern JIT_EXPORT void *jit_register_gl_buffer(/* GLuint */ unsigned int gl_buffer);

extern JIT_EXPORT void *jit_register_gl_texture(/* GLuint */ unsigned int gl_texture);

extern JIT_EXPORT void jit_unregister_cuda_resource(void *cuda_resource);

extern JIT_EXPORT void *jit_map_graphics_resource_ptr(void *cuda_resource,
                                                      size_t *n_bytes);

extern JIT_EXPORT void *
jit_map_graphics_resource_array(void *cuda_resource, uint32_t array_index = 0,
                                uint32_t mip_level = 0);

extern JIT_EXPORT void jit_unmap_graphics_resource(void *cuda_resource);

// extern JIT_EXPORT void jit_memcpy_2d(void *dst, size_t dst_pitch, void *src,
//                                      size_t src_pitch, size_t width,
//                                      size_t height);

// extern JIT_EXPORT void jit_memcpy_2d_to_array(void *dst, size_t w_offset,
//                                               size_t h_offset, void *src,
//                                               size_t src_pitch, size_t width,
//                                               size_t height);

extern JIT_EXPORT void jit_memcpy_2d_to_array_async(void *dst, void *src,
                                                    size_t src_pitch,
                                                    size_t component_size_bytes,
                                                    size_t width,
                                                    size_t height,
                                                    bool from_host);

#if defined(__cplusplus)
}
#endif
