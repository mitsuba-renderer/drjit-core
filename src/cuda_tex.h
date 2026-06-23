#pragma once

#include "cuda_api.h"

extern void *jitc_cuda_tex_create(size_t ndim, const size_t *shape,
                                  size_t n_channels, int format,
                                  int filter_mode, int wrap_mode, int writable,
                                  int srgb);
extern void jitc_cuda_tex_get_shape(const void *handle, size_t *shape);
extern void jitc_cuda_tex_get_indices(const void *handle,
                                      uint32_t *indices);
extern void jitc_cuda_tex_memcpy_d2t(const void *src_ptr,
                                     void *dst_handle);
extern void jitc_cuda_tex_memcpy_t2d(const void *src_handle,
                                     void *dst_ptr);
extern void jitc_cuda_tex_lookup(const void *handle,
                                 const uint32_t *pos, uint32_t active,
                                 uint32_t *out);
extern void jitc_cuda_tex_write(void *handle, const uint32_t *pos,
                                const uint32_t *value, uint32_t active);
extern void jitc_cuda_tex_bilerp_fetch(const void *handle,
                                       const uint32_t *pos, uint32_t active,
                                       uint32_t *out);
extern void *jitc_cuda_tex_wrap(uintptr_t handle, size_t ndim, int format,
                                int writable, int filter_mode, int wrap_mode,
                                int srgb);
extern void jitc_cuda_tex_map(void *handle);
extern void jitc_cuda_tex_unmap(void *handle);
extern uintptr_t jitc_cuda_tex_native_handle(const void *handle,
                                             size_t sub_index);
extern void jitc_cuda_tex_destroy(void *handle);
