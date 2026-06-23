/*
    src/metal_tex.h -- hardware texture support for the Metal backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "tex.h"

#if defined(DRJIT_ENABLE_METAL)

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <memory>
#include <vector>

struct MetalTexture;

/// Opaque sub-resource of a texture: one of the 1/2/4-channel ``MTLTexture``
/// objects or the shared ``MTLSamplerState``.
struct MetalTexResource {
    MetalTexture *parent = nullptr;
    void *object = nullptr; // owned by ``parent``
};

/// Host-side state for one Metal hardware texture
struct MetalTexture : TextureBase {
    /// One ``id<MTLTexture>`` per sub-texture, owned.
    std::vector<void *> textures;
    /// Shared ``id<MTLSamplerState>``, owned.
    void *sampler = nullptr;

    /// Pointer backing variable per sub-texture
    std::vector<uint32_t> indices;
    uint32_t sampler_index = 0;

    /// One resource record per sub-texture, plus a trailing sampler record
    std::unique_ptr<MetalTexResource[]> records;

    MetalTexture(size_t type_size, size_t n_channels, bool writable)
        : TextureBase(JitBackend::Metal, type_size, n_channels, writable),
          textures(n_textures, nullptr), indices(n_textures, 0),
          records(std::make_unique<MetalTexResource[]>(n_textures + 1)) { }
};

// ---------------------------------------------------------------------
//  Public texture API
// ---------------------------------------------------------------------

extern void *jitc_metal_tex_create(size_t ndim, const size_t *shape,
                                   size_t n_channels, int format,
                                   int filter_mode, int wrap_mode,
                                   int writable, int srgb);
extern void *jitc_metal_tex_wrap(uintptr_t handle, size_t ndim, int format,
                                 int writable, int filter_mode, int wrap_mode,
                                 int srgb);
extern uintptr_t jitc_metal_tex_native_handle(const void *handle,
                                              size_t sub_index);
extern void jitc_metal_tex_get_shape(const void *handle, size_t *shape);
extern void jitc_metal_tex_get_indices(const void *handle,
                                       uint32_t *indices);
extern void jitc_metal_tex_memcpy_d2t(const void *src_ptr,
                                      void *dst_handle);
extern void jitc_metal_tex_memcpy_t2d(const void *src_handle,
                                      void *dst_ptr);
extern void jitc_metal_tex_lookup(const void *handle,
                                  const uint32_t *pos, uint32_t active,
                                  uint32_t *out);
extern void jitc_metal_tex_write(void *handle, const uint32_t *pos,
                                 const uint32_t *value, uint32_t active);
extern void jitc_metal_tex_bilerp_fetch(const void *handle,
                                        const uint32_t *pos, uint32_t active,
                                        uint32_t *out);
extern void jitc_metal_tex_destroy(void *handle);

#endif // defined(DRJIT_ENABLE_METAL)
