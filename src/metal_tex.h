/*
    src/metal_tex.h -- hardware texture support for the Metal backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_METAL)

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <memory>
#include <vector>

struct MetalTexture;

/// Opaque sub-resource of a texture: one of the 1/2/4-channel ``MTLTexture`` or
/// or the shared ``MTLSamplerState``.
struct MetalTexResource {
    MetalTexture *parent = nullptr;
    /// `id<MTLTexture> (sub-texture) or id<MTLSamplerState> owned by ``parent``
    void *object = nullptr;
};

/// Host-side state for one hardware texture
/// Metal has no 3-channel pixel formats, so a logical texture with
/// ``n_channels`` components is split into ``n_textures = 1 + (n_channels-1)/4``
/// 1/2/4-channel sub-textures (the last one padded as needed).
struct MetalTexture {
    uint32_t ndim = 0;       ///< 1, 2, or 3.
    int format = 0;          ///< Storage type (``VarType::Float32`` / ``Float16``).
    size_t type_size = 0;    ///< Storage element size (4 or 2 bytes).
    size_t n_channels = 0;   ///< Total logical channels.
    size_t n_textures = 0;   ///< Number of sub-textures.
    size_t shape[3] = { 0, 0, 0 };

    /// One ``__bridge_retained`` id<MTLTexture> per sub-texture.
    std::vector<void *> textures;
    /// Shared ``__bridge_retained`` id<MTLSamplerState>.
    void *sampler = nullptr;

    /// Per-sub-texture pointer-literal backing variable (``jitc_var_mem_map``).
    std::vector<uint32_t> indices;
    uint32_t sampler_index = 0;

    /// Resource records (one per sub-texture, plus one for the sampler at index
    /// ``n_textures``).
    std::unique_ptr<MetalTexResource[]> records;

    /// Outstanding references to the sub-textures. The struct is deleted once
    /// the last sub-texture is released (see the per-index JIT callback).
    std::atomic_size_t n_referenced_textures{ 0 };

    /// Raw channel count backing sub-texture ``index`` (1..4).
    size_t channels(size_t index) const {
        size_t c = 4;
        if (index == n_textures - 1) {
            c = n_channels % 4;
            if (c == 0)
                c = 4;
        }
        return c;
    }

    /// Internal channel count of sub-texture ``index`` (1, 2, or 4 — a raw
    /// count of 3 is padded to a 4-channel format).
    size_t channels_internal(size_t index) const {
        size_t c = channels(index);
        return (c == 3) ? 4 : c;
    }
};

// ---------------------------------------------------------------------
//  Public texture API
// ---------------------------------------------------------------------

extern void *jitc_metal_tex_create(size_t ndim, const size_t *shape,
                                   size_t n_channels, int format,
                                   int filter_mode, int wrap_mode);
extern void jitc_metal_tex_get_shape(size_t ndim, const void *texture_handle,
                                     size_t *shape);
extern void jitc_metal_tex_get_indices(const void *texture_handle,
                                       uint32_t *indices);
extern void jitc_metal_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                                      const void *src_ptr,
                                      void *dst_texture_handle);
extern void jitc_metal_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                                      const void *src_texture_handle,
                                      void *dst_ptr);
extern void jitc_metal_tex_lookup(size_t ndim, const void *texture_handle,
                                  const uint32_t *pos, uint32_t active,
                                  uint32_t *out);
extern void jitc_metal_tex_bilerp_fetch(size_t ndim, const void *texture_handle,
                                        const uint32_t *pos, uint32_t active,
                                        uint32_t *out);
extern void jitc_metal_tex_destroy(void *texture_handle);

#endif // defined(DRJIT_ENABLE_METAL)
