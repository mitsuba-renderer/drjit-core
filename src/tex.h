/*
    src/tex.h -- payload for texture-lookup IR nodes

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include <drjit-core/jit.h>
#include <atomic>
#include <cstddef>

/// Backend-agnostic state shared by ``CUDATexture`` and ``MetalTexture``.
///
/// Textures are split into 1/2/4-channel sub-textures, where the last
/// one is padded as needed.
struct TextureBase {
    JitBackend backend = JitBackend::None; // Backend that owns this handle
    size_t ndim = 0;       // Dimensionality (1, 2, or 3)
    size_t n_channels = 0; // Total number of channels
    size_t n_textures = 0; // Number of sub-textures
    size_t type_size = 0;  // Storage element size in bytes (2 or 4)
    bool writable = false; // Created/wrapped for kernel stores?
    size_t shape[3] = { 0, 0, 0 }; // Per-dimension texel counts (0 past ndim)
    std::atomic_size_t n_referenced_textures{ 0 }; // Outstanding sub-texture refs

    TextureBase() = default;
    TextureBase(JitBackend backend, size_t type_size, size_t n_channels,
                bool writable)
        : backend(backend), n_channels(n_channels),
          n_textures(1 + ((n_channels - 1) / 4)), type_size(type_size),
          writable(writable), n_referenced_textures(n_textures) { }

    /// Number of logical channels in sub-texture ``index``
    size_t channels(size_t index) const {
        size_t c = 4;
        if (index == n_textures - 1) {
            c = n_channels % 4;
            if (c == 0)
                c = 4;
        }
        return c;
    }

    /// Channels physically stored for sub-texture ``index`` (3 padded to 4)
    size_t channels_storage(size_t index) const {
        size_t c = channels(index);
        return (c == 3) ? 4 : c;
    }
};

// Payload for ``TexLookup`` / ``TexFetchBilerp`` / ``TexWrite`` nodes
struct TexData {
    /// Coordinate variable indices, one per dimension.
    uint32_t indices[3] { };
    /// Texture dimensionality (1–3).
    uint32_t ndim = 0;
    /// Gathered channel (bilerp fetch only).
    uint32_t component = 0;

    // --- TexWrite only ---
    /// Per-channel value variable indices (up to 4).
    uint32_t values[4] { };
    /// Number of valid entries in ``values``.
    uint32_t n_values = 0;
    /// Per-channel storage bytes (CUDA: scales the surface coord, picks ``sust`` b16/b32).
    uint32_t comp_bytes = 0;

    ~TexData() {
        for (uint32_t i = 0; i < ndim; ++i)
            jitc_var_dec_ref(indices[i]);
        for (uint32_t i = 0; i < n_values; ++i)
            jitc_var_dec_ref(values[i]);
    }
};
