/*
    src/tex.h -- payload for texture-lookup IR nodes

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

// Payload for the Metal backend's texture-lookup nodes (``TexLookup`` /
// ``TexFetchBilerp``): the lookup coordinates and, for a bilerp fetch, the
// gathered texture component. The texture and sampler handles occupy the
// node's dependencies, and ``literal`` aliases ``data``.
struct TexData {
    /// Coordinate variable indices, one per dimension (``ndim`` of 3 used).
    uint32_t indices[3] = { 0, 0, 0 };
    /// Texture dimensionality (1–3), i.e. the number of valid ``indices``.
    uint32_t ndim = 0;
    /// Gathered channel, for a bilerp fetch.
    uint32_t component = 0;

    ~TexData() {
        for (uint32_t i = 0; i < ndim; ++i)
            jitc_var_dec_ref(indices[i]);
    }
};
