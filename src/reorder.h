/*
    src/reorder.h -- Thread reordering utilities

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

// Trigger a reordering of the GPU threads
extern void jitc_reorder(uint32_t key, uint32_t num_bits,
                         uint32_t n_values, uint32_t *values,
                         uint32_t *out);
