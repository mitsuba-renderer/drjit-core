/*
    src/metal_packet.h -- Specialized memory operations that read or write
    multiple adjacent values at once (MSL codegen).

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

struct Variable;

extern void jitc_metal_render_gather_packet(const Variable *v,
                                            const Variable *ptr,
                                            const Variable *index,
                                            const Variable *mask);

extern void jitc_metal_render_scatter_packet(const Variable *v,
                                             const Variable *ptr,
                                             const Variable *index,
                                             const Variable *mask);
