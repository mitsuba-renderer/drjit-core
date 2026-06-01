/*
    src/metal_coop_vec.h -- Metal code generation for Cooperative Vectors

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Mirrors the LLVM cooperative-vector backend (src/llvm_coop_vec.{h,cpp}):
    each coopvec variable lowers to ``array_length`` independent MSL scalars
    named ``$v_0, $v_1, ..., $v_{N-1}``, and per-element ops emit one MSL
    expression per element.
*/

#pragma once

extern void jitc_metal_render_coop_vec(const Variable *v, const Variable *a0,
                                       const Variable *a1, const Variable *a2,
                                       const Variable *a3);
extern void jitc_metal_render_coop_vec_unpack(const Variable *v,
                                              const Variable *a0);
extern void jitc_metal_render_coop_vec_accum(const Variable *v,
                                             const Variable *a0,
                                             const Variable *a1,
                                             const Variable *a2);
extern void jitc_metal_render_coop_vec_outer_product_accum(const Variable *v,
                                                           const Variable *a0,
                                                           const Variable *a1,
                                                           const Variable *a2,
                                                           const Variable *a3);
