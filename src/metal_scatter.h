/*
    src/metal_scatter.h -- Scatter / scatter-reduce MSL codegen.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_METAL)

struct Variable;
enum class ReduceOp : uint32_t;

/// SIMD-local scatter-reduction over \c n packet channels (shared with the
/// scalar scatter path). \c values are variable indices.
extern void jitc_metal_emit_reduce_block(uint32_t n, const uint32_t *values,
                                         const Variable *ptr, const Variable *index,
                                         ReduceOp op, bool aggregate);

/// Emit MSL code for a scatter or scatter-reduce operation.
extern void jitc_metal_render_scatter(Variable *v);

/// Emit MSL code for an atomic compare-and-swap (ScatterCAS).
extern void jitc_metal_render_scatter_cas(Variable *v);

/// Emit MSL code for an atomic exchange (ScatterExch).
extern void jitc_metal_render_scatter_exch(Variable *v);

/// Emit MSL code for an atomic increment (ScatterInc).
extern void jitc_metal_render_scatter_inc(Variable *v);

#endif // defined(DRJIT_ENABLE_METAL)
