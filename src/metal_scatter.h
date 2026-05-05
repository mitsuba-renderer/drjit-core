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

/// Emit MSL code for a scatter or scatter-reduce operation.
extern void jitc_metal_render_scatter(Variable *v);

/// Emit MSL code for an atomic compare-and-swap (ScatterCAS).
extern void jitc_metal_render_scatter_cas(Variable *v);

/// Emit MSL code for an atomic exchange (ScatterExch).
extern void jitc_metal_render_scatter_exch(Variable *v);

/// Emit MSL code for a Kahan-summation scatter (ScatterKahan).
extern void jitc_metal_render_scatter_kahan(Variable *v);

/// Emit MSL code for an atomic increment (ScatterInc).
extern void jitc_metal_render_scatter_inc(Variable *v);

#endif // defined(DRJIT_ENABLE_METAL)
