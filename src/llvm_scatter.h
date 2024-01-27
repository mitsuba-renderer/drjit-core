/*
    src/llvm_scatter.h -- Indirectly writing to memory aka. "scatter" is a
    nuanced and performance-critical operation. This file provides LLVM IR
    templates for a variety of different scatter implementations to address
    diverse use cases.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"

void jitc_llvm_render_scatter(const Variable *v, const Variable *ptr,
                              const Variable *value, const Variable *index,
                              const Variable *mask);

void jitc_llvm_render_scatter_reduce(const Variable *v,
                                     const Variable *ptr,
                                     const Variable *value,
                                     const Variable *index,
                                     const Variable *mask);

void jitc_llvm_render_scatter_add_kahan(const Variable *v,
                                        const Variable *ptr_1,
                                        const Variable *ptr_2,
                                        const Variable *index,
                                        const Variable *value);

void jitc_llvm_render_scatter_inc(Variable *v, const Variable *ptr,
                                  const Variable *index, const Variable *mask);

