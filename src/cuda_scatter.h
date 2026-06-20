/*
    src/cuda_scatter.h -- Indirectly writing to memory aka. "scatter" is a
    nuanced and performance-critical operation. This file provides PTX IR
    templates for a variety of different scatter implementations to address
    diverse use cases.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"

extern const char *cuda_reduce_op_name[];

extern const char *jitc_cuda_reduce_tp(VarType &vt, ReduceOp op);

extern void jitc_cuda_render_scatter(const Variable *v, const Variable *ptr,
                                     const Variable *value,
                                     const Variable *index,
                                     const Variable *mask);

extern void jitc_cuda_render_scatter_reduce(const Variable *v,
                                            const Variable *ptr,
                                            const Variable *value,
                                            const Variable *index,
                                            const Variable *mask);

extern void jitc_cuda_render_scatter_inc(Variable *v, const Variable *ptr,
                                         const Variable *index,
                                         const Variable *mask);
extern void jitc_cuda_render_warp_reduce(uint32_t n, const uint32_t *values,
                                         VarType vt, ReduceOp op,
                                         bool use_packet_atomics);

extern void jitc_cuda_render_scatter_exch(Variable *v,
                                          const Variable *ptr,
                                          const Variable *value,
                                          const Variable *index,
                                          const Variable *mask);

extern void jitc_cuda_render_scatter_cas(Variable *v,
                                         const Variable *ptr,
                                         const Variable *compare,
                                         const Variable *value,
                                         const Variable *index);
