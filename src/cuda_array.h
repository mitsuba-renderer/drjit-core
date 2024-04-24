/*
    src/cuda_array.h -- Functionality to create, read, and write
    variable arrays / CUDA code generation component.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

struct Variable;

extern void jitc_cuda_render_array(Variable *v, Variable *pred);
extern void jitc_cuda_render_array_init(Variable *v, Variable *pred,
                                        Variable *value);
extern void jitc_cuda_render_array_read(Variable *v, Variable *source,
                                        Variable *mask, Variable *offset);
extern void jitc_cuda_render_array_write(Variable *v, Variable *target,
                                         Variable *value, Variable *mask,
                                         Variable *offset);
extern void jitc_cuda_render_array_memcpy_in(const Variable *v);
extern void jitc_cuda_render_array_memcpy_out(const Variable *v);
