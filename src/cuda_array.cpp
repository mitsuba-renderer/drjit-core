/*
    src/cuda_array.cpp -- Functionality to create, read, and write
    variable arrays / CUDA code generation component.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"
#include "cuda_eval.h"
#include "var.h"
#include "array.h"
#include "log.h"

void jitc_cuda_render_array(Variable *v, Variable *pred) {
    if (pred && pred->array_state != (uint32_t) ArrayState::Conflicted) {
        v->reg_index = pred->reg_index;
        return;
    }
    fmt("    .local .align $a .$B arr_$u[$u];\n",
        v, v, v->reg_index, (uint32_t) v->array_length);
}

extern void jitc_cuda_render_array_init(Variable *v, Variable *pred, Variable *value) {
    v->reg_index = pred->reg_index;

    if (value->type == (uint32_t) VarType::Bool)
        fmt("    selp.u16 %w0, 1, 0, $v;\n", value);

    fmt("    mov.u32 %r3, 0;\n\n"
        "l_$u_init:\n"
        "    st.local.$B arr_$u[%r3], $V;\n"
        "    add.u32 %r3, %r3, 1;\n"
        "    setp.lt.u32 %p2, %r3, $u;\n"
        "    @%p2 bra l_$u_init;\n\n",
        v->reg_index,
        v, v->reg_index, value,
        (uint32_t) v->array_length,
        v->reg_index);
}

void jitc_cuda_render_array_read(Variable *v, Variable *source, Variable *mask,
                                 Variable *offset) {
    if (!mask->is_literal())
        fmt("    mov.$b $v, 0;\n", v, v);

    put("    ");

    if (!mask->is_literal())
        fmt("@$v ", mask);

    if (offset)
        fmt("ld.local.$B $V, arr_$u[$v];\n",
            v, v, source->reg_index, offset);
    else
        fmt("ld.local.$B $V, arr_$u[$u];\n",
            v, v, source->reg_index, (uint32_t) v->literal);

    if (v->type == (uint32_t) VarType::Bool)
        fmt("    setp.ne.u16 $v, %w0, 0;\n", v);
}

void jitc_cuda_render_array_write(Variable *v, Variable *target,
                                  Variable *value, Variable *mask,
                                  Variable *offset) {
    if (offset && offset->is_array())
        offset = nullptr;

    bool copy = target->array_state == (uint32_t) ArrayState::Conflicted;
    uint32_t target_buffer = target->reg_index;

    if (copy) {
        target_buffer = jitc_var(jitc_array_buffer(v))->reg_index;

        fmt("    mov.u32 %r3, 0;\n\n"
            "l_$u_copy:\n"
            "    ld.local.$B $V, arr_$u[%r3];\n"
            "    st.local.$B arr_$u[%r3], $V;\n"
            "    add.u32 %r3, %r3, 1;\n"
            "    setp.lt.u32 %p2, %r3, $u;\n"
            "    @%p2 bra l_$u_copy;\n\n",
            v->reg_index,
            v, v, target->reg_index,
            v, target_buffer, v,
            (uint32_t) v->array_length,
            v->reg_index);
    }

    if (v->type == (uint32_t) VarType::Bool)
        fmt("    selp.u16 %w0, 1, 0, $v;\n", value);

    put("    ");
    if (!mask->is_literal())
        fmt("@$v ", mask);

    if (offset)
        fmt("st.local.$B arr_$u[$v], $V;\n",
            v, target_buffer, offset, value);
    else
        fmt("st.local.$B arr_$u[$u], $V;\n",
            v, target_buffer, (uint32_t) v->literal, value);

    v->reg_index = target_buffer;
}

void jitc_cuda_render_array_memcpy_in(const Variable *v) {
    fmt("    mov.u32 %r3, 0;\n"
        "    .local .align $a .$B arr_$u[$u];\n\n"
        "l_$u_read:\n"
        "    ld.global.$B $V, [%rd0];\n"
        "    st.local.$B arr_$u[%r3], $V;\n"
        "    add.u32 %r3, %r3, 1;\n"
        "    mad.wide.u32 %rd0, %r2, $a, %rd0;\n"
        "    setp.lt.u32 %p2, %r3, $u;\n"
        "    @%p2 bra l_$u_read;\n\n",
        v, v, v->reg_index, (uint32_t) v->array_length,
        v->reg_index,
        v, v,
        v, v->reg_index, v,
        v,
        (uint32_t) v->array_length,
        v->reg_index);
}

void jitc_cuda_render_array_memcpy_out(const Variable *v) {
    fmt("    mov.u32 %r3, 0;\n\n"
        "l_$u_write:\n"
        "    ld.local.$B $V, arr_$u[%r3];\n"
        "    st.global.$B [%rd0], $V;\n"
        "    add.u32 %r3, %r3, 1;\n"
        "    mad.wide.u32 %rd0, %r2, $a, %rd0;\n"
        "    setp.lt.u32 %p2, %r3, $u;\n"
        "    @%p2 bra l_$u_write;\n\n",
        v->reg_index,
        v, v, v->reg_index,
        v, v,
        v,
        (uint32_t) v->array_length,
        v->reg_index
    );
}
