/*
    src/metal_array.cpp -- Functionality to create, read, and write
    variable arrays / Metal (MSL) code generation component.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"
#include "metal_array.h"
#include "array.h"
#include "log.h"

#include "metal_eval.h"

void jitc_metal_render_array(Variable *v, Variable *pred) {
    if (pred && pred->array_state != (uint32_t) ArrayState::Conflicted) {
        v->reg_index = pred->reg_index;
        return;
    }
    fmt("$t arr_$u[$u];\n",
              v, v->reg_index, (uint32_t) v->array_length);
}

void jitc_metal_render_array_init(Variable *v, Variable *pred, Variable *value) {
    v->reg_index = pred->reg_index;

    fmt("for (uint _i = 0; _i < $uu; _i++)\n"
              "    arr_$u[_i] = $v;\n",
              (uint32_t) v->array_length,
              v->reg_index, value);
}

void jitc_metal_render_array_read(Variable *v, Variable *source, Variable *mask,
                                  Variable *offset) {
    if (!mask->is_literal())
        fmt("$t $v = ($t) 0;\n", v, v, v);

    if (offset) {
        if (!mask->is_literal())
            fmt("if ($v) $v = arr_$u[$v];\n",
                      mask, v, source->reg_index, offset);
        else
            fmt("$t $v = arr_$u[$v];\n",
                      v, v, source->reg_index, offset);
    } else {
        if (!mask->is_literal())
            fmt("if ($v) $v = arr_$u[$u];\n",
                      mask, v, source->reg_index, (uint32_t) v->literal);
        else
            fmt("$t $v = arr_$u[$u];\n",
                      v, v, source->reg_index, (uint32_t) v->literal);
    }
}

void jitc_metal_render_array_write(Variable *v, Variable *target,
                                   Variable *value, Variable *mask,
                                   Variable *offset) {
    if (offset && offset->is_array())
        offset = nullptr;

    bool copy = target->array_state == (uint32_t) ArrayState::Conflicted;
    uint32_t target_buffer = target->reg_index;

    if (copy) {
        target_buffer = jitc_array_buffer(v)->reg_index;

        fmt("for (uint _i = 0; _i < $uu; _i++)\n"
                  "    arr_$u[_i] = arr_$u[_i];\n",
                  (uint32_t) v->array_length,
                  target_buffer, target->reg_index);
    }

    if (offset) {
        if (!mask->is_literal())
            fmt("if ($v) arr_$u[$v] = $v;\n",
                      mask, target_buffer, offset, value);
        else
            fmt("arr_$u[$v] = $v;\n",
                      target_buffer, offset, value);
    } else {
        if (!mask->is_literal())
            fmt("if ($v) arr_$u[$u] = $v;\n",
                      mask, target_buffer, (uint32_t) v->literal, value);
        else
            fmt("arr_$u[$u] = $v;\n",
                      target_buffer, (uint32_t) v->literal, value);
    }

    v->reg_index = target_buffer;
}

void jitc_metal_render_array_memcpy_in(const Variable *v) {
    fmt("$t arr_$u[$u];\n",
              v, v->reg_index, (uint32_t) v->array_length);
    fmt("for (uint _i = 0; _i < $uu; _i++)\n"
              "    arr_$u[_i] = ((device const $t*) p$v)[_i * params.size + r0];\n",
              (uint32_t) v->array_length,
              v->reg_index, v, v);
}

void jitc_metal_render_array_memcpy_out(const Variable *v) {
    fmt("for (uint _i = 0; _i < $uu; _i++)\n"
              "    ((device $t*) p$v)[_i * params.size + r0] = arr_$u[_i];\n",
              (uint32_t) v->array_length,
              v, v, v->reg_index);
}

void jitc_metal_render_array_select(Variable *v, Variable *mask, Variable *t, Variable *f) {
    uint32_t reg_index = jitc_array_buffer(v)->reg_index;
    fmt("if ($v) {\n"
              "    for (uint _i = 0; _i < $uu; _i++)\n"
              "        arr_$u[_i] = arr_$u[_i];\n"
              "} else {\n"
              "    for (uint _i = 0; _i < $uu; _i++)\n"
              "        arr_$u[_i] = arr_$u[_i];\n"
              "}\n",
              mask,
              (uint32_t) f->array_length, reg_index, t->reg_index,
              (uint32_t) f->array_length, reg_index, f->reg_index);

    v->reg_index = reg_index;
}
