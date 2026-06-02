/*
    src/metal_scatter.cpp -- Metal scatter[-reduce] code generation

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "metal_scatter.h"
#include "eval.h"
#include "internal.h"
#include "var.h"
#include "op.h"
#include "log.h"

#include "metal_eval.h"

static const char *metal_reduce_op_name[(int) ReduceOp::Count] = {
    "", "add", "mul", "min", "max", "and", "or"
};

void jitc_metal_render_scatter(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *value = jitc_var(v->dep[1]);
    Variable *index = jitc_var(v->dep[2]);
    Variable *mask  = jitc_var(v->dep[3]);

    ReduceOp op = (ReduceOp) (uint32_t) v->literal;
    VarType vt = (VarType) value->type;
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    if (op == ReduceOp::Identity) {
        if (is_unmasked)
            fmt("    ((device $t*) $v)[$v] = $v;\n",
                      value, ptr, index, value);
        else
            fmt("    if ($v) ((device $t*) $v)[$v] = $v;\n",
                      mask, value, ptr, index, value);
        return;
    }

    // Native atomics exist for 32-bit ints (add/min/max/and/or) and float32 (add).
    // float32 min/max use a CAS loop. Everything else is either rejected or handled
    // via promotion to float32.
    const char *op_name = metal_reduce_op_name[(int) op];

    bool native = vt == VarType::Int32 || vt == VarType::UInt32 ||
                  (vt == VarType::Float32 && op == ReduceOp::Add);

    if (!is_unmasked)
        fmt("    if ($v) {\n    ", mask);

    if (native) {
        const char *atomic_t = vt == VarType::Float32 ? "atomic_float"
                             : vt == VarType::Int32   ? "atomic_int"
                                                      : "atomic_uint";
        fmt("    atomic_fetch_$s_explicit((device $s*)((device $t*) $v + $v), "
            "$v, memory_order_relaxed);\n",
            op_name, atomic_t, value, ptr, index, value);
    } else {
        // CAS loop fallback (float32 min/max).
        fmt("    {\n"
            "        device atomic_uint *_addr = (device atomic_uint*)((device $t*) $v + $v);\n"
            "        uint _old = atomic_load_explicit(_addr, memory_order_relaxed);\n"
            "        while (true) {\n",
            value, ptr, index);

        switch (op) {
            case ReduceOp::Min:
                fmt("            $t _new = min(as_type<$t>(_old), $v);\n", value, value, value);
                break;
            case ReduceOp::Max:
                fmt("            $t _new = max(as_type<$t>(_old), $v);\n", value, value, value);
                break;
            default:
                jitc_fail("jitc_metal_render_scatter(): unsupported ReduceOp %s "
                          "for the CAS fallback.", op_name);
        }

        put("            uint _expected = _old;\n"
            "            if (atomic_compare_exchange_weak_explicit(_addr, &_expected, as_type<uint>(_new), memory_order_relaxed, memory_order_relaxed)) break;\n"
            "            _old = _expected;\n"
            "        }\n"
            "    }\n");
    }

    if (!is_unmasked)
        put("    }\n");
}

void jitc_metal_render_scatter_cas(Variable *v) {
    Variable *ptr     = jitc_var(v->dep[0]);
    Variable *compare = jitc_var(v->dep[1]);
    Variable *value   = jitc_var(v->dep[2]);
    Variable *index   = jitc_var(v->dep[3]);

    ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
    Variable *mask = jitc_var(cas_data->mask);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    // Initialize outputs to zero
    fmt("    $t $v_out_0 = ($t) 0;\n"
              "    bool $v_out_1 = false;\n",
              value, v, value,
              v);

    if (!is_unmasked)
        fmt("    if ($v) {\n    ", mask);

    fmt("    {\n"
              "        device atomic_uint *_addr = (device atomic_uint*) ((device $t*) $v + $v);\n"
              "        uint _expected = as_type<uint>($v);\n"
              "        bool _swapped = atomic_compare_exchange_weak_explicit(_addr, &_expected, as_type<uint>($v), memory_order_relaxed, memory_order_relaxed);\n"
              "        $v_out_0 = as_type<$t>(_expected);\n"
              "        $v_out_1 = _swapped;\n"
              "    }\n",
              value, ptr, index,
              compare,
              value,
              v, value,
              v);

    if (!is_unmasked)
        put("    }\n");

    v->consumed = 1;
}

void jitc_metal_render_scatter_exch(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *value = jitc_var(v->dep[1]);
    Variable *index = jitc_var(v->dep[2]);
    Variable *mask  = jitc_var(v->dep[3]);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    fmt("    $t $v = ($t) 0;\n", value, v, value);
    if (!is_unmasked)
        fmt("    if ($v)\n    ", mask);
    fmt("    $v = as_type<$t>(atomic_exchange_explicit((device atomic_uint*)((device $t*) $v + $v), as_type<uint>($v), memory_order_relaxed));\n",
              v, value, value, ptr, index, value);

    v->consumed = 1;
}

void jitc_metal_render_scatter_kahan(Variable *v) {
    Variable *ptr_t  = jitc_var(v->dep[0]);
    Variable *ptr_c  = jitc_var(v->dep[1]);
    Variable *index  = jitc_var(v->dep[2]);
    Variable *value  = jitc_var(v->dep[3]);

    fmt("    if ($v != 0.f) {\n"
              "        #pragma clang fp contract(off)\n"
              "        volatile float _kahan_before = atomic_fetch_add_explicit("
              "(device atomic_float*)((device float*) $v + $v), "
              "$v, memory_order_relaxed);\n"
              "        volatile float _kahan_after = _kahan_before + $v;\n"
              "        volatile float _kahan_c1 = (_kahan_before - _kahan_after) + $v;\n"
              "        volatile float _kahan_c2 = ($v - _kahan_after) + _kahan_before;\n"
              "        float _kahan_comp = abs(_kahan_before) >= abs($v) ? _kahan_c1 : _kahan_c2;\n"
              "        atomic_fetch_add_explicit("
              "(device atomic_float*)((device float*) $v + $v), "
              "_kahan_comp, memory_order_relaxed);\n"
              "    }\n",
              value,
              ptr_t, index,
              value,
              value,
              value,
              value,
              value,
              ptr_c, index);
}

void jitc_metal_render_scatter_inc(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *index = jitc_var(v->dep[1]);
    Variable *mask  = jitc_var(v->dep[2]);

    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    fmt("    uint $v = 0;\n", v);
    if (!is_unmasked)
        fmt("    if ($v)\n    ", mask);
    fmt("    $v = atomic_fetch_add_explicit((device atomic_uint*)((device uint*) $v + $v), 1u, memory_order_relaxed);\n",
        v, ptr, index);

    v->consumed = 1;
}
