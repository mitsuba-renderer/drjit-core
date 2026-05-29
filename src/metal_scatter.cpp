/*
    src/metal_scatter.cpp -- Metal scatter / scatter-reduce codegen.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Generates MSL code fragments for scatter operations with optional
    atomic reductions. Metal provides:
      * atomic_fetch_{add,min,max,and,or,xor}_explicit for 32-bit int/uint
      * atomic_fetch_add_explicit for float (Metal 3.0+ / Apple7+)
      * atomic_exchange / compare_exchange for CAS-based fallbacks
*/

#if defined(DRJIT_ENABLE_METAL)

#include "metal_scatter.h"
#include "eval.h"
#include "internal.h"
#include "var.h"
#include "op.h"
#include "log.h"
#include "strbuf.h"

#define fmt_metal(fmt, ...) buffer.fmt_metal(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)
#define put(...)            buffer.put(__VA_ARGS__)

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
        // Plain scatter (no reduction)
        if (is_unmasked) {
            fmt_metal("    ((device $t*) $v)[$v] = $v;\n",
                      value, ptr, index, value);
        } else {
            fmt_metal("    if ($v) ((device $t*) $v)[$v] = $v;\n",
                      mask, value, ptr, index, value);
        }
        return;
    }

    // Scatter-reduce via Metal atomics
    const char *op_name = metal_reduce_op_name[(int) op];

    // Metal atomic function names
    const char *atomic_fn = nullptr;
    bool use_cas = false;

    switch (op) {
        case ReduceOp::Add:
            if (vt == VarType::Float32)
                atomic_fn = "atomic_fetch_add_explicit";
            else if (vt == VarType::UInt32 || vt == VarType::Int32)
                atomic_fn = "atomic_fetch_add_explicit";
            else
                use_cas = true;
            break;
        case ReduceOp::Min:
            if (vt == VarType::UInt32 || vt == VarType::Int32)
                atomic_fn = "atomic_fetch_min_explicit";
            else
                use_cas = true;
            break;
        case ReduceOp::Max:
            if (vt == VarType::UInt32 || vt == VarType::Int32)
                atomic_fn = "atomic_fetch_max_explicit";
            else
                use_cas = true;
            break;
        case ReduceOp::And:
            if (vt == VarType::UInt32 || vt == VarType::Int32)
                atomic_fn = "atomic_fetch_and_explicit";
            else
                use_cas = true;
            break;
        case ReduceOp::Or:
            if (vt == VarType::UInt32 || vt == VarType::Int32)
                atomic_fn = "atomic_fetch_or_explicit";
            else
                use_cas = true;
            break;
        default:
            use_cas = true;
            break;
    }

    if (!is_unmasked)
        fmt_metal("    if ($v) {\n    ", mask);

    if (atomic_fn && !use_cas) {
        // Direct atomic
        if (vt == VarType::Float32) {
            fmt_metal("    $s((device atomic_float*)((device $t*) $v + $v), "
                      "$v, memory_order_relaxed);\n",
                      atomic_fn, value, ptr, index, value);
        } else if (vt == VarType::UInt32) {
            fmt_metal("    $s((device atomic_uint*)((device $t*) $v + $v), "
                      "$v, memory_order_relaxed);\n",
                      atomic_fn, value, ptr, index, value);
        } else if (vt == VarType::Int32) {
            fmt_metal("    $s((device atomic_int*)((device $t*) $v + $v), "
                      "$v, memory_order_relaxed);\n",
                      atomic_fn, value, ptr, index, value);
        }
    } else {
        if (vt == VarType::Int64 || vt == VarType::UInt64)
            jitc_raise("jitc_metal_render_scatter(): the Metal backend does "
                       "not support the requested type of atomic reduction "
                       "(64-bit atomics are not available on Metal).");

        // CAS loop fallback for float min/max/mul, half, etc.
        // MSL atomic_compare_exchange_weak_explicit returns bool and
        // updates the expected value in-place on failure.
        if (vt == VarType::Float16) {
            // Half-precision: CAS via uint on the containing 32-bit word
            fmt_metal("    {\n"
                      "        uint _idx = (uint)((ulong)((device ushort*) $v + $v) - "
                      "(ulong)((device ushort*) $v)) / 2u;\n"
                      "        bool _odd = _idx & 1u;\n"
                      "        device atomic_uint *_addr32 = (device atomic_uint*)"
                      "((device ushort*) $v + $v) - (_odd ? 1 : 0);\n"
                      "        uint _old32 = atomic_load_explicit(_addr32, "
                      "memory_order_relaxed);\n"
                      "        while (true) {\n"
                      "            ushort _bits = _odd ? (ushort)(_old32 >> 16) "
                      ": (ushort)_old32;\n"
                      "            half _val = as_type<half>(_bits);\n",
                      ptr, index, ptr,
                      ptr, index);

            switch (op) {
                case ReduceOp::Add:
                    fmt_metal("            half _new_val = _val + $v;\n", value);
                    break;
                case ReduceOp::Min:
                    fmt_metal("            half _new_val = min(_val, $v);\n", value);
                    break;
                case ReduceOp::Max:
                    fmt_metal("            half _new_val = max(_val, $v);\n", value);
                    break;
                default:
                    jitc_fail("jitc_metal_render_scatter(): unsupported ReduceOp "
                              "for half CAS fallback.");
            }

            put("            ushort _new_bits = as_type<ushort>(_new_val);\n"
                "            uint _new32 = _odd ? "
                "(_old32 & 0xFFFFu) | ((uint)_new_bits << 16) : "
                "(_old32 & 0xFFFF0000u) | (uint)_new_bits;\n"
                "            uint _expected = _old32;\n"
                "            if (atomic_compare_exchange_weak_explicit("
                "_addr32, &_expected, _new32, "
                "memory_order_relaxed, memory_order_relaxed)) break;\n"
                "            _old32 = _expected;\n"
                "        }\n"
                "    }\n");
        } else {
            // 32-bit types (float32, int32, uint32): CAS via atomic_uint
            fmt_metal("    {\n"
                      "        device atomic_uint *_addr = "
                      "(device atomic_uint*)((device $t*) $v + $v);\n"
                      "        uint _old = atomic_load_explicit(_addr, "
                      "memory_order_relaxed);\n"
                      "        while (true) {\n",
                      value, ptr, index);

            switch (op) {
                case ReduceOp::Add:
                    fmt_metal("            $t _new = as_type<$t>(_old) + $v;\n",
                              value, value, value);
                    break;
                case ReduceOp::Mul:
                    fmt_metal("            $t _new = as_type<$t>(_old) * $v;\n",
                              value, value, value);
                    break;
                case ReduceOp::Min:
                    fmt_metal("            $t _new = min(as_type<$t>(_old), $v);\n",
                              value, value, value);
                    break;
                case ReduceOp::Max:
                    fmt_metal("            $t _new = max(as_type<$t>(_old), $v);\n",
                              value, value, value);
                    break;
                default:
                    jitc_fail("jitc_metal_render_scatter(): unsupported ReduceOp %s "
                              "for CAS fallback.", op_name);
            }

            put("            uint _expected = _old;\n"
                "            if (atomic_compare_exchange_weak_explicit("
                "_addr, &_expected, as_type<uint>(_new), "
                "memory_order_relaxed, memory_order_relaxed)) break;\n"
                "            _old = _expected;\n"
                "        }\n"
                "    }\n");
        }
    }

    if (!is_unmasked)
        put("    }\n");
}

void jitc_metal_render_scatter_cas(Variable *v) {
    // ScatterCAS: atomic compare-and-swap
    // deps: [0]=ptr, [1]=compare, [2]=value, [3]=index
    Variable *ptr     = jitc_var(v->dep[0]);
    Variable *compare = jitc_var(v->dep[1]);
    Variable *value   = jitc_var(v->dep[2]);
    Variable *index   = jitc_var(v->dep[3]);

    ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
    Variable *mask = jitc_var(cas_data->mask);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    // Initialize outputs to zero
    fmt_metal("    $t $v_out_0 = ($t) 0;\n"
              "    bool $v_out_1 = false;\n",
              value, v, value,
              v);

    if (!is_unmasked)
        fmt_metal("    if ($v) {\n    ", mask);

    fmt_metal("    {\n"
              "        device atomic_uint *_addr = (device atomic_uint*)"
              "((device $t*) $v + $v);\n"
              "        uint _expected = as_type<uint>($v);\n"
              "        bool _swapped = atomic_compare_exchange_weak_explicit("
              "_addr, &_expected, as_type<uint>($v), "
              "memory_order_relaxed, memory_order_relaxed);\n"
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
    // ScatterExch: atomic exchange, returns old value
    // deps: [0]=ptr, [1]=value, [2]=index, [3]=mask
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *value = jitc_var(v->dep[1]);
    Variable *index = jitc_var(v->dep[2]);
    Variable *mask  = jitc_var(v->dep[3]);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    fmt_metal("    $t $v = ($t) 0;\n", value, v, value);
    if (!is_unmasked)
        fmt_metal("    if ($v)\n    ", mask);
    fmt_metal("    $v = as_type<$t>(atomic_exchange_explicit("
              "(device atomic_uint*)((device $t*) $v + $v), "
              "as_type<uint>($v), memory_order_relaxed));\n",
              v, value, value, ptr, index, value);

    v->consumed = 1;
}

void jitc_metal_render_scatter_kahan(Variable *v) {
    Variable *ptr_t  = jitc_var(v->dep[0]);
    Variable *ptr_c  = jitc_var(v->dep[1]);
    Variable *index  = jitc_var(v->dep[2]);
    Variable *value  = jitc_var(v->dep[3]);

    fmt_metal("    if ($v != 0.f) {\n"
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
    // ScatterInc: atomic increment, returns old value
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *index = jitc_var(v->dep[1]);
    Variable *mask  = jitc_var(v->dep[2]);

    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    fmt_metal("    uint $v = 0;\n", v);
    if (!is_unmasked)
        fmt_metal("    if ($v)\n    ", mask);
    fmt_metal("    $v = atomic_fetch_add_explicit("
              "(device atomic_uint*)((device uint*) $v + $v), "
              "1u, memory_order_relaxed);\n",
              v, ptr, index);

    v->consumed = 1;
}

#undef fmt_metal
#undef put

#endif // defined(DRJIT_ENABLE_METAL)
