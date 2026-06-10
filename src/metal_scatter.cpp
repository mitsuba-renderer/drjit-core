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

/// Local aggregation is only available for 32-bit types and some reduction operations
static bool jitc_metal_can_reduce_local(VarType vt, ReduceOp op) {
    bool t32 = vt == VarType::UInt32 ||
               vt == VarType::Int32 ||
               vt == VarType::Float32;
    if (!t32)
        return false;
    switch (op) {
        case ReduceOp::Add:
        case ReduceOp::Min:
        case ReduceOp::Max:
            return true;
        case ReduceOp::And:
        case ReduceOp::Or:
            return vt != VarType::Float32;
        default:
            return false;
    }
}

/// Emit a MSL helper to detect the set of active SIMD lanes sharing a
/// 32-bit key. It uses an adaptive bitwise match, whose cost scales with
/// log2(distinct lanes). It returns once every group is a singleton, which
/// helps reduce cost for atomics targeting incoherent addresses.
static void jitc_metal_emit_peer_mask_helper() {
    fmt_intrinsic(
        "inline uint drjit_peer_mask(uint key, uint peers) {\n"
        "    uint varying = simd_or(key ^ simd_broadcast_first(key));\n"
        "    while (varying) {\n"
        "        if (simd_all(popcount(peers) <= 1u)) break;\n"
        "        uint b = ctz(varying);\n"
        "        varying &= varying - 1u;\n"
        "        uint b_mask = (uint) ((simd_vote::vote_t) simd_ballot((key >> b) & 1u));\n"
        "        peers &= ((key >> b) & 1u) ? b_mask : ~b_mask;\n"
        "    }\n"
        "    return peers;\n"
        "}");
}

/// Emit the shared SIMD peer-detection prologue, declaring ``lane``, the active
/// lane mask ``active``, and the per-lane peer group ``peers``.
static void jitc_metal_emit_warp_match(const char *t, const Variable *ptr,
                                       const Variable *index, uint32_t shiftamt) {
    const char *tid = callable_depth > 0 ? "index" : "r0";
    uint32_t simd_width = thread_state_metal->metal_simd_width;
    jitc_metal_emit_peer_mask_helper();
    fmt("uint lane = ($s) & $uu;\n"
        "uint active = (uint) ((simd_vote::vote_t) simd_active_threads_mask());\n"
        "uint key = (uint) (((ulong) ((device $s*) $v + $v)) >> $u);\n"
        "uint peers = drjit_peer_mask(key, active);\n",
        tid, simd_width - 1, t, ptr, index, shiftamt);
}

/// Emit a scatter-reduction that separately reduces ``n``variables of type
/// ``vt`` within the simdgroup, then atomically scatters the per-group result
/// to memory. If ``aggregate`` is set, lanes sharing the base address
/// pre-reduce through a SIMD butterfly so the group leader issues a single
/// device atomic per channel.
void jitc_metal_emit_reduce_block(uint32_t n, const uint32_t *values,
                                  const Variable *ptr, const Variable *index,
                                  ReduceOp op, bool aggregate) {
    fmt_intrinsic("#include <metal_atomic>");
    const Variable *v0  = jitc_var(values[0]);
    VarType vt          = (VarType) v0->type;
    const char *op_name = metal_reduce_op_name[(int) op];
    const char *t       = type_name_metal[(int) vt];

    // Pre-aggregation is only possible for the type/op combinations
    aggregate = aggregate && jitc_metal_can_reduce_local(vt, op);

    // Native atomics exist for 32-bit ints (add/min/max/and/or) and float32
    // (add). float32 min/max fall back to a CAS loop.
    bool native = vt == VarType::Int32 || vt == VarType::UInt32 ||
                  (vt == VarType::Float32 && op == ReduceOp::Add);
    const char *atomic_t = vt == VarType::Float32 ? "atomic_float"
                         : vt == VarType::Int32   ? "atomic_int"
                                                  : "atomic_uint";

    // Per-channel butterfly combine ``wval_i <op>= shuf_i``
    auto emit_combine = [&](uint32_t i) {
        switch (op) {
            case ReduceOp::Add: fmt("wval$u = wval$u + shuf$u;\n", i, i, i); break;
            case ReduceOp::Min: fmt("wval$u = min(wval$u, shuf$u);\n", i, i, i); break;
            case ReduceOp::Max: fmt("wval$u = max(wval$u, shuf$u);\n", i, i, i); break;
            case ReduceOp::And: fmt("wval$u = wval$u & shuf$u;\n", i, i, i); break;
            case ReduceOp::Or:  fmt("wval$u = wval$u | shuf$u;\n", i, i, i); break;
            default: jitc_fail("jitc_metal_emit_reduce_block(): unsupported reduction.");
        }
    };

    put("{\n");

    if (aggregate)
        jitc_metal_emit_warp_match(t, ptr, index, log2i_ceil(type_size[(int) vt]));

    for (uint32_t i = 0; i < n; ++i)
        fmt("$s wval$u = $v;\n", t, i, jitc_var(values[i]));

    if (aggregate) {
        const char *simd_fn = op == ReduceOp::Add ? "simd_sum"
                            : op == ReduceOp::Min ? "simd_min"
                            : op == ReduceOp::Max ? "simd_max"
                            : op == ReduceOp::And ? "simd_and"
                                                  : "simd_or";

        // Fully-coherent fast path: when every active lane shares the address,
        // a single hardware reduction replaces the peer-group butterfly.
        put("if (peers == active) {\n");
        for (uint32_t i = 0; i < n; ++i)
            fmt("    wval$u = $s(wval$u);\n", i, simd_fn, i);
        put("} else {\n");
        // Precompute the peer group structure once and then reuse across the packet
        put("uint rank = popcount(peers & ((1u << lane) - 1u));\n"
            "uint remaining = peers;\n"
            "while (simd_any(popcount(remaining) > 1u)) {\n"
            "    bool in_group  = ((remaining >> lane) & 1u) != 0u;\n"
            "    uint upper     = remaining & ((~0u << lane) << 1);\n"
            "    uint next_lane = upper ? ctz(upper) : lane;\n"
            "    bool recv      = in_group && (rank & 1u) == 0u && upper != 0u;\n");
        for (uint32_t i = 0; i < n; ++i)
            fmt("$s shuf$u = simd_shuffle(wval$u, (ushort) next_lane);\n", t, i, i);
        put("if (recv) {\n");
        for (uint32_t i = 0; i < n; ++i)
            emit_combine(i);
        put("    }\n"
            "    uint drop = (uint) ((simd_vote::vote_t) simd_ballot(in_group && (rank & 1u) != 0u));\n"
            "    remaining &= ~drop;\n"
            "    if (in_group) rank >>= 1;\n"
            "}\n"
            "}\n"
            "if (lane == ctz(peers)) {\n");
    }

    for (uint32_t i = 0; i < n; ++i) {
        if (native) {
            fmt("atomic_fetch_$s_explicit((device $s*)((device $t*) $v + ($v + $u)), "
                "wval$u, memory_order_relaxed);\n",
                op_name, atomic_t, v0, ptr, index, i, i);
        } else {
            // CAS loop fallback (float32 min/max).
            fmt("{\n"
                "    device atomic_uint *_addr = (device atomic_uint*)((device $t*) $v + ($v + $u));\n"
                "    uint _old = atomic_load_explicit(_addr, memory_order_relaxed);\n"
                "    while (true) {\n",
                v0, ptr, index, i);
            if (op == ReduceOp::Min)
                fmt("$t _new = min(as_type<$t>(_old), wval$u);\n", v0, v0, i);
            else if (op == ReduceOp::Max)
                fmt("$t _new = max(as_type<$t>(_old), wval$u);\n", v0, v0, i);
            else
                jitc_fail("jitc_metal_emit_reduce_block(): unsupported ReduceOp "
                          "%s for the CAS fallback.", op_name);
            put("        uint _expected = _old;\n"
                "        if (atomic_compare_exchange_weak_explicit(_addr, &_expected, as_type<uint>(_new), memory_order_relaxed, memory_order_relaxed)) break;\n"
                "        _old = _expected;\n"
                "    }\n"
                "}\n");
        }
    }

    if (aggregate)
        put("}\n");

    put("}\n");
}

void jitc_metal_render_scatter(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *value = jitc_var(v->dep[1]);
    Variable *index = jitc_var(v->dep[2]);
    Variable *mask  = jitc_var(v->dep[3]);

    ReduceOp   op   = (ReduceOp) (uint32_t) v->literal;
    ReduceMode mode = (ReduceMode) (uint32_t) (v->literal >> 32);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    if (op == ReduceOp::Identity) {
        if (is_unmasked)
            fmt("((device $t*) $v)[$v] = $v;\n",
                      value, ptr, index, value);
        else
            fmt("if ($v) ((device $t*) $v)[$v] = $v;\n",
                      mask, value, ptr, index, value);
        return;
    }

    if (!is_unmasked)
        fmt("if ($v) {\n", mask);

    // Treat as a single-channel packet reduce
    uint32_t vi = v->dep[1];
    jitc_metal_emit_reduce_block(1, &vi, ptr, index, op, mode == ReduceMode::Local);

    if (!is_unmasked)
        put("}\n");
}

void jitc_metal_render_scatter_cas(Variable *v) {
    Variable *ptr     = jitc_var(v->dep[0]);
    Variable *compare = jitc_var(v->dep[1]);
    Variable *value   = jitc_var(v->dep[2]);
    Variable *index   = jitc_var(v->dep[3]);

    fmt_intrinsic("#include <metal_atomic>");

    ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
    Variable *mask = jitc_var(cas_data->mask);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    // Initialize outputs to zero
    fmt("$t $v_out_0 = ($t) 0;\n"
              "bool $v_out_1 = false;\n",
              value, v, value,
              v);

    if (!is_unmasked)
        fmt("if ($v) {\n", mask);

    fmt("{\n"
              "    device atomic_uint *_addr = (device atomic_uint*) ((device $t*) $v + $v);\n"
              "    uint _expected = as_type<uint>($v);\n"
              "    bool _swapped = atomic_compare_exchange_weak_explicit(_addr, &_expected, as_type<uint>($v), memory_order_relaxed, memory_order_relaxed);\n"
              "    $v_out_0 = as_type<$t>(_expected);\n"
              "    $v_out_1 = _swapped;\n"
              "}\n",
              value, ptr, index,
              compare,
              value,
              v, value,
              v);

    if (!is_unmasked)
        put("}\n");

    v->consumed = 1;
}

void jitc_metal_render_scatter_exch(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *value = jitc_var(v->dep[1]);
    Variable *index = jitc_var(v->dep[2]);
    Variable *mask  = jitc_var(v->dep[3]);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    fmt_intrinsic("#include <metal_atomic>");

    fmt("$t $v = ($t) 0;\n", value, v, value);
    if (!is_unmasked)
        fmt("if ($v)\n", mask);
    fmt("$v = as_type<$t>(atomic_exchange_explicit((device atomic_uint*)((device $t*) $v + $v), as_type<uint>($v), memory_order_relaxed));\n",
              v, value, value, ptr, index, value);

    v->consumed = 1;
}

void jitc_metal_render_scatter_kahan(Variable *v) {
    Variable *ptr_t  = jitc_var(v->dep[0]);
    Variable *ptr_c  = jitc_var(v->dep[1]);
    Variable *index  = jitc_var(v->dep[2]);
    Variable *value  = jitc_var(v->dep[3]);

    fmt_intrinsic("#include <metal_atomic>");

    fmt("if ($v != 0.f) {\n"
              "    #pragma clang fp contract(off)\n"
              "    volatile float _kahan_before = atomic_fetch_add_explicit("
              "(device atomic_float*)((device float*) $v + $v), "
              "$v, memory_order_relaxed);\n"
              "    volatile float _kahan_after = _kahan_before + $v;\n"
              "    volatile float _kahan_c1 = (_kahan_before - _kahan_after) + $v;\n"
              "    volatile float _kahan_c2 = ($v - _kahan_after) + _kahan_before;\n"
              "    float _kahan_comp = abs(_kahan_before) >= abs($v) ? _kahan_c1 : _kahan_c2;\n"
              "    atomic_fetch_add_explicit("
              "(device atomic_float*)((device float*) $v + $v), "
              "_kahan_comp, memory_order_relaxed);\n"
              "}\n",
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

    fmt_intrinsic("#include <metal_atomic>");

    fmt("uint $v = 0;\n", v);

    if (!is_unmasked)
        fmt("if ($v) {\n", mask);

    // Perform a warp-aggregated atomic increment
    put("{\n");
    jitc_metal_emit_warp_match("uint", ptr, index, 2);
    fmt("    uint rank = popcount(peers & ((1u << lane) - 1u));\n"
        "    uint base = 0;\n"
        "    if (rank == 0u)\n"
        "        base = atomic_fetch_add_explicit((device atomic_uint*)((device uint*) $v + $v), popcount(peers), memory_order_relaxed);\n"
        "    base = simd_shuffle(base, (ushort) ctz(peers));\n"
        "    $v = base + rank;\n"
        "}\n",
        ptr, index,
        v);

    if (!is_unmasked)
        put("}\n");

    v->consumed = 1;
}
