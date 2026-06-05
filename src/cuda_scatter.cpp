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
#include "src/log.h"
#include "var.h"
#include "op.h"
#include "cuda_eval.h"
#include "cuda_scatter.h"

const char *cuda_reduce_op_name[(int) ReduceOp::Count] = {
    "", "add", "mul", "min", "max", "and", "or"
};

void jitc_cuda_prepare_index(const Variable *ptr, const Variable *index, const Variable *value) {
    if (index->is_literal() && index->literal == 0) {
        fmt("    mov.u64 %rd3, $v;\n", ptr);
    } else if (type_size[value->type] == 1) {
        fmt("    cvt.u64.$t %rd3, $v;\n"
            "    add.u64 %rd3, %rd3, $v;\n", index, index, ptr);
    } else {
        fmt("    mad.wide.$t %rd3, $v, $a, $v;\n",
            index, index, value, ptr);
    }
}

/// Perform an ordinary scatter operation
void jitc_cuda_render_scatter(const Variable *, const Variable *ptr,
                              const Variable *value, const Variable *index,
                              const Variable *mask) {

    jitc_cuda_prepare_index(ptr, index, value);

    bool is_bool = value->type == (uint32_t) VarType::Bool,
         is_unmasked = mask->is_literal() && mask->literal == 1;

    if (is_bool)
        fmt("    selp.u16 %w0, 1, 0, $v;\n", value);

    put("    ");

    if (!is_unmasked)
        fmt("@$v ", mask);

    if (is_bool)
        put("st.global.u8 [%rd3], %w0;\n");
    else
        fmt("st.global.$b [%rd3], $v;\n", value, value);
}

/// Emit the shared warp peer-detection prologue
static void jitc_cuda_emit_warp_match(uint32_t shiftamt) {
    fmt("        activemask.b32 %active;\n"
        "        {\n"
        "            .reg .b64 %ptr_shift;\n"
        "            shr.b64 %ptr_shift, %rd3, $u;\n"
        "            cvt.u32.u64 %index, %ptr_shift;\n"
        "        }\n"
        "        match.any.sync.b32 %peers, %index, %active;\n",
        shiftamt);
}

const char *jitc_cuda_reduce_tp(VarType &vt, ReduceOp op) {
    if (op == ReduceOp::Add) {
        switch (vt) {
            case VarType::Int32: vt = VarType::UInt32; break;
            case VarType::Int64: vt = VarType::UInt64; break;
            default: break;
        }
    }

    return (op == ReduceOp::And || op == ReduceOp::Or) ? type_name_ptx_bin[(int) vt]
                                                       : type_name_ptx[(int) vt];
}

/// Emit a segmented butterfly warp-reduction that separately reduces ``n``
/// variables of type ``vt`` within the warp, then atomically scatters the
/// per-group result to memory. The packet base address is assumed to be in
/// ``%rd3``.  Uses packet atomics if ``use_packet_atomics`` is set.
void jitc_cuda_render_warp_reduce(uint32_t n, const uint32_t *values, VarType vt,
                                  ReduceOp op, bool use_packet_atomics) {
    const char *tp      = jitc_cuda_reduce_tp(vt, op);
    const char *op_name = cuda_reduce_op_name[(int) op];
    const char *op_ftz  = op_name;
    if (op == ReduceOp::Add && (vt == VarType::Float32 || vt == VarType::Float16))
        op_ftz = "add.ftz";

    uint32_t tsize    = type_size[(int) vt];
    uint32_t shiftamt = log2i_ceil(tsize);
    bool     is64     = tsize == 8;

    put("    {\n"
        "        .reg .b32 %active, %index, %mask_lt, %mask_gt, %peers,\n"
        "                  %peers_lt, %peers_rev, %rank, %rank_bit, %rank_ballot;\n");
    if (is64)
        put("        .reg .b32 %q0l, %q0h, %q1l, %q1h;\n");
    fmt("        .reg .$s %q0_<$u>, %q1;\n"
        "        .reg .pred %leader, %partial, %done, %valid, %rank_even, %unused;\n\n",
        tp, n);

    for (uint32_t i = 0; i < n; ++i)
        fmt("        mov.$s %q0_$u, $v;\n", tp, i, jitc_var(values[i]));

    jitc_cuda_emit_warp_match(shiftamt);

    put("        setp.ne.s32 %partial, %peers, -1;\n"
        "        @%partial bra reduce_partial;\n\n");

    // If the warp is fully coherent, do a normal butterfly reduction and scatter
    put("        mov.b32 %index, %laneid;\n"
        "        setp.eq.u32 %leader, %index, 0;\n");
    for (uint32_t delta : {1u, 2u, 4u, 8u, 16u}) {
        for (uint32_t i = 0; i < n; ++i) {
            if (!is64)
                fmt("        shfl.sync.bfly.b32 %q1|%unused, %q0_$u, $u, 31, %active;\n"
                    "        $s.$s %q0_$u, %q0_$u, %q1;\n",
                    i, delta, op_ftz, tp, i, i);
            else
                fmt("        mov.b64 {%q0l, %q0h}, %q0_$u;\n"
                    "        shfl.sync.bfly.b32 %q1l|%unused, %q0l, $u, 31, %active;\n"
                    "        shfl.sync.bfly.b32 %q1h|%unused, %q0h, $u, 31, %active;\n"
                    "        mov.b64 %q1, {%q1l, %q1h};\n"
                    "        $s.$s %q0_$u, %q0_$u, %q1;\n",
                    i, delta, delta, op_ftz, tp, i, i);
        }
    }

    // Otherwise, do a reduction within segments
    put("        bra reduce_done;\n\n"
        "    reduce_partial:\n"
        "        mov.u32 %mask_lt, %lanemask_lt;\n"
        "        mov.u32 %mask_gt, %lanemask_gt;\n"
        "        and.b32 %peers_lt, %peers, %mask_lt;\n"
        "        popc.b32 %rank, %peers_lt;\n"
        "        setp.eq.u32 %leader, %rank, 0;\n"
        "        and.b32 %peers, %peers, %mask_gt;\n\n"
        "    reduce_partial_loop:\n"
        "        setp.eq.u32 %done, %peers, 0;\n"
        "        vote.sync.all.pred %done, %done, %active;\n"
        "        @%done bra reduce_done;\n\n"
        "        brev.b32 %peers_rev, %peers;\n"
        "        bfind.shiftamt.u32 %index, %peers_rev;\n"
        "        setp.ne.s32 %valid, %index, -1;\n");

    for (uint32_t i = 0; i < n; ++i) {
        if (!is64)
            fmt("        shfl.sync.idx.b32 %q1|%unused, %q0_$u, %index, 31, %active;\n"
                "        @%valid $s.$s %q0_$u, %q0_$u, %q1;\n",
                i, op_ftz, tp, i, i);
        else
            fmt("        mov.b64 {%q0l, %q0h}, %q0_$u;\n"
                "        shfl.sync.idx.b32 %q1l|%unused, %q0l, %index, 31, %active;\n"
                "        shfl.sync.idx.b32 %q1h|%unused, %q0h, %index, 31, %active;\n"
                "        mov.b64 %q1, {%q1l, %q1h};\n"
                "        @%valid $s.$s %q0_$u, %q0_$u, %q1;\n",
                i, op_ftz, tp, i, i);
    }

    put("        and.b32 %rank_bit, %rank, 1;\n"
        "        setp.eq.u32 %rank_even, %rank_bit, 0;\n"
        "        vote.sync.ballot.b32 %rank_ballot, %rank_even, %active;\n"
        "        and.b32 %peers, %peers, %rank_ballot;\n"
        "        shr.u32 %rank, %rank, 1;\n"
        "        bra reduce_partial_loop;\n\n"
        "    reduce_done:\n");

    // Generate scalar atomics or packet atomics (cap to 128bit/thread)
    uint32_t per_atomic = 1;
    if (use_packet_atomics) {
        uint32_t bytes_per_atomic = 16;
        while ((n * tsize) % bytes_per_atomic != 0)
            bytes_per_atomic /= 2;
        per_atomic = bytes_per_atomic / tsize;
    }

    for (uint32_t base = 0; base < n; base += per_atomic) {
        if (per_atomic == 1) {
            fmt("        @%leader red.global.$s.$s [%rd3+$u], %q0_$u;\n",
                op_name, tp, base * tsize, base);
        } else {
            fmt("        @%leader red.global.v$u.$s.$s [%rd3+$u], {",
                per_atomic, tp, op_name, base * tsize);
            for (uint32_t i = 0; i < per_atomic; ++i)
                fmt("%q0_$u$s", base + i, i + 1 < per_atomic ? ", " : "");
            put("};\n");
        }
    }
    put("    }\n");
}

void jitc_cuda_render_scatter_reduce(const Variable *v,
                                     const Variable *ptr,
                                     const Variable *value,
                                     const Variable *index,
                                     const Variable *mask) {
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    if (!is_unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    jitc_cuda_prepare_index(ptr, index, value);

    ReduceOp op = (ReduceOp) (uint32_t) v->literal;
    VarType vt = (VarType) value->type;
    const char *tp = jitc_cuda_reduce_tp(vt, op);
    const char *op_name = cuda_reduce_op_name[(int) op];

    ReduceMode mode = (ReduceMode) (uint32_t) (v->literal >> 32);
    const ThreadState *ts = thread_state_cuda;
    bool warp_reduce_supported = ts->ptx_version >= 63 && ts->compute_capability >= 70 &&
                       (vt == VarType::UInt32 || vt == VarType::Int32 ||
                        vt == VarType::Float32 || vt == VarType::UInt64 ||
                        vt == VarType::Int64 || vt == VarType::Float64);

    if (mode == ReduceMode::NoConflicts) {
        const char *tp_b = type_name_ptx_bin[(int) vt];
        fmt("    {\n"
            "        .reg.$s %tmp;\n"
            "        ld.global.$s %tmp, [%rd3];\n"
            "        $s.$s %tmp, %tmp, $v;\n"
            "        st.global.$s [%rd3], %tmp;\n"
            "    }\n",
            tp, tp_b, op_name, tp, value, tp_b);
    } else if (mode == ReduceMode::Local && warp_reduce_supported) {
        uint32_t vi = v->dep[1];
        jitc_cuda_render_warp_reduce(1, &vi, (VarType) value->type, op,
                                     /* use_packet_atomics */ false);
    } else if ((VarType) value->type == VarType::Float16) {
        // NVIDIA hardware provides f16x2 (double bandwidth) half precision
        // atomics but no 1x version. Attempting to use the 1x-wide instructions
        // requires software emulation, and this seems poorly implemented on
        // some driver versions. (we ran into issues/miscompilations with
        // OptiX). The solution is to emulate the 1x scatter *ourselves* by
        // reducing it to the 2x version.
        uint64_t identity = jitc_reduce_identity((VarType) value->type, op);

        if (op == ReduceOp::Add) {
            // Use the more broadly supported `.f16x2` instructions, only available for addition.
            fmt("    {\n"
                "        .reg .f16 %identity;\n"
                "        .reg .f16x2 %packed;\n"
                "        cvt.u32.u64 %r3, %rd3;\n"
                "        and.b32 %r3, %r3, 2;\n"
                "        setp.eq.b32 %p3, %r3, 0;\n"
                "        mov.b16 %identity, $u;\n"
                "        mov.b32 %r3, {%identity, $v};\n"
                "        @%p3 prmt.b32 %r3, %r3, 0, 0x1032;\n"
                "        mov.b32 %packed, %r3;\n"
                "        and.b64 %rd3, %rd3, ~0x2;\n"
                "        red.global.add.noftz.f16x2 [%rd3], %packed;\n"
                "    }\n", (uint32_t) identity, value);
        } else if (ts->compute_capability > 90 && !uses_optix) {
            // Use the new `.v2.f16` instructions to enable min & max.
            switch (op) {
                case ReduceOp::Add: op_name = "red.global.v2.f16.add.noftz"; break;
                case ReduceOp::Min: op_name = "red.global.v2.f16.min.noftz"; break;
                case ReduceOp::Max: op_name = "red.global.v2.f16.max.noftz"; break;
                default: break;
            }
            fmt("    {\n"
                "        .reg .f16 %op1, %op2;\n"
                // Determine whether we are trying to scatter to the
                // first or the second f16 value.
                "        cvt.u32.u64 %r3, %rd3;\n"
                "        and.b32 %r3, %r3, 2;\n"
                "        setp.eq.b32 %p3, %r3, 0;\n"
                // Set operands based on the above:
                //     op1 = select(is_even, $v, identity)
                //     op2 = select(is_even, identity, $v)
                "        selp.b16 %op1, $v, $u, %p3;\n"
                "        selp.b16 %op2, $u, $v, %p3;\n"
                "        and.b64 %rd3, %rd3, ~0x2;\n"
                "        $s [%rd3], {%op1, %op2};\n"
                "    }\n",
                value, (uint32_t) identity,
                (uint32_t) identity, value,
                op_name);

        } else {
            jitc_fail("jitc_cuda_render_scatter_reduce(): internal error. The "
                      "requested operation (\"%s\") is not supported on the "
                      "backend and should not have been generated.",
                      cuda_reduce_op_name[(int) op]);
        }

    } else {
        fmt("    red.global.$s.$s [%rd3], $v;\n",
            op_name, tp, value);
    }

    if (!is_unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);
}

void jitc_cuda_render_scatter_inc(Variable *v, const Variable *ptr,
                                  const Variable *index, const Variable *mask) {
    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    uint32_t uid = v->reg_index;

    if (!is_unmasked)
        fmt("    mov.$b $v, 0;\n"
            "    @!$v bra l_$u_done;\n",
            v, v, mask, uid);

    jitc_cuda_prepare_index(ptr, index, index);

    // Perform a warp-aggregated atomic increment
    put("    {\n"
        "        .reg .b32 %active, %index, %peers, %peers_rev, %leader_idx,\n"
        "                  %lt_mask, %lt_active_p, %lt_ct, %peers_ct, %prev, %leader_val;\n"
        "        .reg .pred %leader, %unused;\n");
    jitc_cuda_emit_warp_match(2);
    fmt("        brev.b32 %peers_rev, %peers;\n"
        "        clz.b32 %leader_idx, %peers_rev;\n"
        "        mov.u32 %lt_mask, %lanemask_lt;\n"
        "        and.b32 %lt_active_p, %lt_mask, %peers;\n"
        "        setp.eq.u32 %leader, %lt_active_p, 0;\n"
        "        @!%leader bra l_inc_fetch;\n\n"
        "        popc.b32 %peers_ct, %peers;\n"
        "        atom.global.add.u32 %prev, [%rd3], %peers_ct;\n\n"
        "    l_inc_fetch:\n"
        "        shfl.sync.idx.b32 %leader_val|%unused, %prev, %leader_idx, 31, %active;\n"
        "        popc.b32 %lt_ct, %lt_active_p;\n"
        "        add.u32 $v, %lt_ct, %leader_val;\n"
        "    }\n",
        v);

    if (!is_unmasked)
        fmt("\nl_$u_done:\n", uid);

    v->consumed = 1;
}

void jitc_cuda_render_scatter_add_kahan(const Variable *v,
                                        const Variable *ptr_1,
                                        const Variable *ptr_2,
                                        const Variable *index,
                                        const Variable *value) {
    fmt("    setp.eq.$t %p3, $v, 0.0;\n"
        "    @%p3 bra l_$u_done;\n"
        "    mad.wide.$t %rd2, $v, $a, $v;\n"
        "    mad.wide.$t %rd3, $v, $a, $v;\n",
        value, value,
        v->reg_index,
        index, index, value, ptr_1,
        index, index, value, ptr_2);

    const char* op_suffix = jitc_is_single(value) ? ".ftz" : "";

    fmt("    {\n"
        "        .reg.$t %before, %after, %value, %case_1, %case_2;\n"
        "        .reg.$t %abs_before, %abs_value, %result;\n"
        "        .reg.pred %cond;\n"
        "\n"
        "        mov.$t %value, $v;\n"
        "        atom.global.add.$t %before, [%rd2], %value;\n"
        "        add$s.$t %after, %before, %value;\n"
        "        sub$s.$t %case_1, %before, %after;\n"
        "        add$s.$t %case_1, %case_1, %value;\n"
        "        sub$s.$t %case_2, %value, %after;\n"
        "        add$s.$t %case_2, %case_2, %before;\n"
        "        abs$s.$t %abs_before, %before;\n"
        "        abs$s.$t %abs_value, %value;\n"
        "        setp.ge.$t %cond, %abs_before, %abs_value;\n"
        "        selp.$t %result, %case_1, %case_2, %cond;\n"
        "        red.global.add.$t [%rd3], %result;\n"
        "    }\n",
        value,
        value,
        value, value,
        value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        value,
        value,
        value);

    fmt("\nl_$u_done:\n", v->reg_index);
}

void jitc_cuda_render_scatter_exch(Variable *v,
                                   const Variable *ptr,
                                   const Variable *value,
                                   const Variable *index,
                                   const Variable *mask) {
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    jitc_cuda_prepare_index(ptr, index, value);

    fmt("    mov.$b $v, 0;\n"
        "    ",
        v, v);
    if (!is_unmasked)
        fmt("@$v ", mask);
    fmt("atom.global.exch.$b $v, [%rd3], $v;\n",
        value, v, value);

    v->consumed = 1;
}

void jitc_cuda_render_scatter_cas(Variable *v,
                                  const Variable *ptr,
                                  const Variable *compare,
                                  const Variable *value,
                                  const Variable *index) {
    ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
    Variable *mask = jitc_var(cas_data->mask);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    jitc_cuda_prepare_index(ptr, index, value);

    fmt("    .reg.$b $v_out_0;\n"
        "    mov.$b $v_out_0, 0;\n"
        "    .reg.pred $v_out_1;\n"
        "    mov.pred $v_out_1, 0;\n",
        value, v,
        value, v,
        v,
        v);

    put("    ");
    if (!is_unmasked)
        fmt("@$v ", mask);

    fmt("atom.global.cas.$b $v_out_0, [%rd3], $v, $v;\n"
        "    setp.eq.$b $v_out_1, $v_out_0, $v;\n",
        value, v, compare, value,
        value, v, v, compare);

    v->consumed = 1;
}
