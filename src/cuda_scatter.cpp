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
#include "var.h"
#include "cuda_eval.h"
#include "cuda_scatter.h"

static const char *reduce_op_name[(int) ReduceOp::Count] = {
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

/// Append a callable to the kernel that performs a butterfly reduction
/// over warp elements targeting the same memory region, which reduces
/// the number of needed atomic writes. The implementation has special cases
/// for completely coherent, completely incoherent, and partially coherent
/// scatters.
void jitc_cuda_render_scatter_reduce_bfly_32(const char *tp, const char *op,
                                             const char *op_ftz, uint32_t shiftamt) {
    fmt_intrinsic(
        ".func reduce_$s_$s(.param .u64 ptr, .param .$s value) {\n"
        "    .reg .b32 %active, %index, %mask_lt, %mask_gt, %peers, \n"
        "              %peers_lt, %peers_rev, %rank, %rank_bit, %rank_ballot;\n"
        "    .reg .b64 %ptr, %ptr_shift;\n"
        "    .reg .$s %q0, %q1;\n"
        "    .reg .pred %leader, %partial, %done, %valid, %rank_even;\n\n"
        ""
        "    ld.param.$s %q0, [value];\n"
        "    ld.param.b64 %ptr, [ptr];\n"
        "    activemask.b32 %active;\n"
        "    shr.b64 %ptr_shift, %ptr, $u;\n"
        "    cvt.u32.u64 %index, %ptr_shift;\n"
        "    match.any.sync.b32 %peers, %index, %active;\n"
        "    setp.ne.s32 %partial, %peers, -1;\n"
        "    @%partial bra.uni reduce_partial;\n\n"
        ""
        "reduce_full:\n"
        "    mov.b32 %index, %laneid;\n"
        "    setp.eq.u32 %leader, %index, 0;\n"
        "    shfl.sync.bfly.b32 %q1, %q0, 1, 31, %active;\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    shfl.sync.bfly.b32 %q1, %q0, 2, 31, %active;\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    shfl.sync.bfly.b32 %q1, %q0, 4, 31, %active;\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    shfl.sync.bfly.b32 %q1, %q0, 8, 31, %active;\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    shfl.sync.bfly.b32 %q1, %q0, 16, 31, %active;\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    bra.uni do_write;\n\n"
        ""
        "reduce_partial:\n"
        "    mov.u32 %mask_lt, %lanemask_lt;\n"
        "    mov.u32 %mask_gt, %lanemask_gt;\n"
        "    and.b32 %peers_lt, %peers, %mask_lt;\n"
        "    popc.b32 %rank, %peers_lt;\n"
        "    setp.eq.u32 %leader, %rank, 0;\n"
        "    and.b32 %peers, %peers, %mask_gt;\n\n"
        ""
        "reduce_partial_loop:\n"
        "    setp.eq.u32 %done, %peers, 0;\n"
        "    vote.sync.all.pred %done, %done, %active;\n"
        "    @%done bra.uni do_write;\n\n"
        ""
        "    brev.b32 %peers_rev, %peers;\n"
        "    bfind.shiftamt.u32 %index, %peers_rev;\n"
        "    setp.ne.s32 %valid, %index, -1;\n"
        "    shfl.sync.idx.b32 %q1, %q0, %index, 31, %active;\n"
        "    @%valid $s.$s %q0, %q0, %q1;\n"
        "    and.b32 %rank_bit, %rank, 1;\n"
        "    setp.eq.u32 %rank_even, %rank_bit, 0;\n"
        "    vote.sync.ballot.b32 %rank_ballot, %rank_even, %active;\n"
        "    and.b32 %peers, %peers, %rank_ballot;\n"
        "    shr.u32 %rank, %rank, 1;\n"
        "    bra reduce_partial_loop;\n\n"
        ""
        "do_write:\n"
        "    @%leader red.global.$s.$s [%ptr], %q0;\n"
        "    ret;\n"
        "}",
        op, tp, tp,
        tp,
        tp,
        shiftamt,

        // reduce_full
        op_ftz, tp,
        op_ftz, tp,
        op_ftz, tp,
        op_ftz, tp,
        op_ftz, tp,

        // reduce_partial_loop
        op_ftz, tp,

        // do_write
        op, tp);
}

// Identical to the above, but perform each shuffle twice to move around 64
// bit-sized values
void jitc_cuda_render_scatter_reduce_bfly_64(const char *tp, const char *op,
                                             const char *op_ftz, uint32_t shiftamt) {
    fmt_intrinsic(
        ".func reduce_$s_$s(.param .u64 ptr, .param .$s value) {\n"
        "    .reg .b32 %active, %index, %mask_lt, %mask_gt, %peers, \n"
        "              %peers_lt, %peers_rev, %rank, %rank_bit, %rank_ballot;\n"
        "    .reg .b64 %ptr, %ptr_shift;\n"
        "    .reg .b32 %q0l, %q0h, %q1l, %q1h;\n"
        "    .reg .$s %q0, %q1;\n"
        "    .reg .pred %leader, %partial, %done, %valid, %rank_even;\n\n"
        ""
        "    ld.param.$s %q0, [value];\n"
        "    ld.param.b64 %ptr, [ptr];\n"
        "    activemask.b32 %active;\n"
        "    shr.b64 %ptr_shift, %ptr, $u;\n"
        "    cvt.u32.u64 %index, %ptr_shift;\n"
        "    match.any.sync.b32 %peers, %index, %active;\n"
        "    setp.ne.s32 %partial, %peers, -1;\n"
        "    @%partial bra.uni reduce_partial;\n\n"
        ""
        "reduce_full:\n"
        "    mov.b32 %index, %laneid;\n"
        "    setp.eq.u32 %leader, %index, 0;\n"
        "    mov.b64 {%q0l, %q0h}, %q0;\n"
        "    shfl.sync.bfly.b32 %q1l, %q0l, 1, 31, %active;\n"
        "    shfl.sync.bfly.b32 %q1h, %q0h, 1, 31, %active;\n"
        "    mov.b64 %q1, {%q1l, %q1h};\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    mov.b64 {%q0l, %q0h}, %q0;\n"
        "    shfl.sync.bfly.b32 %q1l, %q0l, 2, 31, %active;\n"
        "    shfl.sync.bfly.b32 %q1h, %q0h, 2, 31, %active;\n"
        "    mov.b64 %q1, {%q1l, %q1h};\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    mov.b64 {%q0l, %q0h}, %q0;\n"
        "    shfl.sync.bfly.b32 %q1l, %q0l, 4, 31, %active;\n"
        "    shfl.sync.bfly.b32 %q1h, %q0h, 4, 31, %active;\n"
        "    mov.b64 %q1, {%q1l, %q1h};\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    mov.b64 {%q0l, %q0h}, %q0;\n"
        "    shfl.sync.bfly.b32 %q1l, %q0l, 8, 31, %active;\n"
        "    shfl.sync.bfly.b32 %q1h, %q0h, 8, 31, %active;\n"
        "    mov.b64 %q1, {%q1l, %q1h};\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    mov.b64 {%q0l, %q0h}, %q0;\n"
        "    shfl.sync.bfly.b32 %q1l, %q0l, 16, 31, %active;\n"
        "    shfl.sync.bfly.b32 %q1h, %q0h, 16, 31, %active;\n"
        "    mov.b64 %q1, {%q1l, %q1h};\n"
        "    $s.$s %q0, %q0, %q1;\n"
        "    bra.uni do_write;\n\n"
        ""
        "reduce_partial:\n"
        "    mov.u32 %mask_lt, %lanemask_lt;\n"
        "    mov.u32 %mask_gt, %lanemask_gt;\n"
        "    and.b32 %peers_lt, %peers, %mask_lt;\n"
        "    popc.b32 %rank, %peers_lt;\n"
        "    setp.eq.u32 %leader, %rank, 0;\n"
        "    and.b32 %peers, %peers, %mask_gt;\n\n"
        ""
        "reduce_partial_loop:\n"
        "    setp.eq.u32 %done, %peers, 0;\n"
        "    vote.sync.all.pred %done, %done, %active;\n"
        "    @%done bra.uni do_write;\n\n"
        ""
        "    brev.b32 %peers_rev, %peers;\n"
        "    bfind.shiftamt.u32 %index, %peers_rev;\n"
        "    setp.ne.s32 %valid, %index, -1;\n"
        "    mov.b64 {%q0l, %q0h}, %q0;\n"
        "    shfl.sync.idx.b32 %q1l, %q0l, %index, 31, %active;\n"
        "    shfl.sync.idx.b32 %q1h, %q0h, %index, 31, %active;\n"
        "    mov.b64 %q1, {%q1l, %q1h};\n"
        "    @%valid $s.$s %q0, %q0, %q1;\n"
        "    and.b32 %rank_bit, %rank, 1;\n"
        "    setp.eq.u32 %rank_even, %rank_bit, 0;\n"
        "    vote.sync.ballot.b32 %rank_ballot, %rank_even, %active;\n"
        "    and.b32 %peers, %peers, %rank_ballot;\n"
        "    shr.u32 %rank, %rank, 1;\n"
        "    bra reduce_partial_loop;\n\n"
        ""
        "do_write:\n"
        "    @%leader red.global.$s.$s [%ptr], %q0;\n"
        "    ret;\n"
        "}",
        op, tp, tp,
        tp,
        tp,
        shiftamt,

        // reduce_full
        op_ftz, tp,
        op_ftz, tp,
        op_ftz, tp,
        op_ftz, tp,
        op_ftz, tp,

        // reduce_partial
        op_ftz, tp,

        // do_write
        op, tp);
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

    if (op == ReduceOp::Add) {
        switch (vt) {
            case VarType::Int32: vt = VarType::UInt32; break;
            case VarType::Int64: vt = VarType::UInt64; break;
            default: break;
        }
    }

    const char *tp = type_name_ptx[(int) vt];

    if (op == ReduceOp::And || op == ReduceOp::Or)
        tp = type_name_ptx_bin[(int) vt];

    const char *op_name = reduce_op_name[(int) op],
               *op_name_ftz = op_name;

    if (op == ReduceOp::Add && (vt == VarType::Float32 || vt == VarType::Float16))
        op_name_ftz = "add.ftz";

    ReduceMode mode = (ReduceMode) (uint32_t) (v->literal >> 32);
    const ThreadState *ts = thread_state_cuda;
    bool reduce_bfly_32 = ts->ptx_version >= 63 && ts->compute_capability >= 70 &&
                          (vt == VarType::UInt32 || vt == VarType::Int32 || vt == VarType::Float32),
         reduce_bfly_64 = ts->ptx_version >= 63 && ts->compute_capability >= 70 &&
                          (vt == VarType::UInt64 || vt == VarType::Int64 || vt == VarType::Float64);

    if (mode == ReduceMode::NoConflicts) {
        const char *tp_b = type_name_ptx_bin[(int) vt];
        fmt("    {\n"
            "        .reg.$s %tmp;\n"
            "        ld.global.$s %tmp, [%rd3];\n"
            "        $s.$s %tmp, %tmp, $v;\n"
            "        st.global.$s [%rd3], %tmp;\n"
            "    }\n",
            tp, tp_b, op_name, tp, value, tp_b);
    } else if (mode == ReduceMode::Local && (reduce_bfly_32 || reduce_bfly_64)) {
        uint32_t shiftamt = log2i_ceil(type_size[(int) vt]);
        if (reduce_bfly_32)
            jitc_cuda_render_scatter_reduce_bfly_32(tp, op_name, op_name_ftz, shiftamt);
        else if (reduce_bfly_64)
            jitc_cuda_render_scatter_reduce_bfly_64(tp, op_name, op_name_ftz, shiftamt);

        fmt("    {\n"
            "        .func reduce_$s_$s(.param .u64 ptr, .param .$s value);\n"
            "        call.uni reduce_$s_$s, (%rd3, $v);\n"
            "    }\n",
            op_name, tp, tp, op_name, tp, value);
    } else if ((VarType) value->type == VarType::Float16) {
        // FTZ f16 add does not exist, must generate .noftz variant
        switch (op) {
            case ReduceOp::Add: op_name = "add.noftz"; break;
            case ReduceOp::Min: op_name = "min.noftz"; break;
            case ReduceOp::Max: op_name = "max.noftz"; break;
            default: break;
        }

        // NVIDIA hardware apparently provides f16x2 (double bandwidth) half
        // precision atomics but no 1x version. Attempting to use the 1x-wide
        // instructions requires software emulation, and this seems poorly
        // implemented on some driver versions. (we ran into
        // issues/miscompilations with OptiX). The solution is to emulate the
        // 1x scatter *ourselves* by reducing it to the 2x version.
        fmt("    {\n"
            "        .reg .f16x2 %packed;\n"
            "        .reg .b64 %align, %offset;\n"
            "        .reg .b32 %offset_32;\n"
            "        .reg .f16 %initial;\n"
            "        mov.b16 %initial, 0;\n"
            "        and.b64 %align, %rd3, ~0x3;\n"
            "        and.b64 %offset, %rd3, 0x2;\n"
            "        cvt.u32.s64 %offset_32, %offset;\n"
            "        shl.b32 %offset_32, %offset_32, 3;\n"
            "        mov.b32 %packed, {$v, %initial};\n"
            "        shl.b32 %packed, %packed, %offset_32;\n"
            "        red.global.$s.f16x2 [%align], %packed;\n"
            "    }\n", value, op_name);
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

    fmt_intrinsic(
        ".func (.param .u32 rv) reduce_inc_u32 (.param .u64 ptr) {\n"
        "    .reg .b64 %ptr, %ptr_shift;\n"
        "    .reg .b32 %active, %ptr_id, %active_p, %active_p_rev,\n"
        "              %leader_idx, %lt_mask, %lt_active_p, %lt_ct,\n"
        "              %active_p_ct, %prev, %leader_val, %rv;\n"
        "    .reg .pred %leader;\n\n"
        ""
        "    ld.param.u64 %ptr, [ptr];\n"
        "    activemask.b32 %active;\n"
        "    shl.b64 %ptr_shift, %ptr, 2;\n"
        "    cvt.u32.u64 %ptr_id, %ptr_shift;\n"
        "    match.any.sync.b32 %active_p, %ptr_id, %active;\n"
        "    brev.b32 %active_p_rev, %active_p;\n"
        "    clz.b32 %leader_idx, %active_p_rev;\n"
        "    mov.u32 %lt_mask, %lanemask_lt;\n"
        "    and.b32 %lt_active_p, %lt_mask, %active_p;\n"
        "    setp.eq.u32 %leader, %lt_active_p, 0;\n"
        "    @!%leader bra fetch_from_leader;\n\n"
        ""
        "    popc.b32 %active_p_ct, %active_p;\n"
        "    atom.global.add.u32 %prev, [%ptr], %active_p_ct;\n\n"
        ""
        "fetch_from_leader:\n"
        "    shfl.sync.idx.b32 %leader_val, %prev, %leader_idx, 31, %active;\n"
        "    popc.b32 %lt_ct, %lt_active_p;\n"
        "    add.u32 %rv, %lt_ct, %leader_val;\n"
        "    st.param.u32 [rv], %rv;\n"
        "    ret;\n"
        "}"
    );

    if (!is_unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    jitc_cuda_prepare_index(ptr, index, index);

    fmt("    {\n"
        "        .func (.param .u32 rv) reduce_inc_u32 (.param .u64 ptr);\n"
        "        call.uni ($v), reduce_inc_u32, (%rd3);\n"
        "    }\n", v);

    if (!is_unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);

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

