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

/// NVIDIA hardware lacks floating point min/max atomics, emulate them
bool jitc_cuda_atomic_minmax_emulated(VarType vt, ReduceOp op) {
    return (vt == VarType::Float32 || vt == VarType::Float64) &&
           (op == ReduceOp::Min || op == ReduceOp::Max);
}

/// Declare the temporaries used by ``jitc_cuda_render_atomic_minmax``
void jitc_cuda_declare_atomic_minmax(VarType vt) {
    fmt("        .reg .$s %red_val;\n"
        "        .reg .pred %red_pos, %red_neg;\n",
        vt == VarType::Float64 ? "b64" : "b32");
}

/// Reduce the float min/max atomic ``[%rd3+offset] <op>= %red_val`` to an
/// integer one, exploiting that IEEE-754 bit patterns order like signed
/// integers when non-negative, and like unsigned integers in reverse
/// otherwise. If ``leader`` is set, the reduction is predicated on ``%leader``.
void jitc_cuda_render_atomic_minmax(VarType vt, ReduceOp op, uint32_t offset,
                                    bool leader) {
    bool is64 = vt == VarType::Float64;
    const char *st  = is64 ? "s64" : "s32",
               *ut  = is64 ? "u64" : "u32",
               *bop = leader ? ".and" : "",
               *arg = leader ? ", %leader" : "";

    fmt("        setp.ge$s.$s %red_pos, %red_val, 0$s;\n"
        "        setp.lt$s.$s %red_neg, %red_val, 0$s;\n"
        "        @%red_pos red.global.$s.$s [%rd3+$u], %red_val;\n"
        "        @%red_neg red.global.$s.$s [%rd3+$u], %red_val;\n",
        bop, st, arg,
        bop, st, arg,
        op == ReduceOp::Min ? "min" : "max", st, offset,
        op == ReduceOp::Min ? "max" : "min", ut, offset);
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

/// In-register PTX operation used to combine two partial reductions
static const char *jitc_cuda_reduce_op_ftz(VarType vt, ReduceOp op) {
    bool ftz = vt == VarType::Float32 || vt == VarType::Float16;
    switch (op) {
        case ReduceOp::Add: return ftz ? "add.ftz" : "add";
        case ReduceOp::Mul: return ftz ? "mul.ftz" : jitc_is_float(vt) ? "mul" : "mul.lo";
        default: return cuda_reduce_op_name[(int) op];
    }
}

/// Declare the registers shared by the warp reduction routines below
static void jitc_cuda_declare_warp_regs(uint32_t n, const char *tp, bool is64) {
    put("        .reg .b32 %active, %index, %mask_lt, %mask_gt, %peers,\n"
        "                  %peers_lt, %peers_rev, %rank, %rank_bit, %rank_ballot;\n");
    if (is64)
        put("        .reg .b32 %q0l, %q0h, %q1l, %q1h;\n");
    fmt("        .reg .$s %q0_<$u>, %q1;\n"
        "        .reg .pred %leader, %partial, %done, %valid, %rank_even, %unused;\n",
        tp, n);
}

/// Emit a butterfly reduction of ``%q0_<n>`` across lanes at distance < ``width``.
/// Lanes whose partner lies beyond the clamp value ``clamp`` keep their value.
static void jitc_cuda_emit_butterfly(uint32_t n, const char *tp, const char *op,
                                     bool is64, uint32_t width,
                                     const char *clamp, const char *mask) {
    for (uint32_t delta = 1; delta < width; delta *= 2) {
        for (uint32_t i = 0; i < n; ++i) {
            if (!is64)
                fmt("        shfl.sync.bfly.b32 %q1|%valid, %q0_$u, $u, $s, $s;\n"
                    "        @%valid $s.$s %q0_$u, %q0_$u, %q1;\n",
                    i, delta, clamp, mask, op, tp, i, i);
            else
                fmt("        mov.b64 {%q0l, %q0h}, %q0_$u;\n"
                    "        shfl.sync.bfly.b32 %q1l|%valid, %q0l, $u, $s, $s;\n"
                    "        shfl.sync.bfly.b32 %q1h|%valid, %q0h, $u, $s, $s;\n"
                    "        mov.b64 %q1, {%q1l, %q1h};\n"
                    "        @%valid $s.$s %q0_$u, %q0_$u, %q1;\n",
                    i, delta, clamp, mask, delta, clamp, mask, op, tp, i, i);
        }
    }
}

/// Emit a reduction of ``%q0_<n>`` among the lanes in ``%peers`` (a subset
/// of ``%active``). The result ends up in the lowest peer, which is marked by
/// the ``%leader`` predicate. Control flow continues at ``reduce_done``.
static void jitc_cuda_emit_peer_reduce(uint32_t n, const char *tp,
                                       const char *op, bool is64) {
    put("        mov.u32 %mask_lt, %lanemask_lt;\n"
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
                i, op, tp, i, i);
        else
            fmt("        mov.b64 {%q0l, %q0h}, %q0_$u;\n"
                "        shfl.sync.idx.b32 %q1l|%unused, %q0l, %index, 31, %active;\n"
                "        shfl.sync.idx.b32 %q1h|%unused, %q0h, %index, 31, %active;\n"
                "        mov.b64 %q1, {%q1l, %q1h};\n"
                "        @%valid $s.$s %q0_$u, %q0_$u, %q1;\n",
                i, op, tp, i, i);
    }

    put("        and.b32 %rank_bit, %rank, 1;\n"
        "        setp.eq.u32 %rank_even, %rank_bit, 0;\n"
        "        vote.sync.ballot.b32 %rank_ballot, %rank_even, %active;\n"
        "        and.b32 %peers, %peers, %rank_ballot;\n"
        "        shr.u32 %rank, %rank, 1;\n"
        "        bra reduce_partial_loop;\n\n");
}

/// Let the leader lane atomically combine ``%q0_<n>`` into ``[%rd3]``.
/// Uses packet atomics if ``use_packet_atomics`` is set.
static void jitc_cuda_emit_leader_atomics(uint32_t n, VarType vt, const char *tp,
                                          ReduceOp op, bool use_packet_atomics) {
    // Emulated float min/max atomics have no vectorized counterpart
    bool emulated = jitc_cuda_atomic_minmax_emulated(vt, op);
    const char *op_name = cuda_reduce_op_name[(int) op];
    uint32_t tsize = type_size[(int) vt];

    // Generate scalar atomics or packet atomics (cap to 128bit/thread)
    uint32_t per_atomic = 1;
    if (use_packet_atomics && !emulated) {
        uint32_t bytes_per_atomic = 16;
        while ((n * tsize) % bytes_per_atomic != 0)
            bytes_per_atomic /= 2;
        per_atomic = bytes_per_atomic / tsize;
    }

    for (uint32_t base = 0; base < n; base += per_atomic) {
        if (emulated) {
            fmt("        mov.$s %red_val, %q0_$u;\n", type_name_ptx_bin[(int) vt], base);
            jitc_cuda_render_atomic_minmax(vt, op, base * tsize, true);
        } else if (per_atomic == 1) {
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
}

/// Emit a segmented butterfly warp-reduction that separately reduces ``n``
/// variables of type ``vt`` within the warp, then atomically scatters the
/// per-group result to memory. The packet base address is assumed to be in
/// ``%rd3``.  Uses packet atomics if ``use_packet_atomics`` is set.
void jitc_cuda_render_warp_reduce(uint32_t n, const uint32_t *values, VarType vt,
                                  ReduceOp op, bool use_packet_atomics) {
    const char *tp     = jitc_cuda_reduce_tp(vt, op),
               *op_ftz = jitc_cuda_reduce_op_ftz(vt, op);
    uint32_t tsize = type_size[(int) vt];
    bool is64 = tsize == 8;

    put("    {\n");
    jitc_cuda_declare_warp_regs(n, tp, is64);
    if (jitc_cuda_atomic_minmax_emulated(vt, op))
        jitc_cuda_declare_atomic_minmax(vt);
    put("\n");

    for (uint32_t i = 0; i < n; ++i)
        fmt("        mov.$s %q0_$u, $v;\n", tp, i, jitc_var(values[i]));

    jitc_cuda_emit_warp_match(log2i_ceil(tsize));

    put("        setp.ne.s32 %partial, %peers, -1;\n"
        "        @%partial bra reduce_partial;\n\n");

    // If the warp is fully coherent, do a normal butterfly reduction and scatter
    put("        mov.b32 %index, %laneid;\n"
        "        setp.eq.u32 %leader, %index, 0;\n");
    jitc_cuda_emit_butterfly(n, tp, op_ftz, is64, 32, "31", "%active");

    // Otherwise, do a reduction within segments
    put("        bra reduce_done;\n\n"
        "    reduce_partial:\n");
    jitc_cuda_emit_peer_reduce(n, tp, op_ftz, is64);
    put("    reduce_done:\n");
    jitc_cuda_emit_leader_atomics(n, vt, tp, op, use_packet_atomics);
    put("    }\n");
}

void jitc_cuda_render_simd_reduce(const Variable *v, const Variable *ptr,
                                  const Variable *value) {
    ReduceOp op  = (ReduceOp) (uint32_t) v->literal;
    VarType vt   = (VarType) value->type;
    uint32_t eff = (uint32_t) (v->literal >> 32),
             tsize = type_size[(int) vt];
    bool is_bool = vt == VarType::Bool,
         is64    = tsize == 8;

    // Masks are shuffled and combined as 32-bit values, but stored as bytes
    const char *tp     = is_bool ? "b32" : jitc_cuda_reduce_tp(vt, op),
               *st     = is_bool ? "u8" : tp,
               *op_ftz = jitc_cuda_reduce_op_ftz(vt, op);

    fmt("    {\n"
        "        .reg .b32 %index, %active, %live, %lane;\n"
        "        .reg .$s %q0_0, %q1;\n"
        "        .reg .pred %leader, %valid, %fail;\n", tp);
    if (is64)
        put("        .reg .b32 %q0l, %q0h, %q1l, %q1h;\n");

    // Load the value, compute the address of the block result, and mark the
    // first lane of each block as its leader
    if (is_bool)
        fmt("        selp.b32 %q0_0, 1, 0, $v;\n", value);
    else
        fmt("        mov.$s %q0_0, $v;\n", tp, value);
    fmt("        shr.u32 %index, %r0, $u;\n"
        "        mad.wide.u32 %rd3, %index, $u, $v;\n"
        "        and.b32 %index, %r0, $u;\n"
        "        setp.eq.u32 %leader, %index, 0;\n",
        log2i_ceil(eff), tsize, ptr, eff - 1);

    // Lanes hold consecutive entries, and lanes past the end of the array have
    // exited. Clamp the butterfly to the remaining lanes. On OptiX, the
    // shuffle mask and lane count come from 'activemask'.
    const char *mask;
    if (!uses_optix) {
        put("        and.b32 %live, %r0, 0xffffffe0;\n"
            "        sub.u32 %live, %r2, %live;\n"
            "        min.u32 %live, %live, 32;\n");
        mask = "0xffffffff";
    } else {
        // OptiX guarantees neither that launch indices map to lanes in order
        // nor that the warp is converged here. Trap if a lane is out of order
        // or if lane 0 is missing, which happens in some group of a diverged warp.
        put("        activemask.b32 %active;\n"
            "        popc.b32 %live, %active;\n"
            "        mov.u32 %lane, %laneid;\n"
            "        and.b32 %index, %r0, 31;\n"
            "        setp.ne.u32 %fail, %index, %lane;\n"
            "        and.b32 %index, %active, 1;\n"
            "        setp.eq.or.u32 %fail, %index, 0, %fail;\n"
            "        @%fail trap;\n");
        mask = "%active";
    }
    put("        sub.u32 %live, %live, 1;\n");
    jitc_cuda_emit_butterfly(1, tp, op_ftz, is64, eff, "%live", mask);
    fmt("        @%leader st.global.$s [%rd3], %q0_0;\n"
        "    }\n", st);
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

    } else if (jitc_cuda_atomic_minmax_emulated(vt, op)) {
        put("    {\n");
        jitc_cuda_declare_atomic_minmax(vt);
        fmt("        mov.$b %red_val, $v;\n", value, value);
        jitc_cuda_render_atomic_minmax(vt, op, 0, false);
        put("    }\n");
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
