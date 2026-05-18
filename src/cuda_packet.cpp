/*
    src/cuda_packet.cpp -- Specialized memory operations that read or write
    multiple adjacent values at once.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"
#include "cuda_eval.h"
#include "var.h"
#include "op.h"
#include "log.h"

static const char *reduce_op_name[(int) ReduceOp::Count] = {
    "", "add", "mul", "min", "max", "and", "or"
};

/// Packet analogue of \ref jitc_cuda_render_scatter_reduce_bfly_32:
/// performs one butterfly reduction shared across all N packet channels,
/// then the leader emits either `red.global.vM` vector atomics (when
/// \c use_vector_atomic is set) or N scalar atomics otherwise.
static void jitc_cuda_render_scatter_reduce_packet_bfly_32(const char *tp,
                                                           const char *op,
                                                           const char *op_ftz,
                                                           uint32_t shiftamt,
                                                           uint32_t n,
                                                           uint32_t tsize,
                                                           bool use_vector_atomic) {
    size_t tmpoff = buffer.size();

    const char *suffix = use_vector_atomic ? "_vatom" : "";
    fmt(".func reduce_$s_$s_v$u$s(.param .u64 ptr",
        op, tp, n, suffix);
    for (uint32_t i = 0; i < n; ++i)
        fmt(", .param .$s v$u", tp, i);
    put(") {\n");

    put("    .reg .b32 %active, %index, %mask_lt, %mask_gt, %peers,\n"
        "              %peers_lt, %peers_rev, %rank, %rank_bit, %rank_ballot;\n"
        "    .reg .b64 %ptr;\n");
    fmt("    .reg .$s %q0_<$u>, %q1;\n", tp, n);
    put("    .reg .pred %leader, %partial, %done, %valid, %rank_even, %unused;\n\n");

    for (uint32_t i = 0; i < n; ++i)
        fmt("    ld.param.$s %q0_$u, [v$u];\n", tp, i, i);
    put("    ld.param.b64 %ptr, [ptr];\n");

    put("    activemask.b32 %active;\n");
    fmt("    {\n"
        "        .reg .b64 %ptr_shift;\n"
        "        shr.b64 %ptr_shift, %ptr, $u;\n"
        "        cvt.u32.u64 %index, %ptr_shift;\n"
        "    }\n",
        shiftamt);
    put("    match.any.sync.b32 %peers, %index, %active;\n"
        "    setp.ne.s32 %partial, %peers, -1;\n"
        "    @%partial bra.uni reduce_partial;\n\n");

    put("reduce_full:\n"
        "    mov.b32 %index, %laneid;\n"
        "    setp.eq.u32 %leader, %index, 0;\n");
    for (uint32_t delta : {1u, 2u, 4u, 8u, 16u}) {
        for (uint32_t i = 0; i < n; ++i) {
            fmt("    shfl.sync.bfly.b32 %q1|%unused, %q0_$u, $u, 31, %active;\n",
                i, delta);
            fmt("    $s.$s %q0_$u, %q0_$u, %q1;\n", op_ftz, tp, i, i);
        }
    }
    put("    bra.uni do_write;\n\n");

    put("reduce_partial:\n"
        "    mov.u32 %mask_lt, %lanemask_lt;\n"
        "    mov.u32 %mask_gt, %lanemask_gt;\n"
        "    and.b32 %peers_lt, %peers, %mask_lt;\n"
        "    popc.b32 %rank, %peers_lt;\n"
        "    setp.eq.u32 %leader, %rank, 0;\n"
        "    and.b32 %peers, %peers, %mask_gt;\n\n");

    put("reduce_partial_loop:\n"
        "    setp.eq.u32 %done, %peers, 0;\n"
        "    vote.sync.all.pred %done, %done, %active;\n"
        "    @%done bra.uni do_write;\n\n");

    put("    brev.b32 %peers_rev, %peers;\n"
        "    bfind.shiftamt.u32 %index, %peers_rev;\n"
        "    setp.ne.s32 %valid, %index, -1;\n");
    for (uint32_t i = 0; i < n; ++i) {
        fmt("    shfl.sync.idx.b32 %q1|%unused, %q0_$u, %index, 31, %active;\n",
            i);
        fmt("    @%valid $s.$s %q0_$u, %q0_$u, %q1;\n",
            op_ftz, tp, i, i);
    }
    put("    and.b32 %rank_bit, %rank, 1;\n"
        "    setp.eq.u32 %rank_even, %rank_bit, 0;\n"
        "    vote.sync.ballot.b32 %rank_ballot, %rank_even, %active;\n"
        "    and.b32 %peers, %peers, %rank_ballot;\n"
        "    shr.u32 %rank, %rank, 1;\n"
        "    bra reduce_partial_loop;\n\n");

    put("do_write:\n");
    if (use_vector_atomic) {
        // `red.global.vN.<tp>` caps at 128 bits per atomic.
        uint32_t bytes_per_atomic = 16;
        while ((n * tsize) % bytes_per_atomic != 0)
            bytes_per_atomic /= 2;
        uint32_t per_atomic = bytes_per_atomic / tsize;

        for (uint32_t base = 0; base < n; base += per_atomic) {
            if (per_atomic == 1) {
                fmt("    @%leader red.global.$s.$s [%ptr+$u], %q0_$u;\n",
                    op, tp, base * tsize, base);
            } else {
                fmt("    @%leader red.global.v$u.$s.$s [%ptr+$u], {",
                    per_atomic, tp, op, base * tsize);
                for (uint32_t i = 0; i < per_atomic; ++i)
                    fmt("%q0_$u$s", base + i, i + 1 < per_atomic ? ", " : "");
                put("};\n");
            }
        }
    } else {
        for (uint32_t i = 0; i < n; ++i)
            fmt("    @%leader red.global.$s.$s [%ptr+$u], %q0_$u;\n",
                op, tp, i * tsize, i);
    }
    put("    ret;\n"
        "}");

    jitc_register_global(buffer.get() + tmpoff);
    buffer.rewind_to(tmpoff);
}

// Identical to the above, but perform each shuffle twice to move around 64
// bit-sized values. The leader always emits N scalar atomics (PTX has no
// 64-bit vector atomic).
static void jitc_cuda_render_scatter_reduce_packet_bfly_64(const char *tp,
                                                           const char *op,
                                                           const char *op_ftz,
                                                           uint32_t shiftamt,
                                                           uint32_t n,
                                                           uint32_t tsize) {
    size_t tmpoff = buffer.size();

    fmt(".func reduce_$s_$s_v$u(.param .u64 ptr", op, tp, n);
    for (uint32_t i = 0; i < n; ++i)
        fmt(", .param .$s v$u", tp, i);
    put(") {\n");

    put("    .reg .b32 %active, %index, %mask_lt, %mask_gt, %peers,\n"
        "              %peers_lt, %peers_rev, %rank, %rank_bit, %rank_ballot;\n"
        "    .reg .b64 %ptr;\n"
        "    .reg .b32 %q0l, %q0h, %q1l, %q1h;\n");
    fmt("    .reg .$s %q0_<$u>, %q1;\n", tp, n);
    put("    .reg .pred %leader, %partial, %done, %valid, %rank_even, %unused;\n\n");

    for (uint32_t i = 0; i < n; ++i)
        fmt("    ld.param.$s %q0_$u, [v$u];\n", tp, i, i);
    put("    ld.param.b64 %ptr, [ptr];\n");

    put("    activemask.b32 %active;\n");
    fmt("    {\n"
        "        .reg .b64 %ptr_shift;\n"
        "        shr.b64 %ptr_shift, %ptr, $u;\n"
        "        cvt.u32.u64 %index, %ptr_shift;\n"
        "    }\n",
        shiftamt);
    put("    match.any.sync.b32 %peers, %index, %active;\n"
        "    setp.ne.s32 %partial, %peers, -1;\n"
        "    @%partial bra.uni reduce_partial;\n\n");

    put("reduce_full:\n"
        "    mov.b32 %index, %laneid;\n"
        "    setp.eq.u32 %leader, %index, 0;\n");
    for (uint32_t delta : {1u, 2u, 4u, 8u, 16u}) {
        for (uint32_t i = 0; i < n; ++i) {
            fmt("    mov.b64 {%q0l, %q0h}, %q0_$u;\n"
                "    shfl.sync.bfly.b32 %q1l|%unused, %q0l, $u, 31, %active;\n"
                "    shfl.sync.bfly.b32 %q1h|%unused, %q0h, $u, 31, %active;\n"
                "    mov.b64 %q1, {%q1l, %q1h};\n"
                "    $s.$s %q0_$u, %q0_$u, %q1;\n",
                i, delta, delta, op_ftz, tp, i, i);
        }
    }
    put("    bra.uni do_write;\n\n");

    put("reduce_partial:\n"
        "    mov.u32 %mask_lt, %lanemask_lt;\n"
        "    mov.u32 %mask_gt, %lanemask_gt;\n"
        "    and.b32 %peers_lt, %peers, %mask_lt;\n"
        "    popc.b32 %rank, %peers_lt;\n"
        "    setp.eq.u32 %leader, %rank, 0;\n"
        "    and.b32 %peers, %peers, %mask_gt;\n\n");

    put("reduce_partial_loop:\n"
        "    setp.eq.u32 %done, %peers, 0;\n"
        "    vote.sync.all.pred %done, %done, %active;\n"
        "    @%done bra.uni do_write;\n\n");

    put("    brev.b32 %peers_rev, %peers;\n"
        "    bfind.shiftamt.u32 %index, %peers_rev;\n"
        "    setp.ne.s32 %valid, %index, -1;\n");
    for (uint32_t i = 0; i < n; ++i)
        fmt("    mov.b64 {%q0l, %q0h}, %q0_$u;\n"
            "    shfl.sync.idx.b32 %q1l|%unused, %q0l, %index, 31, %active;\n"
            "    shfl.sync.idx.b32 %q1h|%unused, %q0h, %index, 31, %active;\n"
            "    mov.b64 %q1, {%q1l, %q1h};\n"
            "    @%valid $s.$s %q0_$u, %q0_$u, %q1;\n",
            i, op_ftz, tp, i, i);
    put("    and.b32 %rank_bit, %rank, 1;\n"
        "    setp.eq.u32 %rank_even, %rank_bit, 0;\n"
        "    vote.sync.ballot.b32 %rank_ballot, %rank_even, %active;\n"
        "    and.b32 %peers, %peers, %rank_ballot;\n"
        "    shr.u32 %rank, %rank, 1;\n"
        "    bra reduce_partial_loop;\n\n");

    put("do_write:\n");
    for (uint32_t i = 0; i < n; ++i)
        fmt("    @%leader red.global.$s.$s [%ptr+$u], %q0_$u;\n",
            op, tp, i * tsize, i);
    put("    ret;\n"
        "}");

    jitc_register_global(buffer.get() + tmpoff);
    buffer.rewind_to(tmpoff);
}

void jitc_cuda_render_gather_packet(const Variable *v, const Variable *ptr,
                                    const Variable *index, const Variable *mask) {
    bool is_masked = !mask->is_literal() || mask->literal != 1,
         is_bool   = (VarType) v->type == VarType::Bool;

    uint32_t count = (uint32_t) v->literal,
             tsize = type_size[v->type],
             total_bytes = count * tsize;

    // Get compute capability for current device
    const ThreadState *ts = thread_state_cuda;
    uint32_t compute_capability = state.devices[ts->device].compute_capability;

    // 256-bit operations require CC 12.0+, and for OptiX: CUDA driver 13.2+
    bool supports_256bit = compute_capability >= 120 &&
                          (!uses_optix ||
                           (jitc_cuda_version_major > 13 ||
                            (jitc_cuda_version_major == 13 && jitc_cuda_version_minor >= 2)));

    fmt("    mad.wide.$t %rd3, $v, $u, $v;\n"
        "    .reg.$t $v_out_<$u>;\n",
        index, index, total_bytes, ptr,
        v, v, count);

    const char *suffix, *out_name = "out";

    if (is_bool) {
        fmt("    .reg.b8 $v_tmp2_<$u>;\n", v, count);
        out_name = "tmp2";
    }

    // Number of output/temporary output registers and their size
    uint32_t dst_count = count,
             out_bits = tsize * 8,
             dst_bits = out_bits;

    if (tsize >= 4) {
        suffix = "out";
    } else {
        // Source size is sub-word, gather larger values and decode
        if (total_bytes % 4 == 0) {
            dst_bits = 32;
            dst_count = total_bytes / 4;
        } else {
            dst_bits = 16;
            dst_count = total_bytes / 2;
        }
        fmt("    .reg.b$u $v_tmp_<$u>;\n",
            dst_bits, v, dst_count);
        suffix = "tmp";
    }

    if (is_masked) {
        for (uint32_t i = 0; i < dst_count; ++i)
            fmt("    mov.b$u $v_$s_$u, 0;\n", dst_bits, v, suffix, i);
    }

    // Try to load 256b/iteration if supported, otherwise 128b/iteration
    // Potentially reduce if the total size of the load isn't divisible
    uint32_t bytes_per_it = supports_256bit ? 32 : 16;
    while ((total_bytes & (bytes_per_it - 1)) != 0)
        bytes_per_it /= 2;
    uint32_t regs_per_it = (bytes_per_it * 8) / dst_bits;

    // Actually load the packets into the `_tmp` registers (or directly into the `_out` registers
    // if the output type size is large enough).
    for (uint32_t byte_offset = 0; byte_offset < total_bytes; byte_offset += bytes_per_it) {
        uint32_t reg_offset = (byte_offset * 8) / dst_bits;
        if (is_masked)
            fmt("    @$v ", mask);
        else
            put("    ");

        switch (regs_per_it) {
            case 1:
                fmt("ld.global.nc.b$u $v_$s_$u, [%rd3+$u];\n",
                    dst_bits, v, suffix, reg_offset, byte_offset);
                break;

            case 2:
                fmt("ld.global.nc.v2.b$u {$v_$s_$u, $v_$s_$u}, [%rd3+$u];\n",
                    dst_bits, v, suffix, reg_offset, v, suffix, reg_offset + 1, byte_offset);
                break;

            case 4:
                fmt("ld.global.nc.v4.b$u {$v_$s_$u, $v_$s_$u, $v_$s_$u, $v_$s_$u}, [%rd3+$u];\n",
                    dst_bits,
                    v, suffix, reg_offset,
                    v, suffix, reg_offset+1,
                    v, suffix, reg_offset+2,
                    v, suffix, reg_offset+3,
                    byte_offset);
                break;

            case 8:
                fmt("ld.global.nc.v8.b$u {$v_$s_$u, $v_$s_$u, $v_$s_$u, $v_$s_$u, "
                    "$v_$s_$u, $v_$s_$u, $v_$s_$u, $v_$s_$u}, [%rd3+$u];\n",
                    dst_bits,
                    v, suffix, reg_offset,
                    v, suffix, reg_offset+1,
                    v, suffix, reg_offset+2,
                    v, suffix, reg_offset+3,
                    v, suffix, reg_offset+4,
                    v, suffix, reg_offset+5,
                    v, suffix, reg_offset+6,
                    v, suffix, reg_offset+7,
                    byte_offset);
                break;

            default:
                jitc_fail("jitc_cuda_render_gather_packet(): internal failure!");
        }
    }

    // Unpack the values into the `_out` registers, if needed.
    if (tsize == 1) {
        uint32_t outputs_per_tmp = dst_bits / out_bits;

        for (uint32_t i = 0; i < dst_count; ++i) {
            if (outputs_per_tmp == 1) {
                fmt("    mov.b$u $v_$s_$u, $v_tmp_$u;\n",
                    dst_bits, v, out_name, i, v, i);
            } else if (outputs_per_tmp == 2) {
                fmt("    mov.b$u {$v_$s_$u, $v_$s_$u}, $v_tmp_$u;\n",
                    dst_bits,
                    v, out_name, 2 * i,
                    v, out_name, 2 * i + 1,
                    v, i);
            } else if (outputs_per_tmp == 4) {
                fmt("    mov.b$u {$v_$s_$u, $v_$s_$u, $v_$s_$u, $v_$s_$u}, $v_tmp_$u;\n",
                    dst_bits,
                    v, out_name, 4 * i,
                    v, out_name, 4 * i + 1,
                    v, out_name, 4 * i + 2,
                    v, out_name, 4 * i + 3,
                    v, i);
            } else {
                jitc_fail("jitc_cuda_render_gather_packet: internal error!");
            }
        }
    } else if (tsize == 2) {
        for (uint32_t i = 0; i < count/2; ++i) {
            fmt("    mov.b$u {$v_$s_$u, $v_$s_$u}, $v_tmp_$u;\n",
                dst_bits, v, out_name, 2*i, v, out_name, 2*i+1, v, i);
        }
    }

    if (is_bool) {
        for (uint32_t i = 0; i < count; ++i) {
            fmt("    cvt.u16.u8 %w3, $v_tmp2_$u;\n"
                "    setp.ne.u16 $v_out_$u, %w3, 0;\n",
                v, i,
                v, i);
        }
    }
}

/**
 * Render the code required to scatter reduce a packet of variables.
 */
void jitc_cuda_render_scatter_reduce_packet(const Variable *v,
                                            const Variable *ptr,
                                            const Variable *index,
                                            const Variable *mask) {
    bool is_masked         = !mask->is_literal() || mask->literal != 1;
    PacketScatterData *psd = (PacketScatterData *) v->data;
    const std::vector<uint32_t> &values = psd->values;
    const Variable *v0                  = jitc_var(values[0]);
    const ReduceOp op = psd->op;
    const char *op_name = reduce_op_name[(uint32_t) psd->op];

    const ThreadState *ts = thread_state_cuda;

    uint32_t count = (uint32_t) values.size(),
             tsize = type_size[v0->type];

    if (count % 2 != 0)
        jitc_fail("jitc_cuda_render_scatter_reduce_packet(): Number of "
                  "elements not supported by reduction.");

    // Vector reduction instructions require CC 9.0+ and for OptiX: CUDA driver 13.2+
    bool supports_vector_reduction = ts->compute_capability >= 90 &&
                                    (!uses_optix ||
                                     (jitc_cuda_version_major > 13 ||
                                      (jitc_cuda_version_major == 13 && jitc_cuda_version_minor >= 2)));

    // Actually OptiX packed half reduction for vector width > 2 is still broken atm :-(
    if (uses_optix && v0->type == (uint32_t) VarType::Float16)
        supports_vector_reduction = false;

    // `match.any.sync` (used by the butterfly) requires PTX 6.3+ and sm_70+.
    bool reduce_bfly_32 = ts->ptx_version >= 63 && ts->compute_capability >= 70 &&
                          (v0->type == (uint32_t) VarType::UInt32 ||
                           v0->type == (uint32_t) VarType::Int32 ||
                           v0->type == (uint32_t) VarType::Float32),
         reduce_bfly_64 = ts->ptx_version >= 63 && ts->compute_capability >= 70 &&
                          (v0->type == (uint32_t) VarType::UInt64 ||
                           v0->type == (uint32_t) VarType::Int64 ||
                           v0->type == (uint32_t) VarType::Float64);

    if (psd->mode == ReduceMode::Local && (reduce_bfly_32 || reduce_bfly_64)) {
        VarType vt = (VarType) v0->type;
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

        const char *op_ftz = op_name;
        if (op == ReduceOp::Add &&
            (vt == VarType::Float32 || vt == VarType::Float16))
            op_ftz = "add.ftz";

        bool use_vector_atomic = reduce_bfly_32 &&
                                 vt == VarType::Float32 &&
                                 supports_vector_reduction;
        const char *suffix = use_vector_atomic ? "_vatom" : "";

        uint32_t shiftamt = log2i_ceil(type_size[(int) vt]);

        if (reduce_bfly_32)
            jitc_cuda_render_scatter_reduce_packet_bfly_32(
                tp, op_name, op_ftz, shiftamt, count, tsize, use_vector_atomic);
        else
            jitc_cuda_render_scatter_reduce_packet_bfly_64(
                tp, op_name, op_ftz, shiftamt, count, tsize);

        fmt("    mad.wide.$t %rd3, $v, $u, $v;\n", index, index, tsize, ptr);

        fmt("    {\n"
            "        .func reduce_$s_$s_v$u$s(.param .u64 ptr",
            op_name, tp, count, suffix);
        for (uint32_t i = 0; i < count; ++i)
            fmt(", .param .$s v$u", tp, i);
        put(");\n");

        if (is_masked)
            fmt("        @!$v bra l_packet_bfly_done_$u;\n", mask, v->reg_index);

        fmt("        call.uni reduce_$s_$s_v$u$s, (%rd3",
            op_name, tp, count, suffix);
        for (uint32_t i = 0; i < count; ++i)
            fmt(", $v", jitc_var(values[i]));
        put(");\n");

        if (is_masked)
            fmt("    l_packet_bfly_done_$u:\n", v->reg_index);

        put("    }\n");
        return;
    }

    if (supports_vector_reduction &&
        (v0->type == (uint32_t) VarType::Float16 ||
         v0->type == (uint32_t) VarType::Float32)) {
        // Use the new `red.global.vX` instructions. This enables both min & max
        // as well as packet reductions with larger packet sizes per iteration
        // and `f32` types.

        // Find the largest supported packet size dividing the number of
        // variables.
        // Note: PTX limitations:
        // - red.global.vX.f32: supports only .v2 or .v4 (max 128 bits)
        // - red.global.vX.f16: supports up to .v8 (max 128 bits, 8x16bit)
        // - No .v16 instruction exists in PTX
        uint32_t max_bytes = 16;  // Default: max 128 bits
        // For f16, we can use v8 (8 elements * 2 bytes = 16 bytes)
        // but NOT v16 (which would be 32 bytes)
        uint32_t vars_per_it = max_bytes / tsize;
        while ((count & (vars_per_it - 1)) != 0)
            vars_per_it /= 2;

        fmt("    mad.wide.$t %rd3, $v, $u, $v;\n", index, index, tsize,
            ptr);

        const char *qualifier =
            v0->type == (uint32_t) VarType::Float16 ? ".noftz" : "";

        VarType vt = (VarType) v0->type;
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

        for (uint32_t i = 0; i < count; i += vars_per_it) {
            uint32_t byte_offset = i * tsize;

            if (is_masked)
                fmt("    @$v ", mask);
            else
                put("    ");
            fmt("red.global.v$u.$s.$s$s [%rd3+$u], {",
                vars_per_it, tp, op_name, qualifier, byte_offset);

            for (uint32_t j = 0; j < vars_per_it; j++) {
                fmt("$v, ", jitc_var(values[i + j]));
            }
            buffer.delete_trailing_commas();
            put("};\n");
        }
    } else if (v0->type == (uint32_t) VarType::Float16 && op == ReduceOp::Add) {
        // The more broadly supported `.f16x2` instruction is, only available
        // for addition and f16 types.

        fmt("    .reg.f16x2 $v_tmp;\n"
            "    mad.wide.$t %rd3, $v, $u, $v;\n",
            v,
            index, index, tsize, ptr);

        for (uint32_t i = 0; i < count; i += 2) {
            uint32_t byte_offset = i * tsize;

            fmt("    mov.b32 $v_tmp, {$v, $v};\n", v, jitc_var(values[i]),
                jitc_var(values[i + 1]));

            if (is_masked)
                fmt("    @$v ", mask);
            else
                put("        ");
            fmt("red.global.add.noftz.f16x2 [%rd3+$u], $v_tmp;\n",
                byte_offset, v);
        }
    } else {
        const char *qualifier =
            v0->type == (uint32_t) VarType::Float16 ? ".noftz" : "";

        VarType vt = (VarType) v0->type;
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

        fmt("    mad.wide.$t %rd3, $v, $u, $v;\n",
            index, index, tsize, ptr);
        for (uint32_t i = 0; i < count; i++){
            if (is_masked)
                fmt("    @$v ", mask);
            else
                put("        ");
            fmt("red.global.$s.$s$s [%rd3+$u], $v;\n",
                tp, op_name, qualifier, i * tsize, jitc_var(values[i]));
        }
    }
}

void jitc_cuda_render_scatter_packet(const Variable *v, const Variable *ptr,
                                     const Variable *index,
                                     const Variable *mask) {
    bool is_masked = !mask->is_literal() || mask->literal != 1;
    PacketScatterData *psd = (PacketScatterData *) v->data;
    const std::vector<uint32_t> &values = psd->values;
    const Variable *v0 = jitc_var(values[0]);

    // Handle non-Identitiy reduction case
    if (psd->op != ReduceOp::Identity) {
        jitc_cuda_render_scatter_reduce_packet(v, ptr, index, mask);
        return;
    }

    // Get compute capability for current device
    const ThreadState *ts = thread_state_cuda;
    uint32_t compute_capability = state.devices[ts->device].compute_capability;

    // 256-bit operations require CC 12.0+, and for OptiX: CUDA driver 13.2+
    bool supports_256bit = compute_capability >= 120 &&
                          (!uses_optix ||
                           (jitc_cuda_version_major > 13 ||
                            (jitc_cuda_version_major == 13 && jitc_cuda_version_minor >= 2)));

    uint32_t count = (uint32_t) values.size(),
             tsize = type_size[v0->type],
             total_bytes = count * tsize;

    // Number of output/temporary output registers and their size
    uint32_t var_count = count,
             var_bits = tsize*8;

    if (tsize < 4) {
        // Target size is sub-word, merge into 32 bit registers first
        if (total_bytes % 4 == 0) {
            var_bits = 32;
            var_count = total_bytes / 4;
        } else {
            var_bits = 16;
            var_count = total_bytes / 2;
        }
    }

    fmt("    mad.wide.$t %rd3, $v, $u, $v;\n"
        "    .reg.b$u $v_<$u>;\n",
        index, index, tsize, ptr,
        var_bits, v, var_count);

    // Try to store 256b/iteration if supported, otherwise 128b/iteration
    uint32_t bytes_per_it = supports_256bit ? 32 : 16;

    // Potentially reduce if the total size of the store isn't divisible
    while ((total_bytes & (bytes_per_it - 1)) != 0)
        bytes_per_it /= 2;

    uint32_t var_ratio = count / var_count;

    for (uint32_t i = 0; i < var_count; ++i) {
        if (v0->type == (uint32_t) VarType::Bool) {
            fmt("    mov.b$u $v_$u, 0;\n", var_bits, v, i);
            for (uint32_t j = 0; j < var_ratio; ++j)
                fmt("    @$v or.b$u $v_$u, $v_$u, $u;\n",
                    jitc_var(values[var_ratio*i+j]), var_bits, v, i, v, i, 1 << (8*j));
        } else if (count == var_count) {
            fmt("    mov.b$u $v_$u, $v;\n",
                var_bits, v, i, jitc_var(values[i]));
        } else if (count == var_count * 2) {
            fmt("    mov.b$u $v_$u, {$v, $v};\n",
                var_bits, v, i, jitc_var(values[2*i]), jitc_var(values[2*i+1]));
        } else if (count == var_count * 4) {
            fmt("    mov.b$u $v_$u, {$v, $v, $v, $v};\n",
                var_bits, v, i,
                jitc_var(values[4*i]),
                jitc_var(values[4*i+1]),
                jitc_var(values[4*i+2]),
                jitc_var(values[4*i+3]));
        } else {
            jitc_fail("jitc_cuda_render_scatter_packet(): internal failure! (1)");
        }
    }

    uint32_t regs_per_it = (bytes_per_it * 8) / var_bits;
    for (uint32_t byte_offset = 0; byte_offset < total_bytes; byte_offset += bytes_per_it) {
        uint32_t reg_offset = (byte_offset * 8) / var_bits;
        if (is_masked)
            fmt("    @$v ", mask);
        else
            put("    ");

        switch (regs_per_it) {
            case 1:
                fmt("st.global.b$u [%rd3+$u], $v_$u;\n",
                    var_bits, byte_offset, v, reg_offset);
                break;

            case 2:
                fmt("st.global.v2.b$u [%rd3+$u], {$v_$u, $v_$u};\n",
                    var_bits, byte_offset, v, reg_offset, v, reg_offset + 1);
                break;

            case 4:
                fmt("st.global.v4.b$u [%rd3+$u], {$v_$u, $v_$u, $v_$u, $v_$u};\n",
                    var_bits, byte_offset,
                    v, reg_offset,   v, reg_offset+1,
                    v, reg_offset+2, v, reg_offset+3);
                break;

            case 8:
                fmt("st.global.v8.b$u [%rd3+$u], {$v_$u, $v_$u, $v_$u, $v_$u, "
                    "$v_$u, $v_$u, $v_$u, $v_$u};\n",
                    var_bits, byte_offset,
                    v, reg_offset,   v, reg_offset+1,
                    v, reg_offset+2, v, reg_offset+3,
                    v, reg_offset+4, v, reg_offset+5,
                    v, reg_offset+6, v, reg_offset+7);
                break;

            default:
                jitc_fail("jitc_cuda_render_scatter_packet(): internal failure! (2)");
        }
    }
}
