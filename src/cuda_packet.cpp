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

void jitc_cuda_render_gather_packet(const Variable *v, const Variable *ptr,
                                    const Variable *index, const Variable *mask) {
    bool is_masked = !mask->is_literal() || mask->literal != 1;

    uint32_t count = (uint32_t) v->literal,
             tsize = type_size[v->type],
             total_bytes = count * tsize;

    const char *suffix;

    fmt("    mad.wide.$t %rd3, $v, $u, $v;\n"
        "    .reg.$t $v_out_<$u>;\n",
        index, index, total_bytes, ptr,
        v, v, count);

    // Number of output/temporary output registers and their size
    uint32_t dst_count = count,
             dst_bits = tsize*8;

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

    // Try to load 128b/iteration
    uint32_t bytes_per_it = 16;

    // Potentially reduce if the total size of the load isn't divisible
    while ((total_bytes & (bytes_per_it - 1)) != 0)
        bytes_per_it /= 2;

    uint32_t regs_per_it = (bytes_per_it * 8) / dst_bits;

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

            default:
                jitc_fail("jitc_cuda_render_gather_packet(): internal failure!");
        }
    }

    if (tsize == 1) {
        if (dst_bits == 16) {
            for (uint32_t i = 0; i < count; ++i) {
                fmt("    and.b16 %w$u, $v_tmp_$u, $u;\n"
                    "    setp.ne.b16 $v_out_$u, %w$u, 0;\n",
                    v->reg_index, v, i/2, 0xFF << (8 * (i%2)),
                    v, i, v->reg_index);
            }
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                fmt("    and.b32 %r$u, $v_tmp_$u, $u;\n"
                    "    setp.ne.b32 $v_out_$u, %r$u, 0;\n",
                    v->reg_index, v, i/4, 0xFF << (8 * (i%4)),
                    v, i, v->reg_index);
            }
        }
    } else if (tsize == 2) {
        for (uint32_t i = 0; i < count/2; ++i) {
            fmt("    mov.b$u {$v_out_$u, $v_out_$u}, $v_tmp_$u;\n",
                dst_bits, v, 2*i, v, 2*i+1, v, i);
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

    if (ts->compute_capability >= 90 && !uses_optix &&
        (v0->type == (uint32_t) VarType::Float16 ||
         v0->type == (uint32_t) VarType::Float32)) {
        // Use the new `red.global.vX` instructions. This enables both min & max
        // as well as packet reductions with larger packet sizes per iteration
        // and `f32` types.

        // Find the largest supported packet size dividing the number of
        // variables.
        uint32_t vars_per_it = 16 / tsize;
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
        for (uint32_t i = 0; i < count; i++)
            fmt("    red.global.$s.$s$s [%rd3+$u], $v;\n",
                tp, op_name, qualifier, i * tsize, jitc_var(values[i]));
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

    // Try to store 128b/iteration
    uint32_t bytes_per_it = 16;

    // Potentially reduce if the total size of the load isn't divisible
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

            default:
                jitc_fail("jitc_cuda_render_scatter_packet(): internal failure! (2)");
        }
    }
}
