/*
    src/cuda_packet.cpp -- Specialized memory operations that read or write
    multiple adjacent values at once.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"
#include "cuda_eval.h"
#include "llvm_packet.h"
#include "var.h"
#include "op.h"
#include "log.h"

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

void jitc_cuda_render_scatter_packet(const Variable *v, const Variable *ptr,
                                     const Variable *index, const Variable *mask) {
    bool is_masked = !mask->is_literal() || mask->literal != 1;
    PacketScatterData *psd = (PacketScatterData *) v->data;
    const std::vector<uint32_t> &values = psd->values;
    const Variable *v0 = jitc_var(values[0]);

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

    // Try to load 128b/iteration
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
