/*
    src/optix_coop_vec.cpp -- OptiX code generation for Cooperative Vectors

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "var.h"
#include "eval.h"
#include "coop_vec.h"
#include "cuda_eval.h"
#include "optix_api.h"

static uint32_t jitc_optix_coop_vec_op_id(JitOp op) {
    switch (op) {
        case JitOp::Exp2: return 0x2A21;
        case JitOp::Log2: return 0x2A22;
        case JitOp::Tanh: return 0x2A23;
        case JitOp::Max: return 0x2A24;
        case JitOp::Min: return 0x2A25;
        case JitOp::Fma: return 0x2A26;
        case JitOp::Mul: return 0x2A27;
        case JitOp::Add: return 0x2A28;
        case JitOp::Sub: return 0x2A29;
        case JitOp::Step: return 0x2A2B;
        default: jitc_fail("get_coop_vec_type_id(): unsupported operation!");
    }
}

uint32_t jitc_optix_coop_vec_type_id(VarType vt) {
    switch (vt) {
        case VarType::Float16: return 0x2A01;
        case VarType::Float32: return 0x2A03;
        default:
            jitc_fail("jitc_optix_coop_vec_type_id(): unsupported variable type!");
    }
}

uint32_t jitc_optix_coop_vec_layout_id(MatrixLayout ml) {
    switch (ml) {
        case MatrixLayout::TrainingOptimal: return OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL;
        case MatrixLayout::InferencingOptimal: return OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
        case MatrixLayout::RowMajor: return OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
        default:
            jitc_fail("jitc_optix_coop_vec_layout_id(): unsupported layout type!");
    }
}

void jitc_optix_render_coop_vec_get(const Variable *v, const Variable *a0) {
    uint32_t tsize = type_size[v->type];

    put("    // coop_vec_get\n");
    if (a0->array_length <= 64) {
        if (tsize == 4) {
            fmt("    mov.b32 $v, %cv$u_$u;\n", v, a0->reg_index, v->literal);
        } else {
            fmt("    .reg.$b %cv$u_temp;\n"
                "    cvt.u$u.u32 %cv$u_temp, %cv$u_$u;\n"
                "    mov.$b $v, %cv$u_temp;\n",
                v, v->reg_index,
                tsize*8, v->reg_index, a0->reg_index, v->literal,
                v, v, v->reg_index);
        }
    } else {
        fmt("    ld.local.$b $v, [cv$u+$u];\n", v, v, a0->reg_index, v->literal * type_size[v->type]);
    }
}

void jitc_optix_render_coop_vec(const Variable *v) {
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr;

    uint32_t reg_count = 0;
    uint32_t tsize = type_size[v->type];

    fmt("    // $s\n", var_kind_name[v->kind]);

    uint32_t length = v->array_length;
    if ((VarKind) v->kind == VarKind::CoopVecMatVec)
        length = std::max(length, (uint32_t) a1->array_length);

    if (length <= 64) {
        reg_count = length <= 16 ? 16 : 64;
        fmt("    .reg.b32 %cv$u_<$u>;\n", v->reg_index, reg_count);
    } else {
        uint32_t size = length * tsize;
        fmt("    .local .align $u .b8 cv$u[$u];\n", tsize, v->reg_index, size);
    }

    switch ((VarKind) v->kind) {
        case VarKind::CoopVecNew:
            if (reg_count) {
                if (tsize != 4)
                    fmt("    .reg.b$u %cv$u_temp;\n", tsize*8, v->reg_index);
                const std::vector<uint32_t> &indices = ((const CoopVecNewData *) v->data)->indices;
                for (uint32_t i =  0; i < (uint32_t) indices.size(); ++i) {
                    if (tsize != 4) {
                        fmt("    mov.b$u %cv$u_temp, $v;\n"
                            "    cvt.u32.u$u %cv$u_$u, %cv$u_temp;\n",
                            tsize*8, v->reg_index, jitc_var(indices[i]),
                            tsize*8, v->reg_index, i, v->reg_index);
                    } else {
                        fmt("    mov.b32 %cv$u_$u, $v;\n",
                            v->reg_index, i, jitc_var(indices[i]));
                    }
                }
            } else {
                const std::vector<uint32_t> &indices = ((const CoopVecNewData *) v->data)->indices;
                for (uint32_t i =  0; i < (uint32_t) indices.size(); ++i)
                    fmt("    st.local.$b [cv$u+$u], $v;\n",
                        v, v->reg_index,
                        tsize * i, jitc_var(indices[i]));
            }
            break;

        case VarKind::CoopVecSet:
            if (tsize != 4 || !reg_count)
                fmt("    .reg.$b %cv$u_temp;\n",
                    v, v->reg_index);

            // Assignment
            if (reg_count) {
                if (tsize != 4) {
                    fmt("    mov.$b %cv$u_temp, $v;\n"
                        "    cvt.u32.u$u %cv$u_$u, %cv$u_temp;\n",
                        v, v->reg_index, a1,
                        tsize*8, v->reg_index, v->literal, v->reg_index);
                } else {
                    fmt("    mov.b32 %cv$u_$u, $v;\n",
                        v->reg_index, v->literal, a1);
                }
            } else {
                fmt("    st.local.$b [cv$u+$u], $v;\n",
                    v, v->reg_index, v->literal*tsize, a1);
            }

            // Copy remaining entries
            for (uint32_t i = 0; i < v->array_length; ++i) {
                if (i == v->literal)
                    continue;
                if (reg_count) {
                    fmt("    mov.b32 %cv$u_$u, %cv$u_$u;\n", v->reg_index, i, a0->reg_index, i);
                } else {
                    fmt("    ld.local.$b %cv$u_temp, [cv$u+$u];\n"
                        "    st.local.$b [cv$u+$u], %cv$u_temp;\n",
                        v, v->reg_index, a0->reg_index, i*tsize,
                        v, v->reg_index, i*tsize, v->reg_index);
                }
            }

            break;

        case VarKind::CoopVecUnaryOp:
            fmt("    .reg.b32 %cv$u_op, %cv$u_type, %cv$u_size;\n", v->reg_index, v->reg_index, v->reg_index);
            if (!reg_count)
                fmt("    .reg.b64 %cv$u_dst, %cv$u_src_0;\n", v->reg_index, v->reg_index);
            fmt("    mov.b32 %cv$u_op, $u;\n", v->reg_index, jitc_optix_coop_vec_op_id((JitOp) v->literal));
            fmt("    mov.b32 %cv$u_type, $u;\n", v->reg_index, jitc_optix_coop_vec_type_id((VarType) v->type));
            fmt("    mov.b32 %cv$u_size, $u;\n", v->reg_index, v->array_length);
            if (!reg_count) {
                fmt("    cvta.local.u64 %cv$u_dst, cv$u;\n"
                    "    cvta.local.u64 %cv$u_src_0, cv$u;\n",
                    v->reg_index, v->reg_index,
                    v->reg_index, a0->reg_index);
            }

            if (reg_count) {
                put("    call (");
                for (uint32_t i = 0; i < reg_count; ++i) {
                    fmt("%cv$u_$u", v->reg_index, i);
                    if (i + 1 < reg_count)
                        put(", ");
                }
                fmt("), _optix_vector_op1_$uxi32, ("
                    "%cv$u_op, %cv$u_type, %cv$u_size, %cv$u_type, %cv$u_size, ",
                    reg_count,
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index, v->reg_index);
                for (uint32_t i = 0; i < reg_count; ++i)
                    fmt("%cv$u_$u, ", a0->reg_index, i);
                buffer.rewind(2);
                put(");\n");
            } else {
                fmt("    call (), _optix_vector_op1_ptr, ("
                    "%cv$u_op, %cv$u_type, %cv$u_size, %cv$u_type, %cv$u_size, %cv$u_dst);\n",
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                    v->reg_index);
            }
            break;


        case VarKind::CoopVecBinaryOp:
            fmt("    .reg.b32 %cv$u_op, %cv$u_type, %cv$u_size;\n", v->reg_index, v->reg_index, v->reg_index);
            if (!reg_count)
                fmt("    .reg.b64 %cv$u_dst, %cv$u_src_0, %cv$u_src_1;\n", v->reg_index, v->reg_index, v->reg_index);
            fmt("    mov.b32 %cv$u_op, $u;\n", v->reg_index, jitc_optix_coop_vec_op_id((JitOp) v->literal));
            fmt("    mov.b32 %cv$u_type, $u;\n", v->reg_index, jitc_optix_coop_vec_type_id((VarType) v->type));
            fmt("    mov.b32 %cv$u_size, $u;\n", v->reg_index, v->array_length);
            if (!reg_count) {
                fmt("    cvta.local.u64 %cv$u_dst, cv$u;\n"
                    "    cvta.local.u64 %cv$u_src_0, cv$u;\n"
                    "    cvta.local.u64 %cv$u_src_1, cv$u;\n",
                    v->reg_index, v->reg_index,
                    v->reg_index, a0->reg_index,
                    v->reg_index, a1->reg_index);
            }

            if (reg_count) {
                put("    call (");
                for (uint32_t i = 0; i < reg_count; ++i) {
                    fmt("%cv$u_$u", v->reg_index, i);
                    if (i + 1 < reg_count)
                        put(", ");
                }
                fmt("), _optix_vector_op2_$uxi32, ("
                    "%cv$u_op, %cv$u_type, %cv$u_size, %cv$u_type, %cv$u_size, ",
                    reg_count,
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index, v->reg_index);
                for (uint32_t i = 0; i < reg_count; ++i)
                    fmt("%cv$u_$u, ", a0->reg_index, i);
                for (uint32_t i = 0; i < reg_count; ++i)
                    fmt("%cv$u_$u, ", a1->reg_index, i);
                buffer.rewind(2);
                put(");\n");
            } else {
                fmt("    call (), _optix_vector_op2_ptr, ("
                    "%cv$u_op, %cv$u_type, %cv$u_size, %cv$u_type, %cv$u_size, %cv$u_src_0, %cv$u_src_1, %cv$u_dst);\n",
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                    v->reg_index, v->reg_index, v->reg_index);
            }
            break;

        case VarKind::CoopVecTernaryOp:
            fmt("    .reg.b32 %cv$u_op, %cv$u_type, %cv$u_size;\n", v->reg_index, v->reg_index, v->reg_index);
            if (!reg_count)
                fmt("    .reg.b64 %cv$u_dst, %cv$u_src_0, %cv$u_src_1, "
                    "%cv$u_src_2;\n",
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index);
            fmt("    mov.b32 %cv$u_op, $u;\n", v->reg_index, jitc_optix_coop_vec_op_id((JitOp) v->literal));
            fmt("    mov.b32 %cv$u_type, $u;\n", v->reg_index, jitc_optix_coop_vec_type_id((VarType) v->type));
            fmt("    mov.b32 %cv$u_size, $u;\n", v->reg_index, v->array_length);
            if (!reg_count) {
                fmt("    cvta.local.u64 %cv$u_dst, cv$u;\n"
                    "    cvta.local.u64 %cv$u_src_0, cv$u;\n"
                    "    cvta.local.u64 %cv$u_src_1, cv$u;\n"
                    "    cvta.local.u64 %cv$u_src_2, cv$u;\n",
                    v->reg_index, v->reg_index,
                    v->reg_index, a0->reg_index,
                    v->reg_index, a1->reg_index,
                    v->reg_index, a2->reg_index);
            }

            if (reg_count) {
                put("    call (");
                for (uint32_t i = 0; i < reg_count; ++i) {
                    fmt("%cv$u_$u", v->reg_index, i);
                    if (i + 1 < reg_count)
                        put(", ");
                }
                fmt("), _optix_vector_op3_$uxi32, ("
                    "%cv$u_op, %cv$u_type, %cv$u_size, %cv$u_type, %cv$u_size, ",
                    reg_count,
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index, v->reg_index);
                for (uint32_t i = 0; i < reg_count; ++i)
                    fmt("%cv$u_$u, ", a0->reg_index, i);
                for (uint32_t i = 0; i < reg_count; ++i)
                    fmt("%cv$u_$u, ", a1->reg_index, i);
                for (uint32_t i = 0; i < reg_count; ++i)
                    fmt("%cv$u_$u, ", a2->reg_index, i);
                buffer.rewind(2);
                put(");\n");
            } else {
                fmt("    call (), _optix_vector_op3_ptr, ("
                    "%cv$u_op, %cv$u_type, %cv$u_size, %cv$u_type, %cv$u_size, %cv$u_src_0, %cv$u_src_1, %cv$u_src_2, %cv$u_dst);\n",
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index);
            }
            break;

        case VarKind::CoopVecMatVec: {
                CoopVecMatVecData *d = (CoopVecMatVecData *) v->data;
                const Variable *matrix_v = jitc_var(a0->dep[3]);
                const Variable *bias_v = a2 ? jitc_var(a2->dep[3]) : nullptr;

                uint32_t input_type_id = jitc_optix_coop_vec_type_id((VarType) a1->type);
                uint32_t output_type_id = jitc_optix_coop_vec_type_id((VarType) v->type);
                uint32_t matrix_type_id = jitc_optix_coop_vec_type_id((VarType) matrix_v->type);
                uint32_t bias_type_id = bias_v ? jitc_optix_coop_vec_type_id((VarType) bias_v->type) : 0;
                uint32_t mat_tsize = type_size[matrix_v->type];
                if (!bias_type_id)
                    bias_type_id = output_type_id;

                put("    .reg.b32 ");
                for (const char *name :
                     { "out_type", "out_size", "in_type", "in_size",
                       "in_interp", "mat_type", "mat_offset", "mat_stride",
                       "mat_layout", "mat_n", "mat_k", "mat_transpose",
                       "bias_type", "bias_offset" })
                    fmt("%cv$u_$s, ", v->reg_index, name);

                buffer.rewind(2);
                put(";\n");
                fmt("    .reg.b64 %cv$u_mat_ptr, %cv$u_bias_ptr;\n", v->reg_index, v->reg_index);
                fmt("    mov.b32 %cv$u_out_type, $u;\n", v->reg_index, output_type_id);
                fmt("    mov.b32 %cv$u_out_size, $u;\n", v->reg_index, v->array_length);
                fmt("    mov.b32 %cv$u_in_type, $u;\n", v->reg_index, input_type_id);
                fmt("    mov.b32 %cv$u_in_size, $u;\n", v->reg_index, a1->array_length);
                fmt("    mov.b32 %cv$u_in_interp, $u;\n", v->reg_index, matrix_type_id);
                fmt("    mov.b32 %cv$u_mat_type, $u;\n", v->reg_index, matrix_type_id);
                fmt("    mov.b64 %cv$u_mat_ptr, $v;\n", v->reg_index, a0);
                fmt("    mov.b32 %cv$u_mat_offset, $u;\n", v->reg_index, d->A_descr.offset * mat_tsize);
                fmt("    mov.b32 %cv$u_mat_stride, $u;\n", v->reg_index, d->A_descr.stride * mat_tsize);
                fmt("    mov.b32 %cv$u_mat_layout, $u;\n", v->reg_index, jitc_optix_coop_vec_layout_id(d->A_descr.layout));
                fmt("    mov.b32 %cv$u_mat_n, $u;\n", v->reg_index, d->A_descr.rows);
                fmt("    mov.b32 %cv$u_mat_k, $u;\n", v->reg_index, d->A_descr.cols);
                fmt("    mov.b32 %cv$u_mat_transpose, $u;\n", v->reg_index, d->transpose);
                fmt("    mov.b32 %cv$u_bias_type, $u;\n", v->reg_index, bias_type_id);
                if (bias_v) {
                    fmt("    mov.b64 %cv$u_bias_ptr, $v;\n", v->reg_index, a2);
                    fmt("    mov.b32 %cv$u_bias_offset, $u;\n", v->reg_index, d->b_descr.offset * type_size[bias_v->type]);
                } else {
                    fmt("    mov.b64 %cv$u_bias_ptr, 0;\n", v->reg_index);
                    fmt("    mov.b32 %cv$u_bias_offset, 0;\n", v->reg_index);
                }

                if (reg_count) {
                    put("    call (");
                    for (uint32_t i = 0; i < reg_count; ++i) {
                        fmt("%cv$u_$u", v->reg_index, i);
                        if (i + 1 < reg_count)
                            put(", ");
                    }

                    fmt("), _optix_matvecmul_$uxi32, (%cv$u_out_type, %cv$u_out_size, "
                        "%cv$u_in_type, %cv$u_in_size, %cv$u_in_interp, %cv$u_mat_n, "
                        "%cv$u_mat_k, %cv$u_mat_ptr, %cv$u_mat_offset, "
                        "%cv$u_mat_stride, %cv$u_mat_layout, %cv$u_mat_transpose, "
                        "%cv$u_mat_type, %cv$u_bias_ptr, %cv$u_bias_offset, "
                        "%cv$u_bias_type, ",
                        reg_count,
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index);

                    for (uint32_t i = 0; i < reg_count; ++i) {
                        fmt("%cv$u_$u", a1->reg_index, std::min(i, a1->array_length - 1u));

                        if (i + 1 < reg_count)
                            put(", ");
                    }
                    put(");\n");
                } else {
                    fmt("    .reg.b64 %cv$u_dst, %cv$u_src;\n"
                        "    cvta.local.u64 %cv$u_dst, cv$u;\n"
                        "    cvta.local.u64 %cv$u_src, cv$u;\n",
                        v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index,
                        v->reg_index, a1->reg_index);

                    fmt("    call (), _optix_matvecmul_ptr, (%cv$u_out_type, %cv$u_out_size, "
                        "%cv$u_in_type, %cv$u_in_size, %cv$u_in_interp, %cv$u_mat_n, "
                        "%cv$u_mat_k, %cv$u_mat_ptr, %cv$u_mat_offset, "
                        "%cv$u_mat_stride, %cv$u_mat_layout, %cv$u_mat_transpose, "
                        "%cv$u_mat_type, %cv$u_bias_ptr, %cv$u_bias_offset, "
                        "%cv$u_bias_type, %cv$u_src, %cv$u_dst);\n",
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index);
                }
            }
            break;

        default:
            jitc_fail("jitc_optix_render_coop_vec(): unhandled variable kind \"%s\"!",
                      var_kind_name[(uint32_t) v->kind]);
    }
}
