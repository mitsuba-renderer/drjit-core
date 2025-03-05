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

#define OPTIX_COOP_VEC_OP_CVT 0x2A2A

// --------------------------------------------------------------------------
// Mappings to Optix ABI IDs

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
        case VarType::UInt32: return 0x2A08;
        case VarType::Int32: return 0x2A09;
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

// --------------------------------------------------------------------------
// Helper routines to generate frequently used PTX fragments

static uint32_t get_reg_count(uint32_t l) {
    if (l <= 16)
        return 16;
    else if (l <= 64)
        return 64;
    else
        return 0;
}

static void declare_buffer(const char *name, const Variable *v) {
    uint32_t tsize  = type_size[v->type],
             length = v->array_length;

    fmt("        .local .align $u .b8 $s[$u];\n"
        "        .reg.u64 %$s;\n"
        "        cvta.local.u64 %$s, $s;\n",
        tsize, name, tsize * length,
        name,
        name, name);
}

static void copy_to_buffer(const char *name, const Variable *v) {
    uint32_t tsize  = type_size[v->type],
             length = v->array_length;
    for (uint32_t i = 0; i < length; ++i)
        fmt("        st.local.$b [$s+$u], %cv$u_$u;\n", v, name, tsize * i, v->reg_index, i);
}

static void copy_from_buffer(const char *name, const Variable *v) {
    uint32_t tsize  = type_size[v->type],
             length = v->array_length;

    for (uint32_t i = 0; i < length; ++i)
        fmt("        ld.local.$b %cv$u_$u, [$s+$u];\n", v, v->reg_index, i, name, tsize * i);
}

static void put_elems(const Variable *v, uint32_t reg_count, bool trailing_comma, bool is_return_value = false) {
    for (uint32_t i = 0; i < reg_count; ++i) {
        if (i < v->array_length || is_return_value)
            fmt("%cv$u_$u, ", v->reg_index, i);
        else
            put("%u, ");
    }
    if (!trailing_comma)
        buffer.rewind(2);
}

// Packed cooperative vectors are stored in `u32` variables. For float16 values,
// the upper half is unused as opposed to, say, packing two of them into a
// 32-bit integer. The backend compiler is free to perform such optimizations
// later on, but they are not part of the OptiX PTX intrinsic interface.
void jitc_optix_render_coop_vec_unpack(const Variable *v, const Variable *a0) {
    uint32_t tsize = type_size[v->type];

    put("    // coop_vec_unpack\n");
    if (tsize == 4) {
        fmt("    mov.b32 $v, %cv$u_$u;\n", v, a0->reg_index, (uint32_t) v->literal);
    } else {
        fmt("    {\n"
            "        .reg.$b %temp;\n"
            "        cvt.u$u.u32 %temp, %cv$u_$u;\n"
            "        mov.$b $v, %temp;\n"
            "    }\n",
            v,
            tsize*8, a0->reg_index, (uint32_t) v->literal,
            v, v);
    }
}

void jitc_optix_render_coop_vec_accum(const Variable *v, const Variable *target,
                                      const Variable *value, const Variable *mask) {
    bool use_mask = !mask->is_literal() || mask->literal != 1;
    uint32_t length = value->array_length,
             reg_count = get_reg_count(length);

    // Apply any potential mask from the mask stack, see
    // jitc_coop_vec_accum()
    if (use_mask)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    fmt("    {   // coop_vec_accum\n"
        "        .reg.b32 %type, %size, %offset, %u;\n"
        "        .reg.b64 %dst;\n"
        "        mov.b32 %type, $u;\n"
        "        mov.b32 %size, $u;\n"
        "        mov.b32 %offset, $u;\n"
        "        mov.b64 %dst, $v;\n",
        jitc_optix_coop_vec_type_id((VarType) value->type),
        length,
        (uint32_t) v->literal * type_size[value->type],
        target
    );

    if (reg_count) {
        fmt("        call (), _optix_reduce_sum_accumulate_$uxi32, (%type, %size, %dst, %offset, ",
            reg_count);
        put_elems(value, reg_count, false);
        put(");\n");
    } else {
        declare_buffer("src", value);
        copy_to_buffer("src", value);
        put("        call (), _optix_reduce_sum_accumulate_ptr, (%type, %size, %dst, %offset, %src);\n");
    }
    put("    }\n");
    if (use_mask)
        fmt("\nl_$u_done:\n", v->reg_index);
}

void jitc_optix_render_coop_vec_outer_product_accum(const Variable *v, const Variable *target,
                                                    const Variable *v0, const Variable *v1,
                                                    const Variable *mask) {
    uint32_t op_length = std::max(v0->array_length, v1->array_length),
             reg_count = get_reg_count(op_length),
             tsize = type_size[jitc_var(target->dep[3])->type];

    const MatrixDescr *d = (const MatrixDescr *) v->data;

    // Apply any potential mask from the mask stack, see
    // jitc_coop_vec_outer_product_accum()
    bool use_mask = !mask->is_literal() || mask->literal != 1;
    if (use_mask)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    fmt("    {   // coop_vec_outer_product_accum\n"
        "        .reg.b32 %type_0, %type_1, %size_0, %size_1, %offset, %layout, %stride, %u;\n"
        "        .reg.b64 %dst;\n"
        "        mov.b32 %type_0, $u;\n"
        "        mov.b32 %type_1, $u;\n"
        "        mov.b32 %size_0, $u;\n"
        "        mov.b32 %size_1, $u;\n"
        "        mov.b32 %offset, $u;\n"
        "        mov.b32 %layout, $u;\n"
        "        mov.b32 %stride, $u;\n"
        "        mov.b64 %dst, $v;\n",
        jitc_optix_coop_vec_type_id((VarType) v0->type),
        jitc_optix_coop_vec_type_id((VarType) v1->type),
        v0->array_length,
        v1->array_length,
        d->offset * tsize,
        jitc_optix_coop_vec_layout_id(d->layout),
        d->stride * tsize,
        target);

    if (reg_count) {
        fmt("        call (), _optix_outer_product_accumulate_$uxi32, (%type_0, "
            "%size_0, %type_1, %size_1, %dst, %offset, %layout, %stride, ",
            reg_count);
        put_elems(v0, reg_count, true);
        put_elems(v1, reg_count, false);
        put(");\n");
    } else {
        declare_buffer("src_0", v0);
        declare_buffer("src_1", v1);
        copy_to_buffer("src_0", v0);
        copy_to_buffer("src_1", v1);
        put("        call (), _optix_outer_product_accumulate_ptr, (%type_0, "
            "%size_0, %type_1, %size_1, %dst, %offset, %layout, %stride, "
            "%src_0, %src_1);\n");
    }
    put("    }\n");
    if (use_mask)
        fmt("\nl_$u_done:\n", v->reg_index);
}

void jitc_optix_render_coop_vec(const Variable *v, const Variable *a0,
                                const Variable *a1, const Variable *a2,
                                const Variable *a3) {
    uint32_t tsize = type_size[v->type],
             length = v->array_length,
             op_length = length;

    // For matrix multiplications, the size of the required intrinsic (e.g., 16
    // vs 64 vs N) is related to the maximum cooperative vector size of the
    // input and output.
    if ((VarKind) v->kind == VarKind::CoopVecMatVec)
        op_length = std::max(op_length, (uint32_t) a1->array_length);

    uint32_t reg_count = get_reg_count(op_length);

    fmt("    .reg.b32 %cv$u_<$u>;\n", v->reg_index, std::max(reg_count, (uint32_t) v->array_length));

    fmt("    {   // $s\n", var_kind_name[v->kind]);

    switch ((VarKind) v->kind) {
        case VarKind::CoopVecLiteral:
            for (uint32_t i = 0; i < length; ++i)
                fmt("        mov.b32 %cv$u_$u, $l;\n", v->reg_index, i, v);
            break;

        case VarKind::CoopVecPack: {
                if (tsize != 4)
                    fmt("        .reg.b$u %temp;\n", tsize*8);
                const std::vector<uint32_t> &indices = ((const CoopVecPackData *) v->data)->indices;
                for (uint32_t i =  0; i < (uint32_t) indices.size(); ++i) {
                    if (tsize != 4) {
                        fmt("        mov.b$u %temp, $v;\n"
                            "        cvt.u32.u$u %cv$u_$u, %temp;\n",
                            tsize*8, jitc_var(indices[i]),
                            tsize*8, v->reg_index, i);
                    } else {
                        fmt("        mov.b32 %cv$u_$u, $v;\n",
                            v->reg_index, i, jitc_var(indices[i]));
                    }
                }
            }
            break;

        case VarKind::CoopVecLoad:
            fmt("        .reg.b32 %type, %size, %u;\n"
                "        .reg.b64 %src;\n"
                "        mov.b32 %type, $u;\n"
                "        mov.b32 %size, $u;\n"
                "        add.u64 %src, $v, $u;\n",
                jitc_optix_coop_vec_type_id((VarType) v->type),
                length,
                a0, (uint32_t) v->literal * type_size[v->type]
            );

            if (reg_count) {
                put("        call (");
                put_elems(v, reg_count, false, true);
                fmt("), _optix_vector_load_$uxi32, (%type, %size, %src);\n",
                    reg_count);
            } else {
                declare_buffer("dst", v);
                fmt("        call (), _optix_vector_load_ptr, (%type, %size, %src, %dst);\n",
                    v->reg_index);
                copy_from_buffer("dst", v);
            }
            break;

        case VarKind::CoopVecUnaryOp:
            fmt("        .reg.b32 %op, %type, %size, %u;\n"
                "        mov.b32 %op, $u;\n"
                "        mov.b32 %type, $u;\n"
                "        mov.b32 %size, $u;\n",
                jitc_optix_coop_vec_op_id((JitOp) v->literal),
                jitc_optix_coop_vec_type_id((VarType) v->type),
                length
            );

            if (reg_count) {
                put("        call (");
                put_elems(v, reg_count, false, true);
                fmt("), _optix_vector_op1_$uxi32, (%op, %type, %size, %type, %size, ", reg_count);
                put_elems(a0, reg_count, false);
                put(");\n");
            } else {
                declare_buffer("src", a0);
                declare_buffer("dst", v);
                copy_to_buffer("src", a0);
                put("        call (), _optix_vector_op1_ptr, (%op, %type, %size, %type, %size, %src, %dst);\n");
                copy_from_buffer("dst", v);
            }
            break;

        case VarKind::CoopVecBinaryOp:
            fmt("        .reg.b32 %op, %type, %size, %u;\n"
                "        mov.b32 %op, $u;\n"
                "        mov.b32 %type, $u;\n"
                "        mov.b32 %size, $u;\n",
                jitc_optix_coop_vec_op_id((JitOp) v->literal),
                jitc_optix_coop_vec_type_id((VarType) v->type),
                length
            );

            if (reg_count) {
                put("        call (");
                put_elems(v, reg_count, false, true);
                fmt("), _optix_vector_op2_$uxi32, (%op, %type, %size, %type, %size, ", reg_count);
                put_elems(a0, reg_count, true);
                put_elems(a1, reg_count, false);
                put(");\n");
            } else {
                declare_buffer("src_0", a0);
                declare_buffer("src_1", a1);
                declare_buffer("dst", v);
                copy_to_buffer("src_0", a0);
                copy_to_buffer("src_1", a1);
                put("        call (), _optix_vector_op2_ptr, (%op, %type, %size, %type, %size, %src_0, %src_1, %dst);\n");
                copy_from_buffer("dst", v);
            }
            break;

        case VarKind::CoopVecTernaryOp:
            fmt("        .reg.b32 %op, %type, %size, %u;\n"
                "        mov.b32 %op, $u;\n"
                "        mov.b32 %type, $u;\n"
                "        mov.b32 %size, $u;\n",
                jitc_optix_coop_vec_op_id((JitOp) v->literal),
                jitc_optix_coop_vec_type_id((VarType) v->type),
                length
            );

            if (reg_count) {
                put("        call (");
                put_elems(v, reg_count, false, true);
                fmt("), _optix_vector_op3_$uxi32, (%op, %type, %size, %type, %size, ", reg_count);
                put_elems(a0, reg_count, true);
                put_elems(a1, reg_count, true);
                put_elems(a2, reg_count, false);
                put(");\n");
            } else {
                declare_buffer("src_0", a0);
                declare_buffer("src_1", a1);
                declare_buffer("src_2", a2);
                declare_buffer("dst", v);
                copy_to_buffer("src_0", a0);
                copy_to_buffer("src_1", a1);
                copy_to_buffer("src_2", a2);
                put("        call (), _optix_vector_op3_ptr, (%op, %type, %size, %type, %size, %src_0, %src_1, %src_2, %dst);\n");
                copy_from_buffer("dst", v);
            }
            break;

        case VarKind::Bitcast:
                for (uint32_t i =  0; i < reg_count; ++i)
                    fmt("        mov.b32 %cv$u_$u, %cv$u_$u;\n",
                        v->reg_index, i, a0->reg_index, i);
            break;

        case VarKind::CoopVecCast:
            fmt("        .reg.b32 %op, %in_type, %out_type, %size, %u;\n"
                "        mov.b32 %op, $u;\n"
                "        mov.b32 %out_type, $u;\n"
                "        mov.b32 %in_type, $u;\n"
                "        mov.b32 %size, $u;\n",
                OPTIX_COOP_VEC_OP_CVT,
                jitc_optix_coop_vec_type_id((VarType) v->type),
                jitc_optix_coop_vec_type_id((VarType) a0->type),
                length
            );

            if (reg_count) {
                put("        call (");
                put_elems(v, reg_count, false, true);
                fmt("), _optix_vector_op1_$uxi32, (%op, %in_type, %size, %out_type, %size, ", reg_count);
                put_elems(a0, reg_count, false);
                put(");\n");
            } else {
                declare_buffer("src", a0);
                declare_buffer("dst", v);
                copy_to_buffer("src", a0);
                put("        call (), _optix_vector_op1_ptr, (%op, %in_type, %size, %out_type, %size, %src, %dst);\n");
                copy_from_buffer("dst", v);
            }
            break;

        case VarKind::CoopVecMatVec: {
                CoopVecMatVecData *d = (CoopVecMatVecData *) v->data;
                const Variable *matrix_v = jitc_var(a0->dep[3]);
                const Variable *bias_v = a3 ? jitc_var(a3->dep[3]) : nullptr;

                uint32_t input_type_id = jitc_optix_coop_vec_type_id((VarType) a1->type);
                uint32_t output_type_id = jitc_optix_coop_vec_type_id((VarType) v->type);
                uint32_t matrix_type_id = jitc_optix_coop_vec_type_id((VarType) matrix_v->type);
                uint32_t bias_type_id = bias_v ? jitc_optix_coop_vec_type_id((VarType) bias_v->type) : 0;
                uint32_t mat_tsize = type_size[matrix_v->type];
                if (!bias_type_id)
                    bias_type_id = output_type_id;

                fmt("        .reg.b32 %out_type, %out_size, %in_type, %in_size, "
                        "%in_interp, %mat_type, %mat_offset, %mat_stride, "
                        "%mat_layout, %mat_n, %mat_k, %mat_transpose, "
                        "%bias_type, %bias_offset, %u;\n"
                    "        .reg.b64 %mat_ptr, %bias_ptr;\n"
                    "        mov.b32 %out_type, $u;\n"
                    "        mov.b32 %out_size, $u;\n"
                    "        mov.b32 %in_type, $u;\n"
                    "        mov.b32 %in_size, $u;\n"
                    "        mov.b32 %in_interp, $u;\n"
                    "        mov.b32 %mat_type, $u;\n"
                    "        mov.b64 %mat_ptr, $v;\n"
                    "        mov.b32 %mat_offset, $u;\n"
                    "        mov.b32 %mat_stride, $u;\n"
                    "        mov.b32 %mat_layout, $u;\n"
                    "        mov.b32 %mat_n, $u;\n"
                    "        mov.b32 %mat_k, $u;\n"
                    "        mov.b32 %mat_transpose, $u;\n"
                    "        mov.b32 %bias_type, $u;\n",
                    output_type_id,
                    v->array_length,
                    input_type_id,
                    a1->array_length,
                    matrix_type_id,
                    matrix_type_id,
                    a0,
                    d->A_descr.offset * mat_tsize,
                    d->A_descr.stride * mat_tsize,
                    jitc_optix_coop_vec_layout_id(d->A_descr.layout),
                    d->transpose ? d->A_descr.cols : d->A_descr.rows,
                    d->transpose ? d->A_descr.rows : d->A_descr.cols,
                    (uint32_t) d->transpose,
                    bias_type_id
                );
                if (bias_v) {
                    fmt("        mov.b64 %bias_ptr, $v;\n"
                        "        mov.b32 %bias_offset, $u;\n",
                        a3,
                        d->b_descr.offset * type_size[bias_v->type]);
                } else {
                    put("        mov.b64 %bias_ptr, 0;\n"
                        "        mov.b32 %bias_offset, 0;\n");
                }

                if (reg_count) {
                    put("        call (");
                    put_elems(v, reg_count, false, true);
                    fmt("), _optix_matvecmul_$uxi32, (%out_type, %out_size, "
                        "%in_type, %in_size, %in_interp, %mat_n, "
                        "%mat_k, %mat_ptr, %mat_offset, "
                        "%mat_stride, %mat_layout, %mat_transpose, "
                        "%mat_type, %bias_ptr, %bias_offset, "
                        "%bias_type, ",
                        reg_count);
                    put_elems(a1, reg_count, false);
                    put(");\n");
                } else {
                    declare_buffer("src", a1);
                    declare_buffer("dst", v);
                    copy_to_buffer("src", a1);
                    put("        call (), _optix_matvecmul_ptr, (%out_type, %out_size, "
                        "%in_type, %in_size, %in_interp, %mat_n, "
                        "%mat_k, %mat_ptr, %mat_offset, "
                        "%mat_stride, %mat_layout, %mat_transpose, "
                        "%mat_type, %bias_ptr, %bias_offset, "
                        "%bias_type, %src, %dst);\n");

                    copy_from_buffer("dst", v);
                }
            }
            break;

        default:
            jitc_fail("jitc_optix_render_coop_vec(): unhandled variable kind \"%s\"!",
                      var_kind_name[(uint32_t) v->kind]);
    }
    put("    }\n");
}
