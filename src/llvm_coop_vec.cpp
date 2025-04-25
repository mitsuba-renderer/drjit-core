/*
    src/llvm_coop_vec.cpp -- LLVM fallback code generation for Cooperative Vectors

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "src/llvm.h"
#include "var.h"
#include "eval.h"
#include "coop_vec.h"
#include "llvm_eval.h"
#include "llvm_scatter.h"

void jitc_llvm_render_coop_vec_unpack(const Variable *v, const Variable *a0) {
    put("    ; coop_vec_unpack\n");
    fmt("    $v = bitcast $V_$u to $T\n", v, a0, v->literal, v);
}


void jitc_llvm_render_coop_vec_accum(const Variable *v, const Variable *target,
                                     const Variable *value, const Variable *mask) {
    put("    ; coop_vec_accum\n");

    fmt("    $v_p = bitcast $<{i8*}$> $v to $<{$t*}$>\n", v, target, value);

    if (callable_depth == 0) {
        fmt_intrinsic("declare $t @llvm$e.vector.reduce.fadd.v$w$h($t, $T)",
                      value, value, value, value);
    } else {
        jitc_llvm_append_reduce_op_local((VarType) value->type, ReduceOp::Add, value);
        fmt("    $v_mask = bitcast $V to i$w\n",
            v, mask);
    }

    for (uint32_t i = 0; i < value->array_length; ++i) {
            fmt("    $v_p_$u = getelementptr inbounds $t, $<{$t*}$> $v_p, i32 $u\n",
                v, i, value, value, v, (uint32_t) v->literal + i);

        if (callable_depth == 0) {
            fmt("    $v_valid_$u = select $V, $V_$u, $T zeroinitializer\n"
                "    $v_red_$u = call $t @llvm$e.vector.reduce.fadd.v$w$h($t zeroinitializer, $T $v_valid_$u)\n"
                "    atomicrmw fadd {$t*} $v_p_$u, $t $v_red_$u monotonic\n",
                v, i, mask, value, i, value,
                v, i, value, value, value, value, v, i,
                value, v, i, value, v, i);
        } else {
            fmt("    call fastcc void @reduce_add_$h_atomic_local(<$w x $p> $v_p_$u, $V_$u, i$w $v_mask)\n",
                value, value, v, i, value, i, v);
        }
    }
}

void jitc_llvm_render_coop_vec_outer_product_accum(const Variable *v, const Variable *target,
                                                   const Variable *v0, const Variable *v1,
                                                   const Variable *mask) {
    put("    ; coop_vec_outer_product_accum\n");

    uint32_t m = v0->array_length,
             n = v1->array_length,
             tsize = type_size[v0->type],
             vec_size = jitc_llvm_vector_width * tsize;

    const MatrixDescr *d = (const MatrixDescr *) v->data;

    alloca_size  = std::max(alloca_size, (int32_t) (vec_size * (n + m)));
    alloca_align = std::max(alloca_align, (int32_t) (vec_size));

    fmt("    $v_po_0 = getelementptr inbounds i8, $<{i8*}$> $v, i32 $u\n"
        "    $v_po = bitcast $<{i8*}$> $v_po_0 to $<{[$u x $t]*}$>\n"
        "    $v_p0 = bitcast {i8*} %buffer to {$T*}\n"
        "    $v_p1 = getelementptr inbounds $T, {$T*} $v_p0, i32 $u\n"
        "    $v_mask = bitcast $V to i$w\n",
        v, target, d->offset * tsize,
        v, v, d->stride, v0,
        v, v0,
        v, v0, v0, v, m,
        v, mask);

    put("\n    ; Prepare inputs\n");
    for (uint32_t i = 0; i < m; ++i) {
        fmt("    $v_p0_$u = getelementptr inbounds $T, {$T*} $v_p0, i32 $u\n"
            "    store $V_$u, {$T*} $v_p0_$u, align $A\n",
            v, i, v0, v0, v, i,
            v0, i, v0, v, i, v0);
    }
    for (uint32_t i = 0; i < n; ++i) {
        fmt("    $v_p1_$u = getelementptr inbounds $T, {$T*} $v_p1, i32 $u\n"
            "    store $V_$u, {$T*} $v_p1_$u, align $A\n",
            v, i, v1, v1, v, i,
            v1, i, v1, v, i, v1);
    }
    fmt("    br label %l$u_before\n"
        "\n"
        "   ; Outer product\n"
        "l$u_before:\n"
        "    br label %l$u_outer\n"
        "\n"
        "l$u_outer:\n"
        "    $v_i = phi i32 [ 0, %l$u_before ], [ $v_i_next, %l$u_inner ]\n"
        "    $v_i_cont = icmp ult i32 $v_i, $u\n"
        "    $v_i_next = add nuw nsw i32 $v_i, 1\n"
        "    br i1 $v_i_cont, label %l$u_load, label %l$u_done\n"
        "\n"
        "l$u_load:\n"
        "    $v_p0i = getelementptr inbounds $T, {$T*} $v_p0, i32 $v_i\n"
        "    $v_v0i = load $T, {$T*} $v_p0i, align $A\n"
        "    br label %l$u_inner;\n"
        "\n"
        "l$u_inner:\n"
        "    $v_j = phi i32 [ 0, %l$u_load ], [ $v_j_next, %l$u_body ]\n"
        "    $v_j_next = add nuw nsw i32 $v_j, 1\n"
        "    $v_j_cont = icmp ult i32 $v_j, $u\n"
        "    br i1 $v_j_cont, label %l$u_body, label %l$u_outer\n\n"
        "l$u_body:\n",
        v->reg_index,
        v->reg_index,
        v->reg_index,

        // outer:
        v->reg_index,
        v, v->reg_index, v, v->reg_index,
        v, v, m,
        v, v,
        v, v->reg_index,v->reg_index,

        // load:
        v->reg_index,
        v, v0, v0, v, v,
        v, v0, v0, v, v0,
        v->reg_index,

        // inner:
        v->reg_index,
        v, v->reg_index, v, v->reg_index,
        v, v,
        v, v, n,
        v, v->reg_index, v->reg_index,
        v->reg_index
    );

    fmt("    $v_p1j = getelementptr inbounds $T, {$T*} $v_p1, i32 $v_j\n"
        "    $v_v1j = load $T, {$T*} $v_p1j, align $A\n"
        "    $v_ij = fmul $T $v_v0i, $v_v1j\n",
        v, v1, v1, v, v,
        v, v1, v1, v, v1,
        v, v1, v, v);

    fmt("    $v_po_ij = getelementptr inbounds [$u x $t], $<{[$u x $t]*}$> $v_po, i32 $v_i, i32 $v_j\n",
        v, d->stride, v0, d->stride, v0, v, v, v);

    if (callable_depth == 0) {
        fmt_intrinsic("declare $t @llvm$e.vector.reduce.fadd.v$w$h($t, $T)",
                      v1, v1, v1, v1);
        fmt("    $v_valid = select $V, $T $v_ij, $T zeroinitializer\n"
            "    $v_red = call $t @llvm$e.vector.reduce.fadd.v$w$h($t zeroinitializer, $T $v_valid)\n"
            "    atomicrmw fadd {$t*} $v_po_ij, $t $v_red monotonic\n",
            v, mask, v1, v, v1,
            v, v1, v1, v1, v1, v,
            v1, v, v1, v);
    } else {
        jitc_llvm_append_reduce_op_local((VarType) v1->type, ReduceOp::Add, v1);
        fmt("    call fastcc void @reduce_add_$h_atomic_local(<$w x $p> $v_po_ij, $T $v_ij, i$w $v_mask)\n",
            v1, v1, v, v1, v, v);
    }

    fmt("    br label %l$u_inner\n"
        "\n"
        "l$u_done:\n",
        v->reg_index,
        v->reg_index);
}

void jitc_llvm_render_coop_vec(const Variable *v, const Variable *a0,
                               const Variable *a1, const Variable *a2,
                               const Variable *a3) {
    fmt("    ; $s\n", var_kind_name[v->kind]);

    switch ((VarKind) v->kind) {
        case VarKind::CoopVecLiteral:
            fmt("    $v_p = insertelement $T undef, $t $l, i32 0\n"
                "    $v_0 = shufflevector $T $v_p, $T undef, <$w x i32> $z\n",
                v, v, v, v,
                v, v, v, v);
            for (uint32_t i = 1; i < v->array_length; ++i)
                fmt("    $v_$u = bitcast $V_0 to $T\n", v, i, v, v);
            break;

        case VarKind::CoopVecPack: {
                const std::vector<uint32_t> &indices = ((const CoopVecPackData *) v->data)->indices;
                for (uint32_t i =  0; i < (uint32_t) indices.size(); ++i)
                    fmt("    $v_$u = bitcast $V to $T\n", v, i, jitc_var(indices[i]), v);
            }
            break;

        case VarKind::CoopVecBinaryOp: {
                const char *op = nullptr;

                bool is_float = jitc_is_float(v),
                     is_sint  = jitc_is_sint(v);
                bool is_intrinsic = false;

                switch ((JitOp) v->literal) {
                    case JitOp::Add: op = is_float ? "fadd" : "add"; break;
                    case JitOp::Mul: op = is_float ? "fmul" : "mul"; break;
                    case JitOp::Sub: op = is_float ? "fsub" : "sub"; break;
                    case JitOp::Min: op = is_float ? "minnum" : (is_sint ? "smin" : "umin"); is_intrinsic = true; break;
                    case JitOp::Max: op = is_float ? "maxnum" : (is_sint ? "smax" : "umax"); is_intrinsic = true; break;
                    case JitOp::Step: op = is_float ? "fcmp olt" : "icmp lt"; break;
                    default:
                        jitc_fail("CoopVecBinaryOp: unsupported operation!");
                }

                if ((JitOp) v->literal == JitOp::Step) {
                    for (uint32_t i =  0; i < v->array_length; ++i) {
                        fmt("    $v_$u_m = $s $V_$u, $v_$u\n", v, i, op, a0, i, a1, i);
                        fmt("    $v_$u = select <$w x i1> $v_$u_m, $T zeroinitializer, $T $s\n",
                            v, i, v, i, v, v, jitc_llvm_ones_str[v->type]);
                    }
                } else {
                    if (!is_intrinsic) {
                        for (uint32_t i =  0; i < v->array_length; ++i)
                            fmt("    $v_$u = $s $V_$u, $v_$u\n", v, i, op, a0, i, a1, i);
                    } else {
                        fmt_intrinsic("declare $T @llvm.$s.v$w$h($T, $T)", v, op, v, a0, a1);
                        for (uint32_t i =  0; i < v->array_length; ++i)
                            fmt("    $v_$u = call fast $T @llvm.$s.v$w$h($V_$u, $V_$u)\n",
                                v, i, v, op, v, a0, i, a1, i);
                    }
                }
            }
            break;

        case VarKind::CoopVecTernaryOp:
            if ((JitOp) v->literal != JitOp::Fma)
                jitc_fail("CoopVecTernaryOp: unsupported operation!");

            fmt_intrinsic("declare $T @llvm.fma.v$w$h($T, $T, $T)", v, v,
                          a0, a1, a2);
            for (uint32_t i = 0; i < v->array_length; ++i)
                fmt("    $v_$u = call $T @llvm.fma.v$w$h($V_$u, $V_$u, $V_$u)\n",
                    v, i, v, v, a0, i, a1, i, a2, i);
            break;

        case VarKind::Bitcast:
                for (uint32_t i =  0; i < v->array_length; ++i)
                    fmt("    $v_$u = bitcast $V_u to $T\n", v, i, a0, i, v);
            break;

        case VarKind::CoopVecCast: {
                bool bigger = type_size[v->type] > type_size[a0->type],
                     dst_float = jitc_is_float(v),
                     src_signed = jitc_is_float(a0),
                     dst_signed = jitc_is_float(v),
                     src_float = jitc_is_float(a0);

                const char *op;
                if (src_float && dst_float)
                    op = bigger ? "fpext" : "fptrunc";
                else if (!src_float && !dst_float)
                    op = bigger ? (src_signed ? "sext" : "zext") : "trunc";
                else if (src_float && !dst_float)
                    op = dst_signed ? "fptosi" : "fptoui";
                else
                    op = src_signed ? "sitofp" : "uitofp";
                for (uint32_t i =  0; i < v->array_length; ++i)
                    fmt("    $v_$u = $s $V_$u to $T\n", v, i, op, a0, i, v);
            }
            break;

        case VarKind::CoopVecLoad:
            fmt("    $v_p = bitcast $<{i8*}$> $v to $<{$t*}$>\n", v, a0, v);

            for (uint32_t i = 0; i < v->array_length; ++i) {
                fmt("    $v_p_$u = getelementptr inbounds $t, $<{$t*}$> $v_p, i32 $u\n",
                    v, i, v, v, v, (uint32_t) v->literal + i);

                if (callable_depth == 0) {
                    fmt("    $v_$u_0 = load $t, {$t *} $v_p_$u, align $a\n"
                        "    $v_$u_1 = insertelement $T undef, $t $v_$u_0, i32 0\n"
                        "    $v_$u = shufflevector $T $v_$u_1, $T undef, <$w x i32> $z\n",
                        v, i, v, v, v, i, v,
                        v, i, v, v, v, i,
                        v, i, v, v, i, v);
                } else {
                    fmt_intrinsic("declare $T @llvm.masked.gather.v$w$h(<$w x {$t*}>, i32, <$w x i1>, $T)",
                                  v, v, v, v);
                    fmt("    $v_$u = call $T @llvm.masked.gather.v$w$h(<$w x {$t*}> $v_p_$u, i32 $a, $V, $T $z)\n",
                        v, i, v, v, v, v, i, v, a1, v);
                }
            }
            break;

        case VarKind::CoopVecMatVec: {
                CoopVecMatVecData *d = (CoopVecMatVecData *) v->data;
                bool transpose = d->transpose;
                const Variable *mask = a2;
                const Variable *bias = a3;

                uint32_t tsize = type_size[v->type],
                         vec_size = jitc_llvm_vector_width * tsize,
                         n = transpose ? d->A_descr.rows : d->A_descr.cols,
                         m = transpose ? d->A_descr.cols : d->A_descr.rows;

                alloca_size  = std::max(alloca_size, (int32_t) (vec_size * (n + m)));
                alloca_align = std::max(alloca_align, (int32_t) (vec_size));

                fmt( "    $v_pi = bitcast {i8*} %buffer to {$T*}\n"
                     "    $v_po = getelementptr inbounds $T, {$T*} $v_pi, i32 $u\n"
                     "    $v_pa_0 = bitcast $<{i8*}$> $v to $<{$t*}$>\n"
                     "    $v_pa{_1|} = getelementptr inbounds $t, $<{$t*}$> $v_pa_0, i32 $u\n"
                    "{    $v_pa = bitcast $<$t*$> $v_pa_1 to $<[$u x $t]*$>\n|}",
                    v, v,
                    v, v, v, v, n,
                    v, a0, v,
                    v, v, v, v, d->A_descr.offset,
                    v, v, v, transpose ? m : n, v);
                if (bias) {
                    fmt("    $v_pb_0 = bitcast $<{i8*}$> $v to $<{$t*}$>\n"
                        "    $v_pb = getelementptr inbounds $t, $<{$t*}$> $v_pb_0, i32 $u\n",
                        v, bias, v,
                        v, v, v, v, d->b_descr.offset);
                }

                put("\n    ; Prepare input\n");
                for (uint32_t i = 0; i < n; ++i) {
                    fmt("    $v_pi_$u = getelementptr inbounds $T, {$T*} $v_pi, i32 $u\n"
                        "    store $V_$u, {$T*} $v_pi_$u, align $A\n",
                        v, i, v, v, v, i,
                        a1, i, v, v, i, v);
                }

                put("\n    ; Prepare output\n");
                for (uint32_t i = 0; i < m; ++i) {
                    if (bias) {
                        fmt("    $v_b_$u_1 = getelementptr inbounds $t, $<{$t*}$> $v_pb, i32 $u\n",
                            v, i, v, v, v, i);
                        if (callable_depth == 0) {
                            fmt("    $v_b_$u_2 = load $t, {$t *} $v_b_$u_1, align $a\n"
                                "    $v_b_$u_3 = insertelement $T undef, $t $v_b_$u_2, i32 0\n"
                                "    $v_b_$u = shufflevector $T $v_b_$u_3, $T undef, <$w x i32> $z\n",
                                v, i, v, v, v, i, v,
                                v, i, v, v, v, i,
                                v, i, v, v, i, v);
                        } else {
                            fmt("    $v_b_$u = call $T @llvm.masked.gather.v$w$h(<$w x {$t*}> $v_b_$u_1, i32 $a, $V, $T $z)\n",
                                v, i, v, v, v, v, i, v, mask, v);
                        }
                    }

                    fmt("    $v_po_$u = getelementptr inbounds $T, {$T*} $v_po, i32 $u\n", v, i, v, v, v, i);
                    if (bias)
                        fmt("    store $V_b_$u, {$T*} $v_po_$u, align $A\n", v, i, v, v, i, v);
                    else
                        fmt("    store $T zeroinitializer, {$T*} $v_po_$u, align $A\n", v, v, v, i, v);
                }

                put("\n    ; Matrix multiplication\n");
                fmt("    br label %l$u_before\n"
                    "\n"
                    "l$u_before:\n"
                    "    br label %l$u_outer\n"
                    "\n"
                    "l$u_outer:\n"
                    "    $v_j = phi i32 [ 0, %l$u_before ], [ $v_j_next, %l$u_inner ]\n"
                    "    $v_j_cont = icmp ult i32 $v_j, $u\n"
                    "    $v_j_next = add nuw nsw i32 $v_j, 1\n"
                    "    br i1 $v_j_cont, label %l$u_load, label %l$u_done\n"
                    "\n"
                    "l$u_load:\n"
                    "    $v_x1 = getelementptr inbounds $T, {$T*} $v_pi, i32 $v_j\n"
                    "    $v_x = load $T, {$T*} $v_x1, align $A\n"
                    "    br label %l$u_inner;\n"
                    "\n"
                    "l$u_inner:\n"
                    "    $v_i = phi i32 [ 0, %l$u_load ], [ $v_i_next, %l$u_body ]\n"
                    "    $v_i_next = add nuw nsw i32 $v_i, 1\n"
                    "    $v_i_cont = icmp ult i32 $v_i, $u\n"
                    "    br i1 $v_i_cont, label %l$u_body, label %l$u_outer\n\n"
                    "l$u_body:\n",
                    v->reg_index,
                    v->reg_index,
                    v->reg_index,
                    v->reg_index,
                    v, v->reg_index, v, v->reg_index,
                    v, v, n,
                    v, v,

                    v, v->reg_index, v->reg_index,

                    v->reg_index,
                    v, v, v, v, v,
                    v, v, v, v, v,
                    v->reg_index,

                    v->reg_index,
                    v, v->reg_index, v, v->reg_index,
                    v, v,
                    v, v, m,
                    v, v->reg_index, v->reg_index,
                    v->reg_index
                );

                fmt("    $v_a1 = getelementptr inbounds [$u x $t], $<{[$u x $t]*}$> $v_pa, i32 $v_$s, i32 $v_$s\n",
                    v, transpose ? m : n,
                    v, transpose ? m : n,
                    v, v,
                    v, transpose ? "j" : "i",
                    v, transpose ? "i" : "j");

                if (callable_depth == 0) {
                    fmt("    $v_a2 = load $t, {$t *} $v_a1, align $a\n"
                        "    $v_a3 = insertelement $T undef, $t $v_a2, i32 0\n"
                        "    $v_a = shufflevector $T $v_a3, $T undef, <$w x i32> $z\n",
                        v, v, v, v, v,
                        v, v, v, v,
                        v, v, v, v);
                } else {
                    fmt_intrinsic("declare $T @llvm.masked.gather.v$w$h(<$w x {$t*}>, i32, <$w x i1>, $T)",
                                  v, v, v, v);
                    fmt("    $v_a = call $T @llvm.masked.gather.v$w$h(<$w x {$t*}> $v_a1, i32 $a, $V, $T $z)\n",
                        v, v, v, v, v, v, mask, v);
                }

                fmt_intrinsic("declare $T @llvm.fma.v$w$h($T, $T, $T)", v, v,
                              v, v, v);

                fmt("    $v_y1 = getelementptr inbounds $T, {$T*} $v_po, i32 $v_i\n"
                    "    $v_y = load $T, {$T*} $v_y1, align $A\n"
                    "    $v_r = call $T @llvm.fma.v$w$h($V_a, $V_x, $V_y)\n"
                    "    store $V_r, {$T*} $v_y1, align $A\n",
                    v, v, v, v, v,
                    v, v, v, v, v,
                    v, v, v, v, v, v,
                    v, v, v, v);

                fmt("    br label %l$u_inner\n"
                    "\n"
                    "l$u_done:\n",
                    v->reg_index,
                    v->reg_index);

                put("    ; Read back results\n");
                for (uint32_t i = 0; i < m; ++i)
                    fmt("    $v_$u = load $T, {$T*} $v_po_$u, align $A\n", v, i, v, v, v, i, v);
            }
            break;

        default:
            jitc_fail("jitc_llvm_render_coop_vec(): unhandled variable "
                      "kind \"%s\"!",
                      var_kind_name[(uint32_t) v->kind]);
    }
}
