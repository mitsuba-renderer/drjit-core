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

void jitc_llvm_render_coop_vec_unpack(const Variable *v, const Variable *a0) {
    put("    ; coop_vec_unpack\n");
    fmt("    $v = bitcast $V_$u to $T\n", v, a0, v->literal, v);
}

void jitc_llvm_render_coop_vec(const Variable *v) {
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr;

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
                    default:
                        jitc_fail("CoopVecBinaryOp: unsupported operation!");
                }

                if (!is_intrinsic) {
                    for (uint32_t i =  0; i < v->array_length; ++i)
                        fmt("    $v_$u = $s $V_$u, $v_$u\n", v, i, op, a0, i, a1, i);
                } else {
                    fmt_intrinsic("declare $T @llvm.$s.v$w$h($T, $T)", v, op, v, a0, a1);
                    for (uint32_t i =  0; i < v->array_length; ++i)
                        fmt("    $v_$u = call $T @llvm.$s.v$w$h($V_$u, $V_$u)\n", v, i, v, op, v, a0, i, a1, i);
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

        case VarKind::CoopVecMatVec: {
                CoopVecMatVecData *d = (CoopVecMatVecData *) v->data;
                bool transpose = d->transpose;

                uint32_t tsize = type_size[v->type],
                         vec_size = jitc_llvm_vector_width * tsize,
                         n = transpose ? d->A_descr.rows : d->A_descr.cols,
                         m = transpose ? d->A_descr.cols : d->A_descr.rows;

                alloca_size  = std::max(alloca_size, (int32_t) (vec_size * (n + m)));
                alloca_align = std::max(alloca_align, (int32_t) (vec_size));

                fmt("    $v_pi = bitcast {i8*} %buffer to {$T*}\n", v, v);
                fmt("    $v_po = getelementptr inbounds $T, {$T*} $v_pi, i32 $u\n", v, v, v, v, n);
                fmt("    $v_pa_0 = bitcast {i8*} $v to {$t*}\n", v, a0, v);
                fmt("    $v_pa = getelementptr inbounds $t, {$t*} $v_pa_0, i32 $u\n", v, v, v, v, d->A_descr.offset);

                if (a2) {
                    fmt("    $v_pb_0 = bitcast {i8*} $v to {$t*}\n", v, a2, v);
                    fmt("    $v_pb = getelementptr inbounds $t, {$t*} $v_pb_0, i32 $u\n", v, v, v, v, d->b_descr.offset);
                }

                put("\n    ; Prepare input\n");
                for (uint32_t i = 0; i < n; ++i) {
                    fmt("    $v_pi_$u = getelementptr inbounds $T, {$T*} $v_pi, i32 $u\n", v, i, v, v, v, i);
                    fmt("    store $V_$u, {$T*} $v_pi_$u, align $A\n", a1, i, v, v, i, v);
                }

                put("\n    ; Prepare output\n");
                for (uint32_t i = 0; i < m; ++i) {

                    if (a2) {
                        fmt("    $v_b_$u_1 = getelementptr inbounds $t, {$t*} $v_pb, i32 $u\n"
                            "    $v_b_$u_2 = load $t, {$t *} $v_b_$u_1, align $a\n"
                            "    $v_b_$u_3 = insertelement $T undef, $t $v_b_$u_2, i32 0\n"
                            "    $v_b_$u = shufflevector $T $v_b_$u_3, $T undef, <$w x i32> $z\n",
                            v, i, v, v, v, i,
                            v, i, v, v, v, i, v,
                            v, i, v, v, v, i,
                            v, i, v, v, i, v);
                    }

                    fmt("    $v_po_$u = getelementptr inbounds $T, {$T*} $v_po, i32 $u\n", v, i, v, v, v, i);
                    if (a2)
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

                if (transpose) {
                    fmt("    $v_a1 = getelementptr inbounds [$u x $t], {[$u x $t]*} $v_pa, i32 $v_j, i32 $v_i\n"
                        "    $v_a2 = load $t, {$t *} $v_a1, align $a\n"
                        "    $v_a3 = insertelement $T undef, $t $v_a2, i32 0\n"
                        "    $v_a = shufflevector $T $v_a3, $T undef, <$w x i32> $z\n",
                        v, m, v, m, v, v, v, v,
                        v, v, v, v, v,
                        v, v, v, v,
                        v, v, v, v);
                } else {
                    fmt("    $v_a1 = getelementptr inbounds [$u x $t], {[$u x $t]*} $v_pa, i32 $v_i, i32 $v_j\n"
                        "    $v_a2 = load $t, {$t *} $v_a1, align $a\n"
                        "    $v_a3 = insertelement $T undef, $t $v_a2, i32 0\n"
                        "    $v_a = shufflevector $T $v_a3, $T undef, <$w x i32> $z\n",
                        v, n, v, n, v, v, v, v,
                        v, v, v, v, v,
                        v, v, v, v,
                        v, v, v, v);
                }

                fmt("    $v_y1 = getelementptr inbounds $T, {$T*} $v_po, i32 $v_i\n"
                    "    $v_y = load $T, {$T*} $v_y1, align $A\n",
                    v, v, v, v, v,
                    v, v, v, v, v);

                fmt_intrinsic("declare $T @llvm.fma.v$w$h($T, $T, $T)", v, v,
                              v, v, v);

                fmt("    $v_r = call $T @llvm.fma.v$w$h($V_a, $V_x, $V_y)\n",
                    v, v, v, v, v, v);

                fmt("    store $V_r, {$T*} $v_y1, align $A\n",
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
