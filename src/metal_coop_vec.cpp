/*
    src/metal_coop_vec.cpp -- Metal code generation for Cooperative Vectors

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

*/

#include "internal.h"
#include "eval.h"
#include "var.h"
#include "log.h"
#include "coop_vec.h"
#include "metal_coop_vec.h"

#include "metal_eval.h"

void jitc_metal_render_coop_vec(const Variable *v, const Variable *a0,
                                const Variable *a1, const Variable *a2,
                                const Variable *a3) {
    fmt("// $s\n", var_kind_name[(uint32_t) v->kind]);

    switch ((VarKind) v->kind) {
        case VarKind::CoopVecLiteral: {
            VarType vt = (VarType) v->type;
            for (uint32_t i = 0; i < v->array_length; ++i) {
                if (vt == VarType::Float32)
                    fmt("$t $v_$u = as_type<float>($lu);\n",
                        v, v, i, v);
                else if (vt == VarType::Float16)
                    fmt("$t $v_$u = as_type<half>((ushort) $lu);\n",
                        v, v, i, v);
                else if (vt == VarType::Bool)
                    fmt("$t $v_$u = ($t) ($lu);\n",
                        v, v, i, v, v);
                else
                    fmt("$t $v_$u = ($t) $lu;\n",
                        v, v, i, v, v);
            }
            break;
        }

        case VarKind::CoopVecPack: {
            const std::vector<uint32_t> &indices =
                ((const CoopVecPackData *) v->data)->indices;
            for (uint32_t i = 0; i < (uint32_t) indices.size(); ++i) {
                Variable *src = jitc_var(indices[i]);
                fmt("$t $v_$u = $v;\n", v, v, i, src);
            }
            break;
        }

        case VarKind::CoopVecLoad: {
            for (uint32_t i = 0; i < v->array_length; ++i) {
                fmt("$t $v_$u = ((device const $t*) $v)[$u];\n",
                    v, v, i, v, a0,
                    (uint32_t) v->literal + i);
            }
            break;
        }

        case VarKind::CoopVecCast: {
            for (uint32_t i = 0; i < v->array_length; ++i)
                fmt("$t $v_$u = ($t) $v_$u;\n",
                    v, v, i, v, a0, i);
            break;
        }

        case VarKind::Bitcast: {
            for (uint32_t i = 0; i < v->array_length; ++i) {
                if (v->type == a0->type)
                    fmt("$t $v_$u = $v_$u;\n",
                        v, v, i, a0, i);
                else if (type_size[v->type] == type_size[a0->type])
                    fmt("$t $v_$u = as_type<$t>($v_$u);\n",
                        v, v, i, v, a0, i);
                else
                    fmt("$t $v_$u = as_type<$t>(($b) $v_$u);\n",
                        v, v, i, v, v, a0, i);
            }
            break;
        }

        case VarKind::CoopVecUnaryOp: {
            const char *fn;
            switch ((JitOp) v->literal) {
                case JitOp::Exp2: fn = "exp2"; break;
                case JitOp::Log2: fn = "log2"; break;
                case JitOp::Tanh: fn = "tanh"; break;
                default:
                    jitc_fail("jitc_metal_render_coop_vec(): "
                              "CoopVecUnaryOp received unsupported op %u",
                              (uint32_t) v->literal);
            }
            for (uint32_t i = 0; i < v->array_length; ++i)
                fmt("$t $v_$u = $s($v_$u);\n",
                    v, v, i, fn, a0, i);
            break;
        }

        case VarKind::CoopVecBinaryOp: {
            JitOp op = (JitOp) v->literal;

            switch (op) {
                case JitOp::Add:
                case JitOp::Sub:
                case JitOp::Mul: {
                    const char *infix = (op == JitOp::Add) ? "+"
                                      : (op == JitOp::Sub) ? "-"
                                                           : "*";
                    for (uint32_t i = 0; i < v->array_length; ++i)
                        fmt("$t $v_$u = $v_$u $s $v_$u;\n",
                            v, v, i, a0, i, infix, a1, i);
                    break;
                }

                case JitOp::Min:
                case JitOp::Max: {
                    const char *fn = (op == JitOp::Min) ? "min" : "max";
                    for (uint32_t i = 0; i < v->array_length; ++i)
                        fmt("$t $v_$u = $s($v_$u, $v_$u);\n",
                            v, v, i, fn, a0, i, a1, i);
                    break;
                }

                case JitOp::Step:
                    for (uint32_t i = 0; i < v->array_length; ++i)
                        fmt("$t $v_$u = ($v_$u < $v_$u) ? ($t) 0 : ($t) 1;\n",
                            v, v, i, a0, i, a1, i, v, v);
                    break;

                default:
                    jitc_fail("jitc_metal_render_coop_vec(): "
                              "CoopVecBinaryOp received unsupported op %u",
                              (uint32_t) v->literal);
            }
            break;
        }

        case VarKind::CoopVecTernaryOp: {
            if ((JitOp) v->literal != JitOp::Fma)
                jitc_fail("jitc_metal_render_coop_vec(): "
                          "CoopVecTernaryOp received unsupported op %u "
                          "(only Fma is supported)",
                          (uint32_t) v->literal);
            for (uint32_t i = 0; i < v->array_length; ++i)
                fmt("$t $v_$u = fma($v_$u, $v_$u, $v_$u);\n",
                    v, v, i, a0, i, a1, i, a2, i);
            break;
        }

        case VarKind::CoopVecMatVec: {
            // out = (A or A^T) @ x [+ b]

            CoopVecMatVecData *d = (CoopVecMatVecData *) v->data;
            const Variable *bias = a3;
            bool transpose = d->transpose;

            uint32_t m      = v->array_length,   // output length
                     n      = a1->array_length,  // input length
                     cols   = d->A_descr.cols,
                     rows   = d->A_descr.rows,
                     stride = d->A_descr.stride,
                     a_off  = d->A_descr.offset;

            // mpp::matmul2d requires the contraction dimension to be a
            // multiple of 16; fall back to the scalar loop otherwise.
            bool use_metal4 =
                state.metal_devices[thread_state(JitBackend::Metal)->device]
                    .supports_metal4 &&
                (n % 16) == 0;

            if (use_metal4) {
                // Single mpp::tensor_ops::matmul2d call per thread.

                // The packed (2-arg) tensor constructor implies a dense layout
                if (stride != cols)
                    jitc_fail("jitc_metal_render_coop_vec(): CoopVecMatVec "
                              "requires a densely-packed weight matrix");

                bool relaxed = (VarType) v->type == VarType::Float16;

                fmt_intrinsic(
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                    "using namespace mpp::tensor_ops;");
                uses_metal4 = true;

                // Declare the output element locals consumed by downstream nodes.
                for (uint32_t i = 0; i < m; ++i)
                    fmt("$t $v_$u;\n", v, v, i);

                put("{\n");

                // Gather the input vector into an addressable thread array.
                fmt("$t $v_mm_in[$u];\n", a1, v, n);
                for (uint32_t j = 0; j < n; ++j)
                    fmt("$v_mm_in[$u] = $v_$u;\n", v, j, a1, j);

                // Float accumulator (matmul destination).
                fmt("float $v_mm_acc[$u];\n", v, m);

                fmt("constexpr auto $v_mm_d = matmul2d_descriptor(1, $u, $u, false, $s, $s);\n",
                    v, m, n, transpose ? "false" : "true",
                    relaxed ? "true" : "false");
                fmt("matmul2d<$v_mm_d, execution_thread> $v_mm_op;\n", v, v);
                fmt("auto $v_mm_x = tensor($v_mm_in, dextents<int,2>($u, 1));\n", v, v, n);
                fmt("auto $v_mm_w = tensor((device $t*) $v + $u, dextents<int,2>($u, $u));\n", v, v, a0, a_off, cols, rows);
                fmt("auto $v_mm_o = tensor($v_mm_acc, dextents<int,2>($u, 1));\n", v, v, m);
                fmt("$v_mm_op.run($v_mm_x, $v_mm_w, $v_mm_o);\n",
                    v, v, v, v);

                // Write results back to the output locals, folding in the bias.
                if (bias) {
                    uint32_t b_off = d->b_descr.offset;
                    for (uint32_t i = 0; i < m; ++i)
                        fmt("$v_$u = ($t)($v_mm_acc[$u] + (float)((device $t*) $v)[$u]);\n",
                            v, i, v, v, i, v, bias, b_off + i);
                } else {
                    for (uint32_t i = 0; i < m; ++i)
                        fmt("$v_$u = ($t) $v_mm_acc[$u];\n", v, i, v, v, i);
                }

                put("}\n");
            } else {
                // Scalar fallback for Metal < 4

                for (uint32_t i = 0; i < m; ++i)
                    fmt("$t $v_$u;\n", v, v, i);

                put("{\n");

                // Gather the input vector into an addressable thread array.
                fmt("$t _in[$u];\n", a1, n);
                for (uint32_t j = 0; j < n; ++j)
                    fmt("_in[$u] = $v_$u;\n", j, a1, j);

                fmt("$t _out[$u];\n", v, m);

                // transpose swaps which loop variable indexes the matrix rows.
                const char *row = transpose ? "_j" : "_i",
                           *col = transpose ? "_i" : "_j";

                fmt("for (uint _i = 0; _i < $u; ++_i) {\n", m);
                if (bias)
                    fmt("float _acc = (float) ((device const $t*) $v)[$u + _i];\n",
                        v, bias, d->b_descr.offset);
                else
                    put("float _acc = 0.f;\n");
                fmt("for (uint _j = 0; _j < $u; ++_j)\n"
                    "    _acc = fma((float) ((device const $t*) $v)[$u + $s * $u + $s], (float) _in[_j], _acc);\n"
                    "_out[_i] = ($t) _acc;\n"
                    "}\n",
                    n, v, a0, a_off, row, stride, col, v);

                for (uint32_t i = 0; i < m; ++i)
                    fmt("$v_$u = _out[$u];\n", v, i, i);

                put("}\n");
            }
            break;
        }

        default:
            jitc_fail("jitc_metal_render_coop_vec(): unhandled VarKind::%s",
                      var_kind_name[(uint32_t) v->kind]);
    }
}

void jitc_metal_render_coop_vec_unpack(const Variable *v,
                                       const Variable *a0) {
    fmt("$t $v = $v_$u;\n", v, v, a0, (uint32_t) v->literal);
}

void jitc_metal_render_coop_vec_accum(const Variable *v,
                                      const Variable *target,
                                      const Variable *value,
                                      const Variable *mask) {
    fmt("// coop_vec_accum (offset=$u, length=$u)\n",
              (uint32_t) v->literal, (uint32_t) value->array_length);

    jitc_assert((VarType) value->type == VarType::Float32,
                "jitc_metal_render_coop_vec_accum(): expected FP32 operands.");

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    if (!is_unmasked)
        fmt("if ($v) {\n", mask);

    uint32_t base = (uint32_t) v->literal;
    for (uint32_t i = 0; i < value->array_length; ++i)
        fmt("atomic_fetch_add_explicit((device atomic_float*) "
            "((device float*) $v + $u), $v_$u, memory_order_relaxed);\n",
            target, base + i, value, i);

    if (!is_unmasked)
        put("}\n");
}

void jitc_metal_render_coop_vec_outer_product_accum(const Variable *v,
                                                    const Variable *target,
                                                    const Variable *a,
                                                    const Variable *b,
                                                    const Variable *mask) {
    const MatrixDescr *d = (const MatrixDescr *) v->data;
    uint32_t m = a->array_length, n = b->array_length;

    fmt("// coop_vec_outer_product_accum ($u x $u, "
              "offset=$u, stride=$u)\n", m, n, d->offset, d->stride);

    jitc_assert((VarType) a->type == VarType::Float32,
                "jitc_metal_render_coop_vec_outer_product_accum(): expected "
                "FP32 operands.");

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    if (is_unmasked)
        put("{\n");
    else
        fmt("if ($v) {\n", mask);

    fmt("float _a[$u], _b[$u];\n", m, n);
    for (uint32_t i = 0; i < m; ++i)
        fmt("_a[$u] = $v_$u;\n", i, a, i);
    for (uint32_t j = 0; j < n; ++j)
        fmt("_b[$u] = $v_$u;\n", j, b, j);

    fmt("for (uint _i = 0; _i < $u; ++_i)\n"
        "    for (uint _j = 0; _j < $u; ++_j)\n"
        "        atomic_fetch_add_explicit((device atomic_float*) "
        "((device float*) $v + ($u + _i * $u + _j)), _a[_i] * _b[_j], memory_order_relaxed);\n",
        m, n, target, d->offset, d->stride);

    put("}\n");
}
