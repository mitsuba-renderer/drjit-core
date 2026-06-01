/*
    src/metal_coop_vec.cpp -- Metal code generation for Cooperative Vectors

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Mirrors the LLVM cooperative-vector backend (src/llvm_coop_vec.cpp): each
    coopvec variable lowers to ``array_length`` independent MSL scalar
    locals named ``v<reg_index>_<element_index>``. Per-element ops emit one
    MSL expression per element, relying on the Metal compiler to coalesce
    consecutive scalar operations into the FFMA pipeline.

    Implementation status — all coopvec ops are implemented:
      * Pack / Load / Cast / Bitcast / Unary{Exp2,Log2,Tanh} /
        Binary{Add,Sub,Mul,Min,Max,Step} / Ternary{Fma} -- per-element
        scalar lowering.
      * CoopVecMatVec -- unrolled scalar FMA path (out = (A or Aᵀ) @ x + b).
      * CoopVecAccum / OuterProductAccum -- per-element atomic adds
        (Float32/Int32/UInt32 native; Float16 via 32-bit-aligned CAS).
*/

#if defined(DRJIT_ENABLE_METAL)

#include <cstdio>
#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "strbuf.h"
#include "coop_vec.h"
#include "metal_coop_vec.h"

// Mirror the local convenience macro used in metal_eval.cpp so the generated
// MSL is byte-for-byte consistent across translation units.
#define fmt_metal(fmt, ...) buffer.fmt_metal(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)
#define put(...)            buffer.put(__VA_ARGS__)

void jitc_metal_render_coop_vec(const Variable *v, const Variable *a0,
                                const Variable *a1, const Variable *a2,
                                const Variable *a3) {
    fmt_metal("    // $s\n", var_kind_name[(uint32_t) v->kind]);

    switch ((VarKind) v->kind) {
        case VarKind::CoopVecLiteral: {
            // Splat: emit one scalar local per element, all initialised to
            // the same literal. Mirrors VarKind::Literal in metal_eval.cpp.
            VarType vt = (VarType) v->type;
            for (uint32_t i = 0; i < v->array_length; ++i) {
                if (vt == VarType::Float32)
                    fmt_metal("    $t $v_$u = as_type<float>($lu);\n",
                              v, v, i, v);
                else if (vt == VarType::Float16)
                    fmt_metal("    $t $v_$u = as_type<half>((ushort) $lu);\n",
                              v, v, i, v);
                else if (vt == VarType::Bool)
                    fmt_metal("    $t $v_$u = ($t) ($lu);\n",
                              v, v, i, v, v);
                else
                    fmt_metal("    $t $v_$u = ($t) $lu;\n",
                              v, v, i, v, v);
            }
            break;
        }

        case VarKind::CoopVecPack: {
            // Bundle N scalar JIT vars into element locals 0..N-1.
            const std::vector<uint32_t> &indices =
                ((const CoopVecPackData *) v->data)->indices;
            for (uint32_t i = 0; i < (uint32_t) indices.size(); ++i) {
                Variable *src = jitc_var(indices[i]);
                fmt_metal("    $t $v_$u = $v;\n", v, v, i, src);
            }
            break;
        }

        case VarKind::CoopVecLoad: {
            // a0 is a buffer pointer; v->literal stores the start offset
            // (in elements). Load `array_length` consecutive elements into
            // the per-element locals.
            for (uint32_t i = 0; i < v->array_length; ++i) {
                fmt_metal("    $t $v_$u = ((device const $t*) $v)[$u];\n",
                          v, v, i, v, a0,
                          (uint32_t) v->literal + i);
            }
            break;
        }

        case VarKind::CoopVecCast: {
            // Per-element type conversion. Mirrors VarKind::Cast.
            for (uint32_t i = 0; i < v->array_length; ++i)
                fmt_metal("    $t $v_$u = ($t) $v_$u;\n",
                          v, v, i, v, a0, i);
            break;
        }

        case VarKind::Bitcast: {
            // Per-element reinterpret cast. Mirrors VarKind::Bitcast in
            // metal_eval.cpp (which itself handles the same-size and
            // mismatched-size cases).
            for (uint32_t i = 0; i < v->array_length; ++i) {
                if (v->type == a0->type)
                    fmt_metal("    $t $v_$u = $v_$u;\n",
                              v, v, i, a0, i);
                else if (type_size[v->type] == type_size[a0->type])
                    fmt_metal("    $t $v_$u = as_type<$t>($v_$u);\n",
                              v, v, i, v, a0, i);
                else
                    fmt_metal("    $t $v_$u = as_type<$t>(($b) $v_$u);\n",
                              v, v, i, v, v, a0, i);
            }
            break;
        }

        case VarKind::CoopVecUnaryOp: {
            // Currently only Exp2 / Log2 / Tanh are emitted by coop_vec.cpp.
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
                fmt_metal("    $t $v_$u = $s($v_$u);\n",
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
                        fmt_metal("    $t $v_$u = $v_$u $s $v_$u;\n",
                                  v, v, i, a0, i, infix, a1, i);
                    break;
                }

                case JitOp::Min:
                case JitOp::Max: {
                    const char *fn = (op == JitOp::Min) ? "min" : "max";
                    for (uint32_t i = 0; i < v->array_length; ++i)
                        fmt_metal("    $t $v_$u = $s($v_$u, $v_$u);\n",
                                  v, v, i, fn, a0, i, a1, i);
                    break;
                }

                case JitOp::Step:
                    // step(a, b) = (a < b) ? 0 : 1, applied per-element.
                    for (uint32_t i = 0; i < v->array_length; ++i)
                        fmt_metal("    $t $v_$u = ($v_$u < $v_$u) ? "
                                  "($t) 0 : ($t) 1;\n",
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
                fmt_metal("    $t $v_$u = fma($v_$u, $v_$u, $v_$u);\n",
                          v, v, i, a0, i, a1, i, a2, i);
            break;
        }

        case VarKind::CoopVecMatVec: {
            // out = (A or A^T) @ x + b, fully unrolled per-element.
            //
            // Inputs:
            //   a0 = matrix buffer pointer
            //   a1 = input vector x (coopvec, n elements)
            //   a2 = mask (ignored; per-thread MSL code is naturally
            //        masked downstream by the caller's active flag)
            //   a3 = bias vector b (optional, may be null)
            //
            // The mat-vec is unrolled into m * n FMAs that reference
            // distinct compile-time-known offsets into the matrix buffer.
            // For typical coopvec sizes (≤64×64) the resulting MSL is
            // small; the Metal compiler folds the scalar FMAs into the
            // GPU's FMA pipeline.
            CoopVecMatVecData *d = (CoopVecMatVecData *) v->data;
            const Variable *bias = a3;
            bool transpose = d->transpose;

            uint32_t m = transpose ? d->A_descr.cols : d->A_descr.rows;
            uint32_t n = transpose ? d->A_descr.rows : d->A_descr.cols;
            uint32_t stride = d->A_descr.stride;
            uint32_t a_off  = d->A_descr.offset;

            // 1. Initialize output elements with the bias, or zero.
            if (bias) {
                uint32_t b_off = d->b_descr.offset;
                for (uint32_t i = 0; i < m; ++i)
                    fmt_metal("    $t $v_$u = ((device const $t*) $v)[$u];\n",
                              v, v, i, v, bias, b_off + i);
            } else {
                for (uint32_t i = 0; i < m; ++i)
                    fmt_metal("    $t $v_$u = ($t) 0;\n", v, v, i, v);
            }

            // 2. Accumulate A @ x. For non-transpose, A[i][j] lives at
            // offset (i*stride + j); for transpose we swap to (j*stride + i).
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    uint32_t addr = a_off + (transpose
                                             ? (j * stride + i)
                                             : (i * stride + j));
                    fmt_metal("    $v_$u = fma("
                              "((device const $t*) $v)[$u], $v_$u, $v_$u);\n",
                              v, i, v, a0, addr, a1, j, v, i);
                }
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
    // a0 is the source coopvec; v->literal stores the element index.
    // The output variable's type matches the element type of the coopvec
    // (enforced by jitc_coop_vec_unpack), so a plain assignment suffices.
    fmt_metal("    $t $v = $v_$u;\n", v, v, a0, (uint32_t) v->literal);
}

// ---- Atomic-add helper (per-element) ---------------------------------------
//
// Emits one atomic add into ``buffer[offset]`` of ``value``. Mirrors the
// per-type dispatch in metal_scatter.cpp (same CAS strategy for Float16):
//
//   * Float32 / Int32 / UInt32 → native ``atomic_fetch_add_explicit``
//   * Float16                 → 32-bit-aligned CAS loop on the containing word
//
// The ``value`` parameter is a literal MSL expression (e.g. ``v123_4`` or
// ``v123_2 * v124_3``) to allow callers to inline simple products without
// materialising a temporary.
static void emit_atomic_add(VarType vt, const Variable *buf,
                            uint32_t elem_offset, const char *value_expr) {
    if (vt == VarType::Float32) {
        fmt_metal("    atomic_fetch_add_explicit("
                  "(device atomic_float*) ((device float*) $v + $u), "
                  "$s, memory_order_relaxed);\n",
                  buf, elem_offset, value_expr);
    } else if (vt == VarType::Int32) {
        fmt_metal("    atomic_fetch_add_explicit("
                  "(device atomic_int*) ((device int*) $v + $u), "
                  "$s, memory_order_relaxed);\n",
                  buf, elem_offset, value_expr);
    } else if (vt == VarType::UInt32) {
        fmt_metal("    atomic_fetch_add_explicit("
                  "(device atomic_uint*) ((device uint*) $v + $u), "
                  "$s, memory_order_relaxed);\n",
                  buf, elem_offset, value_expr);
    } else if (vt == VarType::Float16) {
        // No native ``atomic_half`` in MSL — CAS on the containing 32-bit word.
        // Same construction as metal_scatter.cpp:129-171; see comments there.
        fmt_metal("    {\n"
                  "        device ushort *_addr16 = "
                  "(device ushort*) $v + $u;\n"
                  "        bool _odd = ((ulong) _addr16 / 2u) & 1u;\n"
                  "        device atomic_uint *_addr32 = "
                  "(device atomic_uint*) (_addr16 - (_odd ? 1 : 0));\n"
                  "        uint _old32 = atomic_load_explicit("
                  "_addr32, memory_order_relaxed);\n"
                  "        while (true) {\n"
                  "            ushort _bits = _odd ? (ushort)(_old32 >> 16) "
                  ": (ushort)_old32;\n"
                  "            half _val = as_type<half>(_bits);\n"
                  "            half _new_val = _val + ($s);\n"
                  "            ushort _new_bits = as_type<ushort>(_new_val);\n"
                  "            uint _new32 = _odd ? "
                  "(_old32 & 0xFFFFu) | ((uint)_new_bits << 16) : "
                  "(_old32 & 0xFFFF0000u) | (uint)_new_bits;\n"
                  "            uint _expected = _old32;\n"
                  "            if (atomic_compare_exchange_weak_explicit("
                  "_addr32, &_expected, _new32, "
                  "memory_order_relaxed, memory_order_relaxed)) break;\n"
                  "            _old32 = _expected;\n"
                  "        }\n"
                  "    }\n",
                  buf, elem_offset, value_expr);
    } else {
        jitc_fail("metal coop_vec atomic add: unsupported type %u "
                  "(only Float32/Float16/Int32/UInt32 are supported on the "
                  "Metal backend).",
                  (uint32_t) vt);
    }
}

void jitc_metal_render_coop_vec_accum(const Variable *v,
                                      const Variable *target,
                                      const Variable *value,
                                      const Variable *mask) {
    // Per-element atomic add: target[v->literal + i] += mask ? value[i] : 0,
    // for i in [0, value->array_length).
    //
    // The output variable v is VarType::Void — Accum is a side-effect node,
    // so nothing materialises in the MSL kernel for v itself.
    fmt_metal("    // coop_vec_accum (offset=$u, length=$u)\n",
              (uint32_t) v->literal, (uint32_t) value->array_length);

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    if (!is_unmasked)
        fmt_metal("    if ($v) {\n", mask);

    VarType vt = (VarType) value->type;
    uint32_t base = (uint32_t) v->literal;

    for (uint32_t i = 0; i < value->array_length; ++i) {
        // Build the per-element MSL expression "v_<reg>_<i>" by hand,
        // since emit_atomic_add takes the value as a string.
        char val_expr[64];
        std::snprintf(val_expr, sizeof(val_expr), "v%u_%u",
                      value->reg_index, i);
        emit_atomic_add(vt, target, base + i, val_expr);
    }

    if (!is_unmasked)
        put("    }\n");
}

void jitc_metal_render_coop_vec_outer_product_accum(const Variable *v,
                                                    const Variable *target,
                                                    const Variable *a,
                                                    const Variable *b,
                                                    const Variable *mask) {
    // Per-element outer-product accumulation:
    //
    //   for (i in 0..m) for (j in 0..n)
    //     target[descr.offset + i*descr.stride + j] += mask ? a[i]*b[j] : 0
    //
    // m = a->array_length, n = b->array_length. The output variable v is
    // VarType::Void; it's a side-effect node.
    const MatrixDescr *d = (const MatrixDescr *) v->data;
    uint32_t m = a->array_length, n = b->array_length;

    fmt_metal("    // coop_vec_outer_product_accum ($u x $u, "
              "offset=$u, stride=$u)\n", m, n, d->offset, d->stride);

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    if (!is_unmasked)
        fmt_metal("    if ($v) {\n", mask);

    VarType vt = (VarType) a->type;

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t addr = d->offset + i * d->stride + j;
            char val_expr[96];
            std::snprintf(val_expr, sizeof(val_expr), "v%u_%u * v%u_%u",
                          a->reg_index, i, b->reg_index, j);
            emit_atomic_add(vt, target, addr, val_expr);
        }
    }

    if (!is_unmasked)
        put("    }\n");
}

#endif // DRJIT_ENABLE_METAL
