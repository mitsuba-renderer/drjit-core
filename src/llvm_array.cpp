/*
    src/llvm_array.cpp -- Functionality to create, read, and write
    variable arrays / LLVM code generation component.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#define DRJIT_LLVM_OPTIMIZE_ARRAY_ACCESSES 1

#include "eval.h"
#include "llvm.h"
#include "llvm_eval.h"
#include "array.h"
#include "var.h"

void jitc_llvm_render_array(Variable *v, Variable *pred) {
    if (pred && pred->array_state != (uint32_t) ArrayState::Conflicted) {
        v->reg_index = pred->reg_index;
        return;
    }
    fmt("    %arr_$u = alloca $m, i32 $u, align $A\n",
        v->reg_index, v, jitc_llvm_vector_width * v->array_length, v);
}

/**
 * When a read to a variable array has a uniform or scalar index, the
 * following kinds of mask types can be safely ignored.
 *
 * - Literal 'true' -- literal false was already optimized away in array.cpp.
 * - DefaultMask -- the current SIMD lane may be inactive because it is in
 *   a partial packet at the end of the work stream.
 * - CallMask: the current SIMD lane may be inactive in a subroutine.
 *
 * In the latter 2 cases, there is no harm in writing to the variable
 * array *as long as the index is uniform*.
 */
static bool mask_safe_to_ignore (const Variable *mask) {
    return mask->is_literal() ||
           (mask->kind == (uint32_t) VarKind::CallMask ||
            mask->kind == (uint32_t) VarKind::DefaultMask);
}

extern void jitc_llvm_render_array_init(Variable *v, Variable *pred, Variable *value) {
    v->reg_index = pred->reg_index;

    if (value->is_literal() && value->literal == 0) {
        fmt_intrinsic("declare void @llvm.memset.inline.p0.i32(ptr, i8, i32, i1)");
        fmt("    call void @llvm.memset.inline.p0.i32(ptr %arr_$u, i8 0, i32 $u, i1 0)\n",
            v->reg_index,
            v->array_length * type_size[v->type] * jitc_llvm_vector_width);
    } else {
        const char *ext = "";
        const Variable *src = value;
        if (value->type == (uint32_t) VarType::Bool) {
            fmt("    $v_e = zext $V to $M\n", v, value, value);
            ext = "_e";
            src = v;
        }
        fmt("    br label %l_$u_pre\n\n"
            "l_$u_pre:\n"
            "    br label %l_$u_loop\n\n"
            "l_$u_loop:\n"
            "    $v_cur = phi i64 [ 0, %l_$u_pre ], [ $v_next, %l_$u_loop ]\n"
            "    $v_ptr = getelementptr inbounds $M, ptr %arr_$u, i64 $v_cur\n"
            "    store $M $v$s, ptr $v_ptr, align $A\n"
            "    $v_next = add i64 $v_cur, 1\n"
            "    $v_cont = icmp ult i64 $v_next, $u\n"
            "    br i1 $v_cont, label %l_$u_loop, label %l_$u_done\n\n"
            "l_$u_done:\n",
            v->reg_index,
            v->reg_index,
            v->reg_index,
            v->reg_index,
            // phi
            v, v->reg_index, v, v->reg_index,
            // gep
            v, v, v->reg_index, v,
            // store
            value, src, ext, v, v,
            v, v,
            v, v, v->array_length,
            v, v->reg_index, v->reg_index,
            v->reg_index
        );
    }
}

void jitc_llvm_render_array_read(Variable *v, Variable *source, Variable *mask,
                                 Variable *offset) {

    const char *ext = "";
    if (v->type == (uint32_t) VarType::Bool)
        ext = "_r";

    if (!offset || offset->size == 1) {
        // Scalar/literal offset case: we can avoid a gather operation
        if (offset) {
            fmt_intrinsic("declare i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32>)");

            // Scalar offset, extract the value (danger: can't use
            // extractelement because masked index entries can be invalid)
            fmt("    $v_0 = select $V, $V, $T $z\n"
                "    $v_1 = call i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32> $v_0)\n"
                "    $v_2 = mul i32 $v_1, $u\n"
                "    $v_3 = getelementptr inbounds $m, ptr %arr_$u, i32 $v_2\n",
                v, mask, offset, offset,
                v, v,
                v, v, jitc_llvm_vector_width,
                v, v, source->reg_index, v);
        } else {
            fmt("    $v_3 = getelementptr inbounds $m, ptr %arr_$u, i32 $u\n",
                v, v, source->reg_index, v->literal * jitc_llvm_vector_width);
        }

        if (mask_safe_to_ignore(mask)) {
            fmt("    $v$s = load $M, ptr $v_3, align $A\n",
                v, ext, v, v, v);
        } else {
            fmt("    $v_5 = load $M, ptr $v_3, align $A\n"
                "    $v$s = select $V, $M $v_5, $M $z\n",
                v, v, v, v,
                v, ext, mask, v, v, v);
        }
    } else if (jitc_llvm_vector_width >= 8 && DRJIT_LLVM_OPTIMIZE_ARRAY_ACCESSES) {
        // Check if the gather can be reduced to a packet load

        fmt_intrinsic("declare i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32>)");
        fmt_intrinsic("declare i32 @llvm.vector.reduce.umin.v$wi32(<$w x i32>)");
        fmt_intrinsic("declare $M @llvm.masked.gather.v$w$H(<$w x ptr>, i32, <$w x i1>, $M)",
                      v, v, v);

        // Check if the wread is coherent
        fmt("    $v_0 = insertelement <$w x i32> undef, i32 -1, i32 0\n"
            "    $v_1 = shufflevector <$w x i32> $v_0, <$w x i32> undef, <$w x i32> $z\n"
            "    $v_2 = select $V, $V, <$w x i32> $z\n"
            "    $v_3 = select $V, $V, <$w x i32> $v_1\n"
            "    $v_4 = call i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32> $v_2)\n"
            "    $v_5 = call i32 @llvm.vector.reduce.umin.v$wi32(<$w x i32> $v_3)\n"
            "    $v_6 = icmp eq i32 $v_4, $v_5\n"
            "    br i1 $v_6, label %l_$u_u, label %l_$u_n\n\n",
            v,
            v, v,
            v, mask, offset,
            v, mask, offset, v,
            v, v,
            v, v,
            v, v, v,
            v, v->reg_index, v->reg_index);

        // Uniform case
        fmt( "l_$u_u:\n"
             "    $v_7 = mul i32 $v_4, $u\n"
             "    $v_8 = getelementptr inbounds $m, ptr %arr_$u, i32 $v_7\n",
            v->reg_index,
            v, v, jitc_llvm_vector_width,
            v, v, source->reg_index, v);

        if (mask_safe_to_ignore(mask)) {
            fmt("    $v_u = load $M, ptr $v_8, align $A\n",
                v, v, v, v);
        } else {
            fmt("    $v_10 = load $M, ptr $v_8, align $A\n"
                "    $v_u = select $V, $M $v_10, $M $z\n",
                v, v, v, v,
                v, mask, v, v, v);
        }
        fmt("    br label %l_$u_done\n\n", v->reg_index);

        // Nonuniform case
        fmt("l_$u_n:\n"
            "    $v_12 = mul $V, $s\n"
            "    $v_13 = add <$w x i32> $v_12, $s\n"
            "    $v_14 = getelementptr inbounds $m, ptr %arr_$u, <$w x i32> $v_13\n"
            "    $v_n = call $M @llvm.masked.gather.v$w$H(<$w x ptr> $v_14, i32 $a, $V, $M $z)\n"
            "    br label %l_$u_done\n\n",
            v->reg_index,
            v, offset, jitc_llvm_u32_width_str,
            v, v, jitc_llvm_u32_arange_str,
            v, v, source->reg_index, v,
            v, v, v, v, v, mask, v,
            v->reg_index);

        // Trailer
        fmt("l_$u_done:\n"
            "    $v$s = phi $M [ $v_u, %l_$u_u ], [ $v_n, %l_$u_n ]\n",
            v->reg_index,
            v, ext, v, v, v->reg_index, v, v->reg_index);
    } else {
        fmt_intrinsic("declare $M @llvm.masked.gather.v$w$H(<$w x ptr>, i32, <$w x i1>, $M)",
                      v, v, v);

        fmt("    $v_0 = mul $V, $s\n"
            "    $v_1 = add <$w x i32> $v_0, $s\n"
            "    $v_2 = getelementptr inbounds $m, ptr %arr_$u, <$w x i32> $v_1\n"
            "    $v$s = call $M @llvm.masked.gather.v$w$H(<$w x ptr> $v_2, i32 $a, $V, $M $z)\n",
            v, offset, jitc_llvm_u32_width_str,
            v, v, jitc_llvm_u32_arange_str,
            v, v, source->reg_index, v,
            v, ext, v, v, v, v, mask, v);
    }

    if (v->type == (uint32_t) VarType::Bool)
        fmt("    $v = trunc $M $v_r to $T\n", v, v, v, v);
}


void jitc_llvm_render_array_write(Variable *v, Variable *target,
                                  Variable *value, Variable *mask,
                                  Variable *offset) {
    bool copy = target->array_state == (uint32_t) ArrayState::Conflicted;
    uint32_t target_buffer = target->reg_index;

    if (offset && offset->is_array())
        offset = nullptr;

    if (copy) {
        target_buffer = jitc_array_buffer(v)->reg_index;

        fmt_intrinsic("declare void @llvm.memcpy.inline.p0.p0.i32(ptr, ptr, i32, i1)");
        fmt("    call void @llvm.memcpy.inline.p0.p0.i32(ptr align($A) %arr_$u, ptr align($A) %arr_$u, i32 $u, i1 0)\n",
            v, target_buffer, v, target->reg_index,
            target->array_length * type_size[target->type] * jitc_llvm_vector_width);
    }

    const char *ext = "";
    if (value->type == (uint32_t) VarType::Bool) {
        fmt("    $v_e = zext $V to $M\n", v, value, v);
        value = v;
        ext = "_e";
    }

    if (!offset || offset->size == 1) {
        // Scalar/literal offset case: we can avoid a scatter operation
        if (offset) {
            fmt_intrinsic("declare i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32>)");

            // Scalar offset, extract the value (danger: can't use
            // extractelement because masked index entries can be invalid)
            fmt("    $v_0 = select $V, $V, $T $z\n"
                "    $v_1 = call i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32> $v_0)\n"
                "    $v_2 = mul i32 $v_1, $u\n"
                "    $v_3 = getelementptr inbounds $m, ptr %arr_$u, i32 $v_2\n",
                v, mask, offset, offset,
                v, v,
                v, v, jitc_llvm_vector_width,
                v, v, target_buffer, v);
        } else {
            fmt("    $v_3 = getelementptr inbounds $m, ptr %arr_$u, i32 $u\n",
                v, v, target_buffer, v->literal * jitc_llvm_vector_width);
        }

        if (mask_safe_to_ignore(mask)) {
            fmt("    store $M $v$s, ptr $v_3, align $A\n",
                v, value, ext, v, v);
        } else {
            fmt("    $v_5 = load $M, ptr $v_3, align $A\n"
                "    $v_6 = select $V, $M $v$s, $V_5\n"
                "    store $V_6, ptr $v_3, align $A\n",
                v, v, v, v,
                v, mask, v, value, ext, v,
                v, v, v);
        }
    } else if (jitc_llvm_vector_width >= 8 && DRJIT_LLVM_OPTIMIZE_ARRAY_ACCESSES) {
        /// Check if the scatter can be reduced to a packet store

        fmt_intrinsic("declare i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32>)");
        fmt_intrinsic("declare i32 @llvm.vector.reduce.umin.v$wi32(<$w x i32>)");
        fmt_intrinsic("declare void @llvm.masked.scatter.v$w$H($M, <$w x ptr>, i32, <$w x i1>)",
                      v, v);

        // Check if the write is coherent
        fmt("    $v_0 = insertelement <$w x i32> undef, i32 -1, i32 0\n"
            "    $v_1 = shufflevector <$w x i32> $v_0, <$w x i32> undef, <$w x i32> $z\n"
            "    $v_2 = select $V, $V, <$w x i32> $z\n"
            "    $v_3 = select $V, $V, <$w x i32> $v_1\n"
            "    $v_4 = call i32 @llvm.vector.reduce.umax.v$wi32(<$w x i32> $v_2)\n"
            "    $v_5 = call i32 @llvm.vector.reduce.umin.v$wi32(<$w x i32> $v_3)\n"
            "    $v_6 = icmp eq i32 $v_4, $v_5\n"
            "    br i1 $v_6, label %l_$u_u, label %l_$u_n\n\n",
            v,
            v, v,
            v, mask, offset,
            v, mask, offset, v,
            v, v,
            v, v,
            v, v, v,
            v, v->reg_index, v->reg_index);

        // Uniform case
        fmt( "l_$u_u:\n"
             "    $v_7 = mul i32 $v_4, $u\n"
             "    $v_8 = getelementptr inbounds $m, ptr %arr_$u, i32 $v_7\n",
            v->reg_index,
            v, v, jitc_llvm_vector_width,
            v, v, target_buffer, v);

        if (mask_safe_to_ignore(mask)) {
            fmt("    store $M $v$s, ptr $v_8, align $A\n",
                v, value, ext, v, v);
        } else {
            fmt("    $v_10 = load $M, ptr $v_8, align $A\n"
                "    $v_11 = select $V, $M $v$s, $V_10\n"
                "    store $V_11, ptr $v_8, align $A\n",
                v, v, v, v,
                v, mask, v, value, ext, v,
                v, v, v);
        }

        fmt("    br label %l_$u_done\n\n", v->reg_index);

        // Nonuniform case
        fmt("l_$u_n:\n"
            "    $v_12 = mul $V, $s\n"
            "    $v_13 = add <$w x i32> $v_12, $s\n"
            "    $v_14 = getelementptr inbounds $m, ptr %arr_$u, <$w x i32> $v_13\n"
            "    call void @llvm.masked.scatter.v$w$H($M $v$s, <$w x ptr> $v_14, i32 $a, $V)\n"
            "    br label %l_$u_done\n\n",
            v->reg_index,
            v, offset, jitc_llvm_u32_width_str,
            v, v, jitc_llvm_u32_arange_str,
            v, v, target_buffer, v,
            v, v, value, ext, v, v, mask,
            v->reg_index);

        // Trailer
        fmt("l_$u_done:\n", v->reg_index);
    } else {
        fmt_intrinsic("declare void @llvm.masked.scatter.v$w$H($M, <$w x ptr>, i32, <$w x i1>)",
                      v, v);

        fmt("    $v_0 = mul $V, $s\n"
            "    $v_1 = add <$w x i32> $v_0, $s\n"
            "    $v_2 = getelementptr inbounds $m, ptr %arr_$u, <$w x i32> $v_1\n"
            "    call void @llvm.masked.scatter.v$w$H($M $v$s, <$w x ptr> $v_2, i32 $a, $V)\n",
            v, offset, jitc_llvm_u32_width_str,
            v, v, jitc_llvm_u32_arange_str,
            v, v, target_buffer, v,
            v, v, value, ext, v, v, mask);
    }

    v->reg_index = target_buffer;
}

void jitc_llvm_render_array_memcpy_in(const Variable *v) {
    fmt_intrinsic("declare void @llvm.memcpy.inline.p0.p0.i32(ptr, ptr, i32, i1)");
    fmt( "    $v_p1 = getelementptr inbounds ptr, ptr %params, i32 $o\n"
         "    $v_p2 = load ptr, ptr $v_p1, align 8, !alias.scope !2\n"
         "    $v_p3 = mul i64 %index, $u\n"
         "    $v_p4 = getelementptr inbounds $m, ptr $v_p2, i64 $v_p3\n"
         "    call void @llvm.memcpy.inline.p0.p0.i32(ptr align($A) %arr_$u, ptr align($A) $v_p4, i32 $u, i1 0)\n",
        v, v,
        v, v,
        v, v->size == 1 ? 0 : v->array_length,
        v, v, v, v,
        v, v->reg_index, v, v, v->array_length * type_size[v->type] * jitc_llvm_vector_width
    );
}

void jitc_llvm_render_array_memcpy_out(const Variable *v) {
    fmt_intrinsic("declare void @llvm.memcpy.inline.p0.p0.i32(ptr, ptr, i32, i1)");
    fmt( "    $v_p1 = getelementptr inbounds ptr, ptr %params, i32 $o\n"
         "    $v_p2 = load ptr, ptr $v_p1, align 8, !alias.scope !2\n"
         "    $v_p3 = mul i64 %index, $u\n"
         "    $v_p4 = getelementptr inbounds $m, ptr $v_p2, i64 $v_p3\n"
         "    call void @llvm.memcpy.inline.p0.p0.i32(ptr align($A) $v_p4, ptr align($A) %arr_$u, i32 $u, i1 0)\n",
        v, v,
        v, v,
        v, v->array_length,
        v, v, v, v,
        v, v, v, v->reg_index, v->array_length * type_size[v->type] * jitc_llvm_vector_width
    );
}

void jitc_llvm_render_array_select(Variable *v, Variable *mask, Variable *t, Variable *f) {
    uint32_t reg_index = jitc_array_buffer(v)->reg_index;
    v->reg_index = reg_index;

    fmt("    br label %l_$u_pre\n\n"
        "l_$u_pre:\n"
        "    br label %l_$u_loop\n\n"
        "l_$u_loop:\n"
        "    $v_cur = phi i64 [ 0, %l_$u_pre ], [ $v_next, %l_$u_loop ]\n"
        "    $v_ti = getelementptr inbounds $M, ptr %arr_$u, i64 $v_cur\n"
        "    $v_fi = getelementptr inbounds $M, ptr %arr_$u, i64 $v_cur\n"
        "    $v_vi = getelementptr inbounds $M, ptr %arr_$u, i64 $v_cur\n"
        "    $v_t = load $M, ptr $v_ti, align $A\n"
        "    $v_f = load $M, ptr $v_fi, align $A\n"
        "    $v_v = select $V, $M $v_t, $M $v_f\n"
        "    store $M $v_v, ptr $v_vi, align $A\n"
        "    $v_next = add i64 $v_cur, 1\n"
        "    $v_cont = icmp ult i64 $v_next, $u\n"
        "    br i1 $v_cont, label %l_$u_loop, label %l_$u_done\n\n"
        "l_$u_done:\n",
        reg_index,
        reg_index,
        reg_index,
        reg_index,
        v, reg_index, v, reg_index,
        v, v, t->reg_index, v,
        v, v, f->reg_index, v,
        v, v, reg_index, v,
        v, v, v, v,
        v, v, v, v,
        v, mask, v, v, v, v,
        v, v, v, v,
        v, v,
        v, v, v->array_length,
        v, reg_index, reg_index,
        reg_index
    );
}
