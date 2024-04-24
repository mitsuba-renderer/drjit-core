/*
    src/llvm_scatter.cpp -- Indirectly writing to memory aka. "scatter" is a
    nuanced and performance-critical operation. This file provides LLVM IR
    templates for a variety of different scatter implementations to address
    diverse use cases.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"
#include "var.h"
#include "llvm_eval.h"
#include "llvm_scatter.h"

// Simple scatter
void jitc_llvm_render_scatter(const Variable *v, const Variable *ptr,
                              const Variable *value, const Variable *index,
                              const Variable *mask) {
    fmt_intrinsic("declare void @llvm.masked.scatter.v$w$h($T, <$w x $p>, i32, $T)",
         value, value, value, mask);

    fmt("{    $v_0 = bitcast $<i8*$> $v to $<$t*$>\n|}"
         "    $v_1 = getelementptr inbounds $t, $<$p$> {$v_0|$v}, $V\n"
         "    call void @llvm.masked.scatter.v$w$h($V, <$w x $p> $v_1, i32 $a, $V)\n",
         v, ptr, value,
         v, value, value, v, ptr, index,
         value, value, value, v, value, mask);
}

static const char *jitc_llvm_atomicrmw_name(VarType vt, ReduceOp op) {
    if (jitc_is_float(vt)) {
        switch (op) {
            case ReduceOp::Add: return "fadd";
            case ReduceOp::Min: return "fmin";
            case ReduceOp::Max: return "fmax";
            default: break;
        }
    } else if (jitc_is_int(vt)) {
        bool is_signed = jitc_is_sint(vt);
        switch (op) {
            case ReduceOp::Add: return "add";
            case ReduceOp::Min: return is_signed ? "min" : "umin";
            case ReduceOp::Max: return is_signed ? "max" : "umax";
            case ReduceOp::And: return "and";
            case ReduceOp::Or: return "or";
            default: break;
        }
    }

    jitc_fail("jitc_llvm_atomicrmw_name(): this operation is currently not "
              "supported by the LLVM backend (op %i, vt %i)",
              (int) op, (int) vt);
}

static drjit::tuple<const char *, const char *, const char *, const char *, const char*>
jitc_llvm_vector_reduce_config(VarType vt, ReduceOp op) {
    const char *name = nullptr,
               *modifier = "",
               *identity = "",
               *identity_type = "",
               *version ="";

    if (jitc_is_float(vt)) {

        switch (op) {
            case ReduceOp::Add:
                name = "fadd";
                modifier = "reassoc ";

                if (jitc_llvm_version_major < 12)
                    version = ".v2";

                switch (vt) {
                    case VarType::Float16:
                        identity = "half -0.0, ";
                        identity_type = "half, ";
                        break;

                    case VarType::Float32:
                        identity = "float -0.0, ";
                        identity_type = "float, ";
                        break;

                    case VarType::Float64:
                        identity = "double -0.0, ";
                        identity_type = "double, ";
                        break;

                    default:
                        break;
                }
                break;

            case ReduceOp::Min:
                name = "fmin";
                modifier = "nnan ";
                break;

            case ReduceOp::Max:
                name = "fmax";
                modifier = "nnan ";
                break;

            default:
                break;
        }
    } else if (jitc_is_int(vt)) {
        bool is_signed = jitc_is_sint(vt);
        switch (op) {
            case ReduceOp::Add:
                name = "add";
                break;

            case ReduceOp::Min:
                name = is_signed ? "smin" : "umin";
                break;

            case ReduceOp::Max:
                name = is_signed ? "smax" : "umax";
                break;

            case ReduceOp::And:
                name = "and";
                break;

            case ReduceOp::Or:
                name = "or";
                break;

            default:
                break;
        }
    }


    if (!name)
        jitc_fail("jitc_llvm_name(): this operation is currently not "
                  "supported by the LLVM backend (op %i, vt %i)",
                  (int) op, (int) vt);

    return {name, modifier, identity, identity_type ,version};
}

static const char *reduce_op_name[(int) ReduceOp::Count] = {
    "", "add", "mul", "min", "max", "and", "or"
};

static const char *append_reduce_op_direct(VarType vt, ReduceOp op, const Variable *v) {
    if (jitc_llvm_vector_width > 32)
        jitc_fail("append_reduce_op_direct(): internal error -- code generation "
                  "assumes a vector length of <= 32 entries");

    uint32_t ptr_align = (uint32_t) sizeof(void *),
             ptr_align_vec = std::min(ptr_align * jitc_llvm_vector_width, jitc_llvm_max_align);

    const char *op_name = reduce_op_name[(int) op],
               *atomicrmw_name = jitc_llvm_atomicrmw_name(vt, op);

    fmt_intrinsic(
        "define internal fastcc void @reduce_$s_$h_atomic($P %ptr, $T %val, i$w %active) local_unnamed_addr #0 ${\n"
        "prelude:\n"
        "    %ptr_a = alloca [$w x $p], align $u\n"
        "    %val_a = alloca [$w x $t], align $A\n"
        "    %valid = zext i$w %active to i32\n"
       "{    %ptr_a2 = bitcast [$w x $p]* %ptr_a to $P*\n|}"
       "{    %val_a2 = bitcast [$w x $t]* %val_a to $T*\n|}"
        "    store $P %ptr, {$P*} {%ptr_a2|%ptr_a}, align $u\n"
        "    store $T %val, {$T*} {%val_a2|%val_a}, align $A\n"
        "    br label %loop_prefix\n\n"
        ""
        "loop_prefix:\n"
        "    %index = phi i32 [ 0, %prelude ], [ %index_next, %loop_suffix ]\n"
        "    %bit_i = shl nuw i32 1, %index\n"
        "    %valid_i = and i32 %bit_i, %valid\n"
        "    %do_scatter_i = icmp ne i32 %valid_i, 0\n"
        "    br i1 %do_scatter_i, label %do_scatter, label %loop_suffix\n\n"
        ""
        "do_scatter:\n"
        "    %ptr_p = getelementptr inbounds [$w x $p], {[$w x $p]*} %ptr_a, i32 0, i32 %index\n"
        "    %val_p = getelementptr inbounds [$w x $t], {[$w x $t]*} %val_a, i32 0, i32 %index\n"
        "    %ptr_i = load $p, {$p*} %ptr_p, align $u\n"
        "    %val_i = load $t, $p %val_p, align $a\n"
        "    atomicrmw $s $p %ptr_i, $t %val_i monotonic\n"
        "    br label %loop_suffix\n\n"
        ""
        "loop_suffix:\n"
        "    %index_next = add nuw nsw i32 %index, 1\n"
        "    %is_done = icmp eq i32 %index_next, $w\n"
        "    br i1 %is_done, label %loop_end, label %loop_prefix\n\n"
        ""
        "loop_end:\n"
        "    ret void\n"
        "$}",

        // definition
        op_name, v, v, v,

        // prelude
        v, ptr_align_vec,
        v, v,

        v, v,
        v, v,

        v, v, ptr_align_vec,
        v, v, v,

        // do_scatter
        v, v,
        v, v,
        v, v, ptr_align,
        v, v, v,
        atomicrmw_name, v, v
    );

    return "atomic"; // variant name
}

static const char *append_reduce_op_local(VarType vt, ReduceOp op, const Variable *v) {
    uint32_t ptr_align = (uint32_t) sizeof(void *),
             ptr_align_vec = std::min(ptr_align * jitc_llvm_vector_width, jitc_llvm_max_align),
             shiftamt = log2i_ceil(type_size[(int) vt]);

    const char *op_name = reduce_op_name[(int) op],
               *atomicrmw_name = jitc_llvm_atomicrmw_name(vt, op),
               *cmp_op = jitc_is_float(v) ? "fcmp one" : "icmp ne";

    auto [vector_reduce_name, vector_reduce_modifier, vector_reduce_identity,
          vector_reduce_identity_type, vector_reduce_version] 
            = jitc_llvm_vector_reduce_config(vt, op);

    fmt_intrinsic("declare $t @llvm$e.vector.reduce$s.$s.v$w$h($s$T)",
                  v, vector_reduce_version, vector_reduce_name, v, vector_reduce_identity_type, v);

    Variable id_v{};
    id_v.type = (uint32_t) vt;
    id_v.literal = jitc_reduce_identity(vt, op);

    // Failed experiment: skipping to the next element via ctz() wasn't
    // faster because of the resulting inter-instruction dependency
    fmt_intrinsic(
        "define internal fastcc void @reduce_$s_$h_atomic_local($P %ptr, $T %value, i$w %active_in) local_unnamed_addr #0 ${\n"
        "prelude:\n"
        "    %ptr_a = alloca [$w x $p], align $u\n"
       "{    %ptr_a2 = bitcast [$w x $p]* %ptr_a to $P*\n|}"
        "    store $P %ptr, {$P*} {%ptr_a2|%ptr_a}, align $u\n"
        "    %identity_0 = insertelement $T undef, $t $l, i32 0\n"
        "    %identity_1 = shufflevector $T %identity_0, $T undef, <$w x i32> $z\n"
        "    %valid_0 = $s $T %identity_1, %value\n"
        "    %valid_1 = bitcast i$w %active_in to <$w x i1>\n"
        "    %valid_2 = and <$w x i1> %valid_0, %valid_1\n"
        "    %bit_0 = bitcast i$w 1 to <$w x i1>\n"
        "    %shiftamt_0 = insertelement <$w x i64> undef, i64 $u, i64 0\n"
        "    %shiftamt_1 = shufflevector <$w x i64> %shiftamt_0, <$w x i64> undef, <$w x i32> $z\n"
        "    %ptrlo_0 = ptrtoint $P %ptr to <$w x i64>\n"
        "    %ptrlo_1 = lshr <$w x i64> %ptrlo_0, %shiftamt_1\n"
        "    %ptrlo_2 = trunc <$w x i64> %ptrlo_1 to <$w x i32>\n"
        "    %ptrlo_3 = select <$w x i1> %valid_1, <$w x i32> %ptrlo_2, <$w x i32> $z\n"
        "    br label %loop_prefix\n\n"
        ""
        "loop_prefix:\n"
        "    %active = phi <$w x i1> [ %valid_2, %prelude ], [ %active_final, %loop_suffix ]\n"
        "    %bit = phi <$w x i1> [ %bit_0, %prelude ], [ %bit_next, %loop_suffix ]\n"
        "    %index = phi i32 [ 0, %prelude ], [ %index_next, %loop_suffix ]\n"
        "    %active_0 = bitcast <$w x i1> %active to i$w\n"
        "    %active_1 = icmp ne i$w %active_0, 0\n"
        "    br i1 %active_1, label %loop_body, label %loop_end\n\n"
        ""
        "loop_body:\n"
        "    %active_2 = and <$w x i1> %bit, %active\n"
        "    %active_3 = bitcast <$w x i1> %active_2 to i$w\n"
        "    %active_4 = icmp ne i$w %active_3, 0\n"
        "    br i1 %active_4, label %do_scatter, label %loop_suffix\n\n"
        ""
        "do_scatter:\n"
        "    %ptr_p = getelementptr inbounds [$w x $p], {[$w x $p]*} %ptr_a, i32 0, i32 %index\n"
        "    %ptr_i = load $p, {$p*} %ptr_p, align $u\n"
        "    %ptrlo_4 = ptrtoint $p %ptr_i to i64\n"
        "    %ptrlo_5 = lshr i64 %ptrlo_4, $u\n"
        "    %ptrlo_6 = trunc i64 %ptrlo_5 to i32\n"
        "    %ptrlo_7 = insertelement <$w x i32> undef, i32 %ptrlo_6, i32 0\n"
        "    %ptrlo_8 = shufflevector <$w x i32> %ptrlo_7, <$w x i32> undef, <$w x i32> $z\n"
        "    %ptr_diff = icmp ne <$w x i32> %ptrlo_8, %ptrlo_3\n"
        "    %red_in = select <$w x i1> %ptr_diff, $T %identity_1, $T %value\n"
        "    %red_out = call $s$t @llvm$e.vector.reduce$s.$s.v$w$h($s$T %red_in)\n"
        "    atomicrmw $s $p %ptr_i, $t %red_out monotonic\n"
        "    %active_next = and <$w x i1> %active, %ptr_diff\n"
        "    br label %loop_suffix\n\n"
        ""
        "loop_suffix:\n"
        "    %active_final = phi <$w x i1> [ %active, %loop_body ], [ %active_next, %do_scatter ]\n"
        "    %bit_1 = bitcast <$w x i1> %bit to i$w\n"
        "    %bit_2 = shl i$w %bit_1, 1\n"
        "    %bit_next = bitcast i$w %bit_2 to <$w x i1>\n"
        "    %index_next = add nsw nuw i32 %index, 1\n"
        "    br label %loop_prefix\n\n"
        ""
        "loop_end:\n"
        "    ret void\n"
        "$}",

        // definition
        op_name, v, v, v,

        // prelude
        v, ptr_align_vec,
        v, v,
        v, v, ptr_align_vec,
        v, v, &id_v,
        v, v,
        cmp_op, v,
        shiftamt,
        v,

        // do_scatter
        v, v,
        v, v, ptr_align,
        v,
        shiftamt,
        v, v,
        vector_reduce_modifier, v, vector_reduce_version, vector_reduce_name, v, vector_reduce_identity, v,
        atomicrmw_name, v, v
    );

    return "atomic_local"; // variant name
}

static const char *append_reduce_op_noconflict(VarType vt, ReduceOp op, const Variable *v) {
    uint32_t ptr_align = (uint32_t) sizeof(void *),
             ptr_align_vec = std::min(ptr_align * jitc_llvm_vector_width, jitc_llvm_max_align),
             shiftamt = log2i_ceil(type_size[(int) vt]);

    const char *op_name = reduce_op_name[(int) op],
               *cmp_op = jitc_is_float(v) ? "fcmp one" : "icmp ne";


    auto [vector_reduce_name, vector_reduce_modifier, vector_reduce_identity,
          vector_reduce_identity_type, vector_reduce_version] 
            = jitc_llvm_vector_reduce_config(vt, op);

    fmt_intrinsic("declare $t @llvm$e.vector.reduce$s.$s.v$w$h($s$T)",
                  v, vector_reduce_version, vector_reduce_name, v, vector_reduce_identity_type, v);

    bool is_float = jitc_is_float(v),
         is_sint = jitc_is_sint(v);

    const char* scalar_op_name = nullptr;
    bool scalar_intrinsic = false;
    switch (op) {
        case ReduceOp::Add:
                scalar_op_name = is_float ? "fadd" : "add";
                break;

        case ReduceOp::Max:
                scalar_op_name = is_float ? "maxnum" : (is_sint ? "smax" : "umax");
                scalar_intrinsic = true;
                break;

        case ReduceOp::Min:
                scalar_op_name = is_float ? "minnum" : (is_sint ? "smin" : "umin");
                scalar_intrinsic = true;
                break;

        case ReduceOp::Or:
                scalar_op_name = "or";
                break;

        case ReduceOp::And:
                scalar_op_name = "and";
                break;

        default:
            jitc_fail("append_reduce_op_noconflict(): unsupported operation!");
    }

    char scalar_op[128];
    const char *tp  = type_name_llvm[(int) v->type],
               *tph = type_name_llvm_abbrev[(int) v->type];

    bool intrinsic_generated = false;
#if !defined(__aarch64__)
    if ((op == ReduceOp::Min || op == ReduceOp::Max) && vt == VarType::Float16) {
        fmt_intrinsic(
            "define internal fastcc half @$s.f16(half %arg0, half %arg1) #0 ${\n"
            "    %p = fcmp $s half %arg0, %arg1\n"
            "    %f = select i1 %p, half %arg0, half %arg1\n"
            "    ret half %f\n"
            "$}",
            scalar_op_name,
            op == ReduceOp::Max ? "ogt" : "olt"
        );
        snprintf(scalar_op, sizeof(scalar_op), "call fastcc half @%s.%s(half %%before, half %%red_out)",
                 scalar_op_name, tph);
        intrinsic_generated = true;
    }
#endif
    if (intrinsic_generated) {
        ;
    } else if (scalar_intrinsic) {
        fmt_intrinsic("declare $t @llvm.$s.$h($t, $t)", v, scalar_op_name, v, v, v);
        snprintf(scalar_op, sizeof(scalar_op), "call %s @llvm.%s.%s(%s %%before, %s %%red_out)",
                 tp, scalar_op_name, tph, tp, tp);
    } else {
        snprintf(scalar_op, sizeof(scalar_op), "%s %s %%before, %%red_out",
                 scalar_op_name, tp);
    }

    Variable id_v{};
    id_v.type = (uint32_t) vt;
    id_v.literal = jitc_reduce_identity(vt, op);

    fmt_intrinsic(
        "define internal fastcc void @reduce_$s_$h_noconflict($P %ptr, $T %value, i$w %active_in) local_unnamed_addr #0 ${\n"
        "prelude:\n"
        "    %ptr_a = alloca [$w x $p], align $u\n"
       "{    %ptr_a2 = bitcast [$w x $p]* %ptr_a to $P*\n|}"
        "    store $P %ptr, {$P*} {%ptr_a2|%ptr_a}, align $u\n"
        "    %identity_0 = insertelement $T undef, $t $l, i32 0\n"
        "    %identity_1 = shufflevector $T %identity_0, $T undef, <$w x i32> $z\n"
        "    %valid_0 = $s $T %identity_1, %value\n"
        "    %valid_1 = bitcast i$w %active_in to <$w x i1>\n"
        "    %valid_2 = and <$w x i1> %valid_0, %valid_1\n"
        "    %bit_0 = bitcast i$w 1 to <$w x i1>\n"
        "    %shiftamt_0 = insertelement <$w x i64> undef, i64 $u, i64 0\n"
        "    %shiftamt_1 = shufflevector <$w x i64> %shiftamt_0, <$w x i64> undef, <$w x i32> $z\n"
        "    %ptrlo_0 = ptrtoint $P %ptr to <$w x i64>\n"
        "    %ptrlo_1 = lshr <$w x i64> %ptrlo_0, %shiftamt_1\n"
        "    %ptrlo_2 = trunc <$w x i64> %ptrlo_1 to <$w x i32>\n"
        "    %ptrlo_3 = select <$w x i1> %valid_1, <$w x i32> %ptrlo_2, <$w x i32> $z\n"
        "    br label %loop_prefix\n\n"
        ""
        "loop_prefix:\n"
        "    %active = phi <$w x i1> [ %valid_2, %prelude ], [ %active_final, %loop_suffix ]\n"
        "    %bit = phi <$w x i1> [ %bit_0, %prelude ], [ %bit_next, %loop_suffix ]\n"
        "    %index = phi i32 [ 0, %prelude ], [ %index_next, %loop_suffix ]\n"
        "    %active_0 = bitcast <$w x i1> %active to i$w\n"
        "    %active_1 = icmp ne i$w %active_0, 0\n"
        "    br i1 %active_1, label %loop_body, label %loop_end\n\n"
        ""
        "loop_body:\n"
        "    %active_2 = and <$w x i1> %bit, %active\n"
        "    %active_3 = bitcast <$w x i1> %active_2 to i$w\n"
        "    %active_4 = icmp ne i$w %active_3, 0\n"
        "    br i1 %active_4, label %do_scatter, label %loop_suffix\n\n"
        ""
        "do_scatter:\n"
        "    %ptr_p = getelementptr inbounds [$w x $p], {[$w x $p]*} %ptr_a, i32 0, i32 %index\n"
        "    %ptr_i = load $p, {$p*} %ptr_p, align $u\n"
        "    %ptrlo_4 = ptrtoint $p %ptr_i to i64\n"
        "    %ptrlo_5 = lshr i64 %ptrlo_4, $u\n"
        "    %ptrlo_6 = trunc i64 %ptrlo_5 to i32\n"
        "    %ptrlo_7 = insertelement <$w x i32> undef, i32 %ptrlo_6, i32 0\n"
        "    %ptrlo_8 = shufflevector <$w x i32> %ptrlo_7, <$w x i32> undef, <$w x i32> $z\n"
        "    %ptr_diff = icmp ne <$w x i32> %ptrlo_8, %ptrlo_3\n"
        "    %red_in = select <$w x i1> %ptr_diff, $T %identity_1, $T %value\n"
        "    %red_out = call $s$t @llvm$e.vector.reduce$s.$s.v$w$h($s$T %red_in)\n"
        "    %before = load $t, $p %ptr_i, align $a\n"
        "    %after = $s\n"
        "    store $t %after, $p %ptr_i, align $a\n"
        "    %active_next = and <$w x i1> %active, %ptr_diff\n"
        "    br label %loop_suffix\n\n"
        ""
        "loop_suffix:\n"
        "    %active_final = phi <$w x i1> [ %active, %loop_body ], [ %active_next, %do_scatter ]\n"
        "    %bit_1 = bitcast <$w x i1> %bit to i$w\n"
        "    %bit_2 = shl i$w %bit_1, 1\n"
        "    %bit_next = bitcast i$w %bit_2 to <$w x i1>\n"
        "    %index_next = add nsw nuw i32 %index, 1\n"
        "    br label %loop_prefix\n\n"
        ""
        "loop_end:\n"
        "    ret void\n"
        "$}",

        // definition
        op_name, v, v, v,

        // prelude
        v, ptr_align_vec,
        v, v,
        v, v, ptr_align_vec,
        v, v, &id_v,
        v, v,
        cmp_op, v,
        shiftamt,
        v,

        // do_scatter
        v, v,
        v, v, ptr_align,
        v,
        shiftamt,
        v, v,
        vector_reduce_modifier, v, vector_reduce_version, vector_reduce_name, v, vector_reduce_identity, v,

        v, v, v,
        scalar_op,
        v, v, v
    );

    return "noconflict"; // variant name
}

void jitc_llvm_render_scatter_reduce(const Variable *v,
                                     const Variable *ptr,
                                     const Variable *value,
                                     const Variable *index,
                                     const Variable *mask) {

    ReduceOp op = (ReduceOp) (uint32_t) v->literal;
    ReduceMode mode = (ReduceMode) (uint32_t) (v->literal >> 32);
    VarType vt = (VarType) value->type;

    const char *variant;

    switch (mode) {
        case ReduceMode::Direct:
            variant = append_reduce_op_direct(vt, op, value);
            break;

        case ReduceMode::Local:
            variant = append_reduce_op_local(vt, op, value);
            break;

        case ReduceMode::NoConflicts:
            variant = append_reduce_op_noconflict(vt, op, value);
            break;

        default:
            jitc_fail("jitc_llvm_render_scatter_reduce(): unhandled mode (%i)!", (int) mode);
    }

    fmt("{    $v_0 = bitcast $<i8*$> $v to $<$t*$>\n|}"
         "    $v_1 = getelementptr inbounds $t, $<$p$> {$v_0|$v}, $V\n"
         "    $v_2 = bitcast $V to i$w\n"
         "    call fastcc void @reduce_$s_$h_$s(<$w x $p> $v_1, $V, i$w $v_2)\n",
        v, ptr, value,
        v, value, value, v, ptr, index,
        v, mask,
        reduce_op_name[(int) op], value, variant, value, v, value, v);
}

void jitc_llvm_render_scatter_inc(Variable *v, const Variable *ptr,
                                  const Variable *index, const Variable *mask) {
    fmt("{    $v_0 = bitcast $<i8*$> $v to $<i32*$>\n|}"
         "    $v_1 = getelementptr inbounds i32, $<{i32*}$> {$v_0|$v}, $V\n"
         "    $v = call fastcc $T @reduce_inc_u32(<$w x {i32*}> $v_1, $V)\n",
        v, ptr,
        v, v, ptr, index,
        v, v, v, mask);

    fmt_intrinsic("declare i32 @llvm.cttz.i32(i32, i1)");
    fmt_intrinsic("declare i64 @llvm$e.vector.reduce.umax.v$wi64(<$w x i64>)");

    fmt_intrinsic(
        "define internal fastcc <$w x i32> @reduce_inc_u32(<$w x {i32*}> %ptrs_in, <$w x i1> %active_in) local_unnamed_addr #0 ${\n"
        "L0:\n"
        "    %ptrs_start_0 = select <$w x i1> %active_in, <$w x {i32*}> %ptrs_in, <$w x {i32*}> $z\n"
        "    %ptrs_start_1 = ptrtoint <$w x {i32*}> %ptrs_start_0 to <$w x i64>\n"
        "    br label %L1\n\n"
        "L1:\n"
        "    %ptrs = phi <$w x i64> [ %ptrs_start_1, %L0 ], [ %ptrs_next, %L4 ]\n"
        "    %out = phi <$w x i32> [ $z, %L0 ], [ %out_next, %L4 ]\n"
        "    %ptr = call i64 @llvm$e.vector.reduce.umax.v$wi64(<$w x i64> %ptrs)\n"
        "    %done = icmp eq i64 %ptr, 0\n"
        "    br i1 %done, label %L5, label %L2\n\n"
        ""
        "L2:\n"
        "    %ptr_b0 = insertelement <$w x i64> undef, i64 %ptr, i32 0\n"
        "    %ptr_b1 = shufflevector <$w x i64> %ptr_b0, <$w x i64> undef, <$w x i32> $z\n"
        "    %active_v = icmp eq <$w x i64> %ptr_b1, %ptrs\n"
        "    %active_i0 = bitcast <$w x i1> %active_v to i$w\n"
        "    %active_i1 = zext i$w %active_i0 to i32\n"
        "    %ptrs_next = select <$w x i1> %active_v, <$w x i64> $z, <$w x i64> %ptrs\n"
        "    br label %L3\n\n"
        ""
        "L3:\n"
        "    %active = phi i32 [ %active_i1, %L2 ], [ %active_next, %L3 ]\n"
        "    %accum = phi i32 [ 0, %L2 ], [ %accum_next, %L3 ]\n"
        "    %out_2 = phi <$w x i32> [ %out, %L2 ], [ %out_2_next, %L3 ]\n"
        "    %index = call i32 @llvm.cttz.i32(i32 %active, i1 1)\n"
        "    %index_bit = shl nuw nsw i32 1, %index\n"
        "    %active_next = xor i32 %active, %index_bit\n"
        "    %accum_next = add nuw nsw i32 %accum, 1\n"
        "    %out_2_next = insertelement <$w x i32> %out_2, i32 %accum, i32 %index\n"
        "    %done_2 = icmp eq i32 %active_next, 0\n"
        "    br i1 %done_2, label %L4, label %L3\n\n"
        ""
        "L4:\n"
        "    %ptr_p = inttoptr i64 %ptr to {i32*}\n"
        "    %prev = atomicrmw add {i32*} %ptr_p, i32 %accum_next monotonic\n"
        "    %prev_b0 = insertelement <$w x i32> undef, i32 %prev, i32 0\n"
        "    %prev_b1 = shufflevector <$w x i32> %prev_b0, <$w x i32> undef, <$w x i32> $z\n"
        "    %sum = add <$w x i32> %prev_b1, %out_2_next\n"
        "    %out_next = select <$w x i1> %active_v, <$w x i32> %sum, <$w x i32> %out\n"
        "    br label %L1;\n"
        ""
        "L5:\n"
        "    ret <$w x i32> %out\n"
        "$}"
    );

    v->consumed = 1;
}

// Kahan summation scatter-add
void jitc_llvm_render_scatter_add_kahan(const Variable *v,
                                        const Variable *ptr_1,
                                        const Variable *ptr_2,
                                        const Variable *index,
                                        const Variable *value) {
    uint32_t reg_index = v->reg_index;

    fmt_intrinsic("declare $t @llvm.fabs.$h($t)", value, value, value);

    fmt("{    $v_ptr1 = bitcast $<i8*$> $v to $<$t*$>\n|}"
         "    $v_target1 = getelementptr inbounds $t, $<$p$> {$v_ptr1|$v}, $V\n"
        "{    $v_ptr2 = bitcast $<i8*$> $v to $<$t*$>\n|}"
         "    $v_target2 = getelementptr inbounds $t, $<$p$> {$v_ptr2|$v}, $V\n"
         "    br label %l$u_0\n\n"
         "l$u_0:\n"
         "    br label %l$u_1\n\n",
        v, ptr_1, value,
        v, value, value, v, ptr_1, index,
        v, ptr_2, value,
        v, value, value, v, ptr_2, index,
        reg_index,
        reg_index,
        reg_index);

    fmt("l$u_1:\n"
        "    $v_index = phi i32 [ 0, %l$u_0 ], [ $v_index_next, %l$u_3 ]\n"
        "    $v_value_i = extractelement <$w x $t> $v, i32 $v_index\n"
        "    $v_active_i = fcmp une $t $v_value_i, $z\n"
        "    br i1 $v_active_i, label %l$u_2, label %l$u_3\n\n",
        reg_index,
        v, reg_index, v, reg_index,
        v, value, value, v,
        v, value, v,
        v, reg_index, reg_index);

    fmt("l$u_2:\n"
        "    $v_target1_i = extractelement <$w x $p> $v_target1, i32 $v_index\n"
        "    $v_target2_i = extractelement <$w x $p> $v_target2, i32 $v_index\n"
        "    $v_before = atomicrmw fadd $p $v_target1_i, $t $v_value_i monotonic\n"
        "    $v_after = fadd $t $v_before, $v_value_i\n"
        "    $v_case1_0 = fsub $t $v_before, $v_after\n"
        "    $v_case1 = fadd $t $v_case1_0, $v_value_i\n"
        "    $v_case2_0 = fsub $t $v_value_i, $v_after\n"
        "    $v_case2 = fadd $t $v_case2_0, $v_before\n"
        "    $v_abs_before = call $t @llvm.fabs.$h($t $v_before)\n"
        "    $v_abs_value = call $t @llvm.fabs.$h($t $v_value_i)\n"
        "    $v_pred = fcmp oge $t $v_abs_before, $v_abs_value\n"
        "    $v_result = select i1 $v_pred, $t $v_case1, $t $v_case2\n"
        "    atomicrmw fadd $p $v_target2_i, $t $v_result monotonic\n"
        "    br label %l$u_3\n\n",
        reg_index,
        v, value, v, v,
        v, value, v, v,
        v, value, v, value, v,
        v, value, v, v,
        v, value, v, v,
        v, value, v, v,
        v, value, v, v,
        v, value, v, v,
        v, value, value, value, v,
        v, value, value, value, v,
        v, value, v, v,
        v, v, value, v, value, v,
        value, v, value, v,
        reg_index);

    fmt("l$u_3:\n"
        "    $v_index_next = add nuw nsw i32 $v_index, 1\n"
        "    $v_cond = icmp eq i32 $v_index_next, $w\n"
        "    br i1 $v_cond, label %l$u_4, label %l$u_1\n\n"
        "l$u_4:\n",
        reg_index,
        v, v,
        v, v,
        v, reg_index, reg_index,
        reg_index);
}
