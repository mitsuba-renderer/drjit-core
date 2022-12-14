/**
 * This file contains the logic that assembles a LLVM IR representation from a
 * recorded Dr.Jit computation graph. It implements a small template engine
 * involving plentiful use of the 'fmt' formatting routine.
 *
 * Its format interface supports the following format string characters. Note
 * that it uses the '$' (dollar) escape character, since '%' is used for LLVM
 * register prefixes (otherwise, lots of escaping would be needed).
 *
 *  Format  Input          Example result    Description
 * --------------------------------------------------------------------------
 *  $u      uint32_t      `1234`             Decimal number (32 bit)
 *  $U      uint64_t      `1234`             Decimal number (64 bit)
 * --------------------------------------------------------------------------
 *  $x      uint32_t      `4d2`              Hexadecimal number (32 bit)
 *  $X      uint64_t      `4d2`              Hexadecimal number (64 bit)
 * --------------------------------------------------------------------------
 *  $s      const char *  `foo`              Zero-terminated string
 * --------------------------------------------------------------------------
 *  $t      Variable      `i1`               Scalar variable type
 *  $T      Variable      `<8 x i1>`         Vector variable type
 *  $h      Variable      `f32`              Short type abbreviation
 * --------------------------------------------------------------------------
 *  $m      Variable      `i8`               Scalar variable type
 *                                           (masks promoted to 8 bits)
 *  $M      Variable      `<8 x i8>`         Vector variable type
 *                                           (masks promoted to 8 bits)
 * --------------------------------------------------------------------------
 *  $v      Variable      `%p1234`           Variable name
 * --------------------------------------------------------------------------
 *  $V      Variable      `<8 x i1> %p1234`  Type-qualified vector var. name
 * --------------------------------------------------------------------------
 *  $a      Variable      `1`                Scalar variable alignment
 *  $A      Variable      `16`               Vector variable alignment
 * --------------------------------------------------------------------------
 *  $o      Variable      `5`                Variable offset in param. array
 * --------------------------------------------------------------------------
 *  $l      Variable      `1`                Literal value of variable
 * --------------------------------------------------------------------------
 *  $w      (none)        `16`               Vector width of LLVM backend
 *  $z      (none)        `zeroinitializer`  Zero initializer string
 * --------------------------------------------------------------------------
 *
 * Pointers should be wrapped in braces, as in `{i8*}` or `{$t*}`. This will
 * allow them to be replaced by the opaque `ptr` type on newer versions of LLVM
 * that use this convention. An extended form of this syntax `{a|b}` causes `a`
 * and `b` to be generated for LLVM with non-opaque and opaque pointers,
 * respectively.
 *
 * Another syntax pattern used in a few places is `$<foo$>`. It expands to
 * `foo` at the top level and `<16 x foo>` when the generated code is part
 * of a subroutine (where 16 is the vector width in this example).
 */

#include "eval.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "vcall.h"
#include "op.h"

#define put(...)                                                               \
    buffer.put(__VA_ARGS__)

#define fmt(fmt, ...)                                                          \
    buffer.fmt_llvm(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)

#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_llvm(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__);          \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0)

// Forward declaration
static void jitc_render_stmt_llvm(uint32_t index, const Variable *v, bool in_function);
static void jitc_render_node_llvm(Variable *v);

void jitc_assemble_llvm(ThreadState *ts, ScheduledGroup group) {
    bool print_labels = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);

    fmt("define void @drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(i64 %start, i64 "
        "%end, {i8**} noalias %params) #0 ${\n"
        "entry:\n"
        "    br label %body\n"
        "\n"
        "body:\n"
        "    %index = phi i64 [ %index_next, %suffix ], [ %start, %entry ]\n");

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        uint32_t vti = v->type;
        VarType vt = (VarType) vti;
        uint32_t size = v->size;

        /// If a variable has a custom code generation hook, call it
        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_assemble_llvm(): internal error: 'extra' entry not found!");

            const Extra &extra = it->second;
            if (print_labels && vt != VarType::Void) {
                const char *label =  jitc_var_label(index);
                if (label && label[0])
                    fmt("    ; $s\n", label);
            }

            if (extra.assemble) {
                extra.assemble(v, extra);
                continue;
            }
        }

        /// Determine source/destination address of input/output parameters
        if (v->param_type == ParamType::Input && size == 1 && vt == VarType::Pointer) {
            // Case 1: load a pointer address from the parameter array
            fmt("    $v_p1 = getelementptr inbounds {i8*}, {i8**} %params, i32 $o\n"
                "    $v = load {i8*}, {i8**} $v_p1, align 8, !alias.scope !2\n",
                v, v, v, v);
        } else if (v->param_type != ParamType::Register) {
            // Case 2: read an input/output parameter

            fmt( "    $v_p1 = getelementptr inbounds {i8*}, {i8**} %params, i32 $o\n"
                 "    $v_p{2|3} = load {i8*}, {i8**} $v_p1, align 8, !alias.scope !2\n"
                "{    $v_p3 = bitcast i8* $v_p2 to $m*\n|}",
                v, v, v, v, v, v, v);

            // For output parameters and non-scalar inputs
            if (v->param_type != ParamType::Input || size != 1)
                fmt( "    $v_p{4|5} = getelementptr inbounds $m, {$m*} $v_p3, i64 %index\n"
                    "{    $v_p5 = bitcast $m* $v_p4 to $M*\n|}",
                    v, v, v, v, v, v, v, v);
        }

        if (likely(v->param_type == ParamType::Input)) {
            if (v->is_literal())
                continue;

            if (size != 1) {
                // Load a packet of values
                fmt("    $v$s = load $M, {$M*} $v_p5, align $A, !alias.scope !2, !nontemporal !3\n",
                    v, vt == VarType::Bool ? "_0" : "", v, v, v, v);
                if (vt == VarType::Bool)
                    fmt("    $v = trunc $M $v_0 to $T\n", v, v, v, v);
            } else {
                // Load a scalar value and broadcast it
                fmt("    $v_0 = load $m, {$m*} $v_p3, align $a, !alias.scope !2\n",
                    v, v, v, v, v);

                if (vt == VarType::Bool)
                    fmt("    $v_1 = trunc i8 $v_0 to i1\n", v, v);

                uint32_t src = vt == VarType::Bool ? 1 : 0,
                         dst = vt == VarType::Bool ? 2 : 1;

                fmt("    $v_$u = insertelement $T undef, $t $v_$u, i32 0\n"
                    "    $v = shufflevector $T $v_$u, $T undef, <$w x i32> $z\n",
                    v, dst, v, v, v, src,
                    v, v, v, dst, v);
            }
        } else if (v->is_literal()) {
            fmt("    $v_1 = insertelement $T undef, $t $l, i32 0\n"
                "    $v = shufflevector $T $v_1, $T undef, <$w x i32> $z\n",
                v, v, v, v,
                v, v, v, v);
        } else if (v->is_node()) {
            jitc_render_node_llvm(v);
        } else {
            jitc_render_stmt_llvm(index, v, false);
        }

        if (v->param_type == ParamType::Output) {
            if (vt != VarType::Bool) {
                fmt("    store $V, {$T*} $v_p5, align $A, !noalias !2, !nontemporal !3\n",
                    v, v, v, v);
            } else {
                fmt("    $v_e = zext $V to $M\n"
                    "    store $M $v_e, {$M*} $v_p5, align $A, !noalias !2, !nontemporal !3\n",
                    v, v, v, v, v, v, v, v);
            }
        }
    }

    put("    br label %suffix\n"
        "\n"
        "suffix:\n");
    fmt("    %index_next = add i64 %index, $w\n");
    put("    %cond = icmp uge i64 %index_next, %end\n"
        "    br i1 %cond, label %done, label %body, !llvm.loop !4\n\n"
        "done:\n"
        "    ret void\n"
        "}\n");

    /* The program requires extra memory or uses callables. Insert
       setup code the top of the function to accomplish this */
    if (callable_count > 0 || alloca_size >= 0) {
        size_t suffix_start = buffer.size(),
               suffix_target = (char *) strchr(buffer.get(), ':') - buffer.get() + 2;

        if (callable_count > 0)
            fmt("    %callables = load {i8**}, {i8***} @callables\n");

        if (alloca_size >= 0)
            fmt("    %buffer = alloca i8, i32 $u, align $u\n",
                alloca_size, alloca_align);

        buffer.move_suffix(suffix_start, suffix_target);
    }

    uint32_t ctr = 0;
    for (auto &it : globals_map) {
        put('\n');
        put(globals.get() + it.second.start, it.second.length);
        put('\n');
        if (!it.first.callable)
            continue;
        it.second.callable_index = 1 + ctr++;
    }

    put("\n"
        "!0 = !{!0}\n"
        "!1 = !{!1, !0}\n"
        "!2 = !{!1}\n"
        "!3 = !{i32 1}\n"
        "!4 = !{!\"llvm.loop.unroll.disable\", !\"llvm.loop.vectorize.enable\", i1 0}\n\n");

    fmt("attributes #0 = ${ norecurse nounwind \"frame-pointer\"=\"none\" "
        "\"no-builtins\" \"no-stack-arg-probe\" \"target-cpu\"=\"$s\" "
        "\"target-features\"=\"", jitc_llvm_target_cpu);

#if !defined(__aarch64__)
    put("-vzeroupper");
    if (jitc_llvm_target_features)
        put(",");
#endif

    if (jitc_llvm_target_features)
        put(jitc_llvm_target_features, strlen(jitc_llvm_target_features));

    put("\" }");

    jitc_vcall_upload(ts);
}

void jitc_assemble_llvm_func(const char *name, uint32_t inst_id,
                             uint32_t in_size, uint32_t data_offset,
                             const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                             uint32_t n_out, const uint32_t *out_nested,
                             bool use_self) {
    bool print_labels = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);
    uint32_t width = jitc_llvm_vector_width, callables_local = callable_count;
    if (use_self)
        fmt("define void @func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^("
                    "<$w x i1> %mask, <$w x i32> %self, {i8*} noalias %params");
    else
        fmt("define void @func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^("
                    "<$w x i1> %mask, {i8*} noalias %params");

    if (!data_map.empty()) {
        if (callable_depth == 1)
            fmt(", {i8*} noalias %data, <$w x i32> %offsets");
        else
            fmt(", <$w x {i8*}> %data, <$w x i32> %offsets");
    }

    fmt(") #0 ${\n"
        "entry:\n"
        "    ; VCall: $s\n", name);

    alloca_size = alloca_align = -1;

    for (ScheduledVariable &sv : schedule) {
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;

        if (unlikely(v->extra)) {
            auto it = state.extra.find(sv.index);
            if (it == state.extra.end())
                jitc_fail("jit_assemble_llvm(): internal error: 'extra' entry "
                          "not found!");

            const Extra &extra = it->second;
            if (print_labels && vt != VarType::Void) {
                const char *label =  jitc_var_label(sv.index);
                if (label && label[0])
                    fmt("    ; $s\n", label);
            }

            if (extra.assemble) {
                extra.assemble(v, extra);
                continue;
            }
        }

        if (v->vcall_iface) {
            fmt( "    $v_i{0|1} = getelementptr inbounds i8, {i8*} %params, i64 $u\n"
                "{    $v_i1 = bitcast i8* $v_i0 to $M*\n|}"
                 "    $v$s = load $M, {$M*} $v_i1, align $A\n",
                v, v->param_offset * width,
                v, v, v,
                v, vt == VarType::Bool ? "_i2" : "", v, v, v, v);

            if (vt == VarType::Bool)
                fmt("    $v = trunc $M $v_i2 to $T\n",
                    v, v, v, v);
        } else if (v->is_data() || vt == VarType::Pointer) {
            uint64_t key = (uint64_t) sv.index + (((uint64_t) inst_id) << 32);
            auto it = data_map.find(key);
            if (unlikely(it == data_map.end()))
                jitc_fail("jitc_assemble_llvm_func(): could not find entry for "
                          "variable r%u in 'data_map'", sv.index);
            if (it->second == (uint32_t) -1)
                jitc_fail(
                    "jitc_assemble_llvm_func(): variable r%u is referenced by "
                    "a recorded function call. However, it was evaluated "
                    "between the recording step and code generation (which is "
                    "happening now). This is not allowed.", sv.index);

            fmt_intrinsic("declare $M @llvm.masked.gather.v$w$h(<$w x {$m*}>, i32, <$w x i1>, $M)",
                          v, v, v, v);

            uint32_t offset = it->second - data_offset;
            // Expand $<..$> only when we are compiling a recursive function call
            callable_depth--;
            fmt( "    $v_p1 = getelementptr inbounds i8, $<{i8*}$> %data, i32 $u\n"
                 "    $v_p2 = getelementptr inbounds i8, $<{i8*}$> $v_p1, <$w x i32> %offsets\n"
                "{    $v_p3 = bitcast <$w x i8*> $v_p2 to <$w x $m*>\n|}"
                 "    $v$s = call $M @llvm.masked.gather.v$w$h(<$w x {$m*}> $v_p{3|2}, i32 $a, <$w x i1> %mask, $M $z)\n",
                v, offset,
                v, v,
                v, v, v,
                v, vt == VarType::Pointer ? "_p4" : "", v, v, v, v, v, v);
            callable_depth++;

            if (vt == VarType::Pointer)
                fmt("    $v = inttoptr <$w x i64> $v_p4 to <$w x {i8*}>\n",
                    v, v);
        } else if (v->is_literal()) {
            fmt("    $v_1 = insertelement $T undef, $t $l, i32 0\n"
                "    $v = shufflevector $T $v_1, $T undef, <$w x i32> $z\n",
                v, v, v, v,
                v, v, v, v);
        } else if (v->is_node()) {
            jitc_render_node_llvm(v);
        } else {
            jitc_render_stmt_llvm(sv.index, v, true);
        }
    }

    uint32_t output_offset = in_size * width;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = out_nested[i];
        if (!index)
            continue;
        const Variable *v = jitc_var(index);
        uint32_t vti = v->type;
        const VarType vt = (VarType) vti;

        fmt( "    %out_$u_{0|1} = getelementptr inbounds i8, {i8*} %params, i64 $u\n"
            "{    %out_$u_1 = bitcast i8* %out_$u_0 to $M*\n|}"
             "    %out_$u_2 = load $M, $M* %out_$u_1, align $A\n",
            i, output_offset,
            i, i, v,
            i, v, v, i, v);

        if (vt == VarType::Bool)
            fmt("    %out_$u_zext = zext $V to $M\n"
                "    %out_$u_3 = select <$w x i1> %mask, $M %out_$u_zext, $M %out_$u_2\n",
                i, v, v,
                i, v, i, v, i);
        else
            fmt("    %out_$u_3 = select <$w x i1> %mask, $V, $T %out_$u_2\n",
                i, v, v, i);

        fmt("    store $M %out_$u_3, {$M*} %out_$u_1, align $A\n",
            v, i, v, i, v);

        output_offset += type_size[vti] * width;
    }

    /* The function requires extra memory or uses callables. Insert
       setup code the top of the function to accomplish this */
    if (alloca_size >= 0 || callables_local != callable_count) {
        size_t suffix_start = buffer.size(),
               suffix_target =
                   (char *) strrchr(buffer.get(), '{') - buffer.get() + 9;

        if (callables_local != callable_count)
            fmt("    %callables = load {i8**}, {i8***} @callables\n");

        if (alloca_size >= 0)
            fmt("    %buffer = alloca i8, i32 $u, align $u\n",
                alloca_size, alloca_align);

        buffer.move_suffix(suffix_start, suffix_target);
    }

    put("    ret void\n"
        "}");
}

inline bool is_float(const Variable *v) {
    VarType type = (VarType) v->type;
    return type == VarType::Float16 ||
           type == VarType::Float32 ||
           type == VarType::Float64;
}

inline bool is_single(const Variable *v) {
    return (VarType) v->type == VarType::Float32;
}

inline bool is_double(const Variable *v) {
    return (VarType) v->type == VarType::Float64;
}

inline bool is_uint(const Variable *v) {
    VarType type = (VarType) v->type;
    return type == VarType::UInt8 ||
           type == VarType::UInt16 ||
           type == VarType::UInt32 ||
           type == VarType::UInt64;
}

static void jitc_render_node_llvm(Variable *v) {
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
             *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;

    switch (v->node) {
        case NodeType::Add:
            fmt(is_float(v) ? "    $v = fadd $V, $v\n"
                            : "    $v = add $V, $v\n",
                v, a0, a1);
            break;

        case NodeType::Gather: {
                bool is_bool = v->type == (uint32_t) VarType::Bool;
                if (is_bool) // Temporary change
                    v->type = (uint32_t) VarType::UInt8;

                fmt_intrinsic(
                    "declare $T @llvm.masked.gather.v$w$h(<$w x {$t*}>, i32, $T, $T)",
                    v, v, v, a2, v);

                fmt("{    $v_0 = bitcast $<i8*$> $v to $<$t*$>\n|}"
                     "    $v_1 = getelementptr $t, {$<$t*$>} {$v_0|$v}, $V\n"
                     "    $v$s = call $T @llvm.masked.gather.v$w$h(<$w x {$t*}> $v_1, i32 $a, $V, $T $z)\n",
                     v, a0, v,
                     v, v, v, v, a0, a1,
                     v, is_bool ? "_2" : "", v, v, v, v, v, a2, v);

                if (is_bool) { // Restore
                    v->type = (uint32_t) VarType::Bool;
                    fmt("    $v = trunc <$w x i8> %b$u_2 to <$w x i1>\n", v, v->reg_index);
                }
            }
            break;

        case NodeType::Scatter: {
                // ptr, value, index, mask
                fmt("{    $v_0 = bitcast $<i8*$> $v to $<$t*$>\n|}"
                     "    $v_1 = getelementptr $t, $<{$t*}$> {$v_0|$v}, $V\n",
                    v, a0, a1,
                    v, a1, a1, v, a0, a2);

                if (!v->payload) {
                    fmt_intrinsic("declare void @llvm.masked.scatter.v$w$h($T, <$w x {$t*}>, i32, $T)",
                         a1, a1, a1, a3);
                    fmt("    call void @llvm.masked.scatter.v$w$h($V, <$w x {$t*}> $v_1, i32 $a, $V)\n",
                         a1, a1, a1, v, a1, a3);
                } else {
                    const char *op, *zero_elem = nullptr, *intrinsic_name = nullptr;
                    switch ((ReduceOp) v->payload) {
                        case ReduceOp::Add:
                            op = "fadd";
                            if (is_single(a1)) {
                                zero_elem = "float -0.0, ";
                                intrinsic_name = "v2.fadd.f32";
                            } else if (is_double(a1)) {
                                zero_elem = "double -0.0, ";
                                intrinsic_name = "v2.fadd.f64";
                            } else {
                                op = "add";
                            }
                            break;

                        case ReduceOp::Mul:
                            op = "fmul";
                            if (is_single(a1)) {
                                zero_elem = "float -0.0, ";
                                intrinsic_name = "v2.fmul.f32";
                            } else if (is_double(a1)) {
                                zero_elem = "double -0.0, ";
                                intrinsic_name = "v2.fmul.f64";
                            } else {
                                op = "mul";
                            }
                            break;

                        case ReduceOp::Min:
                            op = is_float(a1) ? "fmin" : (is_uint(a1) ? "umin" : "smin");
                            break;

                        case ReduceOp::Max:
                            op = is_float(a1) ? "fmax" : (is_uint(a1) ? "umax" : "smax");
                            break;

                        case ReduceOp::And: op = "and"; break;
                        case ReduceOp::Or : op = "or"; break;
                        default: op = nullptr;
                    }

                    if (!intrinsic_name)
                        intrinsic_name = op;

                    fmt_intrinsic("declare i1 @llvm.experimental.vector.reduce.or.v$wi1(<$w x i1>)");

                    if (zero_elem)
                        fmt_intrinsic(
                            "declare $t @llvm.experimental.vector.reduce.$s.v$w$h($t, $T)", a1,
                            intrinsic_name, a1, a1, a1);
                    else
                        fmt_intrinsic("declare $t @llvm.experimental.vector.reduce.$s.v$w$h($T)",
                                      a1, op, a1, a1);

                    const char *reassoc = is_float(a1) ? "reassoc " : "";

                    fmt_intrinsic(
                        "define internal void @reduce_$s_$h(<$w x {$t*}> %ptr, $T %value, <$w x i1> %active_in) #0 ${\n"
                        "L0:\n"
                        "   br label %L1\n\n"
                        "L1:\n"
                        "   %index = phi i32 [ 0, %L0 ], [ %index_next, %L3 ]\n"
                        "   %active = phi <$w x i1> [ %active_in, %L0 ], [ %active_next_2, %L3 ]\n"
                        "   %active_i = extractelement <$w x i1> %active, i32 %index\n"
                        "   br i1 %active_i, label %L2, label %L3\n\n"
                        "L2:\n"
                        "   %ptr_0 = extractelement <$w x {$t*}> %ptr, i32 %index\n"
                        "   %ptr_1 = insertelement <$w x {$t*}> undef, {$t*} %ptr_0, i32 0\n"
                        "   %ptr_2 = shufflevector <$w x {$t*}> %ptr_1, <$w x {$t*}> undef, <$w x i32> $z\n"
                        "   %ptr_eq = icmp eq <$w x {$t*}> %ptr, %ptr_2\n"
                        "   %active_cur = and <$w x i1> %ptr_eq, %active\n"
                        "   %value_cur = select <$w x i1> %active_cur, $T %value, $T $z\n"
                        "   %sum = call $s$t @llvm.experimental.vector.reduce.$s.v$w$h($s$T %value_cur)\n"
                        "   atomicrmw $s {$t*} %ptr_0, $t %sum monotonic\n"
                        "   %active_next = xor <$w x i1> %active, %active_cur\n"
                        "   %active_red = call i1 @llvm.experimental.vector.reduce.or.v$wi1(<$w x i1> %active_next)\n"
                        "   br i1 %active_red, label %L3, label %L4\n\n"
                        "L3:\n"
                        "   %active_next_2 = phi <$w x i1> [ %active, %L1 ], [ %active_next, %L2 ]\n"
                        "   %index_next = add nuw nsw i32 %index, 1\n"
                        "   %cond_2 = icmp eq i32 %index_next, $w\n"
                        "   br i1 %cond_2, label %L4, label %L1\n\n"
                        "L4:\n"
                        "   ret void\n"
                        "$}",
                        op, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, reassoc,
                        a1, intrinsic_name, a1, zero_elem ? zero_elem : "", a1, op, a1, a1
                    );

                    fmt("    call void @reduce_$s_$h(<$w x {$t*}> $v_1, $V, $V)\n",
                        op, a1, a1, v, a1, a3);
                }
            }
            break;

        default:
            jitc_fail("jitc_render_node_llvm(): unhandled node type!");
    }
}

/// Convert an IR template with '$' expressions into valid IR
static void jitc_render_stmt_llvm(uint32_t index, const Variable *v, bool in_function) {
    const char *s = v->stmt;
    if (unlikely(!s || *s == '\0'))
        return;
    put("    ");
    char c;
    size_t intrinsic_start = 0;
    do {
        const char *start = s;
        while (c = *s, c != '\0' && c != '$')
            s++;
        put(start, s - start);

        if (c == '$') {
            s++;
            const char **prefix_table = nullptr, tname = *s++;
            switch (tname) {
                case '[':
                    intrinsic_start = buffer.size();
                    continue;

                case ']':
                    jitc_register_global(buffer.get() + intrinsic_start);
                    buffer.rewind_to(intrinsic_start);
                    continue;

                case 'n': put("\n    "); continue;
                case 'w': put(jitc_llvm_vector_width_str,
                                     strlen(jitc_llvm_vector_width_str)); continue;
                case 't': prefix_table = type_name_llvm; break;
                case 'T': prefix_table = type_name_llvm_big; break;
                case 'b': prefix_table = type_name_llvm_bin; break;
                case 'a': prefix_table = type_name_llvm_abbrev; break;
                case 's': prefix_table = type_size_str; break;
                case 'r': prefix_table = type_prefix; break;
                case 'i': prefix_table = nullptr; break;
                case '<': if (in_function) {
                              put('<');
                              put(jitc_llvm_vector_width_str,
                                         strlen(jitc_llvm_vector_width_str));
                              put(" x ");
                           }
                           continue;
                case '>': if (in_function)
                              put('>');
                           continue;
                case 'o': prefix_table = (const char **) jitc_llvm_ones_str; break;
                default:
                    jitc_fail("jit_render_stmt_llvm(): encountered invalid \"$\" "
                              "expression (unknown character \"%c\") in \"%s\"!", tname, v->stmt);
            }

            uint32_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 4))
                jitc_fail("jit_render_stmt_llvm(%s): encountered invalid \"$\" "
                          "expression (argument out of bounds)!", v->stmt);

            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            if (unlikely(dep_id == 0))
                jitc_fail("jit_render_stmt_llvm(%s): encountered invalid \"$\" "
                          "expression (referenced variable %u is missing)!", v->stmt, arg_id);

            const Variable *dep = jitc_var(dep_id);
            if (likely(prefix_table)) {
                const char *prefix = prefix_table[(int) dep->type];
                put(prefix, strlen(prefix));
            }

            if (tname == 'r' || tname == 'i')
                buffer.put_u32(dep->reg_index);
        }
    } while (c != '\0');

    put('\n');
}

static void jitc_llvm_ray_trace_assemble(const Variable *v, const Extra &extra);

void jitc_llvm_ray_trace(uint32_t func, uint32_t scene, int shadow_ray,
                         const uint32_t *in, uint32_t *out) {
    const uint32_t n_args = 14;
    bool double_precision = ((VarType) jitc_var(in[2])->type) == VarType::Float64;
    VarType float_type = double_precision ? VarType::Float64 : VarType::Float32;

    VarType types[]{ VarType::Bool,   VarType::Bool,  float_type,
                     float_type,      float_type,     float_type,
                     float_type,      float_type,     float_type,
                     float_type,      float_type,     VarType::UInt32,
                     VarType::UInt32, VarType::UInt32 };

    bool placeholder = false, dirty = false;
    uint32_t size = 0;
    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *v = jitc_var(in[i]);
        if ((VarType) v->type != types[i])
            jitc_raise("jitc_llvm_ray_trace(): type mismatch for arg. %u (got %s, "
                       "expected %s)",
                       i, type_name[v->type], type_name[(int) types[i]]);
        size = std::max(size, v->size);
        placeholder |= (bool) v->placeholder;
        dirty |= v->is_dirty();
    }

    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jitc_llvm_ray_trace(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if ((VarType) jitc_var(func)->type != VarType::Pointer ||
        (VarType) jitc_var(scene)->type != VarType::Pointer)
        jitc_raise("jitc_llvm_ray_trace(): 'func', and 'scene' must be pointer variables!");

    if (dirty) {
        jitc_eval(thread_state(JitBackend::LLVM));
        dirty = false;

        for (uint32_t i = 0; i < n_args; ++i)
            dirty |= jitc_var(in[i])->is_dirty();

        if (dirty)
            jitc_raise(
                "jit_llvm_ray_trace(): inputs remain dirty after evaluation!");
    }

    // ----------------------------------------------------------
    Ref valid = steal(jitc_var_mask_apply(in[1], size));
    {
        int32_t minus_one_c = -1, zero_c = 0;
        Ref minus_one = steal(jitc_var_new_literal(
                JitBackend::LLVM, VarType::Int32, &minus_one_c, 1, 0)),
            zero = steal(jitc_var_new_literal(
                JitBackend::LLVM, VarType::Int32, &zero_c, 1, 0));

        uint32_t deps_2[3] = { valid, minus_one, zero };
        valid = steal(jitc_var_new_op(JitOp::Select, 3, deps_2));
    }

    jitc_log(InfoSym, "jitc_llvm_ray_trace(): tracing %u %sray%s%s%s", size,
             shadow_ray ? "shadow " : "", size != 1 ? "s" : "",
             placeholder ? " (part of a recorded computation)" : "",
             double_precision ? " (double precision)" : "");

    Ref op = steal(jitc_var_new_stmt_n(JitBackend::LLVM, VarType::Void,
                               shadow_ray ? "// Ray trace (shadow ray)"
                                          : "// Ray trace",
                               1, func, scene));
    Variable *v_op = jitc_var(op);
    v_op->size = size;
    v_op->extra = 1;

    Extra &e = state.extra[op];
    e.dep = (uint32_t *) malloc_check(sizeof(uint32_t) * n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        uint32_t index = i != 1 ? in[i] : valid;
        jitc_var_inc_ref(index);
        e.dep[i] = index;
    }
    e.n_dep = n_args;
    e.assemble = jitc_llvm_ray_trace_assemble;

    char tmp[128];
    for (int i = 0; i < (shadow_ray ? 1 : 6); ++i) {
        snprintf(tmp, sizeof(tmp),
                 "$r0 = bitcast <$w x $t0> $r1_out_%u to <$w x $t0>", i);
        VarType vt = (i < 3) ? float_type : VarType::UInt32;
        out[i] = jitc_var_new_stmt_n(JitBackend::LLVM, vt, tmp, 0, op);
    }
}

static void jitc_llvm_ray_trace_assemble(const Variable *v, const Extra &extra) {
    const uint32_t width = jitc_llvm_vector_width;
    const uint32_t id = v->reg_index;
    bool shadow_ray = strstr(v->stmt, "(shadow ray)") != nullptr;
    bool double_precision = ((VarType) jitc_var(extra.dep[2])->type) == VarType::Float64;
    VarType float_type = double_precision ? VarType::Float64 : VarType::Float32;
    uint32_t float_size = double_precision ? 8 : 4;

    uint32_t ctx_size = 6 * 4, alloca_size_rt;

    if (shadow_ray)
        alloca_size_rt = (9 * float_size + 4 * 4) * width;
    else
        alloca_size_rt = (14 * float_size + 7 * 4) * width;

    alloca_size  = std::max(alloca_size, (int32_t) (alloca_size_rt + ctx_size));
    alloca_align = std::max(alloca_align, (int32_t) (float_size * width));

    /* Offsets:
        0  bool coherent
        1  uint32_t valid
        1  float org_x
        2  float org_y
        3  float org_z
        4  float tnear
        5  float dir_x
        6  float dir_y
        7  float dir_z
        8  float time
        9  float tfar
        10 uint32_t mask
        11 uint32_t id
        12 uint32_t flags
        13 float Ng_x
        14 float Ng_y
        15 float Ng_z
        16 float u
        17 float v
        18 uint32_t primID
        19 uint32_t geomID
        20 uint32_t instID[] */
    fmt("\n    ; -------- Ray $s -------\n", shadow_ray ? "test" : "trace");

    uint32_t offset = 0;
    for (int i = 0; i < 13; ++i) {
        if (jitc_llvm_vector_width == 1 && i == 0)
            continue; // valid flag not needed for 1-lane versions

        const Variable *v2 = jitc_var(extra.dep[i + 1]);
        fmt( "    %u$u_in_$u_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    %u$u_in_$u_1 = bitcast i8* %u$u_in_$u_0 to $T*\n|}"
             "    store $V, {$T*} %u$u_in_$u_1, align $A\n",
             id, i, offset,
             id, i, id, i, v2,
             v2, v2, v2);

        offset += type_size[v2->type] * width;
    }

    if (!shadow_ray) {
        fmt( "    %u$u_in_geomid_0 = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    %u$u_in_geomid_1 = bitcast i8* %u$u_in_geomid_0 to <$w x i32> *\n|}"
             "    store <$w x i32> $s, {<$w x i32>*} %u$u_in_geomid_1, align $u\n",
            id, (14 * float_size + 5 * 4) * width,
            id, id,
            jitc_llvm_ones_str[(int) VarType::Int32], id, float_size * width);
    }

    const Variable *coherent = jitc_var(extra.dep[0]);

    fmt( "    %u$u_in_ctx_0 = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
        "{    %u$u_in_ctx_1 = bitcast i8* %u$u_in_ctx_0 to <6 x i32>*\n|}",
        id, alloca_size_rt, id, id);

    if (coherent->is_literal() && coherent->literal == 0) {
        fmt("    store <6 x i32> <i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0>, {<6 x i32>*} %u$u_in_ctx_1, align 4\n", id);
    } else if (coherent->is_literal() && coherent->literal == 1) {
        fmt("    store <6 x i32> <i32 1, i32 0, i32 0, i32 0, i32 -1, i32 0>, {<6 x i32>*} %u$u_in_ctx_1, align 4\n", id);
    } else {
        fmt_intrinsic("declare i1 @llvm.experimental.vector.reduce.and.v$wi1(<$w x i1>)");

        fmt("    %u$u_coherent = call i1 @llvm.experimental.vector.reduce.and.v$wi1($V)\n"
            "    %u$u_ctx = select i1 %u$u_coherent, <6 x i32> <i32 1, i32 0, i32 0, i32 0, i32 -1, i32 0>, "
                                                    "<6 x i32> <i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0>\n"
            "    store <6 x i32> %u$u_ctx, {<6 x i32>*} %u$u_in_ctx_1, align 4\n",
            id, coherent,
            id, id,
            id, id);
    }

    const Variable *func  = jitc_var(v->dep[0]),
                   *scene = jitc_var(v->dep[1]);

    /* When the ray tracing operation occurs inside a recorded function, it's
       possible that multiple different scenes are being traced simultaneously.
       In that case, it's necessary to perform one ray tracing call per scene,
       which is implemented by the following loop. */
    if (callable_depth == 0) {
        if (jitc_llvm_vector_width > 1) {
            fmt("{    %u$u_func = bitcast i8* $v to void (i8*, i8*, i8*, i8*)*\n|}"
                 "    call void {%u$u_func|$v}({i8*} %u$u_in_0_0, {i8*} $v, {i8*} %u$u_in_ctx_0, {i8*} %u$u_in_1_0)\n",
                id, func,
                id, func, id, scene, id, id
            );
        } else {
            fmt(
                "{    %u$u_func = bitcast i8* $v to void (i8*, i8*, i8*)*\n|}"
                "     call void {%u$u_func|$v}({i8*} $v, {i8*} %u$u_in_ctx_0, {i8*} %u$u_in_1_0)\n",
                id, func,
                id, func, scene, id, id
            );
        }
    } else {
        fmt_intrinsic("declare i64 @llvm.experimental.vector.reduce.umax.v$wi64(<$w x i64>)");

        uint32_t offset_tfar = (8 * float_size + 4) * width;
        const char *tname_tfar = type_name_llvm[(int) float_type];

        // =====================================================
        // 1. Prepare the loop for the ray tracing calls
        // =====================================================

        fmt( "    br label %l$u_start\n"
             "\nl$u_start:\n"
             "    ; Ray tracing\n"
             "    %u$u_func_i64 = call i64 @llvm.experimental.vector.reduce.umax.v$wi64(<$w x i64> %rd$u_p4)\n"
             "    %u$u_func_ptr = inttoptr i64 %u$u_func_i64 to {i8*}\n"
             "    %u$u_tfar_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    %u$u_tfar_1 = bitcast i8* %u$u_tfar_0 to <$w x $s> *\n|}",
            id, id,
            id, func->reg_index,
            id, id,
            id, offset_tfar,
            id, id, tname_tfar);

        if (jitc_llvm_vector_width > 1) {
            fmt("    %u$u_func = bitcast {i8*} %u$u_func_ptr to {void (i8*, i8*, i8*, i8*)*}\n",
                id, id);
        } else {
            fmt("    %u$u_func = bitcast {i8*} %u$u_func_ptr to {void (i8*, i8*, i8*)*}\n",
                id, id);
        }

        // Get original mask, to be overwritten at every iteration
        fmt("    %u$u_mask_value = load <$w x i32>, {<$w x i32>*} %u$u_in_0_1, align 64\n"
            "    br label %l$u_check\n",
            id, id, id);

        // =====================================================
        // 2. Move on to the next instance & check if finished
        // =====================================================

        fmt("\nl$u_check:\n"
            "    %u$u_scene = phi <$w x {i8*}> [ %rd$u, %l$u_start ], [ %u$u_scene_next, %l$u_call ]\n"
            "    %u$u_scene_i64 = ptrtoint <$w x {i8*}> %u$u_scene to <$w x i64>\n"
            "    %u$u_next_i64 = call i64 @llvm.experimental.vector.reduce.umax.v%wi64(<$w x i64> %u$u_scene_i64)\n"
            "    %u$u_next = inttoptr i64 %u$u_next_i64 to {i8*}\n"
            "    %u$u_valid = icmp ne {i8*} %u$u_next, null\n"
            "    br i1 %u$u_valid, label %l$u_call, label %l$u_end\n",
            id,
            id, scene->reg_index, id, id, id,
            id, id,
            id, id,
            id, id,
            id, id,
            id, id, id);

        // =====================================================
        // 3. Perform ray tracing call to each unique instance
        // =====================================================

        fmt("\nl$u_call:\n"
            "    %u$u_tfar_prev = load <$w x $s>, {<$w x $s>*} %u$u_tfar_1, align $u\n"
            "    %u$u_bcast_0 = insertelement <$w x i64> undef, i64 %u$u_next_i64, i32 0\n"
            "    %u$u_bcast_1 = shufflevector <$w x i64> %u$u_bcast_0, <$w x i64> undef, <$w x i32> $z\n"
            "    %u$u_bcast_2 = inttoptr <$w x i64> %u$u_bcast_1 to <$w x {i8*}>\n"
            "    %u$u_active = icmp eq <$w x {i8*}> %u$u_scene, %u$u_bcast_2\n"
            "    %u$u_active_2 = select <$w x i1> %u$u_active, <$w x i32> %u$u_mask_value, <$w x i32> $z\n"
            "    store <$u x i32> %u$u_active_2, {<$w x i32>*} %u$u_in_0_1, align 64\n",
            id,
            id, tname_tfar, tname_tfar, id, float_size * width,
            id, id,
            id, id,
            id, id,
            id, id, id,
            id, id, id,
            id, id);

        if (jitc_llvm_vector_width > 1) {
            fmt("    call void %u$u_func({i8*} %u$u_in_0_0, {i8*} %u$u_next, {i8*} %u$u_in_ctx_0, {i8*} %u$u_in_1_0)\n",
                id, id, id, id, id);
        } else {
            fmt("    call void %u$u_func({i8*} %u$u_next, {i8*} %u$u_in_ctx_0, {i8*} %u$u_in_1_0)\n",
                id, id, id, id);
        }

        fmt("    %u$u_tfar_new = load <$w x $s>, {<$w x $s>*} %u$u_tfar_1, align $u\n"
            "    %u$u_tfar_masked = select <$w x i1> %u$u_active, <$w x $s> %u$u_tfar_new, <$w x $s> %u$u_tfar_prev\n"
            "    store <$w x $s> %u$u_tfar_masked, {<$w x $s>*} %u$u_tfar_1, align $u\n"
            "    %u$u_scene_next = select <$w x i1> %u$u_active, <$w x {i8*}> $z, <$w x {i8*}> %u$u_scene\n"
            "    br label %l$u_check\n"
            "\nl%u_end:\n",
            id, tname_tfar, tname_tfar, id, float_size * width,
            id, id, tname_tfar, id, tname_tfar, id,
            width, tname_tfar, id, tname_tfar, id, float_size * width,
            id, id, id,
            id, id);
    }

    offset = (8 * float_size + 4) * width;

    for (int i = 0; i < (shadow_ray ? 1 : 6); ++i) {
        VarType vt = (i < 3) ? float_type : VarType::UInt32;
        const char *tname = type_name_llvm[(int) vt];
        uint32_t tsize = type_size[(int) vt];
        fmt( "    %u$u_out_%u_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    %u$u_out_%u_1 = bitcast i8* %u$u_out_%u_0 to <$w x $s> *\n|}"
             "    %u$u_out_%u = load <$w x $s>, {<$w x $s>*} %u$u_out_%u_1, align $u\n",
            id, i, offset,
            id, i, id, i, tname,
            id, i, tname, tname, id, i, float_size * width);

        if (i == 0)
            offset += (4 * float_size + 3 * 4) * width;
        else
            offset += tsize * width;
    }

    put("    ; -------------------\n\n");
}


/// Virtual function call code generation -- LLVM IR-specific bits
void jitc_var_vcall_assemble_llvm(VCall *vcall, uint32_t vcall_reg,
                                  uint32_t self_reg, uint32_t mask_reg,
                                  uint32_t offset_reg, uint32_t data_reg,
                                  uint32_t n_out, uint32_t in_size,
                                  uint32_t in_align, uint32_t out_size,
                                  uint32_t out_align) {

    uint32_t width = jitc_llvm_vector_width;
    alloca_size  = std::max(alloca_size, (int32_t) ((in_size + out_size) * width));
    alloca_align = std::max(alloca_align, (int32_t) (std::max(in_align, out_align) * width));

    // =====================================================
    // 1. Declare a few intrinsics that we will use
    // =====================================================

    fmt_intrinsic("@callables = dso_local local_unnamed_addr global {i8**} null, align 8");

    /* How to prevent @callables from being optimized away as a constant, while
       at the same time not turning it an external variable that would require a
       global offset table (GOT)? Let's make a dummy function that writes to it.. */
    fmt_intrinsic("define void @set_callables({i8**} %ptr) local_unnamed_addr #0 ${\n"
                  "    store {i8**} %ptr, {i8***} @callables\n"
                  "    ret void\n"
                  "$}");

    fmt_intrinsic("declare i32 @llvm.experimental.vector.reduce.umax.v$wi32(<$w x i32>)");
    fmt_intrinsic("declare <$w x i64> @llvm.masked.gather.v$wi64(<$w x {i64*}>, i32, <$w x i1>, <$w x i64>)");

    fmt( "\n"
         "    br label %l$u_start\n"
         "\nl$u_start:\n"
         "    ; VCall: $s\n"
        "{    %u$u_self_ptr_0 = bitcast $<i8*$> %rd$u to $<i64*$>\n|}"
         "    %u$u_self_ptr = getelementptr i64, $<{i64*}$> {%u$u_self_ptr_0|%rd$u}, <$w x i32> %r$u\n"
         "    %u$u_self_combined = call <$w x i64> @llvm.masked.gather.v$wi64(<$w x i64*> %u$u_self_ptr, i32 8, <$w x i1> %p$u, <$w x i64> $z)\n"
         "    %u$u_self_initial = trunc <$w x i64> %u$u_self_combined to <$w x i32>\n",
        vcall_reg, vcall_reg, vcall->name,
        vcall_reg, offset_reg,
        vcall_reg, vcall_reg, offset_reg, self_reg,
        vcall_reg, vcall_reg, mask_reg,
        vcall_reg, vcall_reg);

    if (data_reg) {
        fmt("    %u$u_offset_1 = lshr <$w x i64> %u$u_self_combined, <",
                 vcall_reg, vcall_reg);
        for (uint32_t i = 0; i < width; ++i)
            fmt("i64 32$s", i + 1 < width ? ", " : "");
        fmt(">\n"
            "    %u$u_offset = trunc <$w x i64> %u$u_offset_1 to <$w x i32>\n",
            vcall_reg, vcall_reg);
    }

    // =====================================================
    // 2. Pass the input arguments
    // =====================================================

    uint32_t offset = 0;
    for (uint32_t i = 0; i < (uint32_t) vcall->in.size(); ++i) {
        uint32_t index = vcall->in[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;

        fmt(
             "    %u$u_in_$u_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    %u$u_in_$u_1 = bitcast i8* %u$u_in_$u_0 to $M*\n|}",
            vcall_reg, i, offset,
            vcall_reg, i, vcall_reg, i, v2
        );

        if ((VarType) v2->type != VarType::Bool) {
            fmt("    store $V, {$T*} %u$u_in_$u_1, align $A\n",
                v2, v2, vcall_reg, i, v2);
        } else {
            fmt("    %u$u_$u_zext = zext $V to $M\n"
                "    store $M %u$u_$u_zext, {$M*} %u$u_in_$u_1, align $A\n",
                vcall_reg, i, v2, v2,
                v2, vcall_reg, i, v2, vcall_reg, i, v2);
        }

        offset += type_size[v2->type] * width;
    }

    if (out_size)
        fmt("    %u$u_out = getelementptr i8, {i8*} %buffer, i32 $u\n",
            vcall_reg, in_size * width);

    offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;

        fmt( "    %u$u_tmp_$u_{0|1} = getelementptr inbounds i8, {i8*} %u$u_out, i64 $u\n"
            "{    %u$u_tmp_$u_1 = bitcast i8* %u$u_tmp_$u_0 to $M*\n|}"
             "    store $M $z, {$M*} %u$u_tmp_$u_1, align $A\n",
            vcall_reg, i, vcall_reg, offset,
            vcall_reg, i, vcall_reg, i, v2,
            v2, v2, vcall_reg, i, v2);

        offset += type_size[v2->type] * width;
    }

    // =====================================================
    // 3. Perform one call to each unique instance
    // =====================================================

    fmt("    br label %l$u_check\n"
        "\nl$u_check:\n"
        "    %u$u_self = phi <$w x i32> [ %u$u_self_initial, %l$u_start ], [ %u$u_self_next, %l$u_call ]\n",
        vcall_reg,
        vcall_reg,
        vcall_reg, vcall_reg, vcall_reg, vcall_reg, vcall_reg);

    fmt("    %u$u_next = call i32 @llvm.experimental.vector.reduce.umax.v$wi32(<$w x i32> %u$u_self)\n"
        "    %u$u_valid = icmp ne i32 %u$u_next, 0\n"
        "    br i1 %u$u_valid, label %l$u_call, label %l$u_end\n",
        vcall_reg, vcall_reg,
        vcall_reg, vcall_reg,
        vcall_reg, vcall_reg, vcall_reg);

    fmt("\nl$u_call:\n"
        "    %u$u_bcast_0 = insertelement <$w x i32> undef, i32 %u$u_next, i32 0\n"
        "    %u$u_bcast = shufflevector <$w x i32> %u$u_bcast_0, <$w x i32> undef, <$w x i32> $z\n"
        "    %u$u_active = icmp eq <$w x i32> %u$u_self, %u$u_bcast\n"
        "    %u$u_func_0 = getelementptr inbounds {i8*}, {i8**} %callables, i32 %u$u_next\n"
        "    %u$u_func{_1|} = load {i8*}, {i8**} %u$u_func_0\n",
        vcall_reg,
        vcall_reg, vcall_reg, // bcast_0
        vcall_reg, vcall_reg, // bcast
        vcall_reg, vcall_reg, vcall_reg, // active
        vcall_reg, vcall_reg, // func_0
        vcall_reg, vcall_reg // func_1
    );

    // Cast into correctly typed function pointer
    if (!jitc_llvm_opaque_pointers) {
        fmt("    %u$u_func = bitcast i8* %u$u_func_1 to void (<$w x i1>",
                 vcall_reg, vcall_reg);

        if (vcall->use_self)
            fmt(", <$w x i32>");

        fmt(", i8*");
        if (data_reg)
            fmt(", $<i8*$>, <$w x i32>");

        fmt(")*\n");
    }

    // Perform the actual function call
    fmt("    call void %u$u_func(<$w x i1> %u$u_active",
        vcall_reg, vcall_reg);

    if (vcall->use_self)
        fmt(", <$w x i32> %r$u", self_reg);

    fmt(", {i8*} %buffer");

    if (data_reg)
        fmt(", $<{i8*}$> %rd$u, <$w x i32> %u$u_offset", data_reg, vcall_reg);

    fmt(")\n"
        "    %u$u_self_next = select <$w x i1> %u$u_active, <$w x i32> $z, <$w x i32> %u$u_self\n"
        "    br label %l$u_check\n"
        "\nl$u_end:\n",
        vcall_reg, vcall_reg, vcall_reg, vcall_reg,
        vcall_reg);

    // =====================================================
    // 5. Read back the output arguments
    // =====================================================

    offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i],
                 index_2 = vcall->out[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        uint32_t size = type_size[it->second.type],
                 load_offset = offset;
        offset += size * width;

        // Skip if outer access expired
        auto it2 = state.variables.find(index_2);
        if (it2 == state.variables.end())
            continue;

        const Variable *v2 = &it2.value();
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        VarType vt = (VarType) v2->type;

        fmt( "    %u$u_out_$u_{0|1} = getelementptr inbounds i8, {i8*} %u$u_out, i64 $u\n"
            "{    %u$u_out_$u_1 = bitcast i8* %u$u_out_$u_0 to $M*\n|}"
             "    $v$s = load $M, {$M*} %u$u_out_$u_1, align $A\n",
            vcall_reg, i, vcall_reg, load_offset,
            vcall_reg, i, vcall_reg, i, v2,
            v2, vt == VarType::Bool ? "_0" : "", v2, v2, vcall_reg, i, v2);

            if (vt == VarType::Bool)
                fmt("    $v = trunc $M $v_0 to $T\n", v2, v2, v2, v2);
    }

    fmt("    br label %l$u_done\n"
        "\nl$u_done:\n", vcall_reg, vcall_reg);
}
