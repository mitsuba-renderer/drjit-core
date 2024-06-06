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
 *  $t      Variable      `float`            Scalar variable type
 *  $T      Variable      `<8 x float>`      Vector variable type
 *  $h      Variable      `f32`              Type abbreviation for intrinsics
 * --------------------------------------------------------------------------
 *  $b      Variable      `i32`              Scalar variable type (as int)
 *  $B      Variable      `<8 x i32>`        Vector variable type (as int)
 * --------------------------------------------------------------------------
 *  $d      Variable      `i64`              Double-size variable type
 *  $D      Variable      `<8 x i64>`        Vector double-size variable type
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
 *  $a      Variable      `4`                Scalar variable alignment
 *  $A      Variable      `64`               Vector variable alignment
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
static void jitc_llvm_render_stmt(uint32_t index, const Variable *v, bool in_function);
static void jitc_llvm_render_var(uint32_t index, Variable *v);
static void jitc_llvm_render_scatter(const Variable *v, const Variable *ptr,
                                     const Variable *value, const Variable *index,
                                     const Variable *mask);
static void jitc_llvm_render_scatter_kahan(const Variable *v, uint32_t index);
static void jitc_llvm_render_scatter_inc(Variable *v,
                                         const Variable *ptr,
                                         const Variable *index,
                                         const Variable *mask);
static void jitc_llvm_render_printf(uint32_t index, const Variable *v,
                                    const Variable *mask, const Variable *target);
static void jitc_llvm_render_trace(uint32_t index, const Variable *v,
                                   const Variable *func,
                                   const Variable *scene);

void jitc_llvm_assemble(ThreadState *ts, ScheduledGroup group) {
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
        } else if (!v->is_stmt()) {
            jitc_llvm_render_var(index, v);
        } else {
            jitc_llvm_render_stmt(index, v, false);
        }

        v = jitc_var(index); // `v` might have been invalidated during its assembly

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
            fmt("    %callables = load {i8**}, {i8***} @callables, align 8\n");

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

void jitc_llvm_assemble_func(const char *name, uint32_t inst_id,
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
                jitc_fail("jitc_llvm_assemble_func(): could not find entry for "
                          "variable r%u in 'data_map'", sv.index);
            if (it->second == (uint32_t) -1)
                jitc_fail(
                    "jitc_llvm_assemble_func(): variable r%u is referenced by "
                    "a recorded function call. However, it was evaluated "
                    "between the recording step and code generation (which is "
                    "happening now). This is not allowed.", sv.index);

            fmt_intrinsic("declare $M @llvm.masked.gather.v$w$h(<$w x {$m*}>, i32, <$w x i1>, $M)",
                          v, v, v, v);

            uint32_t offset = it->second - data_offset;
            bool is_pointer_or_bool =
                (vt == VarType::Pointer) || (vt == VarType::Bool);
            // Expand $<..$> only when we are compiling a recursive function call
            callable_depth--;
            fmt( "    $v_p1 = getelementptr inbounds i8, $<{i8*}$> %data, i32 $u\n"
                 "    $v_p2 = getelementptr inbounds i8, $<{i8*}$> $v_p1, <$w x i32> %offsets\n"
                "{    $v_p3 = bitcast <$w x i8*> $v_p2 to <$w x $m*>\n|}"
                 "    $v$s = call $M @llvm.masked.gather.v$w$h(<$w x {$m*}> $v_p{3|2}, i32 $a, <$w x i1> %mask, $M $z)\n",
                v, offset,
                v, v,
                v, v, v,
                v, is_pointer_or_bool ? "_p4" : "", v, v, v, v, v, v);
            callable_depth++;

            if (vt == VarType::Pointer)
                fmt("    $v = inttoptr <$w x i64> $v_p4 to <$w x {i8*}>\n",
                    v, v);
            else if (vt == VarType::Bool)
                fmt("    $v = trunc <$w x i8> $v_p4 to <$w x i1>\n",
                    v, v);
        } else if (!v->is_stmt()) {
            jitc_llvm_render_var(sv.index, v);
        } else {
            jitc_llvm_render_stmt(sv.index, v, true);
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
             "    %out_$u_2 = load $M, {$M*} %out_$u_1, align $A\n",
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
            fmt("    %callables = load {i8**}, {i8***} @callables, align 8\n");

        if (alloca_size >= 0)
            fmt("    %buffer = alloca i8, i32 $u, align $u\n",
                alloca_size, alloca_align);

        buffer.move_suffix(suffix_start, suffix_target);
    }

    put("    ret void\n"
        "}");
}

static void jitc_llvm_render_var(uint32_t index, Variable *v) {
    const char *stmt = nullptr;
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
             *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;

    switch (v->kind) {
        case VarKind::Literal:
            fmt("    $v_1 = insertelement $T undef, $t $l, i32 0\n"
                "    $v = shufflevector $T $v_1, $T undef, <$w x i32> $z\n",
                v, v, v, v,
                v, v, v, v);
            break;

        case VarKind::Nop:
            break;

        case VarKind::Neg:
            if (jitc_is_float(v))
                fmt("    $v = fneg $V\n", v, a0);
            else
                fmt("    $v = sub $T $z, $v\n", v, v, a0);
            break;

        case VarKind::Not:
            fmt("    $v = xor $V, $s\n", v, a0, jitc_llvm_ones_str[v->type]);
            break;

        case VarKind::Sqrt:
            fmt_intrinsic("declare $T @llvm.sqrt.v$w$h($T)", v, v, a0);
            fmt("    $v = call $T @llvm.sqrt.v$w$h($V)\n", v, v, v, a0);
            break;

        case VarKind::Abs:
            if (jitc_is_float(v)) {
                fmt_intrinsic("declare $T @llvm.fabs.v$w$h($T)", v, v, a0);
                fmt("    $v = call $T @llvm.fabs.v$w$h($V)\n", v, v, v, a0);
            } else {
                fmt("    $v_0 = icmp slt $V, $z\n"
                    "    $v_1 = sub nsw $T $z, $v\n"
                    "    $v = select <$w x i1> $v_0, $V_1, $V\n",
                    v, a0,
                    v, v, a0,
                    v, v, v, a0);
            }
            break;

        case VarKind::Add:
            fmt(jitc_is_float(v) ? "    $v = fadd $V, $v\n"
                                 : "    $v = add $V, $v\n",
                v, a0, a1);
            break;

        case VarKind::Sub:
            fmt(jitc_is_float(v) ? "    $v = fsub $V, $v\n"
                                 : "    $v = sub $V, $v\n",
                v, a0, a1);
            break;

        case VarKind::Mul:
            fmt(jitc_is_float(v) ? "    $v = fmul $V, $v\n"
                                 : "    $v = mul $V, $v\n",
                v, a0, a1);
            break;

        case VarKind::Div:
            if (jitc_is_float(v))
                stmt = "    $v = fdiv $V, $v\n";
            else if (jitc_is_uint(v))
                stmt = "    $v = udiv $V, $v\n";
            else
                stmt = "    $v = sdiv $V, $v\n";
            fmt(stmt, v, a0, a1);
            break;

        case VarKind::Mod:
            fmt(jitc_is_uint(v) ? "    $v = urem $V, $v\n"
                                : "    $v = srem $V, $v\n",
                v, a0, a1);
            break;

        case VarKind::Mulhi:
            fmt("    $v_0 = $sext $V to $D\n"
                "    $v_1 = $sext $V to $D\n"
                "    $v_3 = insertelement $D undef, $d $u, i32 0\n"
                "    $v_4 = shufflevector $D $v_3, $D undef, <$w x i32> $z\n"
                "    $v_5 = mul $D $v_0, $v_1\n"
                "    $v_6 = lshr $D $v_5, $v_4\n"
                "    $v = trunc $D $v_6 to $T\n",
                v, jitc_is_uint(v) ? "z" : "s", a0, a0,
                v, jitc_is_uint(v) ? "z" : "s", a1, a1,
                v, v, v, type_size[v->type] * 8,
                v, v, v, v,
                v, v, v, v,
                v, v, v, v,
                v, v, v, v);
            break;

        case VarKind::Fma:
            if (jitc_is_float(v)) {
                fmt_intrinsic("declare $T @llvm.fma.v$w$h($T, $T, $T)\n",
                    v, v, a0, a1, a2);
                fmt("    $v = call $T @llvm.fma.v$w$h($V, $V, $V)\n",
                    v, v, v, a0, a1, a2);
            } else {
                fmt("    $v_0 = mul $V, $v\n"
                    "    $v = add $V_0, $v\n",
                    v, a0, a1, v, v, a2);
            }
            break;

        case VarKind::Min:
            if (jitc_llvm_version_major >= 12 || jitc_is_float(v)) {
                if (jitc_is_float(v))
                    stmt = "minnum";
                else if (jitc_is_uint(v))
                    stmt = "umin";
                else
                    stmt = "smin";
                fmt_intrinsic("declare $T @llvm.$s.v$w$h($T, $T)", v, stmt, v, a0, a1);
                fmt("    $v = call $T @llvm.$s.v$w$h($V, $V)\n", v, v, stmt, v, a0, a1);
            } else {
                fmt("    $v_0 = icmp $s $V, $v\n"
                    "    $v = select <$w x i1> $v_0, $V, $V\n",
                    v, jitc_is_uint(v) ? "ult" : "slt", a0, a1,
                    v, v, a0, a1);
            }
            break;

        case VarKind::Max:
            if (jitc_llvm_version_major >= 12 || jitc_is_float(v)) {
                if (jitc_is_float(v))
                    stmt = "maxnum";
                else if (jitc_is_uint(v))
                    stmt = "umax";
                else
                    stmt = "smax";
                fmt_intrinsic("declare $T @llvm.$s.v$w$h($T, $T)", v, stmt, v, a0, a1);
                fmt("    $v = call $T @llvm.$s.v$w$h($V, $V)\n", v, v, stmt, v, a0, a1);
            } else {
                fmt("    $v_0 = icmp $s $V, $v\n"
                    "    $v = select <$w x i1> $v_0, $V, $V\n",
                    v, jitc_is_uint(v) ? "ugt" : "sgt", a0, a1,
                    v, v, a0, a1);
            }
            break;

        case VarKind::Ceil:
            fmt_intrinsic("declare $T @llvm.ceil.v$w$h($T)", v, v, a0);
            fmt("    $v = call $T @llvm.ceil.v$w$h($V)\n", v, v, v, a0);
            break;

        case VarKind::Floor:
            fmt_intrinsic("declare $T @llvm.floor.v$w$h($T)", v, v, a0);
            fmt("    $v = call $T @llvm.floor.v$w$h($V)\n", v, v, v, a0);
            break;

        case VarKind::Round:
            fmt_intrinsic("declare $T @llvm.nearbyint.v$w$h($T)", v, v, a0);
            fmt("    $v = call $T @llvm.nearbyint.v$w$h($V)\n", v, v, v, a0);
            break;

        case VarKind::Trunc:
            fmt_intrinsic("declare $T @llvm.trunc.v$w$h($T)", v, v, a0);
            fmt("    $v = call $T @llvm.trunc.v$w$h($V)\n", v, v, v, a0);
            break;

        case VarKind::Eq:
            fmt(jitc_is_float(a0) ? "    $v = fcmp oeq $V, $v\n"
                                  : "    $v = icmp eq $V, $v\n", v, a0, a1);
            break;

        case VarKind::Neq:
            fmt(jitc_is_float(a0) ? "    $v = fcmp one $V, $v\n"
                                  : "    $v = icmp ne $V, $v\n", v, a0, a1);
            break;

        case VarKind::Lt:
            if (jitc_is_float(a0))
                stmt = "    $v = fcmp olt $V, $v\n";
            else if (jitc_is_uint(a0))
                stmt = "    $v = icmp ult $V, $v\n";
            else
                stmt = "    $v = icmp slt $V, $v\n";
            fmt(stmt, v, a0, a1);
            break;

        case VarKind::Le:
            if (jitc_is_float(a0))
                stmt = "    $v = fcmp ole $V, $v\n";
            else if (jitc_is_uint(a0))
                stmt = "    $v = icmp ule $V, $v\n";
            else
                stmt = "    $v = icmp sle $V, $v\n";
            fmt(stmt, v, a0, a1);
            break;

        case VarKind::Gt:
            if (jitc_is_float(a0))
                stmt = "    $v = fcmp ogt $V, $v\n";
            else if (jitc_is_uint(a0))
                stmt = "    $v = icmp ugt $V, $v\n";
            else
                stmt = "    $v = icmp sgt $V, $v\n";
            fmt(stmt, v, a0, a1);
            break;

        case VarKind::Ge:
            if (jitc_is_float(a0))
                stmt = "    $v = fcmp oge $V, $v\n";
            else if (jitc_is_uint(a0))
                stmt = "    $v = icmp uge $V, $v\n";
            else
                stmt = "    $v = icmp sge $V, $v\n";
            fmt(stmt, v, a0, a1);
            break;

        case VarKind::Select:
            fmt("    $v = select $V, $V, $V\n", v, a0, a1, a2);
            break;

        case VarKind::Popc:
            fmt_intrinsic("declare $T @llvm.ctpop.v$w$h($T)", v, a0, a0);
            fmt("    $v = call $T @llvm.ctpop.v$w$h($V)\n", v, v, a0, a0);
            break;

        case VarKind::Clz:
            fmt_intrinsic("declare $T @llvm.ctlz.v$w$h($T, i1)", v, a0, a0);
            fmt("    $v = call $T @llvm.ctlz.v$w$h($V, i1 0)\n", v, v, a0, a0);
            break;

        case VarKind::Ctz:
            fmt_intrinsic("declare $T @llvm.cttz.v$w$h($T, i1)", v, a0, a0);
            fmt("    $v = call $T @llvm.cttz.v$w$h($V, i1 0)\n", v, v, a0, a0);
            break;

        case VarKind::And:
            if (a0->type != a1->type)
                fmt("    $v = select $V, $V, $T $z\n",
                    v, a1, a0, a0);
            else if (jitc_is_float(v))
                fmt("    $v_0 = bitcast $V to $B\n"
                    "    $v_1 = bitcast $V to $B\n"
                    "    $v_2 = and $B $v_0, $v_1\n"
                    "    $v = bitcast $B $v_2 to $T\n",
                    v, a0, v, v, a1, v, v, v, v, v, v, v, v, v);
            else
                fmt("    $v = and $V, $v\n", v, a0, a1);
            break;

        case VarKind::Or:
            if (a0->type != a1->type)
                fmt("    $v_0 = bitcast $V to $B\n"
                    "    $v_1 = sext $V to $B\n"
                    "    $v_2 = or $B $v_0, $v_1\n"
                    "    $v = bitcast $B $v_2 to $T\n",
                    v, a0, v, v, a1, v, v, v, v, v, v, v, v, v);
            else if (jitc_is_float(v))
                fmt("    $v_0 = bitcast $V to $B\n"
                    "    $v_1 = bitcast $V to $B\n"
                    "    $v_2 = or $B $v_0, $v_1\n"
                    "    $v = bitcast $B $v_2 to $T\n",
                    v, a0, v, v, a1, v, v, v, v, v, v, v, v, v);
            else
                fmt("    $v = or $V, $v\n", v, a0, a1);
            break;

        case VarKind::Xor:
            if (jitc_is_float(v))
                fmt("    $v_0 = bitcast $V to $B\n"
                    "    $v_1 = bitcast $V to $B\n"
                    "    $v_2 = xor $B $v_0, $v_1\n"
                    "    $v = bitcast $B $v_2 to $T\n",
                    v, a0, v, v, a1, v, v, v, v, v, v, v, v, v);
            else
                fmt("    $v = xor $V, $v\n", v, a0, a1);
            break;

        case VarKind::Shl:
            fmt("    $v = shl $V, $v\n", v, a0, a1);
            break;

        case VarKind::Shr:
            fmt(jitc_is_uint(v) ? "    $v = lshr $V, $v\n"
                                : "    $v = ashr $V, $v\n",
                v, a0, a1);
            break;

        case VarKind::Cast:
            if (jitc_is_bool(v)) {
                fmt(jitc_is_float(a0) ? "    $v = fcmp one $V, $z\n"
                                      : "    $v = icmp ne $V, $z\n",
                    v, a0);
            } else if (jitc_is_bool(a0)) {
                fmt("    $v_1 = insertelement $T undef, $t $s, i32 0\n"
                    "    $v_2 = shufflevector $T $v_1, $T undef, <$w x i32> $z\n"
                    "    $v = select $V, $T $v_2, $T $z\n",
                    v, v, v, jitc_is_float(v) ? "1.0" : "1",
                    v, v, v, v,
                    v, a0, v, v, v);
            } else if (jitc_is_float(v) && !jitc_is_float(a0)) {
                fmt(jitc_is_uint(a0) ? "    $v = uitofp $V to $T\n"
                                     : "    $v = sitofp $V to $T\n",
                    v, a0, v);
            } else if (!jitc_is_float(v) && jitc_is_float(a0)) {
                fmt(jitc_is_uint(v) ? "    $v = fptoui $V to $T\n"
                                    : "    $v = fptosi $V to $T\n",
                    v, a0, v);
            } else if (jitc_is_float(v) && jitc_is_float(a0)) {
                fmt(type_size[v->type] > type_size[a0->type]
                        ? "    $v = fpext $V to $T\n"
                        : "    $v = fptrunc $V to $T\n",
                    v, a0, v);
            } else if (type_size[v->type] < type_size[a0->type]) {
                fmt("    $v = trunc $V to $T\n", v, a0, v);
            } else {
                fmt(jitc_is_uint(a0) ? "    $v = zext $V to $T\n"
                                     : "    $v = sext $V to $T\n",
                    v, a0, v);
            }
            break;

        case VarKind::Bitcast:
            fmt("    $v = bitcast $V to $T\n", v, a0, v);
            break;

        case VarKind::Gather: {
                bool is_bool = v->type == (uint32_t) VarType::Bool;
                if (is_bool) // Temporary change
                    v->type = (uint32_t) VarType::UInt8;

                fmt_intrinsic(
                    "declare $T @llvm.masked.gather.v$w$h(<$w x {$t*}>, i32, $T, $T)",
                    v, v, v, a2, v);

                fmt("{    $v_0 = bitcast $<i8*$> $v to $<$t*$>\n|}"
                     "    $v_1 = getelementptr $t, $<{$t*}$> {$v_0|$v}, $V\n"
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

        case VarKind::Scatter:
            jitc_llvm_render_scatter(v, a0, a1, a2, a3);
            break;

        case VarKind::ScatterInc:
            jitc_llvm_render_scatter_inc(v, a0, a1, a2);
            break;

        case VarKind::ScatterKahan:
            jitc_llvm_render_scatter_kahan(v, index);
            break;

        case VarKind::VCallMask:
            fmt("    $v = bitcast <$w x i1> %mask to <$w x i1>\n", v);
            break;

        case VarKind::VCallSelf:
            fmt("    $v = bitcast <$w x i32> %self to <$w x i32>\n", v);
            break;

        case VarKind::Counter:
            fmt("    $v_0 = trunc i64 %index to $t\n"
                "    $v_1 = insertelement $T undef, $t $v_0, i32 0\n"
                "    $v_2 = shufflevector $V_1, $T undef, <$w x i32> $z\n"
                "    $v = add $V_2, <",
                v, v, v, v, v, v, v, v, v, v, v);
            for (uint32_t i = 0; i < jitc_llvm_vector_width; ++i)
                fmt("i32 $u$s", i, i + 1 < jitc_llvm_vector_width ? ", " : ">\n");
            break;

        case VarKind::DefaultMask:
            fmt("    $v_0 = trunc i64 %end to i32\n"
                "    $v_1 = insertelement <$w x i32> undef, i32 $v_0, i32 0\n"
                "    $v_2 = shufflevector <$w x i32> $v_1, <$w x i32> undef, <$w x i32> zeroinitializer\n"
                "    $v = icmp ult <$w x i32> $v, $v_2\n",
                v, v, v, v, v, v, a0, v);
            break;

        case VarKind::Printf:
            jitc_llvm_render_printf(index, v, a0, a1);
            break;

        case VarKind::Dispatch:
            jitc_var_vcall_assemble((VCall *) state.extra[index].callback_data,
                                    a0->reg_index, a1->reg_index, a2->reg_index,
                                    a3 ? a3->reg_index : 0);
            break;

        case VarKind::TraceRay:
            jitc_llvm_render_trace(index, v, a0, a1);
            break;

        case VarKind::Extract:
            fmt("    $v = bitcast $T $v_out_$u to $T\n", v, v, a0,
                (uint32_t) v->literal, v);
            break;

        default:
            jitc_fail("jitc_llvm_render_var(): unhandled node kind \"%s\"!",
                      var_kind_name[(uint32_t) v->kind]);
    }
}

static void jitc_llvm_render_scatter(const Variable *v,
                                     const Variable *ptr,
                                     const Variable *value,
                                     const Variable *index,
                                     const Variable *mask) {
    fmt("{    $v_0 = bitcast $<i8*$> $v to $<$t*$>\n|}"
         "    $v_1 = getelementptr $t, $<{$t*}$> {$v_0|$v}, $V\n",
        v, ptr, value,
        v, value, value, v, ptr, index);

    if (!v->literal) {
        fmt_intrinsic("declare void @llvm.masked.scatter.v$w$h($T, <$w x {$t*}>, i32, $T)",
             value, value, value, mask);
        fmt("    call void @llvm.masked.scatter.v$w$h($V, <$w x {$t*}> $v_1, i32 $a, $V)\n",
             value, value, value, v, value, mask);
    } else {
        const char *op, *zero_elem = nullptr, *intrinsic_name = nullptr;
        switch ((ReduceOp) v->literal) {
            case ReduceOp::Add:
                if (jitc_is_single(value)) {
                    op = "fadd";
                    zero_elem = "float -0.0, ";
                    intrinsic_name = "v2.fadd.f32";
                } else if (jitc_is_double(value)) {
                    op = "fadd";
                    zero_elem = "double -0.0, ";
                    intrinsic_name = "v2.fadd.f64";
                } else {
                    op = "add";
                }
                break;

            case ReduceOp::Mul:
                if (jitc_is_single(value)) {
                    op = "fmul";
                    zero_elem = "float -0.0, ";
                    intrinsic_name = "v2.fmul.f32";
                } else if (jitc_is_double(value)) {
                    op = "fmul";
                    zero_elem = "double -0.0, ";
                    intrinsic_name = "v2.fmul.f64";
                } else {
                    op = "mul";
                }
                break;

            case ReduceOp::Min:
                op = jitc_is_float(value)
                         ? "fmin"
                         : (jitc_is_uint(value) ? "umin" : "smin");
                break;

            case ReduceOp::Max:
                op = jitc_is_float(value)
                         ? "fmax"
                         : (jitc_is_uint(value) ? "umax" : "smax");
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
                "declare $t @llvm.experimental.vector.reduce.$s.v$w$h($t, $T)", value,
                intrinsic_name, value, value, value);
        else
            fmt_intrinsic("declare $t @llvm.experimental.vector.reduce.$s.v$w$h($T)",
                          value, op, value, value);

        const char *reassoc = jitc_is_float(value) ? "reassoc " : "";

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
            op, value, value, value, value, value, value, value, value, value, value, value, reassoc,
            value, intrinsic_name, value, zero_elem ? zero_elem : "", value, op, value, value
        );

        fmt("    call void @reduce_$s_$h(<$w x {$t*}> $v_1, $V, $V)\n",
            op, value, value, v, value, mask);
    }
}

static void jitc_llvm_render_scatter_inc(Variable *v,
                                         const Variable *ptr,
                                         const Variable *index,
                                         const Variable *mask) {
    fmt( "    $v_1 = extractelement $V, i32 0\n"
        "{    $v_2 = bitcast i8* $v to i32*\n"
         "    $v_3 = getelementptr i32, i32* $v_2, i32 $v_1\n|"
         "    $v_3 = getelementptr i32, ptr $v, i32 $v_1\n}"
         "    $v = call $T @reduce_inc_u32({$t*} $v_3, $V)\n",
        v, index,
        v, ptr,
        v, v, v,
        v, ptr, v,
        v, v, v, v, mask);

    fmt_intrinsic(
        "define internal <$w x i32> @reduce_inc_u32({i32*} %ptr, <$w x i1> %active) #0 ${\n"
        "L0:\n"
        "   br label %L1\n\n"
        "L1:\n"
        "   %index = phi i32 [ 0, %L0 ], [ %index_next, %L1 ]\n"
        "   %sum = phi i32 [ 0, %L0 ], [ %sum_next, %L1 ]\n"
        "   %sum_vec = phi <$w x i32> [ undef, %L0 ], [ %sum_vec_next, %L1 ]\n"
        "   %active_i = extractelement <$w x i1> %active, i32 %index\n"
        "   %active_u = zext i1 %active_i to i32\n"
        "   %sum_next = add nuw i32 %sum, %active_u\n"
        "   %sum_vec_next = insertelement <$w x i32> %sum_vec, i32 %sum, i32 %index\n"
        "   %index_next = add nuw nsw i32 %index, 1\n"
        "   %cond_1 = icmp eq i32 %index_next, $w\n"
        "   br i1 %cond_1, label %L2, label %L1\n\n"
        "L2:\n"
        "   %cond_2 = icmp eq i32 %sum_next, 0\n"
        "   br i1 %cond_2, label %L4, label %L3\n\n"
        "L3:\n"
        "   %old_1 = atomicrmw add {i32*} %ptr, i32 %sum_next monotonic\n"
        "   %old_2 = insertelement <$w x i32> undef, i32 %old_1, i32 0\n"
        "   %old_3 = shufflevector <$w x i32> %old_2, <$w x i32> undef, <$w x i32> $z\n"
        "   %sum_vec_final = add <$w x i32> %sum_vec_next, %old_3\n"
        "   br label %L4;\n\n"
        "L4:\n"
        "   %sum_vec_combined = phi <$w x i32> [ %sum_vec_next, %L2 ], [ %sum_vec_final, %L3 ]\n"
        "   ret <$w x i32> %sum_vec_combined\n"
        "$}"
    );

    v->consumed = 1;
}

static void jitc_llvm_render_scatter_kahan(const Variable *v, uint32_t v_index) {
    const Extra &extra = state.extra[v_index];
    const Variable *ptr_1 = jitc_var(extra.dep[0]),
                   *ptr_2 = jitc_var(extra.dep[1]),
                   *index = jitc_var(extra.dep[2]),
                   *mask = jitc_var(extra.dep[3]),
                   *value = jitc_var(extra.dep[4]);

    fmt_intrinsic("declare $t @llvm.fabs.$h($t)", value, value, value);

    fmt("{    $v_ptr1 = bitcast $<i8*$> $v to $<$t*$>\n|}"
         "    $v_target1 = getelementptr $t, $<{$t*}$> {$v_ptr1|$v}, $V\n",
        v, ptr_1, value,
        v, value, value, v, ptr_1, index);
    fmt("{    $v_ptr2 = bitcast $<i8*$> $v to $<$t*$>\n|}"
         "    $v_target2 = getelementptr $t, $<{$t*}$> {$v_ptr2|$v}, $V\n"
         "    br label %l$u_0\n\n",
        v, ptr_2, value,
        v, value, value, v, ptr_2, index,
        v_index);

    fmt("l$u_0:\n"
        "    br label %l$u_1\n\n",
        v_index,
        v_index);

    fmt("l$u_1:\n"
        "    $v_index = phi i32 [ 0, %l$u_0 ], [ $v_index_next, %l$u_3 ]\n"
        "    $v_active_i = extractelement <$w x i1> $v, i32 $v_index\n"
        "    br i1 $v_active_i, label %l$u_2, label %l$u_3\n\n",
        v_index,
        v, v_index, v, v_index,
        v, mask, v,
        v, v_index, v_index);

    fmt("l$u_2:\n"
        "    $v_target1_i = extractelement <$w x {$t*}> $v_target1, i32 $v_index\n"
        "    $v_target2_i = extractelement <$w x {$t*}> $v_target2, i32 $v_index\n"
        "    $v_value_i = extractelement <$w x $t> $v, i32 $v_index\n"
        "    $v_before = atomicrmw fadd {$t*} $v_target1_i, $t $v_value_i monotonic\n"
        "    $v_after = fadd $t $v_before, $v_value_i\n"
        "    $v_case1_0 = fsub $t $v_before, $v_after\n"
        "    $v_case1 = fadd $t $v_case1_0, $v_value_i\n"
        "    $v_case2_0 = fsub $t $v_value_i, $v_after\n"
        "    $v_case2 = fadd $t $v_case2_0, $v_before\n"
        "    $v_abs_before = call $t @llvm.fabs.$h($t $v_before)\n"
        "    $v_abs_value = call $t @llvm.fabs.$h($t $v_value_i)\n"
        "    $v_pred = fcmp oge $t $v_abs_before, $v_abs_value\n"
        "    $v_result = select i1 $v_pred, $t $v_case1, $t $v_case2\n"
        "    atomicrmw fadd {$t*} $v_target2_i, $t $v_result monotonic\n"
        "    br label %l$u_3\n\n",
        v_index,
        v, value, v, v,
        v, value, v, v,
        v, value, value, v,
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
        v_index);

    fmt("l$u_3:\n"
        "    $v_index_next = add nuw nsw i32 $v_index, 1\n"
        "    $v_cond = icmp eq i32 $v_index_next, $w\n"
        "    br i1 $v_cond, label %l$u_4, label %l$u_1\n\n"
        "l$u_4:\n",
        v_index,
        v, v,
        v, v,
        v, v_index, v_index,
        v_index);
}


static void jitc_llvm_render_printf(uint32_t index, const Variable *v,
                                    const Variable *mask,
                                    const Variable *target) {
    const char *format_str = (const char *) v->literal;
    const uint32_t len = (uint32_t) strlen(format_str);
    XXH128_hash_t hash = XXH128(format_str, len, 0);

    size_t buffer_offset = buffer.size();
    fmt("@data_$Q$Q = private unnamed_addr constant [$u x i8] [",
        hash.high64, hash.low64, len + 1);
    for (uint32_t i = 0; ; ++i) {
        fmt("i8 $u", (uint32_t) format_str[i]);
        if (!format_str[i])
            break;
        put(", ");
    }
    put("], align 1");
    jitc_register_global(buffer.get() + buffer_offset);
    buffer.rewind_to(buffer_offset);

    uint32_t idx = v->reg_index;

    fmt("    br label %l$u_start\n\n"
        "l$u_start: ; ---- printf_async() ----\n"
        "    $v_fmt = getelementptr [$u x i8], {[$u x i8]*} @data_$Q$Q, i64 0, i64 0\n"
        "    br label %l$u_cond\n\n"
        "l$u_cond: ; ---- printf_async() ----\n"
        "    $v_idx = phi i32 [ 0, %l$u_start ], [ $v_next, %l$u_tail ]\n"
        "    $v_msk = extractelement $V, i32 $v_idx\n"
        "    br i1 $v_msk, label %l$u_body, label %l$u_tail\n\n"
        "l$u_body: ; ---- printf_async() ----\n",
        idx, idx, v, len + 1, len + 1, hash.high64, hash.low64, idx, idx, v, idx, v,
        idx, v, mask, v, v, idx, idx, idx);

    const Extra &extra = state.extra[index];
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        VarType vt = (VarType) v2->type;

        fmt("    $v_$u$s = extractelement $V, i32 $v_idx\n",
            v, i, vt == VarType::Float32 ? "_0" : "", v2, v);

        if (vt == VarType::Float32)
            fmt("    $v_$u = fpext float $v_$u_0 to double\n", v, i, v, i);
    }

    if (callable_depth == 0)
        fmt("    $v_fun = bitcast {i8*} $v to {i32 (i8*, ...)*}\n", v, target);
    else
        fmt( "    $v_fun{_0|} = extractelement <$w x {i8*}> $v, i32 $v_idx\n"
            "{    $v_fun = bitcast i8* $v_fun_0 to i32 (i8*, ...)*\n|}",
            v, target, v,
            v, v);

    fmt("    $v_rv = call i32 ({i8*}, ...) $v_fun ({i8*} $v_fmt", v, v, v);

    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        VarType vt = (VarType) v2->type;
        if (vt == VarType::Float32)
            vt = VarType::Float64;
        fmt(", $s $v_$u", type_name_llvm[(int) vt], v, i);
    }

    fmt(")\n"
        "    br label %l$u_tail\n\n"
        "l$u_tail: ; ---- printf_async() ----\n"
        "    $v_next = add i32 $v_idx, 1\n"
        "    $v_again = icmp ult i32 $v_next, $w\n"
        "    br i1 $v_again, label %l$u_cond, label %l$u_end\n\n"
        "l$u_end:\n",
        idx, idx, v, v, v, v, v, idx, idx, idx);
}

/// Convert an IR template with '$' expressions into valid IR
static void jitc_llvm_render_stmt(uint32_t index, const Variable *v, bool in_function) {
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
                case 'z': put("zeroinitializer"); continue;
                case 'w': buffer.put_u32(jitc_llvm_vector_width); continue;
                case 't': prefix_table = type_name_llvm; break;
                case 'T': prefix_table = type_name_llvm_big; break;
                case 'b': prefix_table = type_name_llvm_bin; break;
                case 'a': prefix_table = type_name_llvm_abbrev; break;
                case 's': prefix_table = type_size_str; break;
                case 'r': prefix_table = type_prefix; break;
                case 'i': prefix_table = nullptr; break;
                case '<': if (in_function) {
                              put('<');
                              buffer.put_u32(jitc_llvm_vector_width);
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

void jitc_llvm_ray_trace(uint32_t func, uint32_t scene, int shadow_ray,
                         const uint32_t *in, uint32_t *out) {
    const uint32_t n_args = 14;
    VarType float_type = (VarType) jitc_var(in[2])->type;

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

    if (jitc_var_type(func) != VarType::Pointer ||
        jitc_var_type(scene) != VarType::Pointer)
        jitc_raise("jitc_llvm_ray_trace(): 'func', and 'scene' must be pointer variables!");

    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jitc_llvm_ray_trace(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::LLVM));
        dirty = false;

        for (uint32_t i = 0; i < n_args; ++i)
            dirty |= jitc_var(in[i])->is_dirty();

        if (dirty)
            jitc_raise("jit_llvm_ray_trace(): inputs remain dirty after evaluation!");
    }

    // ----------------------------------------------------------
    Ref valid = steal(jitc_var_mask_apply(in[1], size));
    {
        int32_t minus_one_c = -1, zero_c = 0;
        Ref minus_one = steal(jitc_var_literal(
                JitBackend::LLVM, VarType::Int32, &minus_one_c, 1, 0)),
            zero = steal(jitc_var_literal(
                JitBackend::LLVM, VarType::Int32, &zero_c, 1, 0));

        valid = steal(jitc_var_select(valid, minus_one, zero));
    }

    jitc_log(InfoSym, "jitc_llvm_ray_trace(): tracing %u %sray%s%s", size,
             shadow_ray ? "shadow " : "", size != 1 ? "s" : "",
             placeholder ? " (part of a recorded computation)" : "");

    Ref index = steal(jitc_var_new_node_2(
        JitBackend::LLVM, VarKind::TraceRay, VarType::Void, size,
        placeholder, func, jitc_var(func), scene,
        jitc_var(scene), (uint64_t) shadow_ray));

    Variable *v = jitc_var(index);
    v->extra = 1;

    Extra &extra = state.extra[index];
    extra.n_dep = n_args;
    extra.dep = (uint32_t *) malloc_check(sizeof(uint32_t) * n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        uint32_t id = i != 1 ? in[i] : valid;
        extra.dep[i] = id;
        jitc_var_inc_ref(id);
    }

    for (int i = 0; i < (shadow_ray ? 1 : 6); ++i)
        out[i] = jitc_var_new_node_1(JitBackend::LLVM, VarKind::Extract,
                                     i < 3 ? float_type : VarType::UInt32, size,
                                     placeholder, index, jitc_var(index),
                                     (uint64_t) i);
}

static void jitc_llvm_render_trace(uint32_t index, const Variable *v,
                                   const Variable *func,
                                   const Variable *scene) {
    /* Intersection data structure layout:
        0  uint32_t valid
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
    const Extra &extra = state.extra[index];
    bool shadow_ray = v->literal == 1;
    VarType float_type = jitc_var_type(extra.dep[2]);

    uint32_t width          = jitc_llvm_vector_width,
             ctx_size       = 6 * 4,
             float_size     = type_size[(int) float_type],
             alloca_size_rt = (shadow_ray ? (9 * float_size + 4 * 4)
                                          : (14 * float_size + 7 * 4)) * width;

    alloca_size  = std::max(alloca_size, (int32_t) (alloca_size_rt + ctx_size));
    alloca_align = std::max(alloca_align, (int32_t) (float_size * width));

    fmt("\n    ; -------- Ray $s -------\n", shadow_ray ? "test" : "trace");

	// Copy input parameters to staging area
    uint32_t offset = 0;
    for (uint32_t i = 0; i < 13; ++i) {
        if (jitc_llvm_vector_width == 1 && i == 0)
            continue; // valid flag not needed for 1-lane versions

        const Variable *v2 = jitc_var(extra.dep[i + 1]);
        fmt( "    $v_in_$u_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    $v_in_$u_1 = bitcast i8* $v_in_$u_0 to $T*\n|}"
             "    store $V, {$T*} $v_in_$u_1, align $A\n",
             v, i, offset,
             v, i, v, i, v2,
             v2, v2, v, i, v2);

        offset += type_size[v2->type] * width;
    }

	// Reset geomID field to ones as required
    if (!shadow_ray) {
        fmt( "    $v_in_geomid_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    $v_in_geomid_1 = bitcast i8* $v_in_geomid_0 to <$w x i32> *\n|}"
             "    store <$w x i32> $s, {<$w x i32>*} $v_in_geomid_1, align $u\n",
            v, (14 * float_size + 5 * 4) * width,
            v, v,
            jitc_llvm_ones_str[(int) VarType::Int32], v, float_size * width);
    }

	/// Determine whether to mark the rays as coherent or incoherent
    const Variable *coherent = jitc_var(extra.dep[0]);

    fmt( "    $v_in_ctx_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
        "{    $v_in_ctx_1 = bitcast i8* $v_in_ctx_0 to <6 x i32>*\n|}",
        v, alloca_size_rt, v, v);

    if (coherent->is_literal()) {
        fmt("    store <6 x i32> <i32 $u, i32 0, i32 0, i32 0, i32 -1, i32 0>, {<6 x i32>*} $v_in_ctx_1, align 4\n",
            (uint32_t) coherent->literal, v);
    } else {
        fmt_intrinsic("declare i1 @llvm.experimental.vector.reduce.and.v$wi1(<$w x i1>)");

        fmt("    $v_coherent_0 = call i1 @llvm.experimental.vector.reduce.and.v$wi1($V)\n"
			"    $v_coherent_1 = zext i1 $v_coherent_0 to i32\n"
            "    $v_ctx = insertelement <6 x i32> <i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0>, i32 $v_coherent_1, i32 0\n"
            "    store <6 x i32> $v_ctx, {<6 x i32>*} $v_in_ctx_1, align 4\n",
            v, coherent, v, v, v, v, v, v);
    }

    /* When the ray tracing operation occurs inside a recorded function, it's
       possible that multiple different scenes are being traced simultaneously.
       In that case, it's necessary to perform one ray tracing call per scene,
       which is implemented by the following loop. */
    if (callable_depth == 0) {
        if (jitc_llvm_vector_width > 1) {
            fmt("{    $v_func = bitcast i8* $v to void (i8*, i8*, i8*, i8*)*\n|}"
                 "    call void {$v_func|$v}({i8*} $v_in_0_{0|1}, {i8*} $v, {i8*} $v_in_ctx_{0|1}, {i8*} $v_in_1_{0|1})\n",
                v, func,
                v, func, v, scene, v, v
            );
        } else {
            fmt(
                "{    $v_func = bitcast i8* $v to void (i8*, i8*, i8*)*\n|}"
                "     call void {$v_func|$v}({i8*} $v, {i8*} $v_in_ctx_{0|1}, {i8*} $v_in_1_{0|1})\n",
                v, func,
                v, func, scene, v, v
            );
        }
    } else {
        fmt_intrinsic("declare i64 @llvm.experimental.vector.reduce.umax.v$wi64(<$w x i64>)");

        // =====================================================
        // 1. Prepare the loop for the ray tracing calls
        // =====================================================

        uint32_t offset_tfar = (8 * float_size + 4) * width;
        const char *tname_tfar = type_name_llvm[(int) float_type];

        fmt( "    br label %l$u_start\n"
             "\nl$u_start:\n"
             "    ; Ray tracing\n"
             "    $v_func_i64 = call i64 @llvm.experimental.vector.reduce.umax.v$wi64(<$w x i64> %rd$u_p4)\n"
             "    $v_func_ptr = inttoptr i64 $v_func_i64 to {i8*}\n"
             "    $v_tfar_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    $v_tfar_1 = bitcast i8* $v_tfar_0 to <$w x $s> *\n|}",
            v->reg_index,
            v->reg_index,
            v, func->reg_index,
            v, v,
            v, offset_tfar,
            v, v, tname_tfar);

        if (jitc_llvm_vector_width > 1)
            fmt("    $v_func = bitcast {i8*} $v_func_ptr to {void (i8*, i8*, i8*, i8*)*}\n", v, v);
        else
            fmt("    $v_func = bitcast {i8*} $v_func_ptr to {void (i8*, i8*, i8*)*}\n", v, v);

        // Get original mask, to be overwritten at every iteration
        fmt("    $v_mask_value = load <$w x i32>, {<$w x i32>*} $v_in_0_1, align 64\n"
            "    br label %l$u_check\n",
            v, v, v->reg_index);

        // =====================================================
        // 2. Move on to the next instance & check if finished
        // =====================================================

        fmt("\nl$u_check:\n"
            "    $v_scene = phi <$w x {i8*}> [ %rd$u, %l$u_start ], [ $v_scene_next, %l$u_call ]\n"
            "    $v_scene_i64 = ptrtoint <$w x {i8*}> $v_scene to <$w x i64>\n"
            "    $v_next_i64 = call i64 @llvm.experimental.vector.reduce.umax.v$wi64(<$w x i64> $v_scene_i64)\n"
            "    $v_next = inttoptr i64 $v_next_i64 to {i8*}\n"
            "    $v_valid = icmp ne {i8*} $v_next, null\n"
            "    br i1 $v_valid, label %l$u_call, label %l$u_end\n",
            v->reg_index,
            v, scene->reg_index, v->reg_index, v, v->reg_index,
            v, v,
            v, v,
            v, v,
            v, v,
            v, v->reg_index, v->reg_index);

        // =====================================================
        // 3. Perform ray tracing call to each unique instance
        // =====================================================

        fmt("\nl$u_call:\n"
            "    $v_tfar_prev = load <$w x $s>, {<$w x $s>*} $v_tfar_1, align $u\n"
            "    $v_bcast_0 = insertelement <$w x i64> undef, i64 $v_next_i64, i32 0\n"
            "    $v_bcast_1 = shufflevector <$w x i64> $v_bcast_0, <$w x i64> undef, <$w x i32> $z\n"
            "    $v_bcast_2 = inttoptr <$w x i64> $v_bcast_1 to <$w x {i8*}>\n"
            "    $v_active = icmp eq <$w x {i8*}> $v_scene, $v_bcast_2\n"
            "    $v_active_2 = select <$w x i1> $v_active, <$w x i32> $v_mask_value, <$w x i32> $z\n"
            "    store <$w x i32> $v_active_2, {<$w x i32>*} $v_in_0_1, align 64\n",
            v->reg_index,
            v, tname_tfar, tname_tfar, v, float_size * width,
            v, v,
            v, v,
            v, v,
            v, v, v,
            v, v, v,
            v, v);

        if (jitc_llvm_vector_width > 1)
            fmt("    call void $v_func({i8*} $v_in_0_{0|1}, {i8*} $v_next, {i8*} $v_in_ctx_{0|1}, {i8*} $v_in_1_{0|1})\n",
                v, v, v, v, v);
        else
            fmt("    call void $v_func({i8*} $v_next, {i8*} $v_in_ctx_{0|1}, {i8*} $v_in_1_{0|1})\n",
                v, v, v, v);

        fmt("    $v_tfar_new = load <$w x $s>, {<$w x $s>*} $v_tfar_1, align $u\n"
            "    $v_tfar_masked = select <$w x i1> $v_active, <$w x $s> $v_tfar_new, <$w x $s> $v_tfar_prev\n"
            "    store <$w x $s> $v_tfar_masked, {<$w x $s>*} $v_tfar_1, align $u\n"
            "    $v_scene_next = select <$w x i1> $v_active, <$w x {i8*}> $z, <$w x {i8*}> $v_scene\n"
            "    br label %l$u_check\n"
            "\nl$u_end:\n",
            v, tname_tfar, tname_tfar, v, float_size * width,
            v, v, tname_tfar, v, tname_tfar, v,
            tname_tfar, v, tname_tfar, v, float_size * width,
            v, v, v,
            v->reg_index, v->reg_index);
    }

    offset = (8 * float_size + 4) * width;

    for (int i = 0; i < (shadow_ray ? 1 : 6); ++i) {
        VarType vt = (i < 3) ? float_type : VarType::UInt32;
        const char *tname = type_name_llvm[(int) vt];
        uint32_t tsize = type_size[(int) vt];

        fmt( "    $v_out_$u_{0|1} = getelementptr inbounds i8, {i8*} %buffer, i32 $u\n"
            "{    $v_out_$u_1 = bitcast i8* $v_out_$u_0 to <$w x $s> *\n|}"
             "    $v_out_$u = load <$w x $s>, {<$w x $s>*} $v_out_$u_1, align $u\n",
            v, i, offset,
            v, i, v, i, tname,
            v, i, tname, tname, v, i, float_size * width);

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
         "    %u$u_self_combined = call <$w x i64> @llvm.masked.gather.v$wi64(<$w x {i64*}> %u$u_self_ptr, i32 8, <$w x i1> %p$u, <$w x i64> $z)\n"
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
