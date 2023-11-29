/**
 * This file contains the logic that assembles a CUDA PTX representation from a
 * recorded Dr.Jit computation graph. It implements a small template engine
 * involving plentiful use of the 'fmt' formatting routine.
 *
 * Its format interface supports the following format string characters. Note
 * that it uses the '$' (dollar) escape character, since '%' is used for CUDA
 * register prefixes (otherwise, lots of escaping would be needed).
 *
 *  Format  Input          Example result    Description
 * --------------------------------------------------------------------------
 *  $u      uint32_t      `1234`             Decimal number (32 bit)
 *  $U      uint64_t      `1234`             Decimal number (64 bit)
 * --------------------------------------------------------------------------
 *  $x      uint32_t      `4d2`              Hexadecimal number (32 bit)
 *  $X      uint64_t      `4d2`              Hexadecimal number (64 bit)
 *  $Q      uint64_t      `00000000000004d2` Hex. number, 0-filled (64 bit)
 * --------------------------------------------------------------------------
 *  $s      const char *  `foo`              Zero-terminated string
 *  $c      char          `f`                A single ASCII character
 * --------------------------------------------------------------------------
 *  $t      Variable      `f32`              Variable type
 * --------------------------------------------------------------------------
 *  $b      Variable      `b32`              Variable type, binary format
 * --------------------------------------------------------------------------
 *  $v      Variable      `%f1234`           Variable name
 * --------------------------------------------------------------------------
 *  $o      Variable      `5`                Variable offset in param. array
 * --------------------------------------------------------------------------
 *  $l      Variable      `1`                Literal value of variable (hex)
 * --------------------------------------------------------------------------
 */

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "call.h"
#include "loop.h"
#include "optix.h"

#define fmt(fmt, ...) buffer.fmt_cuda(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)
#define put(...)      buffer.put(__VA_ARGS__)
#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_cuda(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__);          \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0);

// Forward declarations
static void jitc_cuda_render(uint32_t index, Variable *v);
static void jitc_cuda_render_scatter(const Variable *v, const Variable *ptr,
                                     const Variable *value, const Variable *index,
                                     const Variable *mask);
static void jitc_cuda_render_scatter_inc(Variable *v, const Variable *ptr,
                                         const Variable *index, const Variable *mask);
static void jitc_cuda_render_scatter_kahan(const Variable *v, uint32_t index);

#if defined(DRJIT_ENABLE_OPTIX)
static void jitc_cuda_render_trace(uint32_t index, const Variable *v,
                                   const Variable *valid,
                                   const Variable *pipeline,
                                   const Variable *sbt);
#endif

void jitc_cuda_assemble(ThreadState *ts, ScheduledGroup group,
                        uint32_t n_regs, uint32_t n_params) {
    bool params_global = !uses_optix && n_params > DRJIT_CUDA_ARG_LIMIT;
    bool print_labels  = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);

#if defined(DRJIT_ENABLE_OPTIX)
    /* If use optix and the kernel contains no ray tracing operations,
       fall back to the default OptiX pipeline and shader binding table. */
    if (uses_optix) {
        /// Ensure OptiX is initialized
        (void) jitc_optix_context();
        ts->optix_pipeline = state.optix_default_pipeline;
        ts->optix_sbt = state.optix_default_sbt;
    }
#endif

    /* Special registers:

         %r0   :  Index
         %r1   :  Step
         %r2   :  Size
         %p0   :  Stopping predicate
         %rd0  :  Temporary for parameter pointers
         %rd1  :  Pointer to parameter table in global memory if too big

         %b3, %w3, %r3, %rd3, %f3, %d3, %p3: reserved for use in compound
         statements that must write a temporary result to a register.
    */

    fmt(".version $u.$u\n"
        ".target sm_$u\n"
        ".address_size 64\n\n",
        ts->ptx_version / 10, ts->ptx_version % 10,
        ts->compute_capability);

    if (!uses_optix) {
        fmt(".entry drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^("
            ".param .align 8 .b8 params[$u]) { \n",
            params_global ? 8u : (n_params * (uint32_t) sizeof(void *)));
    } else {
        fmt(".const .align 8 .b8 params[$u];\n\n"
            ".entry __raygen__^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^() {\n",
            n_params * (uint32_t) sizeof(void *));
    }

    fmt("    .reg.b8   %b <$u>; .reg.b16  %w<$u>; .reg.b32 %r<$u>;\n"
        "    .reg.b64  %rd<$u>; .reg.f16  %h<$u>; .reg.f32 %f<$u>;\n"
        "    .reg.f64  %d <$u>; .reg.pred %p <$u>;\n\n",
        n_regs, n_regs, n_regs, n_regs, n_regs, n_regs, n_regs, n_regs);

    if (!uses_optix) {
        put("    mov.u32 %r0, %ctaid.x;\n"
            "    mov.u32 %r1, %ntid.x;\n"
            "    mov.u32 %r2, %tid.x;\n"
            "    mad.lo.u32 %r0, %r0, %r1, %r2;\n");

        if (likely(!params_global)) {
           put("    ld.param.u32 %r2, [params];\n");
        } else {
           put("    ld.param.u64 %rd1, [params];\n"
               "    ldu.global.u32 %r2, [%rd1];\n");
        }

        put("    setp.ge.u32 %p0, %r0, %r2;\n"
            "    @%p0 bra done;\n"
            "\n"
            "    mov.u32 %r3, %nctaid.x;\n"
            "    mul.lo.u32 %r1, %r3, %r1;\n"
            "\n");

        fmt("body: // sm_$u\n", state.devices[ts->device].compute_capability);
    } else {
        put("    call (%r0), _optix_get_launch_index_x, ();\n"
            "    ld.const.u32 %r1, [params + 4];\n"
            "    add.u32 %r0, %r0, %r1;\n\n"
            "body:\n");
    }

    const char *params_base = "params",
               *params_type = "param";

    if (uses_optix) {
        params_type = "const";
    } else if (params_global) {
        params_base = "%rd1";
        params_type = "global";
    }

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        const uint32_t vti = v->type,
                       size = v->size;
        const VarType vt = (VarType) vti;
        const VarKind kind = (VarKind) v->kind;

        if (unlikely(print_labels && v->extra)) {
            const char *label = jitc_var_label(index);
            if (label && *label && vt != VarType::Void && kind != VarKind::CallOutput)
                fmt("    // $s\n", label);
        }

        if (likely(v->param_type == ParamType::Input)) {
            if (v->is_literal()) {
                fmt("    ld.$s.u64 $v, [$s+$o];\n", params_type, v, params_base, v);
                continue;
            } else {
                fmt("    ld.$s.u64 %rd0, [$s+$o];\n", params_type, params_base, v);
            }

            if (size > 1)
                fmt("    mad.wide.u32 %rd0, %r0, $a, %rd0;\n", v);

            if (vt != VarType::Bool) {
                fmt("    $s$b $v, [%rd0];\n",
                    size > 1 ? "ld.global.cs." : "ldu.global.", v, v);
            } else {
                fmt("    $s %w0, [%rd0];\n"
                    "    setp.ne.u16 $v, %w0, 0;\n",
                    size > 1 ? "ld.global.cs.u8" : "ldu.global.u8", v);
            }
            continue;
        } else {
            jitc_cuda_render(index, v);
        }

        if (v->param_type == ParamType::Output) {
            fmt("    ld.$s.u64 %rd0, [$s+$o];\n"
                "    mad.wide.u32 %rd0, %r0, $a, %rd0;\n",
                params_type, params_base, v, v);

            if (vt != VarType::Bool) {
                fmt("    st.global.cs.$b [%rd0], $v;\n", v, v);
            } else {
                fmt("    selp.u16 %w0, 1, 0, $v;\n"
                    "    st.global.cs.u8 [%rd0], %w0;\n", v);
            }
        }
    }

    if (!uses_optix) {
        put("\n"
            "    add.u32 %r0, %r0, %r1;\n"
            "    setp.ge.u32 %p0, %r0, %r2;\n"
            "    @!%p0 bra body;\n"
            "\n"
            "done:\n");
    }

    put("    ret;\n"
        "}\n");

    uint32_t ctr = 0;
    for (auto &it : globals_map) {
        put('\n');
        put(globals.get() + it.second.start, it.second.length);
        put('\n');
        if (!it.first.callable)
            continue;
        it.second.callable_index = ctr++;
    }

    if (callable_count > 0 && !uses_optix) {
        size_t suffix_start = buffer.size(),
               suffix_target =
                   (char *) strstr(buffer.get(), ".address_size 64\n\n") -
                   buffer.get() + 18;

        fmt(".extern .global .u64 callables[$u];\n\n", callable_count_unique);
        buffer.move_suffix(suffix_start, suffix_target);

        fmt("\n.visible .global .align 8 .u64 callables[$u] = {\n",
            callable_count_unique);
        for (auto const &it : globals_map) {
            if (!it.first.callable)
                continue;

            fmt("    func_$Q$Q$s\n",
                it.first.hash.high64, it.first.hash.low64,
                it.second.callable_index + 1 < callable_count_unique ? "," : "");
        }

        put("};\n\n");
    }

    jitc_call_upload(ts);
}

void jitc_cuda_assemble_func(const CallData *call, uint32_t inst,
                             uint32_t in_size, uint32_t in_align,
                             uint32_t out_size, uint32_t out_align,
                             uint32_t n_regs) {
    bool print_labels = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);

    put(".visible .func");
    if (out_size)
        fmt(" (.param .align $u .b8 result[$u])", out_align, out_size);
    fmt(" $s^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(",
        uses_optix ? "__direct_callable__" : "func_");

    if (call->use_index)
        put(".reg .u32 index, ");
    if (call->use_self)
        put(".reg .u32 self, ");
    if (!call->data_map.empty())
        put(".reg .u64 data, ");
    if (in_size)
        fmt(".param .align $u .b8 params[$u], ", in_align, in_size);
    buffer.delete_trailing_commas();

    fmt(") {\n"
        "    // Call: $s\n"
        "    .reg.b8   %b <$u>; .reg.b16  %w<$u>; .reg.b32 %r<$u>;\n"
        "    .reg.b64  %rd<$u>; .reg.f16  %h<$u>; .reg.f32 %f<$u>;\n"
        "    .reg.f64  %d <$u>; .reg.pred %p<$u>;\n",
        call->name.c_str(), n_regs, n_regs, n_regs, n_regs, n_regs, n_regs,
        n_regs, n_regs);

    for (ScheduledVariable &sv : schedule) {
        Variable *v = jitc_var(sv.index);
        const uint32_t vti = v->type;
        const VarType vt = (VarType) vti;
        const VarKind kind = (VarKind) v->kind;

        if (unlikely(print_labels && v->extra)) {
            const char *label = jitc_var_label(sv.index);
            if (label && *label && vt != VarType::Void && kind != VarKind::CallOutput)
                fmt("    // $s\n", label);
        }

        if (kind == VarKind::Counter) {
            fmt("    mov.$b $v, index;\n", v, v);
        } else if (kind == VarKind::CallInput) {
            Variable *a = jitc_var(v->dep[0]);
            if (vt != VarType::Bool) {
                fmt("    ld.param.$b $v, [params+$o];\n", v, v, a);
            } else {
                fmt("    ld.param.u8 %w0, [params+$o];\n"
                    "    setp.ne.u16 $v, %w0, 0;\n", v, a);
            }
        } else if (v->is_evaluated() || vt == VarType::Pointer) {
            uint64_t key = (uint64_t) sv.index + (((uint64_t) inst) << 32);
            auto it = call->data_map.find(key);

            if (unlikely(it == call->data_map.end())) {
                jitc_fail("jitc_cuda_assemble_func(): could not find entry for "
                          "variable r%u in 'data_map'", sv.index);
                continue;
            }

            if (it->second == (uint32_t) -1)
                jitc_fail(
                    "jitc_cuda_assemble_func(): variable r%u is referenced by "
                    "a recorded function call. However, it was evaluated "
                    "between the recording step and code generation (which "
                    "is happening now). This is not allowed.", sv.index);

            uint32_t offset = it->second - call->data_offset[inst];
            if (vt != VarType::Bool)
                fmt("    ld.global.$b $v, [data+$u];\n",
                    v, v, offset);
            else
                fmt("    ld.global.u8 %w0, [data+$u];\n"
                    "    setp.ne.u16 $v, %w0, 0;\n",
                    offset, v);
        } else if (v->is_literal()) {
            fmt("    mov.$b $v, $l;\n", v, v, v);
        } else {
            jitc_cuda_render(sv.index, v);
        }
    }

    for (uint32_t i = 0; i < call->n_out; ++i) {
        const Variable *v = jitc_var(call->inner_out[inst * call->n_out + i]);
        const uint32_t offset = call->out_offset[i];

        if (offset == (uint32_t) -1)
            continue;

        if ((VarType) v->type != VarType::Bool) {
            fmt("    st.param.$b [result+$u], $v;\n", v, offset, v);
        } else {
            fmt("    selp.u16 %w0, 1, 0, $v;\n"
                "    st.param.u8 [result+$u], %w0;\n",
                v, offset);
        }
    }

    put("    ret;\n"
        "}");
}

static const char *reduce_op_name[(int) ReduceOp::Count] = {
    "", "add", "mul", "min", "max", "and", "or"
};

static void jitc_cuda_render(uint32_t index, Variable *v) {
    const char *stmt = nullptr;
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
             *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;

    const ThreadState *ts = thread_state_cuda;

    bool f32_upcast = jitc_is_half(v) && ts->compute_capability < var_kind_fp16_min_compute_cuda[v->kind];

    if (f32_upcast) {
        Variable* b = const_cast<Variable*>(v);
        b->type = (uint32_t)VarType::Float32;
        for (size_t i = 0; i < 4; ++i) {
            Variable* dep = b->dep[i] ? jitc_var(b->dep[i]) : nullptr;
            if (dep) {
                fmt("    cvt.f32.f16 %f$u, %h$u;\n", dep->reg_index, dep->reg_index);
                dep->type = (uint32_t)VarType::Float32;
            }
        }
    }

    switch (v->kind) {
        case VarKind::Undefined:
        case VarKind::Literal:
            fmt("    mov.$b $v, $l;\n", v, v, v);
            break;

        case VarKind::Nop:
            break;

        case VarKind::Neg:
            if (jitc_is_uint(v))
                fmt("    neg.s$u $v, $v;\n", type_size[v->type]*8, v, a0);
            else
                fmt(jitc_is_single(v) ? "    neg.ftz.$t $v, $v;\n"
                                      : "    neg.$t $v, $v;\n",
                    v, v, a0);
            break;

        case VarKind::Not:
            fmt("    not.$b $v, $v;\n", v, v, a0);
            break;

        case VarKind::Sqrt:
            fmt(jitc_is_single(v) || jitc_is_half(v)
                ? "    sqrt.approx.ftz.$t $v, $v;\n"
                : "    sqrt.rn.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Abs:
            fmt("    abs.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Add:
            fmt(jitc_is_single(v) ? "    add.ftz.$t $v, $v, $v;\n"
                                  : "    add.$t $v, $v, $v;\n",
                v, v, a0, a1);
            break;

        case VarKind::Sub:
            fmt(jitc_is_single(v) ? "    sub.ftz.$t $v, $v, $v;\n"
                                  : "    sub.$t $v, $v, $v;\n",
                v, v, a0, a1);
            break;

        case VarKind::Mul:
            if (jitc_is_single(v) || jitc_is_half(v))
                stmt = "    mul.ftz.$t $v, $v, $v;\n";
            else if (jitc_is_double(v))
                stmt = "    mul.$t $v, $v, $v;\n";
            else
                stmt = "    mul.lo.$t $v, $v, $v;\n";
            fmt(stmt, v, v, a0, a1);
            break;

        case VarKind::Div:
            if (jitc_is_single(v) || jitc_is_half(v))
                stmt = "    div.approx.ftz.$t $v, $v, $v;\n";
            else if (jitc_is_double(v))
                stmt = "    div.rn.$t $v, $v, $v;\n";
            else
                stmt = "    div.$t $v, $v, $v;\n";
            fmt(stmt, v, v, a0, a1);
            break;

        case VarKind::Mod:
            fmt("    rem.$t $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Mulhi:
            fmt("    mul.hi.$t $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Fma:
            if (jitc_is_single(v) || jitc_is_half(v))
                stmt = "    fma.rn.ftz.$t $v, $v, $v, $v;\n";
            else if (jitc_is_double(v))
                stmt = "    fma.rn.$t $v, $v, $v, $v;\n";
            else
                stmt = "    mad.lo.$t $v, $v, $v, $v;\n";
            fmt(stmt, v, v, a0, a1, a2);
            break;

        case VarKind::Min:
            fmt(jitc_is_single(v) ? "    min.ftz.$t $v, $v, $v;\n"
                                  : "    min.$t $v, $v, $v;\n",
                                    v, v, a0, a1);
            break;

        case VarKind::Max:
            fmt(jitc_is_single(v) ? "    max.ftz.$t $v, $v, $v;\n"
                                  : "    max.$t $v, $v, $v;\n",
                                    v, v, a0, a1);
            break;

        case VarKind::Ceil:
            fmt("    cvt.rpi.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Floor:
            fmt("    cvt.rmi.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Round:
            fmt("    cvt.rni.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Trunc:
            fmt("    cvt.rzi.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Eq:
            if (jitc_is_bool(a0))
                fmt("    xor.$t $v, $v, $v;\n"
                    "    not.$t $v, $v;", v, v, a0, a1, v, v, v);
            else
                fmt("    setp.eq.$t $v, $v, $v;\n", a0, v, a0, a1);
            break;

        case VarKind::Neq:
            if (jitc_is_bool(a0))
                fmt("    xor.$t $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    setp.ne.$t $v, $v, $v;\n", a0, v, a0, a1);
            break;

        case VarKind::Lt:
            fmt(jitc_is_uint(a0) ? "    setp.lo.$t $v, $v, $v;\n"
                                 : "    setp.lt.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Le:
            fmt(jitc_is_uint(a0) ? "    setp.ls.$t $v, $v, $v;\n"
                                 : "    setp.le.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Gt:
            fmt(jitc_is_uint(a0) ? "    setp.hi.$t $v, $v, $v;\n"
                                 : "    setp.gt.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Ge:
            fmt(jitc_is_uint(a0) ? "    setp.hs.$t $v, $v, $v;\n"
                                 : "    setp.ge.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Select:
            if (!jitc_is_bool(a1)) {
                fmt("    selp.$b $v, $v, $v, $v;\n", v, v, a1, a2, a0);
            } else {
                fmt("    and.pred %p3, $v, $v;\n"
                    "    and.pred %p2, !$v, $v;\n"
                    "    or.pred $v, %p2, %p3;\n",
                    a0, a1, a0, a2, v);
            }
            break;

        case VarKind::Popc:
            if (type_size[v->type] == 4)
                fmt("    popc.$b $v, $v;\n", v, v, a0);
            else
                fmt("    popc.$b %r3, $v;\n"
                    "    cvt.$t.u32 $v, %r3;\n", v, a0, v, v);
            break;

        case VarKind::Clz:
            if (type_size[v->type] == 4)
                fmt("    clz.$b $v, $v;\n", v, v, a0);
            else
                fmt("    clz.$b %r3, $v;\n"
                    "    cvt.$t.u32 $v, %r3;\n", v, a0, v, v);
            break;

        case VarKind::Ctz:
            if (type_size[v->type] == 4)
                fmt("    brev.$b $v, $v;\n"
                    "    clz.$b $v, $v;\n", v, v, a0, v, v, v);
            else
                fmt("    brev.$b $v, $v;\n"
                    "    clz.$b %r3, $v;\n"
                    "    cvt.$t.u32 $v, %r3;\n", v, v, a0, v, v, v, v);
            break;

        case VarKind::And:
            if (a0->type == a1->type)
                fmt("    and.$b $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    selp.$b $v, $v, 0, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Or:
            if (a0->type == a1->type)
                fmt("    or.$b $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    selp.$b $v, -1, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Xor:
            fmt("    xor.$b $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Shl:
            if (type_size[v->type] == 4)
                fmt("    shl.$b $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    cvt.u32.$t %r3, $v;\n"
                    "    shl.$b $v, $v, %r3;\n", a1, a1, v, v, a0);
            break;

        case VarKind::Shr:
            if (type_size[v->type] == 4)
                fmt("    shr.$t $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    cvt.u32.$t %r3, $v;\n"
                    "    shr.$t $v, $v, %r3;\n", a1, a1, v, v, a0);
            break;

        case VarKind::Rcp:
            fmt(jitc_is_single(v) || jitc_is_half(v)
                ? "    rcp.approx.ftz.$t $v, $v;\n"
                : "    rcp.rn.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Rsqrt:
            if (jitc_is_single(v) || jitc_is_half(v))
                fmt("    rsqrt.approx.ftz.$t $v, $v;\n", v, v, a0);
            else
                fmt("    rcp.rn.$t $v, $v;\n"
                    "    sqrt.rn.$t $v, $v;\n", v, v, a0, v, v, v);
            break;

        case VarKind::Sin:
            fmt("    sin.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Cos:
            fmt("    cos.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Exp2:
            fmt("    ex2.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Log2:
            fmt("    lg2.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;


        case VarKind::Cast:
            if (jitc_is_bool(v)) {
                fmt(jitc_is_float(a0) && !jitc_is_half(a0)
                    ? "    setp.ne.$t $v, $v, 0.0;\n"
                    : "    setp.ne.$b $v, $v, 0;\n",
                    a0, v, a0);
            } else if (jitc_is_bool(a0)) {
                // No selp for fp16 so use b16 view
                fmt(jitc_is_half(v)  ? "    selp.$b $v, 0x3C00, 0, $v;\n" :
                    jitc_is_float(v) ? "    selp.$t $v, 1.0, 0.0, $v;\n" :
                                       "    selp.$t $v, 1, 0, $v;\n",
                    v, v, a0);
            } else if (jitc_is_float(v) && !jitc_is_float(a0)) {
                fmt("    cvt.rn.$t.$t $v, $v;\n", v, a0, v, a0);
            } else if (!jitc_is_float(v) && jitc_is_float(a0)) {
                fmt("    cvt.rzi.$t.$t $v, $v;\n", v, a0, v, a0);
            } else {
                fmt(jitc_is_float(v) && jitc_is_float(a0) &&
                            type_size[v->type] < type_size[a0->type]
                        ? "    cvt.rn.$t.$t $v, $v;\n"
                        : "    cvt.$t.$t $v, $v;\n",
                    v, a0, v, a0);
            }
            break;

        case VarKind::Bitcast:
            fmt("    mov.$b $v, $v;\n", v, v, a0);
            break;

        case VarKind::Gather: {
                bool index_zero = a1->is_literal() && a1->literal == 0;
                bool unmasked = a2->is_literal() && a2->literal == 1;
                bool is_bool = v->type == (uint32_t) VarType::Bool;

                if (!unmasked)
                    fmt("    @!$v bra l_$u_masked;\n", a2, v->reg_index);

                if (index_zero) {
                    fmt("    mov.u64 %rd3, $v;\n", a0);
                } else if (type_size[v->type] == 1) {
                    fmt("    cvt.u64.$t %rd3, $v;\n"
                        "    add.u64 %rd3, %rd3, $v;\n", a1, a1, a0);
                } else {
                    fmt("    mad.wide.$t %rd3, $v, $a, $v;\n",
                        a1, a1, v, a0);
                }

                if (is_bool) {
                    fmt("    ld.global.nc.u8 %w0, [%rd3];\n"
                        "    setp.ne.u16 $v, %w0, 0;\n", v);
                } else {
                    fmt("    ld.global.nc.$b $v, [%rd3];\n", v, v);
                }

                if (!unmasked)
                    fmt("    bra.uni l_$u_done;\n\n"
                        "l_$u_masked:\n"
                        "    mov.$b $v, 0;\n\n"
                        "l_$u_done:\n", v->reg_index,
                        v->reg_index, v, v, v->reg_index);
            }
            break;

        case VarKind::Scatter:
            jitc_cuda_render_scatter(v, a0, a1, a2, a3);
            break;

        case VarKind::ScatterInc:
            jitc_cuda_render_scatter_inc(v, a0, a1, a2);
            break;

        case VarKind::ScatterKahan:
            jitc_cuda_render_scatter_kahan(v, index);
            break;


        case VarKind::Counter:
            fmt("    mov.$b $v, %r0;\n", v, v);
            break;

        case VarKind::Call:
            jitc_var_call_assemble((CallData *) v->data, v->reg_index,
                                   a0->reg_index, a1->reg_index, a2->reg_index,
                                   a3 ? a3->reg_index : 0);
            break;

        case VarKind::CallOutput:
            break;

        case VarKind::CallSelf:
            fmt("    mov.u32 $v, self;\n", v);
            break;

        case VarKind::TexLookup:
            fmt("    .reg.$t $v_out_<4>;\n", v, v);
            if (a3)
                fmt("    tex.3d.v4.$t.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v, $v, $v, $v}];\n",
                    v, v, v, v, v, a0, a1, a2, a3, a3);
            else if (a2)
                fmt("    tex.2d.v4.$t.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v, $v}];\n",
                    v, v, v, v, v, a0, a1, a2);
            else
                fmt("    tex.1d.v4.$t.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v}];\n",
                    v, v, v, v, v, a0, a1);
            break;

        case VarKind::TexFetchBilerp:
            fmt("    .reg.f32 %f$u_out_<4>;\n"
                "    tld4.$c.2d.v4.f32.f32 {%f$u_out_0, %f$u_out_1, %f$u_out_2, %f$u_out_3}, [$v, {$v, $v}];\n",
                v->reg_index, "rgba"[v->literal], v->reg_index, v->reg_index, v->reg_index, v->reg_index, a0, a1, a2);
            if (jitc_is_half(v)) {
                fmt("    .reg.f16 %h$u_out_<4>;\n"
                    "    cvt.rn.f16.f32 %h$u_out_0, %f$u_out_0;\n"
                    "    cvt.rn.f16.f32 %h$u_out_1, %f$u_out_1;\n"
                    "    cvt.rn.f16.f32 %h$u_out_2, %f$u_out_2;\n"
                    "    cvt.rn.f16.f32 %h$u_out_3, %f$u_out_3;\n",
                    v->reg_index,
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index);
            }
            break;

#if defined(DRJIT_ENABLE_OPTIX)
        case VarKind::TraceRay:
            jitc_cuda_render_trace(index, v, a0, a1, a2);
            break;
#endif

        case VarKind::Extract:
            fmt("    mov.$b $v, $v_out_$u;\n", v, v, a0, (uint32_t) v->literal);
            break;

        case VarKind::LoopStart: {
                const LoopData *ld = (LoopData *) v->data;
                for (size_t i = 0; i < ld->size; ++i) {
                    Variable *inner_in  = jitc_var(ld->inner_in[i]),
                             *outer_in  = jitc_var(ld->outer_in[i]),
                             *outer_out = jitc_var(ld->outer_out[i]);

                    if (inner_in == outer_in)
                        continue; // ignore eliminated loop state variables

                    if (inner_in->reg_index && outer_in->reg_index)
                        fmt("    mov.$b $v, $v;\n", inner_in, inner_in, outer_in);

                    if (outer_out && outer_out->reg_index && inner_in->reg_index)
                        outer_out->reg_index = inner_in->reg_index;
                }

                fmt("\nl_$u_cond:\n", v->reg_index);
                if (ld->name != "unnamed")
                    fmt("    // Symbolic loop: $s\n", ld->name.c_str());
            }
            break;

        case VarKind::LoopCond:
            fmt("    @!$v bra l_$u_done;\n\n"
                "l_$u_body:\n", a1, a0->reg_index, a0->reg_index);
            break;

        case VarKind::LoopEnd: {
                const LoopData *ld = (LoopData *) a0->data;
                for (uint32_t i = 0; i < ld->size; ++i) {
                    const Variable *inner_out = jitc_var(ld->inner_out[i]),
                                   *inner_in = jitc_var(ld->inner_in[i]);
                    if (inner_out != inner_in && inner_in->reg_index && inner_out->reg_index)
                        fmt("    mov.$b $v, $v;\n", inner_in, inner_in, inner_out);
                }
                fmt("    bra l_$u_cond;\n\n"
                    "l_$u_done:\n",
                    a0->reg_index, a0->reg_index);
            }
            break;

        case VarKind::LoopPhi:
        case VarKind::LoopOutput:
            // No code generated for this node
            break;

        default:
            jitc_fail("jitc_cuda_render(): unhandled variable kind \"%s\"!",
                      var_kind_name[(uint32_t) v->kind]);
    }

    if (f32_upcast) {
        Variable* b = const_cast<Variable*>(v);
        b->type = (uint32_t)VarType::Float16;
        for (size_t i = 0; i < 4; ++i) {
            Variable* dep = b->dep[i] ? jitc_var(b->dep[i]) : nullptr;
            if (dep)
                dep->type = (uint32_t)VarType::Float16;
        }

        fmt("    cvt.rn.f16.f32 $v, %f$u;\n", v, v->reg_index);
    }
}

static void jitc_cuda_render_scatter(const Variable *v,
                                     const Variable *ptr,
                                     const Variable *value,
                                     const Variable *index,
                                     const Variable *mask) {
    bool index_zero = index->is_literal() && index->literal == 0;
    bool unmasked = mask->is_literal() && mask->literal == 1;
    bool is_bool = value->type == (uint32_t) VarType::Bool;
    bool is_half = value->type == (uint32_t) VarType::Float16;

    if (!unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    if (index_zero) {
        fmt("    mov.u64 %rd3, $v;\n", ptr);
    } else if (type_size[v->type] == 1) {
        fmt("    cvt.u64.$t %rd3, $v;\n"
            "    add.u64 %rd3, %rd3, $v;\n", index, index, ptr);
    } else {
        fmt("    mad.wide.$t %rd3, $v, $a, $v;\n",
            index, index, value, ptr);
    }
    const char *op = reduce_op_name[v->literal];
    const ThreadState *ts = thread_state_cuda;


    if (v->literal && callable_depth == 0 && type_size[value->type] == 4 &&
        ts->ptx_version >= 62 && ts->compute_capability >= 70 &&
        (jitc_flags() & (uint32_t) JitFlag::AtomicReduceLocal)) {
        fmt("    {\n"
            "        .func reduce_$s_$t(.param .u64 ptr, .param .$t value);\n"
            "        call.uni reduce_$s_$t, (%rd3, $v);\n"
            "    }\n",
            op, value, value, op, value, value);

        // Intrinsic to perform an intra-warp reduction before writing to global memory
        fmt_intrinsic(
            ".func reduce_$s_$t(.param .u64 ptr,\n"
            "                   .param .$t value) {\n"
            "    .reg .pred %p<14>;\n"
            "    .reg .$t %q<19>;\n"
            "    .reg .b32 %r<41>;\n"
            "    .reg .b64 %rd<2>;\n"
            "\n"
            "    ld.param.u64 %rd0, [ptr];\n"
            "    ld.param.$t %q3, [value];\n"
            "    activemask.b32 %r1;\n"
            "    match.any.sync.b64 %r2, %rd0, %r1;\n"
            "    setp.eq.s32 %p1, %r2, -1;\n"
            "    @%p1 bra.uni fast_path;\n"
            "\n"
            "    brev.b32 %r10, %r2;\n"
            "    bfind.shiftamt.u32 %r40, %r10;\n"
            "    shf.l.wrap.b32 %r12, -2, -2, %r40;\n"
            "    and.b32 %r39, %r2, %r12;\n"
            "    setp.ne.s32 %p2, %r39, 0;\n"
            "    vote.sync.any.pred %p3, %p2, %r1;\n"
            "    @!%p3 bra maybe_scatter;\n"
            "    mov.b32 %r5, %q3;\n"
            "\n"
            "slow_path_repeat:\n"
            "    brev.b32 %r14, %r39;\n"
            "    bfind.shiftamt.u32 %r15, %r14;\n"
            "    shfl.sync.idx.b32 %r17, %r5, %r15, 31, %r1;\n"
            "    mov.b32 %q6, %r17;\n"
            "    @%p2 $s.$t %q3, %q3, %q6;\n"
            "    shf.l.wrap.b32 %r19, -2, -2, %r15;\n"
            "    and.b32 %r39, %r39, %r19;\n"
            "    setp.ne.s32 %p2, %r39, 0;\n"
            "    vote.sync.any.pred %p3, %p2, %r1;\n"
            "    @!%p3 bra maybe_scatter;\n"
            "    bra.uni slow_path_repeat;\n"
            "\n"
            "fast_path:\n"
            "    mov.b32 %r22, %q3;\n"
            "    shfl.sync.down.b32 %r26, %r22, 16, 31, %r1;\n"
            "    mov.b32 %q7, %r26;\n"
            "    $s.$t %q8, %q7, %q3;\n"
            "    mov.b32 %r27, %q8;\n"
            "    shfl.sync.down.b32 %r29, %r27, 8, 31, %r1;\n"
            "    mov.b32 %q9, %r29;\n"
            "    $s.$t %q10, %q8, %q9;\n"
            "    mov.b32 %r30, %q10;\n"
            "    shfl.sync.down.b32 %r32, %r30, 4, 31, %r1;\n"
            "    mov.b32 %q11, %r32;\n"
            "    $s.$t %q12, %q10, %q11;\n"
            "    mov.b32 %r33, %q12;\n"
            "    shfl.sync.down.b32 %r34, %r33, 2, 31, %r1;\n"
            "    mov.b32 %q13, %r34;\n"
            "    $s.$t %q14, %q12, %q13;\n"
            "    mov.b32 %r35, %q14;\n"
            "    shfl.sync.down.b32 %r37, %r35, 1, 31, %r1;\n"
            "    mov.b32 %q15, %r37;\n"
            "    $s.$t %q3, %q14, %q15;\n"
            "    mov.u32 %r40, 0;\n"
            "\n"
            "maybe_scatter:\n"
            "    mov.u32 %r38, %laneid;\n"
            "    setp.ne.s32 %p13, %r40, %r38;\n"
            "    @%p13 bra done;\n"
            "    red.$s.$t [%rd0], %q3;\n"
            "\n"
            "done:\n"
            "    ret;\n"
            "}",
            op, value, value, value, value, op, value, op, value, op, value, op, value, op,
            value, op, value, op, value
        );
    } else if (v->literal && is_half) {
        // Encountered OptiX link errors attempting to use red.global.add.noftz.f16
        // so use f16x2 instead
        fmt("    {\n"
            "        .reg .f16x2 %packed;\n"
            "        .reg .b64 %align, %offset;\n"
            "        .reg .b32 %offset_32;\n"
            "        .reg .f16 %initial;\n"
            "        mov.b16 %initial, 0;\n"
            "        and.b64 %align, %rd3, ~0x3;\n"
            "        and.b64 %offset, %rd3, 0x2;\n"
            "        cvt.u32.s64 %offset_32, %offset;\n"
            "        shl.b32 %offset_32, %offset_32, 3;\n"
            "        mov.b32 %packed, {$v, %initial};\n"
            "        shl.b32 %packed, %packed, %offset_32;\n"
            "        red.global.add.noftz.f16x2 [%align], %packed;\n"
            "    }\n", value);
    } else {
        const char *op_type = v->literal ? "red" : "st";

        if (is_bool)
            fmt("    selp.u16 %w0, 1, 0, $v;\n"
                "    $s.global$s$s.u8 [%rd3], %w0;\n",
                value, op_type, v->literal ? "." : "", op);
        else
            fmt(v->literal ? "    $s.global$s$s.$t [%rd3], $v;\n"
                           : "    $s.global$s$s.$b [%rd3], $v;\n"
                , op_type,
                v->literal ? "." : "", op, value, value);
    }

    if (!unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);
}

static void jitc_cuda_render_scatter_inc(Variable *v,
                                         const Variable *ptr,
                                         const Variable *index,
                                         const Variable *mask) {
    bool index_zero = index->is_literal() && index->literal == 0;
    bool unmasked = mask->is_literal() && mask->literal == 1;

    fmt_intrinsic(
        ".func  (.param .u32 rv) reduce_inc_u32 (.param .u64 ptr) {\n"
        "    .reg .b64 %rd<1>;\n"
        "    .reg .pred %p<1>;\n"
        "    .reg .b32 %r<12>;\n"
        "\n"
        "    ld.param.u64 %rd0, [ptr];\n"
        "    activemask.b32 %r1;\n"
        "    match.any.sync.b64 %r2, %rd0, %r1;\n"
	    "    brev.b32 %r3, %r2;\n"
	    "    bfind.shiftamt.u32 %r4, %r3;\n"
        "    mov.u32 %r5, %lanemask_lt;\n"
        "    and.b32 %r6, %r5, %r2;\n"
        "    popc.b32 %r7, %r6;\n"
        "    setp.ne.u32 %p0, %r6, 0;\n"
        "    @%p0 bra L0;\n"
        "\n"
        "    popc.b32 %r8, %r2;\n"
        "    atom.global.add.u32 %r9, [%rd0], %r8;\n"
        "\n"
        "L0:\n"
        "    shfl.sync.idx.b32 %r10, %r9, %r4, 31, %r2;\n"
        "    add.u32 %r11, %r7, %r10;\n"
        "    st.param.u32 [rv], %r11;\n"
        "    ret;\n"
        "}\n"
    );

    if (!unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    if (index_zero) {
        fmt("    mov.u64 %rd3, $v;\n", ptr);
    } else {
        fmt("    mad.wide.$t %rd3, $v, 4, $v;\n",
            index, index, ptr);
    }

    fmt("    {\n"
        "        .func (.param .u32 rv) reduce_inc_u32 (.param .u64 ptr);\n"
        "        call.uni ($v), reduce_inc_u32, (%rd3);\n"
        "    }\n", v);

    if (!unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);

    v->consumed = 1;
}

static void jitc_cuda_render_scatter_kahan(const Variable *v, uint32_t v_index) {
#if 1
    (void) v;
    (void) v_index;
#else
    const Extra &extra = state.extra[v_index];

    const Variable *ptr_1 = jitc_var(extra.dep[0]),
                   *ptr_2 = jitc_var(extra.dep[1]),
                   *index = jitc_var(extra.dep[2]),
                   *mask = jitc_var(extra.dep[3]),
                   *value = jitc_var(extra.dep[4]);

    bool unmasked = mask->is_literal() && mask->literal == 1;

    if (!unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    fmt("    mad.wide.$t %rd2, $v, $a, $v;\n"
        "    mad.wide.$t %rd3, $v, $a, $v;\n",
        index, index, value, ptr_1,
        index, index, value, ptr_2);

    const char* op_suffix = jitc_is_single(value) ? ".ftz" : "";

    fmt("    {\n"
        "        .reg.$t %before, %after, %value, %case_1, %case_2;\n"
        "        .reg.$t %abs_before, %abs_value, %result;\n"
        "        .reg.pred %cond;\n"
        "\n"
        "        mov.$t %value, $v;\n"
        "        atom.global.add.$t %before, [%rd2], %value;\n"
        "        add$s.$t %after, %before, %value;\n"
        "        sub$s.$t %case_1, %before, %after;\n"
        "        add$s.$t %case_1, %case_1, %value;\n"
        "        sub$s.$t %case_2, %value, %after;\n"
        "        add$s.$t %case_2, %case_2, %before;\n"
        "        abs$s.$t %abs_before, %before;\n"
        "        abs$s.$t %abs_value, %value;\n"
        "        setp.ge.$t %cond, %abs_before, %abs_value;\n"
        "        selp.$t %result, %case_1, %case_2, %cond;\n"
        "        red.global.add.$t [%rd3], %result;\n"
        "    }\n",
        value,
        value,
        value, value,
        value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        value,
        value,
        value);

    if (!unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);
#endif
}

#if defined(DRJIT_ENABLE_OPTIX)
static void jitc_cuda_render_trace(uint32_t index, const Variable *v,
                                   const Variable *valid,
                                   const Variable *pipeline,
                                   const Variable *sbt) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    OptixPipelineData *pipeline_p = (OptixPipelineData *) pipeline->literal;
    OptixShaderBindingTable *sbt_p = (OptixShaderBindingTable*) sbt->literal;
    bool problem = false;

    if (ts->optix_pipeline == state.optix_default_pipeline) {
        ts->optix_pipeline = pipeline_p;
    } else if (ts->optix_pipeline != pipeline_p) {
        jitc_log(
            Warn,
            "jit_eval(): more than one OptiX pipeline was used within a single "
            "kernel, which is not supported. Please split your kernel into "
            "smaller parts (e.g. using `dr::eval()`). Disabling the ray "
            "tracing operation to avoid potential undefined behavior.");
        problem = true;
    }

    if (ts->optix_sbt == state.optix_default_sbt) {
        ts->optix_sbt = sbt_p;
    } else if (ts->optix_sbt != sbt_p) {
        jitc_log(
            Warn,
            "jit_eval(): more than one OptiX shader binding table was used "
            "within a single kernel, which is not supported. Please split your "
            "kernel into smaller parts (e.g. using `dr::eval()`). Disabling "
            "the ray tracing operation to avoid potential undefined behavior.");
        problem = true;
    }

#if 0
    const Extra &extra = state.extra[index];
    uint32_t payload_count = extra.n_dep - 15;

    fmt("    .reg.u32 $v_out_<32>;\n", v);

    if (problem) {
        for (uint32_t i = 0; i < 32; ++i)
            fmt("    mov.b32 $v_out_$u, 0;\n", v, i);
        return;
    }

    bool masked = !valid->is_literal() || valid->literal != 1;
    if (masked)
        fmt("    @!$v bra l_masked_$u;\n", valid, v->reg_index);

    fmt("    .reg.u32 $v_payload_type, $v_payload_count;\n"
        "    mov.u32 $v_payload_type, 0;\n"
        "    mov.u32 $v_payload_count, $u;\n",
            v, v, v, v, payload_count);

    put("    call (");
    for (uint32_t i = 0; i < 32; ++i)
        fmt("$v_out_$u$s", v, i, i + 1 < 32 ? ", " : "");
    put("), _optix_trace_typed_32, (");

    fmt("$v_payload_type, ", v);
    for (uint32_t i = 0; i < 15; ++i)
        fmt("$v, ", jitc_var(extra.dep[i]));

    fmt("$v_payload_count, ", v);
    for (uint32_t i = 15; i < extra.n_dep; ++i)
        fmt("$v$s", jitc_var(extra.dep[i]), (i - 15 < 32) ? ", " : "");

    for (uint32_t i = payload_count; i < 32; ++i)
        fmt("$v_out_$u$s", v, i, (i + 1 < 32) ? ", " : "");

    put(");\n");

    if (masked)
        fmt("\nl_masked_$u:\n", v->reg_index);
#endif
}
#endif

/// Virtual function call code generation -- CUDA/PTX-specific bits
void jitc_var_call_assemble_cuda(CallData *call, uint32_t call_reg,
                                 uint32_t self_reg, uint32_t mask_reg,
                                 uint32_t offset_reg, uint32_t data_reg,
                                 uint32_t in_size, uint32_t in_align,
                                 uint32_t out_size, uint32_t out_align) {
    // =====================================================
    // 1. Conditional branch
    // =====================================================

    fmt("\n    @!%p$u bra l_masked_$u;\n\n"
        "    { // Call: $s\n", mask_reg, call_reg, call->name.c_str());

    // =====================================================
    // 2. Determine unique callable ID
    // =====================================================

    // %r3: callable ID
    // %rd3: (high 32 bit): data offset
    fmt("\n"
        "        mad.wide.u32 %rd3, %r$u, 8, %rd$u;\n"
        "        ld.global.u64 %rd3, [%rd3];\n"
        "        cvt.u32.u64 %r3, %rd3;\n",
        self_reg, offset_reg);

    // =====================================================
    // 3. Turn callable ID into a function pointer
    // =====================================================

    if (!uses_optix)
        put("        ld.global.u64 %rd2, callables[%r3];\n");
    else
        put("        call (%rd2), _optix_call_direct_callable, (%r3);\n");

    // =====================================================
    // 4. Obtain pointer to supplemental call data
    // =====================================================

    if (data_reg)
        fmt("        shr.u64 %rd3, %rd3, 32;\n"
            "        add.u64 %rd3, %rd3, %rd$u;\n",
            data_reg);

    // %rd2: function pointer (if applicable)
    // %rd3: call data pointer with offset

    // =====================================================
    // 5. Generate the actual function call
    // =====================================================

    put("\n");

    // Special handling for predicates
    for (uint32_t in : call->outer_in) {
        const Variable *v = jitc_var(in);
        if (!v->reg_index)
            continue;

        const Variable *v2 = jitc_var(in);
        if ((VarType) v2->type != VarType::Bool)
            continue;

        fmt("        selp.u16 %w$u, 1, 0, %p$u;\n",
            v2->reg_index, v2->reg_index);
    }

    put("        {\n");

    // Call prototype
    put("            proto: .callprototype");
    if (out_size)
        fmt(" (.param .align $u .b8 result[$u])", out_align, out_size);
    put(" _(");

    if (call->use_index)
        put(".reg .u32 index, ");
    if (call->use_self)
        put(".reg .u32 self, ");
    if (data_reg)
        put(".reg .u64 data, ");
    if (in_size)
        fmt(".param .align $u .b8 params[$u], ", in_align, in_size);
    buffer.delete_trailing_commas();
    put(");\n");

    // Input/output parameter arrays
    if (out_size)
        fmt("            .param .align $u .b8 out[$u];\n", out_align, out_size);
    if (in_size)
        fmt("            .param .align $u .b8 in[$u];\n", in_align, in_size);

    // =====================================================
    // 5.1. Pass the input arguments
    // =====================================================

    for (uint32_t in : call->outer_in) {
        const Variable *v = jitc_var(in);
        if (!v->reg_index)
            continue;
        const Variable *v2 = jitc_var(in);

        const char *tname  = type_name_ptx_bin[v2->type],
                   *prefix = type_prefix[v2->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v2->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        fmt("            st.param.$s [in+$u], $s$u;\n",
            tname, v->param_offset, prefix, v2->reg_index);
    }

    put("            call ");
    if (out_size)
        put("(out), ");
    put("%rd2, (");
    if (call->use_index) {
        if (callable_depth == 0)
            put("%r0, ");
        else
            put("index, ");
    }
    if (call->use_self)
        fmt("%r$u, ", self_reg);
    if (data_reg)
        put("%rd3, ");
    if (in_size)
        put("in, ");
    buffer.delete_trailing_commas();
    put("), proto;\n");


    // =====================================================
    // 5.2. Read back the output arguments
    // =====================================================

    for (uint32_t i = 0; i < call->n_out; ++i) {
        const Variable *v = jitc_var(call->outer_out[i]);
        if (!v || !v->reg_index)
            continue;

        const char *tname = type_name_ptx_bin[v->type],
                   *prefix = type_prefix[v->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        fmt("            ld.param.$s $s$u, [out+$u];\n",
            tname, prefix, v->reg_index, call->out_offset[i]);
    }

    put("        }\n\n");

    // =====================================================
    // 6. Special handling for predicates return value(s)
    // =====================================================

    for (uint32_t i = 0; i < call->n_out; ++i) {
        const Variable *v = jitc_var(call->outer_out[i]);
        if (!v || !v->reg_index || (VarType) v->type != VarType::Bool)
            continue;

        // Special handling for predicates
        fmt("        setp.ne.u16 %p$u, %w$u, 0;\n",
            v->reg_index, v->reg_index);
    }


    fmt("        bra.uni l_done_$u;\n"
        "    }\n", call_reg);

    // =====================================================
    // 7. Prepare output registers for masked lanes
    // =====================================================

    fmt("\nl_masked_$u:\n", call_reg);

    for (uint32_t i = 0; i < call->n_out; ++i) {
        const Variable *v = jitc_var(call->outer_out[i]);
        if (!v || !v->reg_index)
            continue;
        fmt("    mov.$b $v, 0;\n", v, v);
    }

    fmt("\nl_done_$u:\n", call_reg);
}

