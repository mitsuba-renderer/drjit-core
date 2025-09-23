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
 *  $a      Variable      `4`                Variable alignment in bytes
 * --------------------------------------------------------------------------
 *  $l      Variable      `1`                Literal value of variable (hex)
 * --------------------------------------------------------------------------
 *  The following format strings facilitate loading/storing of predicates,
 *  which are a somewhat of a corner case in PTX.
 * --------------------------------------------------------------------------
 *  $B      Variable      `b32`              Like $b. For predicates: 'b8'
 *  $V      Variable      `%f1234`           Like $v. For predicates: '%w0'
 * --------------------------------------------------------------------------
 */

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "call.h"
#include "cond.h"
#include "loop.h"
#include "optix.h"
#include "trace.h"
#include "cuda_eval.h"
#include "cuda_array.h"
#include "cuda_scatter.h"
#include "cuda_packet.h"
#if defined(DRJIT_ENABLE_OPTIX)
#  include <drjit-core/optix.h>
#  include "optix_coop_vec.h"
#endif

// Forward declarations
static void jitc_cuda_render(Variable *v);

#if defined(DRJIT_ENABLE_OPTIX)
static void jitc_cuda_render_trace(const Variable *v,
                                   const Variable *valid,
                                   const Variable *pipeline,
                                   const Variable *sbt);

static void jitc_cuda_render_reorder(const Variable *, const Variable *);
#endif

void jitc_cuda_assemble(ThreadState *ts, ScheduledGroup group,
                        uint32_t n_regs, uint32_t n_params) {
    uint32_t flags = jitc_flags();
    
    bool params_global = !uses_optix && n_params > jitc_cuda_arg_limit;
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
            ".param .align 8 .b8 params[$u]) {\n",
            params_global ? 8u : (n_params * (uint32_t) sizeof(void *)));
        
        if ((flags & (uint32_t) JitFlag::SpillToSharedMemory) && 
           ts->compute_capability >= 75 && ts->ptx_version >= 87)
            fmt("    .pragma \"enable_smem_spilling\";\n\n");
    } else {
        fmt(".const .align 8 .b8 params[$u];\n\n"
            ".entry __raygen__^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^() {\n",
            n_params * (uint32_t) sizeof(void *));
    }

    fmt("    .reg.b8  %b <$u>; .reg.b16  %w<$u>; .reg.b32 %r<$u>;\n"
        "    .reg.b64 %rd<$u>; .reg.f16  %h<$u>; .reg.f32 %f<$u>;\n"
        "    .reg.f64 %d <$u>; .reg.pred %p<$u>;\n\n",
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

        fmt("    setp.ge.u32 %p0, %r0, %r2;\n"
            "    @%p0 ret;\n\n"
            "body: // sm_$u\n",
            state.devices[ts->device].compute_capability);
    } else {
        put("    call (%r0), _optix_get_launch_index_x, ();\n"
            "    call (%r2), _optix_get_launch_dimension_x, ();\n"
            "    ld.const.u32 %r1, [params+4];\n"
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
        ParamType ptype = (ParamType) v->param_type;

        if (unlikely(print_labels && v->extra)) {
            const char *label = jitc_var_label(index);
            if (label && *label && vt != VarType::Void && kind != VarKind::CallOutput)
                fmt("    // $s\n", label);
        }

        if (likely(ptype == ParamType::Input)) {
            if (v->is_literal()) {
                fmt("    ld.$s.u64 $v, [$s+$o];\n", params_type, v, params_base, v);
                continue;
            } else {
                fmt("    ld.$s.u64 %rd0, [$s+$o];\n", params_type, params_base, v);
            }

            if (size > 1)
                fmt("    mad.wide.u32 %rd0, %r0, $a, %rd0;\n", v);

            if (!v->is_array()) {
                fmt("    $s$B $V, [%rd0];\n",
                    size > 1 ? "ld.global.cs." : "ldu.global.", v, v);
                if (vt == VarType::Bool)
                    fmt("    setp.ne.u16 $v, %w0, 0;\n", v);
            } else {
                jitc_cuda_render_array_memcpy_in(v);
            }
            continue;
        } else {
            jitc_cuda_render(v);
        }

        if (ptype == ParamType::Output) {
            jitc_assert(jitc_var(index) == v,
                        "Unexpected mutation of the variable data structure.");

            fmt("    ld.$s.u64 %rd0, [$s+$o];\n"
                "    mad.wide.u32 %rd0, %r0, $a, %rd0;\n",
                params_type, params_base, v, v);

            if (!v->is_array()) {
                if (vt == VarType::Bool)
                    fmt("    selp.u16 %w0, 1, 0, $v;\n", v);
                fmt("    st.global.cs.$B [%rd0], $V;\n", v, v);
            } else {
                jitc_cuda_render_array_memcpy_out(v);
            }
        }
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
    uint32_t flags = jitc_flags();

    bool print_labels = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (flags & (uint32_t) JitFlag::PrintIR);

    put(".visible .func");
    if (out_size)
        fmt(" (.param .align $u .b8 result[$u])", out_align, out_size);
    fmt(" $s^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(",
        uses_optix ? (jitc_optix_use_continuation_callables()
                          ? "__continuation_callable__"
                          : "__direct_callable__") : "func_");

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

    // Warning: do not rewrite this into a range-based for loop.
    // The memory location of 'schedule' may change.
    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
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
                    "    setp.ne.u16 $v, %w0, 0;\n", a, v);
            }
        } else if (v->is_evaluated() || (vt == VarType::Pointer && kind == VarKind::Literal)) {
            uint64_t key = (uint64_t) sv.index + (((uint64_t) inst) << 32);
            auto it = call->data_map.find(key);

            if (unlikely(it == call->data_map.end())) {
                jitc_fail("jitc_cuda_assemble_func(): could not find entry for "
                          "variable r%u in 'data_map' for function %s", sv.index, call->name.c_str());
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
            jitc_cuda_render(v);
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

static inline uint32_t jitc_fp16_min_compute_cuda(VarKind kind) {
    switch (kind) {
        case VarKind::Sqrt:
        case VarKind::Div:
        case VarKind::Rcp:
        case VarKind::RcpApprox:
            return UINT_MAX;
        case VarKind::Min:
        case VarKind::Max:
            return 80;
        default:
            return 53;
    }
}

void jitc_cuda_render_loop_end(Variable *a0) {
    const LoopData *ld = (LoopData *) a0->data;
    uint32_t size = (uint32_t) ld->size;

    // Initialize scratch space
    for (uint32_t i = 0; i < size; ++i) {
        Variable *in  = jitc_var(ld->inner_in[i]),
                 *out = jitc_var(ld->inner_out[i]);
        in->scratch = out->scratch = 0;
    }

    // Mark inputs that will be read
    for (uint32_t i = 0; i < size; ++i) {
        Variable *in  = jitc_var(ld->inner_in[i]),
                 *out = jitc_var(ld->inner_out[i]);
        if (in == out || !in->reg_index || !out->reg_index)
            continue;

        in->scratch = 1;
    }

    /* If inner_out[i] == inner_in[j] (with i != j), we need to copy
       `inner_in[j]` to a temporary to ensure that we can store its
       original value to `inner_out[i]`, rather than its updated
       value (i.e after the `inner_in[j] = inner_out[j]` assignment). */
    for (uint32_t i = 0; i < size; ++i) {
        Variable *in  = jitc_var(ld->inner_in[i]),
                 *out = jitc_var(ld->inner_out[i]);

        // If `out->scratch != 1` then we're certain that `out` is an input
        if (in == out || !in->reg_index || !out->reg_index || out->scratch != 1)
            continue;

        fmt("    .reg.$b $v_tmp;\n"
            "    mov.$b $v_tmp, $v;\n",
            in, in, in, in, out);

        // Mark that we have created a temporary
        out->scratch = 2;
    }

    for (uint32_t i = 0; i < size; ++i) {
        Variable *in  = jitc_var(ld->inner_in[i]),
                 *out = jitc_var(ld->inner_out[i]);

        if (in == out || !in->reg_index || !out->reg_index)
            continue;

        if (out->scratch == 2)
            fmt("    mov.$b $v, $v_tmp;\n", in, in, in);
        else
            fmt("    mov.$b $v, $v;\n", in, in, out);
    }

    fmt("    bra l_$u_cond;\n\n"
        "l_$u_done:\n",
        a0->reg_index, a0->reg_index);
}

static bool jitc_int8_unsupported(VarKind kind) {
    switch (kind) {
        case VarKind::Literal:
        case VarKind::Select:
        case VarKind::Add:
        case VarKind::Sub:
        case VarKind::Mul:
        case VarKind::Fma:
        case VarKind::Abs:
        case VarKind::Neg:
        case VarKind::Not:
        case VarKind::Or:
        case VarKind::And:
        case VarKind::Xor:
        case VarKind::Shl:
        case VarKind::Shr:
        case VarKind::Popc:
        case VarKind::Cast:
        case VarKind::Bitcast:
            return true;
        default:
            return false;
    }
}

static void jitc_cuda_render(Variable *v) {
    const char *stmt = nullptr;
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
             *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;

#if defined(DRJIT_ENABLE_OPTIX)
    if (v->coop_vec)
        return jitc_optix_render_coop_vec(v, a0, a1, a2, a3);
#endif

    const ThreadState *ts = thread_state_cuda;

    VarType orig_vt = (VarType) v->type,
            vt = orig_vt;
    VarKind kind = (VarKind) v->kind;

    switch (orig_vt) {
        case VarType::Float16:
            if (ts->compute_capability < jitc_fp16_min_compute_cuda(kind))
                vt = VarType::Float32;
            break;

        case VarType::Int8:
        case VarType::UInt8:
            if (jitc_int8_unsupported(kind))
                vt = orig_vt == VarType::Int8 ? VarType::Int16 : VarType::UInt16;
            break;

        case VarType::Bool:
            if (a0) {
                if ((VarType) a0->type == VarType::UInt8) {
                    orig_vt = VarType::UInt8;
                    vt = VarType::UInt16;
                } else if ((VarType) a0->type == VarType::Int8) {
                    orig_vt = VarType::Int8;
                    vt = VarType::Int16;
                }
            }
            break;

        default:
            break;
    }

    if (vt != orig_vt) {
        Variable* b = const_cast<Variable*>(v);

        if ((VarType) b->type == orig_vt)
            b->type = (uint32_t) vt;

        for (size_t i = 0; i < 4; ++i) {
            Variable* dep = b->dep[i] ? jitc_var(b->dep[i]) : nullptr;

            if (dep && (VarType) dep->type == orig_vt) {
                fmt("    cvt.$s.$t $s$u, $v;\n",
                    type_name_ptx[(uint32_t) vt],
                    dep,
                    type_prefix[(uint32_t) vt],
                    dep->reg_index,
                    dep
                );

                dep->type = (uint32_t) vt;
            }
        }
    }

    switch (kind) {
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

        case VarKind::SqrtApprox:
            fmt("    sqrt.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Sqrt:
            fmt("    sqrt.rn.$t $v, $v;\n", v, v, a0);
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

        case VarKind::DivApprox:
            stmt = "    div.approx.ftz.$t $v, $v, $v;\n";
            fmt(stmt, v, v, a0, a1);
            break;

        case VarKind::Div:
            if (jitc_is_int(v))
                stmt = "    div.$t $v, $v, $v;\n";
            else
                stmt = "    div.rn.$t $v, $v, $v;\n";

            fmt(stmt, v, v, a0, a1);
            break;

        case VarKind::Mod:
            fmt("    rem.$t $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::MulHi:
            fmt("    mul.hi.$t $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::MulWide:
            fmt("    mul.wide.$t $v, $v, $v;\n", a0, v, a0, a1);
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
            if (!jitc_is_bool(a0)) {
                fmt("    setp.eq.$t $v, $v, $v;\n", a0, v, a0, a1);
            } else {
                fmt("    xor.$t $v, $v, $v;\n"
                    "    not.$t $v, $v;", v, v, a0, a1, v, v, v);
            }
            break;

        case VarKind::Neq:
            if (!jitc_is_bool(a0))
                fmt("    setp.ne.$t $v, $v, $v;\n", a0, v, a0, a1);
            else
                fmt("    xor.$t $v, $v, $v;\n", v, v, a0, a1);
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

        case VarKind::Array:
            jitc_cuda_render_array(v, a0);
            break;

        case VarKind::ArrayInit:
            jitc_cuda_render_array_init(v, a0 ,a1);
            break;

        case VarKind::ArrayWrite:
            jitc_cuda_render_array_write(v, a0, a1, a2, a3);
            break;

        case VarKind::ArrayRead:
            jitc_cuda_render_array_read(v, a0, a1, a2);
            break;

        case VarKind::ArrayPhi:
            v->reg_index = a0->reg_index;
            break;

        case VarKind::ArraySelect:
            jitc_cuda_render_array_select(v, a0, a1, a2);
            break;

#if defined(DRJIT_ENABLE_OPTIX)
        case VarKind::CoopVecUnpack:
            jitc_optix_render_coop_vec_unpack(v, a0);
            break;

        case VarKind::CoopVecAccum:
            jitc_optix_render_coop_vec_accum(v, a0, a1, a2);
            break;

        case VarKind::CoopVecOuterProductAccum:
            jitc_optix_render_coop_vec_outer_product_accum(v, a0, a1, a2, a3);
            break;
#endif

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

        case VarKind::Brev:
            fmt("    brev.$b $v, $v;\n", v, v, a0);
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

        case VarKind::RcpApprox:
            fmt("    rcp.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Rcp:
            fmt("    rcp.rn.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::RSqrtApprox:
            fmt("    rsqrt.approx.ftz.$t $v, $v;\n", v, v, a0);
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

        case VarKind::Tanh:
            fmt("    tanh.approx.$t $v, $v;\n", v, v, a0);
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

        case VarKind::Bitcast: {
            VarType a0_vtype = (VarType) a0->type;
            if (jitc_is_b8(orig_vt) && jitc_is_b8(a0_vtype)) {
                // Cast or bitcast from i8/u8 to i8/u8. Since there's
                // no `mov.b8` instruction, we go through i16/u16. Note that the
                // target variable was already automatically upcast to i16/u16.
                VarType dep_vtype = (a0_vtype == VarType::Int8) ? VarType::Int16 : VarType::UInt16;
                fmt("    cvt.$s.$t $s$u, $v;\n"
                    "    mov.$b $v, $s$u;\n",
                    type_name_ptx[(uint32_t) dep_vtype],
                    a0,
                    type_prefix[(uint32_t) dep_vtype],
                    a0->reg_index,
                    a0,
                    v,
                    v,
                    type_prefix[(uint32_t) dep_vtype],
                    a0->reg_index
                );

            } else {
                fmt("    mov.$b $v, $v;\n", v, v, a0);
            }
            break;
        }

        case VarKind::Gather: {
                bool index_zero = a1->is_literal() && a1->literal == 0;
                bool is_masked = !a2->is_literal() || a2->literal != 1;
                bool is_bool = v->type == (uint32_t) VarType::Bool;

                if (index_zero) {
                    fmt("    mov.u64 %rd3, $v;\n", a0);
                } else if (type_size[v->type] == 1) {
                    fmt("    cvt.u64.$t %rd3, $v;\n"
                        "    add.u64 %rd3, %rd3, $v;\n", a1, a1, a0);
                } else {
                    fmt("    mad.wide.$t %rd3, $v, $a, $v;\n",
                        a1, a1, v, a0);
                }

                if (is_masked) {
                    if (is_bool)
                        fmt("    mov.b16 %w0, 0;\n");
                    else if ((VarType) v->type == VarType::UInt8 || (VarType) v->type == VarType::Int8)
                        // There is no `mov.b8`
                        fmt("    cvt.u8.u16 $v, 0;\n", v);
                    else
                        fmt("    mov.$b $v, 0;\n", v, v);
                    fmt("    @$v ", a2);
                } else {
                    put("    ");
                }

                if (is_bool)
                    fmt("ld.global.nc.u8 %w0, [%rd3];\n"
                        "    setp.ne.u16 $v, %w0, 0;\n", v);
                else
                    fmt("ld.global.nc.$b $v, [%rd3];\n", v, v);
            }
            break;

        case VarKind::Scatter:
            if (v->literal)
                jitc_cuda_render_scatter_reduce(v, a0, a1, a2, a3);
            else
                jitc_cuda_render_scatter(v, a0, a1, a2, a3);
            break;

        case VarKind::PacketGather:
            jitc_cuda_render_gather_packet(v, a0, a1, a2);
            break;

        case VarKind::PacketScatter:
            jitc_cuda_render_scatter_packet(v, a0, a1, a2);
            break;

        case VarKind::ScatterInc:
            jitc_cuda_render_scatter_inc(v, a0, a1, a2);
            break;

        case VarKind::ScatterKahan:
            jitc_cuda_render_scatter_add_kahan(v, a0, a1, a2, a3);
            break;

        case VarKind::ScatterExch:
            jitc_cuda_render_scatter_exch(v, a0, a1, a2, a3);
            break;

        case VarKind::ScatterCAS:
            jitc_cuda_render_scatter_cas(v, a0, a1, a2, a3);
            break;

        case VarKind::BoundsCheck:
            fmt("    setp.ge.and.u32 $v, $v, $u, $v;\n"
                "    @$v st.global.u32 [$v], $v;\n"
                "    xor.pred $v, $v, $v;\n"
                ,
                v, a0, (uint32_t) v->literal, a1,
                v, a2, a0,
                v, a1, v);
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

            if (v->literal /* masked */) {
                fmt("    mov.v4.$b {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, {0, 0, 0, 0};\n"
                    "    setp.ne.b64 %p3, $v, 0;\n"
                    "    @%p3 ",
                    v, v, v, v, v,
                    a0);
            } else {
                put("    ");
            }

            if (a3)
                fmt("tex.3d.v4.$t.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v, $v, $v, $v}];\n",
                    v, v, v, v, v, a0, a1, a2, a3, a3);
            else if (a2)
                fmt("tex.2d.v4.$t.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v, $v}];\n",
                    v, v, v, v, v, a0, a1, a2);
            else
                fmt("tex.1d.v4.$t.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v}];\n",
                    v, v, v, v, v, a0, a1);
            break;

        case VarKind::TexFetchBilerp:
            fmt("    .reg.f32 %f$u_out_<4>;\n",
                v->reg_index);
            if (!(a1->is_literal() && a1->literal == 1)) {
                fmt("    mov.v4.f32 {%f$u_out_0, %f$u_out_1, %f$u_out_2, %f$u_out_3}, {0.0, 0.0, 0.0, 0.0};\n"
                    "    @$v ",
                    v->reg_index, v->reg_index, v->reg_index, v->reg_index,
                    a1);
            } else {
                put("    ");
            }
            fmt("tld4.$c.2d.v4.f32.f32 {%f$u_out_0, %f$u_out_1, %f$u_out_2, %f$u_out_3}, [$v, {$v, $v}];\n",
                "rgba"[v->literal], v->reg_index, v->reg_index, v->reg_index, v->reg_index, a0, a2, a3);
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

        case VarKind::VectorLoad:
            fmt("    mov.$b $v, 0;\n"
                "    .reg.u64 $v_src;\n"
                "    add.u64 $v_src, $v, $u;\n"
                "    @$v ld.global.$b $v, [$v_src];\n",
                v, v,
                v,
                v, a0, (uint32_t) v->literal,
                a1, v, v, v);
            break;

#if defined(DRJIT_ENABLE_OPTIX)
        case VarKind::TraceRay:
            jitc_cuda_render_trace(v, a0, a1, a2);
            break;
#endif

        case VarKind::Extract:
            if (vt == VarType::UInt8 || vt == VarType::Int8) {
                fmt("    cvt.u16.u8 %w3, $v_out_$u;\n", a0, (uint32_t) v->literal);
                fmt("    cvt.u8.u16 $v, %w3;\n", v);
            } else {
                fmt("    mov.$b $v, $v_out_$u;\n", v, v, a0, (uint32_t) v->literal);
            }
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

        case VarKind::LoopEnd:
            jitc_cuda_render_loop_end(a0);
            break;


        case VarKind::LoopPhi:
            if (v->is_array())
                v->reg_index = a3->reg_index;
            break;

        case VarKind::LoopOutput:
            // No code generated for this node
            break;

        case VarKind::CondStart: {
                const CondData *cd = (CondData *) v->data;
                if (cd->name != "unnamed")
                    fmt("\n    // Symbolic conditional: $s\n", cd->name.c_str());
                fmt("    @!$v bra l_$u_f;\n"
                    "\nl_$u_t:\n",
                    a0, v->reg_index, v->reg_index);
            }
            break;

        case VarKind::CondMid: {
                const CondData *cd = (CondData *) a0->data;
                for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                    Variable *vt2 = jitc_var(cd->indices_t[i]),
                             *vo  = jitc_var(cd->indices_out[i]);
                    if (!vo || !vo->reg_index)
                        continue;
                    fmt("    mov.$b $v, $v;\n", vo, vo, vt2);
                }
                fmt("    bra l_$u_end;\n\n"
                    "l_$u_f:\n", a0->reg_index, a0->reg_index);
            }
            break;

        case VarKind::CondEnd: {
                const CondData *cd = (CondData *) a0->data;
                for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                    Variable *vf = jitc_var(cd->indices_f[i]),
                             *vo = jitc_var(cd->indices_out[i]);
                    if (!vo || !vo->reg_index)
                        continue;
                    fmt("    mov.$b $v, $v;\n", vo, vo, vf);
                }
                fmt("\nl_$u_end:\n", a0->reg_index);
            }
            break;

        case VarKind::CondOutput: // No output
            break;

#if defined(DRJIT_ENABLE_OPTIX)
        case VarKind::ReorderThread:
            jitc_cuda_render_reorder(v, a0);
            break;
#endif

        default:
            jitc_fail("jitc_cuda_render(): unhandled variable kind \"%s\"!",
                      var_kind_name[(uint32_t) v->kind]);
    }

    if (vt != orig_vt) {
        Variable* b = const_cast<Variable*>(v);

        for (size_t i = 0; i < 4; ++i) {
            Variable* dep = b->dep[i] ? jitc_var(b->dep[i]) : nullptr;
            if (dep && (VarType) dep->type == vt)
                dep->type = (uint32_t) orig_vt;
        }

        if ((VarType) b->type == vt) {
            b->type = (uint32_t) orig_vt;

            fmt("    cvt$s.$t.$s $v, $s$u;\n",
                orig_vt == VarType::Float16 ? ".rn" : "",
                v,
                type_name_ptx[(uint32_t) vt],
                v,
                type_prefix[(uint32_t) vt],
                v->reg_index);
        }
    }
}

#if defined(DRJIT_ENABLE_OPTIX)
static void jitc_cuda_render_trace(const Variable *v,
                                   const Variable *valid,
                                   const Variable *pipeline,
                                   const Variable *sbt) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    OptixPipelineData *pipeline_p = (OptixPipelineData *) pipeline->literal;
    OptixShaderBindingTable *sbt_p = (OptixShaderBindingTable*) sbt->literal;
    bool disabled = false, some_masked = false;
    TraceData *td = (TraceData *) v->data;

    if (ts->optix_pipeline == state.optix_default_pipeline) {
        ts->optix_pipeline = pipeline_p;
    } else if (ts->optix_pipeline != pipeline_p) {
        jitc_log(
            Warn,
            "jit_eval(): more than one OptiX pipeline was used within a single "
            "kernel, which is not supported. Please split your kernel into "
            "smaller parts (e.g. using `dr::eval()`). Disabling this ray "
            "tracing operation to avoid potential undefined behavior.");
        disabled = true;
    }

    if (ts->optix_sbt == state.optix_default_sbt) {
        ts->optix_sbt = sbt_p;
    } else if (ts->optix_sbt != sbt_p) {
        jitc_log(
            Warn,
            "jit_eval(): more than one OptiX shader binding table was used "
            "within a single kernel, which is not supported. Please split your "
            "kernel into smaller parts (e.g. using `dr::eval()`). Disabling "
            "this ray tracing operation to avoid potential undefined behavior.");
        disabled = true;
    }

    fmt("    .reg.u32 $v_out_<32>;\n", v);

    if (valid->is_literal()) {
        if (valid->literal == 0)
            disabled = true;
    } else if (!disabled) {
        some_masked = true;
        if (td->reorder && jit_flag(JitFlag::ShaderExecutionReordering))
            fmt("    @!$v bra l_reorder_$u;\n", valid, v->reg_index);
        else
            fmt("    @!$v bra l_masked_$u;\n", valid, v->reg_index);
    }

    if (disabled) {
        for (uint32_t i = 0; i < 32; ++i)
            fmt("    mov.b32 $v_out_$u, 0;\n", v, i);
        return;
    }

    uint32_t payload_count = (uint32_t) td->indices.size() - 15;

    fmt("    .reg.u32 $v_z, $v_count;\n"
        "    mov.u32 $v_z, 0;\n"
        "    mov.u32 $v_count, $u;\n",
        v, v, v, v, payload_count);

    // =====================================================
    // 1. Traverse
    // =====================================================
    put("    call (");
    for (uint32_t i = 0; i < 32; ++i)
        fmt("$v_out_$u$s", v, i, i + 1 < 32 ? ", " : "");
    put("), _optix_hitobject_traverse, (");

    fmt("$v_z, ", v);

    for (uint32_t i = 0; i < 15 + 32; ++i) {
        if (i == 15)
            fmt("$v_count, ", v);

        if (i < td->indices.size())
            fmt("$v", jitc_var(td->indices[i]));
        else
            fmt("$v_z", v);

        if (i + 1 < 15 + 32)
            put(", ");
    }

    put(");\n");

    // =====================================================
    // 2. Reorder
    // =====================================================
    if (td->reorder && jit_flag(JitFlag::ShaderExecutionReordering)) {
        if (some_masked)
            fmt("\nl_reorder_$u:\n"
                "    @!$v call (), _optix_hitobject_make_nop, ();\n",
                v->reg_index,
                valid);

        if (td->reorder_hint_num_bits == 0) {
            fmt("    call (), _optix_hitobject_reorder, ($v_z, $v_z);\n", v, v);
        } else {
            fmt("    .reg .u32 $v_hint_bits;\n"
                "    mov.u32 $v_hint_bits, $u;\n"
                "    call (), _optix_hitobject_reorder, ($v, $v_hint_bits);\n",
                v,
                v, td->reorder_hint_num_bits,
                jitc_var(td->reorder_hint), v);
        }

        if (some_masked)
            fmt("    @!$v bra l_masked_$u;\n", valid, v->reg_index);
    }

    // =====================================================
    // 3. Get HitObject fields
    // =====================================================
    size_t n_fields = td->hit_object_fields.size();
    for (uint32_t i = 0; i < n_fields; ++i) {
        uint32_t field_i = td->hit_object_fields[i];
        switch ((OptixHitObjectField) field_i) {
            case OptixHitObjectField::IsHit:
                fmt("    .reg.u32 $v_out_$u;\n"
                    "    call ($v_out_$u), _optix_hitobject_is_hit, ();\n",
                    v, 32 + i,
                    v, 32 + i);
                break;

            case OptixHitObjectField::InstanceId:
                fmt("    .reg.u32 $v_out_$u;\n"
                    "    call ($v_out_$u), _optix_hitobject_get_instance_id, ();\n",
                    v, 32 + i,
                    v, 32 + i);
                break;

            case OptixHitObjectField::PrimitiveIndex:
                fmt("    .reg.u32 $v_out_$u;\n"
                    "    call ($v_out_$u), _optix_hitobject_get_primitive_idx, ();\n",
                    v, 32 + i,
                    v, 32 + i);
                break;

            case OptixHitObjectField::SBTDataPointer:
                fmt("    .reg.b64 $v_out_$u;\n"
                    "    call ($v_out_$u), _optix_hitobject_get_sbt_data_pointer, ();\n",
                    v, 32 + i,
                    v, 32 + i);
                break;

            case OptixHitObjectField::RayTMax:
                fmt("    .reg.f32 $v_out_$u;\n"
                    "    call ($v_out_$u), _optix_hitobject_get_ray_tmax, ();\n",
                    v, 32 + i,
                    v, 32 + i);
                break;

            case OptixHitObjectField::Attribute0:
            case OptixHitObjectField::Attribute1:
            case OptixHitObjectField::Attribute2:
            case OptixHitObjectField::Attribute3:
            case OptixHitObjectField::Attribute4:
            case OptixHitObjectField::Attribute5:
            case OptixHitObjectField::Attribute6:
            case OptixHitObjectField::Attribute7: {
                uint32_t idx = field_i - (uint32_t) OptixHitObjectField::Attribute0;
                fmt("    .reg.u32 $v_out_$u;\n"
                    "    .reg.u32 $v_attr_idx_$u;\n"
                    "    mov.u32 $v_attr_idx_$u, $u;\n"
                    "    call ($v_out_$u), _optix_hitobject_get_attribute, ($v_attr_idx_$u);\n",
                    v, 32 + i,
                    v, idx,
                    v, idx, idx,
                    v, 32 + i, v, idx);
            }
            break;

            default:
                jitc_fail("jitc_cuda_render_trace(): unhandled hit object "
                          "field type (value %u)!", field_i);
        }
    }

    // =====================================================
    // 4. Invoke miss & closest hit programs
    // =====================================================
    if (td->invoke) {
        put("    call (");
        for (uint32_t i = 0; i < 32; ++i)
            fmt("$v_out_$u$s", v, i, i + 1 < 32 ? ", " : "");
        put("), _optix_hitobject_invoke, (");

        fmt("$v_z, $v_count, ", v, v);

        for (uint32_t i = 0; i < 32; ++i) {
            if (i + 15 < td->indices.size())
                fmt("$v", jitc_var(td->indices[i + 15]));
            else
                fmt("$v_z", v);

            if (i + 1 < 32)
                put(", ");
        }

        put(");\n");
    }

    if (some_masked)
        fmt("\nl_masked_$u:\n", v->reg_index);
}

static void jitc_cuda_render_reorder(const Variable *v, const Variable *key) {
    if (jit_flag(JitFlag::ShaderExecutionReordering)) {
        // Create an outgoing Nop hit object on all lanes, to overwrite any existing
        // hit object which could influence the reordering.
        put("    call (), _optix_hitobject_make_nop, ();\n");

        fmt("     .reg .u32 $v_hint_bits;\n"
            "    mov.u32 $v_hint_bits, $u;\n"
            "    call (), _optix_hitobject_reorder, ($v, $v_hint_bits);\n",
            v,
            v, v->literal,
            key, v);
    }
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

    if (!uses_optix) {
        put("        ld.global.u64 %rd2, callables[%r3];\n");
    } else {
        fmt("        call (%rd2), _optix_call_$s_callable, (%r3);\n",
            jitc_optix_use_continuation_callables() ? "continuation"
                                                    : "direct");
    }

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

    for (uint32_t i = 0; i < call->n_in; ++i) {
        const Variable *v = jitc_var(call->outer_in[i]);
        const Variable *v_in = jitc_var(call->inner_in[i]);
        bool unused =
            !v->reg_index ||
            (call->optimize && // Optimizations are on
                (v->is_literal() || // Literals are propagated
                 v_in->ref_count == 1)); // CallInput is never used
        if (unused)
            continue;

        v = jitc_var(call->outer_in[i]);
        const char *tname  = type_name_ptx_bin[v->type],
                   *prefix = type_prefix[v->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        fmt("            st.param.$s [in+$u], $s$u;\n",
            tname, v->param_offset, prefix, v->reg_index);
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
        if (!v || !v->reg_index || v->param_type == ParamType::Input)
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
    // 6. Special handling for predicate return value(s)
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
