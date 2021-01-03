#include "eval.h"
#include "internal.h"
#include "var.h"

// Forward declaration
static void jitc_render_stmt_cuda(uint32_t index, const Variable *v);

void jitc_assemble_cuda(ThreadState *ts, ScheduledGroup group,
                        uint32_t n_regs, uint32_t n_params) {
    bool params_global = !uses_optix && n_params > ENOKI_CUDA_ARG_LIMIT;
    bool log_trace = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace;

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

    buffer.fmt(".version %u.%u\n"
               ".target sm_%u\n"
               ".address_size 64\n\n",
               ts->ptx_version / 10, ts->ptx_version % 10,
               ts->compute_capability);

    if (!uses_optix) {
        buffer.fmt(".entry enoki_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^("
                   ".param .align 8 .b8 params[%u]) { \n",
                   params_global ? 8u : (n_params * (uint32_t) sizeof(void *)));
    } else {
       buffer.fmt(".const .align 8 .b8 params[%u];\n\n"
                  ".entry __raygen__enoki_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^() {\n",
                  n_params * (uint32_t) sizeof(void *));
    }

    buffer.fmt(
        "    .reg.b8   %%b <%u>; .reg.b16 %%w<%u>; .reg.b32 %%r<%u>;\n"
        "    .reg.b64  %%rd<%u>; .reg.f32 %%f<%u>; .reg.f64 %%d<%u>;\n"
        "    .reg.pred %%p <%u>;\n\n",
        n_regs, n_regs, n_regs, n_regs, n_regs, n_regs, n_regs);

    if (!uses_optix) {
        buffer.put("    mov.u32 %r0, %ctaid.x;\n"
                   "    mov.u32 %r1, %ntid.x;\n"
                   "    mov.u32 %r2, %tid.x;\n"
                   "    mad.lo.u32 %r0, %r0, %r1, %r2;\n");

        if (likely(!params_global)) {
           buffer.put("    ld.param.u32 %r2, [params];\n");
        } else {
           buffer.put("    ld.param.u64 %rd1, [params];\n"
                      "    ldu.global.u32 %r2, [%rd1];\n");
        }

        buffer.put("    setp.ge.u32 %p0, %r0, %r2;\n"
                   "    @%p0 bra done;\n"
                   "\n"
                   "    mov.u32 %r3, %nctaid.x;\n"
                   "    mul.lo.u32 %r1, %r3, %r1;\n"
                   "\n");
        buffer.fmt("body: // sm_%u\n",
                   state.devices[ts->device].compute_capability);
    } else {
        buffer.put("    call (%r0), _optix_get_launch_dimension_y, ();\n"
                   "    call (%r1), _optix_get_launch_index_y, ();\n"
                   "    call (%r2), _optix_get_launch_index_x, ();\n"
                   "    mad.lo.u32 %r0, %r0, %r1, %r2;\n"
                   "    call (%r1), _optix_get_launch_dimension_z, ();\n"
                   "    call (%r2), _optix_get_launch_index_z, ();\n"
                   "    mad.lo.u32 %r0, %r0, %r1, %r2;\n\n"
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
        const Variable *v = jitc_var(index);
        const uint32_t vti = v->type,
                       size = v->size;
        const VarType vt = (VarType) vti;

        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_assemble_cuda(): internal error: 'extra' entry not found!");

            const Extra &extra = it->second;
            if (log_trace && extra.label)
                buffer.fmt("    // %s\n", extra.label);

            if (extra.assemble) {
                extra.assemble(v, extra);
                continue;
            }
        }

        if (likely(v->param_type == ParamType::Input)) {
            const char *prefix = "%rd";
            uint32_t id = 0;

            if (v->literal) {
                prefix = type_prefix[vti];
                id = v->reg_index;
            }

            buffer.fmt("    ld.%s.u64 %s%u, [%s+%u];\n", params_type,
                       prefix, id, params_base, v->param_offset);

            if (v->literal)
                continue;

            if (size > 1)
                buffer.fmt("    mad.wide.u32 %%rd0, %%r0, %u, %%rd0;\n",
                           type_size[vti]);

            if (vt != VarType::Bool) {
                buffer.fmt("    %s%s %s%u, [%%rd0];\n",
                           size > 1 ? "ld.global.cs." : "ldu.global.",
                           type_name_ptx[vti],
                           type_prefix[vti],
                           v->reg_index);
            } else {
                buffer.fmt("    %s %%w0, [%%rd0];\n"
                           "    setp.ne.u16 %%p%u, %%w0, 0;\n",
                           size > 1 ? "ld.global.cs.u8" : "ldu.global.u8",
                           v->reg_index);
            }
            continue;
        } else {
            jitc_render_stmt_cuda(index, v);
        }

        if (v->param_type == ParamType::Output) {
            buffer.fmt("    ld.%s.u64 %%rd0, [%s+%u];\n"
                       "    mad.wide.u32 %%rd0, %%r0, %u, %%rd0;\n",
                       params_type, params_base, v->param_offset,
                       type_size[vti]);

            if (vt != VarType::Bool) {
                buffer.fmt("    st.global.cs.%s [%%rd0], %s%u;\n",
                           type_name_ptx[vti],
                           type_prefix[vti],
                           v->reg_index);
            } else {
                buffer.fmt("    selp.u16 %%w0, 1, 0, %%p%u;\n"
                           "    st.global.cs.u8 [%%rd0], %%w0;\n",
                           v->reg_index);
            }
        }
    }

    if (!uses_optix) {
        buffer.put("\n"
                   "    add.u32 %r0, %r0, %r1;\n"
                   "    setp.ge.u32 %p0, %r0, %r2;\n"
                   "    @!%p0 bra body;\n"
                   "\n"
                   "done:\n");
    }

    buffer.put("    ret;\n"
               "}\n");

    size_t globals_strlen = 0;
    for (const std::string &s : globals)
        globals_strlen += s.length();

    if (globals_strlen) {
        size_t body_length = buffer.size();
        buffer.putc(' ', globals_strlen);
        char *p = (char *) strstr(buffer.get(), "\n\n") + 2;
        memmove(p + globals_strlen, p, body_length - (p - buffer.get()));
        for (auto it = globals.rbegin(); it != globals.rend(); ++it) {
            const std::string &s = *it;
            memcpy(p, s.c_str(), s.length());
            p += s.length();
        }
    }
}

void jitc_assemble_cuda_func(uint32_t n_regs, uint32_t in_size,
                             uint32_t in_align, uint32_t out_size,
                             uint32_t out_align, uint32_t extra_size,
                             uint32_t n_out, const uint32_t *out,
                             const uint32_t *out_nested,
                             const char *ret_label) {
    bool log_trace = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace,
         function_interface = ret_label == nullptr;

    if (function_interface) {
        buffer.put(".visible .func");
        if (out_size)
            buffer.fmt(" (.param .align %u .b8 result[%u])", out_align, out_size);

        buffer.fmt(" %s_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(",
                   uses_optix ? "__direct_callable__" : "func");

        if (extra_size) {
            buffer.put(".reg .u64 extra");
            if (in_size)
                buffer.put(", ");
        }
        if (in_size)
            buffer.fmt(".param .align %u .b8 params[%u]", in_align, in_size);

        buffer.fmt(
            ") {\n"
            "    .reg.b8   %%b <%u>; .reg.b16 %%w<%u>; .reg.b32 %%r<%u>;\n"
            "    .reg.b64  %%rd<%u>; .reg.f32 %%f<%u>; .reg.f64 %%d<%u>;\n"
            "    .reg.pred %%p <%u>;\n\n",
            n_regs, n_regs, n_regs, n_regs, n_regs, n_regs, n_regs);
    } else {
        // Branch-based interface
        buffer.put("l_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^:");
        int ctr = 0;
        for (ScheduledVariable &sv : schedule) {
            const Variable *v = jitc_var(sv.index);
            if ((VarType) v->type == VarType::Void ||
                !(v->stmt || v->literal))
                continue;
            buffer.fmt("%s.reg.%s %s%u;", ctr % 4 == 0 ? "\n    " : " ",
                       type_name_ptx[v->type], type_prefix[v->type],
                       v->reg_index);
            ++ctr;
        }
        buffer.put("\n\n");
    }

    for (ScheduledVariable &sv : schedule) {
        const Variable *v = jitc_var(sv.index);
        const uint32_t vti = v->type;
        const VarType vt = (VarType) vti;

        if (unlikely(v->extra)) {
            auto it = state.extra.find(sv.index);
            if (it == state.extra.end()) jitc_fail("jit_assemble_cuda(): internal error: 'extra' entry not found!");

            const Extra &extra = it->second;
            if (log_trace && extra.label) {
                const char *label = strrchr(extra.label, '/');
                if (label && label[1])
                    buffer.fmt("    // %s\n", label + 1);
            }

            if (extra.assemble) {
                extra.assemble(v, extra);
                continue;
            }
        }

        if (v->stmt || v->literal) {
            jitc_render_stmt_cuda(sv.index, v);
        } else if (function_interface) {
            if (vt != VarType::Bool) {
                buffer.fmt("    ld.param.%s %s%u, [params+%u];\n",
                           type_name_ptx[vti], type_prefix[vti],
                           v->reg_index, v->param_offset);
            } else {
                buffer.fmt("    ld.param.u8 %%w0, [params+%u];\n"
                           "    setp.ne.u16 %%p%u, %%w0, 0;\n",
                           v->param_offset, v->reg_index);
            }
        }
    }


    uint32_t offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        const Variable *v = jitc_var(out_nested[i]);
        uint32_t vti = v->type;
        const char *tname = type_name_ptx[vti],
                   *prefix = type_prefix[vti];

        if (function_interface) {
            if ((VarType) vti != VarType::Bool) {
                buffer.fmt("    st.param.%s [result+%u], %s%u;\n", tname,
                           offset, prefix, v->reg_index);
            } else {
                buffer.fmt("    selp.u16 %%w0, 1, 0, %%p%u;\n"
                           "    st.param.u8 [result+%u], %%w0;\n",
                           v->reg_index, offset);
            }
        } else {
            auto it = state.variables.find(out[i]);
            if (it != state.variables.end())
                buffer.fmt("    mov.%s %s%u, %s%u;\n", tname, prefix,
                           it.value().reg_index, prefix, v->reg_index);
        }
        offset += type_size[vti];
    }

    if (function_interface) {
        buffer.put("    ret;\n"
                   "}\n");
    } else {
        buffer.fmt("    bra %s;\n", ret_label);
    }
}

/// Convert an IR template with '$' expressions into valid IR
static void jitc_render_stmt_cuda(uint32_t index, const Variable *v) {
    if (v->literal) {
        const char *prefix = type_prefix[v->type],
                   *tname = type_name_ptx_bin[v->type];

        size_t tname_len = strlen(tname),
               prefix_len = strlen(prefix);

        buffer.put("    mov.");
        buffer.put(tname, tname_len);
        buffer.putc(' ');
        buffer.put(prefix, prefix_len);
        buffer.put_uint32(v->reg_index);
        buffer.put(", 0x");
        buffer.put_uint64_hex(v->value);
        buffer.put(";\n");
    } else {
        const char *s = v->stmt;
        if (unlikely(*s == '\0'))
            return;
        buffer.put("    ");
        char c;
        do {
            const char *start = s;
            while (c = *s, c != '\0' && c != '$')
                s++;
            buffer.put(start, s - start);

            if (c == '$') {
                s++;
                const char **prefix_table = nullptr, type = *s++;
                switch (type) {
                    case 'n': buffer.put(";\n    "); continue;
                    case 't': prefix_table = type_name_ptx; break;
                    case 'b': prefix_table = type_name_ptx_bin; break;
                    case 's': prefix_table = type_size_str; break;
                    case 'r': prefix_table = type_prefix; break;
                    default:
                        jitc_fail("jit_render_stmt_cuda(): encountered invalid \"$\" "
                                  "expression (unknown type \"%c\") in \"%s\"!", type, v->stmt);
                }

                uint32_t arg_id = *s++ - '0';
                if (unlikely(arg_id > 4))
                    jitc_fail("jit_render_stmt_cuda(%s): encountered invalid \"$\" "
                              "expression (argument out of bounds)!", v->stmt);

                uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
                if (unlikely(dep_id == 0))
                    jitc_fail("jit_render_stmt_cuda(%s): encountered invalid \"$\" "
                              "expression (referenced variable %u is missing)!", v->stmt, arg_id);

                const Variable *dep = jitc_var(dep_id);
                const char *prefix = prefix_table[(int) dep->type];
                buffer.put(prefix, strlen(prefix));

                if (type == 'r')
                    buffer.put_uint32(dep->reg_index);
            }
        } while (c != '\0');

        buffer.put(";\n");
    }
}
