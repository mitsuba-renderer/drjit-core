#include "eval.h"
#include "internal.h"
#include "var.h"

// Forward declaration
static void jitc_render_stmt_llvm(uint32_t index, const Variable *v);

void jitc_assemble_llvm(ThreadState *, ScheduledGroup group) {
    uint32_t width = jitc_llvm_vector_width;
    bool log_trace = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace;

    buffer.put("define void @enoki_^^^^^^^^^^^^^^^^(i64 %start, i64 %end, "
               "i8** noalias %params) #0 {\n"
               "entry:\n"
               "    br label %loop\n"
               "\n"
               "loop:\n"
               "    %index = phi i64 [ %index_next, %suffix ], [ %start, %entry ]\n");

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        const Variable *v = jitc_var(index);
        const uint32_t vti = v->type;
        const VarType vt = (VarType) vti;

        const char *prefix = var_type_prefix[vti],
                   *tname = vt == VarType::Bool
                            ? "i8" : var_type_name_llvm[vti];
        uint32_t tsize = var_type_size[vti],
                 id = v->reg_index,
                 align = v->unaligned ? 1 : (tsize * width),
                 size = v->size;

        if (unlikely(log_trace && v->extra)) {
            const char *label = jitc_var_label(index);
            if (label)
                buffer.fmt("    ; %s\n", label);
        }

        if (v->param_type != ParamType::Register) {
            buffer.fmt(
                "    %s%u_p1 = getelementptr inbounds i8*, i8** %%params, i32 %u\n"
                "    %s%u_p2 = load i8*, i8** %s%u_p1, align 8, !alias.scope !1\n"
                "    %s%u_p3 = bitcast i8* %s%u_p2 to %s*\n",
                prefix, id, v->param_index, prefix, id, prefix, id,
                prefix, id, prefix, id, tname);

            if (v->param_type != ParamType::Input || size != 1)
                buffer.fmt("    %s%u_p4 = getelementptr inbounds %s, %s* %s%u_p3, i64 %%index\n"
                           "    %s%u_p5 = bitcast %s* %s%u_p4 to <%u x %s>*\n",
                           prefix, id, tname, tname, prefix, id, prefix, id, tname,
                           prefix, id, width, tname);
        }

        if (likely(v->param_type == ParamType::Input)) {
            if (v->literal)
                continue;

            if (size != 1) {
                if (vt != VarType::Bool) {
                    buffer.fmt("    %s%u = load <%u x %s>, <%u x %s>* %s%u_p5, align %u, !alias.scope !1\n",
                               prefix, id, width, tname, width, tname, prefix, id, align);
                } else {
                    buffer.fmt("    %s%u_0 = load <%u x i8>, <%u x i8>* %s%u_p5, align %u, !alias.scope !1\n"
                              "     %s%u = trunc <%u x i8> %s%u_0 to <%u x i1>\n",
                               prefix, id, width, width, prefix, id, align,
                               prefix, id, width, prefix, id, width);
                }
            } else {
                if (vt != VarType::Bool) {
                    buffer.fmt("    %s%u_0 = load %s, %s* %s%u_p3, align %u, !alias.scope !1\n"
                               "    %s%u_1 = insertelement <%u x %s> undef, %s %s%u_0, i32 0\n"
                               "    %s%u = shufflevector <%u x %s> %s%u_1, <%u x %s> undef, <%u x i32> zeroinitializer\n",
                               prefix, id, tname, tname, prefix, id, tsize,
                               prefix, id, width, tname, tname, prefix, id,
                               prefix, id, width, tname, prefix, id, width, tname, width);
                } else {
                    buffer.fmt("    %s%u_0 = load i8, i8* %s%u_p3, align %u, !alias.scope !1\n"
                               "    %s%u_1 = trunc i8 %s%u_0 to i1\n"
                               "    %s%u_2 = insertelement <%u x i1> undef, i1 %s%u_1, i32 0\n"
                               "    %s%u = shufflevector <%u x i1> %s%u_2, <%u x i1> undef, <%u x i32> zeroinitializer\n",
                               prefix, id, prefix, id, tsize,
                               prefix, id,
                               prefix, id, prefix, id, width, prefix, id,
                               prefix, id, width, prefix, id, width, width);
                }
            }
        } else {
            jitc_render_stmt_llvm(index, v);
        }

        if (v->param_type == ParamType::Output) {
            if (vt != VarType::Bool) {
                buffer.fmt("    store <%u x %s> %s%u, <%u x %s>* %s%u_p5, align %u, !noalias !1\n",
                           width, tname, prefix, id, width, tname, prefix, id, align);
            } else {
                buffer.fmt("    %s%u_e = zext <%u x i1> %s%u to <%u x i8>\n"
                           "    store <%u x i8> %s%u_e, <%u x i8>* %s%u_p5, align %u, !noalias !1\n",
                           prefix, id, width, prefix, id, width,
                           width, prefix, id, width, prefix, id, align);
            }
        }
    }

    buffer.put("\n"
               "    br label %suffix\n"
               "\n"
               "suffix:\n");
    buffer.fmt("    %%index_next = add i64 %%index, %u\n", width);
    buffer.put("    %cond = icmp uge i64 %index_next, %end\n"
               "    br i1 %cond, label %done, label %loop, !llvm.loop !2\n\n"
               "done:\n"
               "    ret void\n"
               "}\n"
               "\n");
    buffer.put(globals.get(), globals.size());
    buffer.put("!0 = !{!0}\n"
               "!1 = !{!1, !0}\n"
               "!2 = !{!\"llvm.loop.unroll.disable\", !\"llvm.loop.vectorize.enable\", i1 0}\n\n");

    buffer.fmt("attributes #0 = { norecurse nounwind alignstack=%u "
               "\"target-cpu\"=\"%s\" \"stack-probe-size\"=\"%u\" \"target-features\"=\"-vzeroupper",
               std::max(16u, width * (uint32_t) sizeof(float)),
               jitc_llvm_target_cpu, 1024 * 1024 * 1024);
    if (jitc_llvm_target_features) {
        buffer.putc(',');
        buffer.put(jitc_llvm_target_features, strlen(jitc_llvm_target_features));
    }
    buffer.put("\" }");

}

/// Insert an intrinsic declaration into 'globals'
static void jitc_llvm_process_intrinsic() {
    const char *s = buffer.cur();
    while (strncmp(s, "call", 4) != 0)
        --s;
    s += 4;

    size_t before = globals.size();
    globals.put("declare");

    char c;
    while (c = *s, c != '\0') {
        if (c == ' ' && s[1]== '%') {
            while (c = *s, c != '\0' && c != ')' && c != ',')
                s++;
        } else if (c == 'i' && s[1]== '1' && s[2] == ' ') {
            globals.put("i1");
            while (c = *s, c != '\0' && c != ')' && c != ',')
                s++;
        } else if (c == 'i' && s[1]== '3' && s[2] == '2' && s[3] == ' ') {
            globals.put("i32");
            while (c = *s, c != '\0' && c != ')' && c != ',')
                s++;
        } else if (c == ' ' && s[1]== 'z' && s[2] == 'e' && s[3] == 'r') {
            while (c = *s, c != '\0' && c != ')' && c != ',')
                s++;
        } else {
            globals.putc(c);
            if (c == ')')
                break;
            s++;
        }
    }
    globals.put("\n\n");
    size_t after = globals.size();

    std::string key(globals.get() + before, globals.get() + after);
    if (!globals_set.insert(key).second)
        globals.rewind(after - before);
}

/// Convert an IR template with '$' expressions into valid IR
static void jitc_render_stmt_llvm(uint32_t index, const Variable *v) {
    if (v->literal) {
        uint32_t reg = v->reg_index, width = jitc_llvm_vector_width;
        uint32_t vt = v->type;
        const char *prefix = var_type_prefix[vt],
                   *tname = var_type_name_llvm[vt];
        uint64_t value = v->value;

        if (vt == (uint32_t) VarType::Float32) {
            float f;
            memcpy(&f, &value, sizeof(float));
            double d = f;
            memcpy(&value, &d, sizeof(uint64_t));
            vt = (uint32_t) VarType::Float64;
        }

#if 0
        /* The commented code here uses snprintf(), which internally relies on
           FILE*-style streams and becomes a performance bottleneck of the
           whole stringification process .. */

        buffer.fmt("    %s%u_1 = insertelement <%u x %s> undef, %s %llu, i32 0\n"
                   "    %s%u = shufflevector <%u x %s> %s%u_1, <%u x %s> undef, <%u x i32> zeroinitializer\n",
                   prefix, reg, width, tname, tname, (unsigned long long) value,
                   prefix, reg, width, tname, prefix, reg, width, tname, width);
#else
        // .. the ridiculous explicit variant below is equivalent and faster.

        size_t tname_len = strlen(tname),
               prefix_len = strlen(prefix);

        buffer.put("    ");
        buffer.put(prefix, strlen(prefix));
        buffer.put_uint32(reg);
        buffer.put("_1 = insertelement <");
        buffer.put_uint32(width);
        buffer.put(" x ");
        buffer.put(tname, tname_len);
        buffer.put("> undef, ");
        buffer.put(tname, tname_len);

        if (vt == (uint32_t) VarType::Float64) {
            buffer.put(" 0x");
            buffer.put_uint64_hex(value);
        } else {
            buffer.putc(' ');
            buffer.put_uint64(value);
        }

        buffer.put(", i32 0\n    ");
        buffer.put(prefix, prefix_len);
        buffer.put_uint32(reg);
        buffer.put(" = shufflevector <");
        buffer.put_uint32(width);
        buffer.put(" x ");
        buffer.put(tname, tname_len);
        buffer.put("> ");
        buffer.put(prefix, prefix_len);
        buffer.put_uint32(reg);
        buffer.put("_1, <");
        buffer.put_uint32(width);
        buffer.put(" x ");
        buffer.put(tname, tname_len);
        buffer.put("> undef, <");
        buffer.put_uint32(width);
        buffer.put(" x i32> zeroinitializer\n");
#endif
    } else {
        const char *s = v->stmt;
        buffer.put("    ");
        char c;
        bool has_intrinsic = false;
        do {
            const char *start = s;
            while (c = *s, c != '\0' && c != '$')
                s++;
            buffer.put(start, s - start);

            if (c == '$') {
                s++;
                const char **prefix_table = nullptr, tname = *s++;
                switch (tname) {
                    case 'c': buffer.putc('c'); has_intrinsic = true; continue;
                    case 'n': buffer.put("\n    "); continue;
                    case 'w': buffer.put(jitc_llvm_vector_width_str,
                                         strlen(jitc_llvm_vector_width_str)); continue;
                    case 't': prefix_table = var_type_name_llvm; break;
                    case 'b': prefix_table = var_type_name_llvm_bin; break;
                    case 'a': prefix_table = var_type_name_llvm_abbrev; break;
                    case 's': prefix_table = var_type_size_str; break;
                    case 'r': prefix_table = var_type_prefix; break;
                    case 'i': prefix_table = nullptr; break;
                    case 'o': prefix_table = (const char **) jitc_llvm_ones_str; break;
                    default:
                        jitc_fail("jit_render_stmt_llvm(): encountered invalid \"$\" "
                                 "expression (unknown tname \"%c\") in \"%s\"!", tname, v->stmt);
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
                    buffer.put(prefix, strlen(prefix));
                }

                if (tname == 'r' || tname == 'i')
                    buffer.put_uint32(dep->reg_index);
            }
        } while (c != '\0');

        if (unlikely(has_intrinsic))
            jitc_llvm_process_intrinsic();

        buffer.putc('\n');
    }
}

