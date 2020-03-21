#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include <tsl/robin_set.h>

#define CUDA_MAX_KERNEL_PARAMETERS 512

// ====================================================================
//  The following data structures are temporarily used during program
//  generation. They are declared as global variables to enable memory
//  reuse across jit_eval() calls.
// ====================================================================

struct ScheduledVariable {
    uint32_t size;
    uint32_t index;

    ScheduledVariable(uint32_t size, uint32_t index)
        : size(size), index(index) { }
};

struct ScheduledGroup {
    uint32_t size;
    uint32_t start;
    uint32_t end;

    ScheduledGroup(uint32_t size, uint32_t start, uint32_t end)
        : size(size), start(start), end(end) { }
};

struct IntrinsicHash {
    size_t operator()(const char *a_s) const {
        const char *a_e = strchr(a_s, '(');
        return (size_t) (a_e ? crc32(0, a_s, a_e - a_s) : 0u);
    }
};

struct IntrinsicEquality {
    size_t operator()(const char *a_s, const char *b_s) const {
        const char *a_e = strchr(a_s, '('),
                   *b_e = strchr(b_s, '(');
        if (!a_e || !b_e || a_e - a_s != b_e - b_s)
            return 0;
        return strncmp(a_s, b_s, a_e - a_s) == 0;
    }
};

/// Ordered list of variables that should be computed
static std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
static std::vector<ScheduledGroup> schedule_groups;

/// Auxiliary data structure needed to compute 'schedule_sizes' and 'schedule'
static tsl::robin_set<std::pair<uint32_t, uint32_t>, pair_hash> visited;

/// Input/output arguments of the kernel being evaluated
static std::vector<void *> kernel_args, kernel_args_extra;

/// Hash code of the last generated kernel
uint32_t kernel_hash = 0;

/// Name of the last generated kernel
static char kernel_name[9] { };

/// Dedicated buffer for intrinsic intrinsics_buffer (LLVM only)
Buffer intrinsics_buffer{1};

/// Intrinsics used by the current program (LLVM only)
static tsl::robin_set<const char *, IntrinsicHash, IntrinsicEquality> intrinsics_set;

// ====================================================================

/// Recursively traverse the computation graph to find variables needed by a computation
static void jit_var_traverse(uint32_t size, uint32_t index) {
    std::pair<uint32_t, uint32_t> key(size, index);

    if (index == 0 || visited.find(key) != visited.end())
        return;

    visited.insert(key);

    Variable *v = jit_var(index);
    const uint32_t *dep = v->dep;

    std::pair<uint32_t, uint32_t> ch[3] = {
        { dep[0], dep[0] ? jit_var(dep[0])->tsize : 0 },
        { dep[1], dep[1] ? jit_var(dep[1])->tsize : 0 },
        { dep[2], dep[2] ? jit_var(dep[2])->tsize : 0 }
    };

    // Simple sorting network
    if (ch[1].second < ch[2].second)
        std::swap(ch[1], ch[2]);
    if (ch[0].second < ch[2].second)
        std::swap(ch[0], ch[2]);
    if (ch[0].second < ch[1].second)
        std::swap(ch[0], ch[1]);

    for (auto const &v: ch)
        jit_var_traverse(size, v.first);

    schedule.emplace_back(size, index);
}

void jit_render_stmt_cuda(uint32_t index, Variable *v) {
    const char *s = v->stmt;
    char c;

    buffer.put("    ");
    while ((c = *s++) != '\0') {
        if (c != '$') {
            buffer.putc(c);
        } else {
            const char **prefix_table = nullptr, type = *s++;
            switch (type) {
                case 'n': buffer.put(";\n    "); continue;
                case 'i': buffer.put("%r0"); continue;
                case 't': prefix_table = var_type_name_ptx;     break;
                case 'b': prefix_table = var_type_name_ptx_bin; break;
                case 'r': prefix_table = var_type_prefix; break;
                default:
                    jit_fail("jit_render_stmt_cuda(): encountered invalid \"$\" "
                             "expression (unknown type \"%c\")!", type);
            }

            uint32_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 3))
                jit_fail("jit_render_stmt_cuda(%s): encountered invalid \"$\" "
                         "expression (argument out of bounds)!", v->stmt);

            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            if (unlikely(dep_id == 0))
                jit_fail("jit_render_stmt_cuda(%s): encountered invalid \"$\" "
                         "expression (dependency %u is missing)!", v->stmt, arg_id);

            Variable *dep = jit_var(dep_id);
            buffer.put(prefix_table[(int) dep->type]);

            if (type == 'r')
                buffer.fmt("%u", dep->reg_index);
        }
    }

    buffer.put(";\n");
}

void jit_render_stmt_llvm(uint32_t index, Variable *v) {
    const char *s = v->stmt;

    buffer.put("    ");
    char c;
    while ((c = *s++) != '\0') {
        if (c != '$') {
            buffer.putc(c);
        } else {
            const char **prefix_table = nullptr, type = *s++;
            switch (type) {
                case 's': continue;
                case 'n': buffer.put("\n    "); continue;
                case 'w': buffer.fmt("%u", jit_llvm_vector_width); continue;
                case 'z':
                case 'l':
                case 't': prefix_table = var_type_name_llvm; break;
                case 'o':
                case 'b': prefix_table = var_type_name_llvm_bin; break;
                case 'a': prefix_table = var_type_name_llvm_abbrev; break;
                case 'r': prefix_table = var_type_prefix; break;

                default:
                    jit_fail("jit_render_stmt_llvm(): encountered invalid \"$\" "
                             "expression (unknown type \"%c\")!", type);
            }

            uint32_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 3))
                jit_fail("jit_render_stmt_llvm(%s): encountered invalid \"$\" "
                         "expression (argument out of bounds)!", v->stmt);

            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            if (unlikely(dep_id == 0))
                jit_fail("jit_render_stmt_llvm(%s): encountered invalid \"$\" "
                         "expression (dependency %u is missing)!", v->stmt, arg_id);

            Variable *dep = jit_var(dep_id);
            const char *prefix = prefix_table[(int) dep->type];
            bool is_float = (VarType) dep->type == VarType::Float16 ||
                            (VarType) dep->type == VarType::Float32 ||
                            (VarType) dep->type == VarType::Float64;

            switch (type) {
                case 'r':
                    buffer.fmt("%s%u", prefix, dep->reg_index);
                    break;

                case 'a':
                case 'b':
                case 't':
                    buffer.put(prefix);
                    break;

                case 'l':
                    buffer.putc('<');
                    for (int i = 0; i < jit_llvm_vector_width; ++i)
                        buffer.fmt("%s %i%s%s", prefix, i, is_float ? ".0" : "",
                                   i + 1 < jit_llvm_vector_width ? ", " : "");
                    buffer.putc('>');
                    break;

                case 'z':
                    buffer.putc('<');
                    for (int i = 0; i < jit_llvm_vector_width; ++i)
                        buffer.fmt("%s 0%s%s", prefix, is_float ? ".0" : "",
                                   i + 1 < jit_llvm_vector_width ? ", " : "");
                    buffer.putc('>');
                    break;

                case 'o':
                    buffer.putc('<');
                    for (int i = 0; i < jit_llvm_vector_width; ++i)
                        buffer.fmt("%s -1%s", prefix, i + 1 < jit_llvm_vector_width ? ", " : "");
                    buffer.putc('>');
                    break;
            }
        }
    }

    buffer.putc('\n');

    // Check for intrinsics
    s = strstr(v->stmt, "call");
    if (likely(!s))
        return;
    s += 5;

    if (intrinsics_set.find(s) != intrinsics_set.end())
        return;
    intrinsics_set.insert(s);

    intrinsics_buffer.put("declare ");
    while ((c = *s++) != '\0') {
        if (c != '$') {
            intrinsics_buffer.putc(c);
        } else {
            const char **prefix_table = nullptr, type = *s++;
            switch (type) {
                case 't': prefix_table = var_type_name_llvm; break;
                case 'b': prefix_table = var_type_name_llvm_bin; break;
                case 'a': prefix_table = var_type_name_llvm_abbrev; break;
                case 'w': intrinsics_buffer.fmt("%u", jit_llvm_vector_width); continue;
                case 'r': s++; intrinsics_buffer.rewind(1); continue;
                case 'n':
                case 'o':
                case 'l':
                case 'z': s++; continue;
                case 's': s+= 2; continue;
                default:
                    jit_fail("jit_render_stmt_llvm(): encountered invalid \"$\" "
                             "expression (unknown type \"%c\")!", type);
            }

            uint32_t arg_id = *s++ - '0';
            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            Variable *dep = jit_var(dep_id);
            intrinsics_buffer.put(prefix_table[(int) dep->type]);
        }
    }
    intrinsics_buffer.putc('\n');
}

void jit_assemble_cuda(ScheduledGroup group, uint32_t n_regs_total) {
    auto get_parameter_addr = [](const Variable *v, uint32_t target = 0) {
        if (v->arg_index < CUDA_MAX_KERNEL_PARAMETERS - 1)
            buffer.fmt("    ld.param.u64 %%rd%u, [arg%u];\n", target, v->arg_index - 1);
        else
            buffer.fmt("    ldu.global.u64 %%rd%u, [%%rd2 + %u];\n",
                       target, (v->arg_index - (CUDA_MAX_KERNEL_PARAMETERS - 1)) * 8);

        if (v->size > 1)
            buffer.fmt("    mul.wide.u32 %%rd1, %%r0, %u;\n"
                       "    add.u64 %%rd%u, %%rd%u, %%rd1;\n",
                       var_type_size[(int) v->type], target, target);
    };

    buffer.clear();
    buffer.put(".version 6.3\n");
    buffer.put(".target sm_61\n");
    buffer.put(".address_size 64\n");

    /* Special registers:

         %r0   :  Index
         %r1   :  Step
         %r2   :  Size
         %p0   :  Stopping predicate
         %rd0  :  Temporary for parameter pointers
         %rd1  :  Temporary for offset calculation
         %rd2  :  'arg_extra' pointer

         %b3, %w3, %r3, %rd3, %f3, %d3, %p3: reserved for use in compound
         statements that must write a temporary result to a register.
    */

    buffer.put(".visible .entry enoki_^^^^^^^^(.param .u32 size,\n");
    for (uint32_t index = 1; index < kernel_args.size(); ++index)
        buffer.fmt("                               .param .u64 arg%u%s\n",
                   index - 1, (index + 1 < kernel_args.size()) ? "," : ") {");

    for (const char *reg_type : { "b8 %b", "b16 %w", "b32 %r", "b64 %rd",
                                  "f32 %f", "f64 %d", "pred %p" })
        buffer.fmt("    .reg.%s<%u>;\n", reg_type, n_regs_total);

    buffer.put("\n    // Grid-stride loop setup\n");

    buffer.put("    mov.u32 %r0, %ctaid.x;\n");
    buffer.put("    mov.u32 %r1, %ntid.x;\n");
    buffer.put("    mov.u32 %r2, %tid.x;\n");
    buffer.put("    mad.lo.u32 %r0, %r0, %r1, %r2;\n");
    buffer.put("    ld.param.u32 %r2, [size];\n");
    buffer.put("    setp.ge.u32 %p0, %r0, %r2;\n");
    buffer.put("    @%p0 bra L0;\n\n");

    buffer.put("    mov.u32 %r3, %nctaid.x;\n");
    buffer.put("    mul.lo.u32 %r1, %r3, %r1;\n");
    if (!kernel_args_extra.empty())
        buffer.fmt("    ld.param.u64 %%rd2, [arg%u];\n",
                   CUDA_MAX_KERNEL_PARAMETERS - 2);

    buffer.put("\nL1: // Loop body\n");

    bool log_trace = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace;

    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        uint32_t index = schedule[group_index].index;
        Variable *v = jit_var(index);

        if (v->arg_type == ArgType::Input) {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Load %s%u%s%s\n",
                           var_type_prefix[(int) v->type],
                           v->reg_index, v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");

            get_parameter_addr(v, v->direct_pointer ? v->reg_index : 0u);

            if (likely(!v->direct_pointer)) {
                if (likely(v->type != (uint32_t) VarType::Bool)) {
                    buffer.fmt("    %s.global.%s %s%u, [%%rd0];\n",
                           v->size == 1 ? "ldu" : "ld",
                           var_type_name_ptx[(int) v->type],
                           var_type_prefix[(int) v->type],
                           v->reg_index);
                } else {
                    buffer.fmt("    %s.global.u8 %%w0, [%%rd0];\n",
                           v->size == 1 ? "ldu" : "ld");
                    buffer.fmt("    setp.ne.u16 %s%u, %%w0, 0;\n",
                           var_type_prefix[(int) v->type],
                           v->reg_index);
                }
            }
        } else {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Evaluate %s%u%s%s\n",
                           var_type_prefix[(int) v->type],
                           v->reg_index,
                           v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");

            jit_render_stmt_cuda(index, v);
        }

        if (v->arg_type == ArgType::Output) {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Store %s%u%s%s\n",
                           var_type_prefix[(int) v->type],
                           v->reg_index, v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");

            get_parameter_addr(v);

            if (likely(v->type != (uint32_t) VarType::Bool)) {
                buffer.fmt("    st.global.%s [%%rd0], %s%u;\n",
                       var_type_name_ptx[(int) v->type],
                       var_type_prefix[(int) v->type],
                       v->reg_index);
            } else {
                buffer.fmt("    selp.u16 %%w0, 1, 0, %%p%u;\n",
                           v->reg_index);
                buffer.put("    st.global.u8 [%rd0], %w0;\n");
            }
        }
    }

    buffer.putc('\n');
    buffer.put("    add.u32 %r0, %r0, %r1;\n");
    buffer.put("    setp.ge.u32 %p0, %r0, %r2;\n");
    buffer.put("    @!%p0 bra L1;\n");
    buffer.put("\n");
    buffer.put("L0:\n");
    buffer.put("    ret;\n");
    buffer.put("}");

    /// Replace '^'s in 'enoki_^^^^^^^^' by CRC32 hash
    kernel_hash = (uint32_t) string_hash()(buffer.get());
    snprintf(kernel_name, 9, "%08x", kernel_hash);
    memcpy((void *) strchr(buffer.get(), '^'), kernel_name, 8);
}

void jit_assemble_llvm(ScheduledGroup group) {
    const int width = jit_llvm_vector_width;

    bool log_trace = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace;

    buffer.clear();
    intrinsics_buffer.clear();
    intrinsics_set.clear();
    buffer.fmt("define void @enoki_^^^^^^^^(i64 %%start, i64 %%end, i8** "
               "%%ptrs) norecurse nosync nounwind alignstack(%i) "
               "\"target-cpu\"=\"%s\"",
               width * (int) sizeof(float), jit_llvm_target_cpu);
    if (jit_llvm_target_features)
        buffer.fmt(" \"target-features\"=\"%s\"", jit_llvm_target_features);
    buffer.put(" {\n");
    buffer.put("entry:\n");
    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        uint32_t index = schedule[group_index].index;
        Variable *v = jit_var(index);
        if (v->arg_type == ArgType::Register)
            continue;
        uint32_t arg_id = v->arg_index - 1;
        const char *type = var_type_name_llvm[(int) v->type];

        if ((VarType) v->type == VarType::Bool)
            type = "i8";
        if (unlikely(log_trace))
            buffer.fmt("\n    ; Prepare argument %u\n", arg_id);

        buffer.fmt("    %%a%u_i = getelementptr inbounds i8*, i8** %%ptrs, i32 %u\n", arg_id, arg_id);
        buffer.fmt("    %%a%u_p = load i8*, i8** %%a%u_i, align 8, !alias.scope !1\n", arg_id, arg_id);

        if (likely(!v->direct_pointer)) {
            buffer.fmt("    %%a%u = bitcast i8* %%a%u_p to %s*\n", arg_id, arg_id, type);
            if (v->size == 1) {
                buffer.fmt("    %%a%u_s = load %s, %s* %%a%u, align %u, !alias.scope !1\n", arg_id,
                           type, type, arg_id, var_type_size[(int) v->type]);
                if ((VarType) v->type == VarType::Bool)
                    buffer.fmt("    %%a%u_s1 = trunc i8 %%a%u_s to i1\n", arg_id, arg_id);
            }
        }
    }
    buffer.put("    br label %loop\n\n");
    buffer.put("done:\n");
    buffer.put("    ret void\n\n");

    buffer.put("loop:\n");
    buffer.put("    %index = phi i64 [ %index_next, %loop ], [ %start, %entry ]\n");

    auto get_parameter_addr = [](uint32_t reg_id, uint32_t arg_id,
                                 const char *reg_prefix, const char *type,
                                 uint32_t size) {
        if (size == 1) {
            buffer.fmt("    %s%u_p = bitcast %s* %%a%u to <%u x %s>*\n", reg_prefix, reg_id,
                       type, arg_id, jit_llvm_vector_width, type);
        } else {
            buffer.fmt("    %s%u_i = getelementptr inbounds %s, %s* %%a%u, "
                       "i64 %%index\n", reg_prefix, reg_id, type, type, arg_id);
            buffer.fmt("    %s%u_p = bitcast %s* %s%u_i to <%u x %s>*\n", reg_prefix, reg_id,
                       type, reg_prefix, reg_id, jit_llvm_vector_width, type);
        }
    };

    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        uint32_t index = schedule[group_index].index;
        Variable *v = jit_var(index);
        uint32_t align = var_type_size[(int) v->type] * width,
                 reg_id = v->reg_index, arg_id = v->arg_index - 1;
        const char *type = var_type_name_llvm[(int) v->type],
                   *reg_prefix = var_type_prefix[(int) v->type];
        uint32_t size = v->size;

        if ((VarType) v->type == VarType::Bool)
            type = "i8";

        if (v->arg_type == ArgType::Input) {
            if (unlikely(log_trace))
                buffer.fmt("\n    ; Load %s%u%s%s\n",
                           reg_prefix, reg_id, v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");

            if (v->direct_pointer) {
                buffer.fmt("    %s%u = ptrtoint i8* %%a%u_p to i64\n",
                           reg_prefix, reg_id, arg_id);
                continue;
            }

            if (size > 1)
                get_parameter_addr(reg_id, arg_id, reg_prefix, type, size);

            if ((VarType) v->type != VarType::Bool) {
                if (size > 1) {
                    buffer.fmt("    %s%u = load <%u x %s>, <%u x %s>* %s%u_p, align %u, !alias.scope !1\n",
                               reg_prefix, reg_id, width, type, width, type, reg_prefix, reg_id, align);
                } else {
                    buffer.fmt("    %s%u_z = insertelement <%u x %s> undef, %s %%a%u_s, i32 0\n",
                               reg_prefix, reg_id, width, type, type, arg_id);
                    buffer.fmt("    %s%u = shufflevector <%u x %s> %s%u_z, <%u x %s> undef, <%u x i32> zeroinitializer\n",
                               reg_prefix, reg_id, width, type, reg_prefix, reg_id, width, type, width);
                }
            } else {
                if (size > 1) {
                    buffer.fmt("    %s%u_z = load <%u x i8>, <%u x i8>* %s%u_p, align %u, !alias.scope !1\n",
                               reg_prefix, reg_id, width, width, reg_prefix, reg_id, align);
                    buffer.fmt("    %s%u = trunc <%u x i8> %s%u_z to <%u x i1>\n",
                               reg_prefix, reg_id, width, reg_prefix, reg_id, width);
                } else {
                    buffer.fmt("    %s%u_z = insertelement <%u x i1> undef, i1 %%a%u_s1, i32 0\n",
                               reg_prefix, reg_id, width, arg_id);
                    buffer.fmt("    %s%u = shufflevector <%u x i1> %s%u_z, <%u x i1> undef, <%u x i32> zeroinitializer\n",
                               reg_prefix, reg_id, width, reg_prefix, reg_id, width, width);
                }
            }
        } else {
            if (unlikely(log_trace))
                buffer.fmt("\n    ; Evaluate %s%u%s%s\n",
                           reg_prefix, reg_id,
                           v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");
            jit_render_stmt_llvm(index, v);
        }

        if (v->arg_type == ArgType::Output) {
            if (unlikely(log_trace))
                buffer.fmt("\n    ; Store %s%u%s%s\n",
                           reg_prefix, reg_id, v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");
            get_parameter_addr(reg_id, arg_id, reg_prefix, type, size);

            if ((VarType) v->type != VarType::Bool) {
                buffer.fmt("    store <%u x %s> %s%u, <%u x %s>* %s%u_p, align %u, !noalias !1\n",
                           width, type, reg_prefix,
                           reg_id, width, type, reg_prefix, reg_id, align);
            } else {
                buffer.fmt("    %s%u_e = zext <%u x i1> %s%u to <%u x i8>\n",
                           reg_prefix, reg_id, width, reg_prefix, reg_id, width);
                buffer.fmt("    store <%u x i8> %s%u_e, <%u x i8>* %s%u_p, align %u, !noalias !1\n",
                           width, reg_prefix, reg_id, width, reg_prefix, reg_id, align);
            }
        }
    }

    buffer.putc('\n');
    buffer.fmt("    %%index_next = add i64 %%index, %u\n", width);
    buffer.put("    %cond = icmp uge i64 %index_next, %end\n");
    buffer.put("    br i1 %cond, label %done, label %loop, !llvm.loop "
               "!{!\"llvm.loop.unroll.disable\", !\"llvm.loop.vectorize.enable\", i1 0}\n");
    buffer.put("}\n\n");
    buffer.put("!0 = !{!0}\n");
    buffer.put("!1 = !{!1, !0}\n");

    if (intrinsics_buffer.size() > 1) {
        buffer.putc('\n');
        buffer.put(intrinsics_buffer.get());
    }

    /// Replace '^'s in 'enoki_^^^^^^^^' by CRC32 hash
    kernel_hash = (uint32_t) string_hash()(buffer.get());
    snprintf(kernel_name, 9, "%08x", kernel_hash);
    memcpy((void *) strchr(buffer.get(), '^'), kernel_name, 8);
}

void jit_assemble(ScheduledGroup group) {
    bool cuda = active_stream != nullptr;

    uint32_t n_args_in    = 0,
             n_args_out   = 0,
             // The first 4 variables are reserved on the CUDA backend
             n_regs_total = cuda ? 4 : 0;

    (void) timer();
    jit_trace("jit_assemble(size=%u): register map:", group.size);

    /// Push the size argument
    void *tmp = 0;
    memcpy(&tmp, &group.size, sizeof(uint32_t));
    kernel_args.clear();
    kernel_args_extra.clear();
    kernel_args.push_back(tmp);
    n_args_in++;

    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        uint32_t index = schedule[group_index].index;
        Variable *v = jit_var(index);
        bool push = false;

        if (unlikely(v->ref_count_int == 0 && v->ref_count_ext == 0))
            jit_fail("jit_assemble(): schedule contains unreferenced variable %u!", index);
        else if (unlikely(v->size != 1 && v->size != group.size))
            jit_fail("jit_assemble(): schedule contains variable %u with incompatible size "
                     "(%u and %u)!", index, v->size, group.size);
        else if (unlikely(v->data == nullptr && !v->direct_pointer && v->stmt == nullptr))
            jit_fail("jit_assemble(): schedule contains variable %u with empty statement!", index);

        if (std::max(state.log_level_stderr, state.log_level_callback) >= LogLevel::Trace) {
            buffer.clear();
            buffer.fmt("   - %s%u -> %u",
                       var_type_prefix[(int) v->type],
                       n_regs_total, index);

            if (v->has_label) {
                const char *label = jit_var_label(index);
                buffer.fmt(" \"%s\"", label ? label : "(null)");
            }
            if (v->size == 1)
                buffer.put(" [scalar]");
            if (v->data != nullptr || v->direct_pointer)
                buffer.put(" [in]");
            else if (v->side_effect)
                buffer.put(" [se]");
            else if (v->ref_count_ext > 0 && v->size == group.size)
                buffer.put(" [out]");

            jit_trace("%s", buffer.get());
        }

        if (v->data || v->direct_pointer) {
            v->arg_index = (uint16_t) (n_args_in + n_args_out);
            v->arg_type = ArgType::Input;
            n_args_in++;
            push = true;
        } else if (!v->side_effect && v->ref_count_ext > 0 &&
                   v->size == group.size) {
            size_t var_size =
                (size_t) group.size * (size_t) var_type_size[(int) v->type];
            v->data = jit_malloc(cuda ? AllocType::Device : AllocType::Host, var_size);
            v->arg_index = (uint16_t) (n_args_in + n_args_out);
            v->arg_type = ArgType::Output;
            v->tsize = 1;
            n_args_out++;
            push = true;
        } else {
            v->arg_index = (uint16_t) 0xFFFF;
            v->arg_type = ArgType::Register;
        }

        if (push) {
            if (cuda && kernel_args.size() < CUDA_MAX_KERNEL_PARAMETERS - 1)
                kernel_args.push_back(v->data);
            else
                kernel_args_extra.push_back(v->data);
        }

        v->reg_index  = n_regs_total++;
    }

    if (unlikely(n_regs_total > 0xFFFFFFu))
        jit_fail("jit_run(): The queued computation involves more than 16 "
                 "million variables, which overflowed an internal counter. "
                 "Even if Enoki could compile such a large program, it would "
                 "not run efficiently. Please periodically run jitc_eval() to "
                 "break down the computation into smaller chunks.");
    else if (unlikely(n_args_in + n_args_out > 0xFFFFu))
        jit_fail("jit_run(): The queued computation involves more than 65536 "
                 "input or output arguments, which overflowed an internal counter. "
                 "Even if Enoki could compile such a large program, it would "
                 "not run efficiently. Please periodically run jitc_eval() to "
                 "break down the computation into smaller chunks.");

    if (unlikely(cuda && !kernel_args_extra.empty())) {
        size_t args_extra_size = kernel_args_extra.size() * sizeof(uint64_t);
        void *args_extra_host = jit_malloc(AllocType::HostPinned, args_extra_size);
        void *args_extra_dev  = jit_malloc(AllocType::Device, args_extra_size);

        memcpy(args_extra_host, kernel_args_extra.data(), args_extra_size);
        cuda_check(cuMemcpyAsync(args_extra_dev, args_extra_host,
                                 args_extra_size, active_stream->handle));

        kernel_args.push_back(args_extra_dev);
        jit_free(args_extra_host);

        /* Safe, because there won't be further allocations on the current
           stream until after this kernel has executed. */
        jit_free(args_extra_dev);
    }

    jit_log(Info, "jit_run(): launching kernel (n=%u, in=%u, out=%u, ops=%u) ..",
            group.size, n_args_in - 1, n_args_out, n_regs_total);

    if (cuda)
        jit_assemble_cuda(group, n_regs_total);
    else
        jit_assemble_llvm(group);

    jit_log(Debug, "%s", buffer.get());
}

void jit_run_llvm(ScheduledGroup group) {
    float codegen_time = timer();
    auto it = state.kernel_cache.find(buffer.get(), (size_t) kernel_hash);
    Kernel kernel;

    if (it == state.kernel_cache.end()) {
        bool cache_hit = false;
        kernel = jit_llvm_compile(buffer.get(), buffer.size(), kernel_hash, cache_hit);
        float link_time = timer();
        jit_log(Info, "jit_run(): cache %s, codegen: %s, %s: %s, %s.",
                cache_hit ? "hit" : "miss",
                std::string(jit_time_string(codegen_time)).c_str(),
                cache_hit ? "load" : "build",
                std::string(jit_time_string(link_time)).c_str(),
                std::string(jit_mem_string(kernel.llvm.size)).c_str());

        char *str = (char *) malloc(buffer.size() + 1);
        memcpy(str, buffer.get(), buffer.size() + 1);
        state.kernel_cache.emplace(str, kernel);
    } else {
        kernel = it.value();
        jit_log(Info, "jit_run(): cache hit, codegen: %s.",
                jit_time_string(codegen_time));
    }

    kernel.llvm.func(0, group.size, kernel_args_extra.data());
}

void jit_run_cuda(ScheduledGroup group) {
    float codegen_time = timer();
    auto it = state.kernel_cache.find(buffer.get(), (size_t) kernel_hash);
    Kernel kernel;

    if (it == state.kernel_cache.end()) {
        const uintptr_t log_size = 8192;
        std::unique_ptr<char[]> error_log(new char[log_size]),
                                 info_log(new char[log_size]);
        int arg[5] = {
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_LOG_VERBOSE
        };

        void *argv[5] = {
            (void *) info_log.get(),
            (void *) log_size,
            (void *) error_log.get(),
            (void *) log_size,
            (void *) 1
        };

        CUlinkState link_state;
        cuda_check(cuLinkCreate(5, arg, argv, &link_state));

        int rt = cuLinkAddData(link_state, CU_JIT_INPUT_PTX, (void *) buffer.get(),
                               buffer.size(), nullptr, 0, nullptr, nullptr);
        if (rt != CUDA_SUCCESS)
            jit_fail(
                "compilation failed. Please see the PTX assembly listing and "
                "error message below:\n\n%s\n\njit_run(): linker error:\n\n%s",
                buffer.get(), error_log.get());

        void *link_output = nullptr;
        size_t link_output_size = 0;
        cuda_check(cuLinkComplete(link_state, &link_output, &link_output_size));
        if (rt != CUDA_SUCCESS)
            jit_fail(
                "compilation failed. Please see the PTX assembly listing and "
                "error message below:\n\n%s\n\njit_run(): linker error:\n\n%s",
                buffer.get(), error_log.get());

        float link_time = timer();
        bool cache_hit = strstr(info_log.get(), "ptxas info") == nullptr;
        jit_log(Debug, "Detailed linker output:\n%s", info_log.get());

        CUresult ret;
        /* Unlock while synchronizing */ {
            unlock_guard guard(state.mutex);
            ret = cuModuleLoadData(&kernel.cuda.cu_module, link_output);
        }
        if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
            jit_malloc_trim();
            /* Unlock while synchronizing */ {
                unlock_guard guard(state.mutex);
                ret = cuModuleLoadData(&kernel.cuda.cu_module, link_output);
            }
        }
        cuda_check(ret);

        // Locate the kernel entry point
        std::string name = std::string("enoki_") + kernel_name;
        cuda_check(cuModuleGetFunction(&kernel.cuda.cu_func, kernel.cuda.cu_module,
                                       name.c_str()));

        /// Enoki doesn't use shared memory at all, prefer to have more L1 cache.
        cuda_check(cuFuncSetAttribute(
            kernel.cuda.cu_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
        cuda_check(cuFuncSetAttribute(
            kernel.cuda.cu_func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            CU_SHAREDMEM_CARVEOUT_MAX_L1));

        int reg_count;
        cuda_check(cuFuncGetAttribute(
            &reg_count, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel.cuda.cu_func));

        // Destroy the linker invocation
        cuda_check(cuLinkDestroy(link_state));

        char *str = (char *) malloc(buffer.size() + 1);
        memcpy(str, buffer.get(), buffer.size() + 1);

        cuda_check(cuOccupancyMaxPotentialBlockSize(
            &kernel.cuda.block_count, &kernel.cuda.thread_count,
            kernel.cuda.cu_func, nullptr, 0, 0));

        kernel.type = KernelType::CUDA;
        state.kernel_cache.emplace(str, kernel);

        jit_log(Debug,
                "jit_run(): cache %s, codegen: %s, %s: %s, %i registers, %i "
                "threads, %i blocks.",
                cache_hit ? "hit" : "miss",
                std::string(jit_time_string(codegen_time)).c_str(),
                cache_hit ? "load" : "build",
                std::string(jit_time_string(link_time)).c_str(), reg_count,
                kernel.cuda.thread_count, kernel.cuda.block_count);
    } else {
        kernel = it.value();
        jit_log(Debug, "jit_run(): cache hit, codegen: %s.",
                jit_time_string(codegen_time));
    }

    size_t kernel_args_size = (size_t) kernel_args.size() * sizeof(uint64_t);

    void *config[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER,
        kernel_args.data(),
        CU_LAUNCH_PARAM_BUFFER_SIZE,
        &kernel_args_size,
        CU_LAUNCH_PARAM_END
    };

    uint32_t thread_count = kernel.cuda.thread_count,
             block_count  = kernel.cuda.block_count;

    /// Reduce the number of blocks when processing a very small amount of data
    if (group.size <= thread_count) {
        block_count = 1;
        thread_count = group.size;
    } else if (group.size <= thread_count * block_count) {
        block_count = (group.size + thread_count - 1) / thread_count;
    }

    cuda_check(cuLaunchKernel(kernel.cuda.cu_func, block_count, 1, 1, thread_count,
                              1, 1, 0, active_stream->handle, nullptr, config));
}

/// Evaluate all computation that is queued on the current device & stream
void jit_eval() {
    Stream *stream = active_stream;
    bool cuda = stream != nullptr;
    auto &todo = cuda ? stream->todo : state.todo_host;

    visited.clear();
    schedule.clear();

    // Collect variables that must be computed and their subtrees
    for (uint32_t index : todo) {
        Variable *v = jit_var(index);
        if (v->ref_count_ext == 0)
            continue;
        jit_var_traverse(v->size, index);
    }
    todo.clear();

    if (schedule.empty())
        return;

    // Group them from large to small sizes while preserving dependencies
    std::stable_sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            return a.size > b.size;
        });

    schedule_groups.clear();
    if (schedule[0].size == schedule[schedule.size() - 1].size) {
        schedule_groups.emplace_back(schedule[0].size, 0,
                                     (uint32_t) schedule.size());
    } else {
        uint32_t cur = 0;
        for (uint32_t i = 1; i < (uint32_t) schedule.size(); ++i) {
            if (schedule[i - 1].size != schedule[i].size) {
                schedule_groups.emplace_back(schedule[cur].size, cur, i);
                cur = i;
            }
        }
        schedule_groups.emplace_back(schedule[cur].size,
                                     cur, (uint32_t) schedule.size());
    }

    // Are there independent groups of work that could be dispatched in parallel?
    bool parallel_dispatch =
        state.parallel_dispatch && cuda && schedule_groups.size() > 1;

    if (!parallel_dispatch) {
        jit_log(Debug, "jit_eval(): begin.");
    } else {
        jit_log(Debug, "jit_eval(): begin (parallel dispatch to %zu streams).",
                schedule_groups.size());
        cuda_check(cuEventRecord(stream->event, stream->handle));
    }

    uint32_t group_idx = 1;
    for (ScheduledGroup &group : schedule_groups) {
        jit_assemble(group);

        Stream *sub_stream = stream;
        if (parallel_dispatch) {
            uint32_t stream_index = 1000 * stream->stream + group_idx++;
            jit_device_set(stream->device, stream_index);
            sub_stream = active_stream;
            cuda_check(cuStreamWaitEvent(sub_stream->handle, stream->event, 0));
        }

        if (cuda)
            jit_run_cuda(group);
        else
            jit_run_llvm(group);

        if (parallel_dispatch) {
            cuda_check(cuEventRecord(sub_stream->event, sub_stream->handle));
            cuda_check(cuStreamWaitEvent(stream->handle, sub_stream->event, 0));
        }
    }

    if (parallel_dispatch)
        jit_device_set(stream->device, stream->stream);

    /* At this point, all variables and their dependencies are computed, which
       means that we can remove internal edges between them. This in turn will
       cause many of the variables to be garbage-collected. */
    jit_log(Debug, "jit_eval(): cleaning up..");

    for (ScheduledVariable sv : schedule) {
        uint32_t index = sv.index;

        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;

        Variable *v = &it.value();
        jit_cse_drop(index, v);

        bool side_effect = v->side_effect;
        uint32_t dep[3], extra_dep = v->extra_dep;
        memcpy(dep, v->dep, sizeof(uint32_t) * 3);

        memset(v->dep, 0, sizeof(uint32_t) * 3);
        v->extra_dep = 0;
        for (int j = 0; j < 3; ++j)
            jit_var_dec_ref_int(dep[j]);
        jit_var_dec_ref_ext(extra_dep);

        if (unlikely(side_effect)) {
            fprintf(stderr, "Variable with side effect on %u\n", extra_dep);
            if (extra_dep) {
                Variable *v2 = jit_var(extra_dep);
                v2->dirty = false;
            }

            Variable *v2 = jit_var(index);
            if (unlikely(v2->ref_count_ext != 1 || v2->ref_count_int != 0))
                jit_fail("jit_eval(): invalid invalid reference for statment "
                         "with side effects!");
            jit_var_dec_ref_ext(index, v2);
        }
    }

    if (cuda)
        jit_free_flush();

    jit_log(Debug, "jit_eval(): done.");
}

