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

/// A single variable that is scheduled to execute for a launch with 'size' entries
struct ScheduledVariable {
    uint32_t size;
    uint32_t index;

    ScheduledVariable(uint32_t size, uint32_t index)
        : size(size), index(index) { }
};

/// Start and end index of a group of variables that will be merged into the same kernel
struct ScheduledGroup {
    uint32_t size;
    uint32_t start;
    uint32_t end;

    ScheduledGroup(uint32_t size, uint32_t start, uint32_t end)
        : size(size), start(start), end(end) { }
};

struct Intrinsic {
    uint32_t width;
    const char *str;

    Intrinsic(uint32_t width, const char *str) : width(width), str(str) { }
};

/// Hash function for keeping track of LLVM intrinsics
struct IntrinsicHash {
    size_t operator()(const Intrinsic &a) const {
        const char *end = strchr(a.str, '(');
        return end ? hash(a.str, end - a.str, (uint32_t) a.width) : (size_t) 0u;
    }
};

/// Equality operation for keeping track of LLVM intrinsics
struct IntrinsicEquality {
    size_t operator()(const Intrinsic &a, const Intrinsic &b) const {
        const char *end_a = strchr(a.str, '('),
                   *end_b = strchr(b.str, '(');
        const size_t strlen_a = end_a - a.str,
                     strlen_b = end_b - b.str;
        if (a.width != b.width || !end_a || !end_b || strlen_a != strlen_b)
            return 0;
        return strncmp(a.str, b.str, strlen_a) == 0;
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
size_t kernel_hash = 0;

/// Name of the last generated kernel
static char kernel_name[17] { };

/// Dedicated buffer for intrinsic intrinsics_buffer (LLVM only)
Buffer intrinsics_buffer{1};

/// Intrinsics used by the current program (LLVM only)
static tsl::robin_set<Intrinsic, IntrinsicHash, IntrinsicEquality> intrinsics_set;

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

/// Convert an IR template with '$' expressions into valid IR (PTX variant)
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

/// Convert an IR template with '$' expressions into valid IR (LLVM variant)
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
                case 'n': buffer.put("\n    "); continue;
                case 'i': buffer.put("%index"); continue;
                case 'w': buffer.fmt("%u", jit_llvm_vector_width); continue;
                case 'z':
                case 'l':
                case 't': prefix_table = var_type_name_llvm; break;
                case 's': prefix_table = var_type_size_str; break;
                case 'o':
                case 'b': prefix_table = var_type_name_llvm_bin; break;
                case 'a': prefix_table = var_type_name_llvm_abbrev; break;
                case 'r': prefix_table = var_type_prefix; break;
                case 'O':
                    buffer.putc('<');
                    for (uint32_t i = 0; i < jit_llvm_vector_width; ++i)
                        buffer.fmt("i1 1%s", i + 1 < jit_llvm_vector_width ? ", " : "");
                    buffer.putc('>');
                    continue;

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
                case 's':
                case 't':
                    buffer.put(prefix);
                    break;

                case 'l':
                    buffer.putc('<');
                    for (uint32_t i = 0; i < jit_llvm_vector_width; ++i)
                        buffer.fmt("%s %i%s%s", prefix, i, is_float ? ".0" : "",
                                   i + 1 < jit_llvm_vector_width ? ", " : "");
                    buffer.putc('>');
                    break;

                case 'z':
                    buffer.putc('<');
                    for (uint32_t i = 0; i < jit_llvm_vector_width; ++i)
                        buffer.fmt("%s 0%s%s", prefix, is_float ? ".0" : "",
                                   i + 1 < jit_llvm_vector_width ? ", " : "");
                    buffer.putc('>');
                    break;

                case 'o':
                    buffer.putc('<');
                    for (uint32_t i = 0; i < jit_llvm_vector_width; ++i)
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

    if (intrinsics_set.find({ jit_llvm_vector_width, s }) != intrinsics_set.end())
        return;
    intrinsics_set.insert({ jit_llvm_vector_width, s });

    intrinsics_buffer.put("declare ");
    while ((c = *s++) != '\0') {
        if (c != '$') {
            intrinsics_buffer.putc(c);
        } else {
            const char **prefix_table = nullptr, type = *s++;
            switch (type) {
                case 'O': continue;
                case 't': prefix_table = var_type_name_llvm; break;
                case 'b': prefix_table = var_type_name_llvm_bin; break;
                case 'a': prefix_table = var_type_name_llvm_abbrev; break;
                case 'w': intrinsics_buffer.fmt("%u", jit_llvm_vector_width); continue;
                case 's':
                case 'r': s++; intrinsics_buffer.rewind(1); continue;
                case 'n':
                case 'o':
                case 'l':
                case 'z': s++; continue;
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
    auto get_parameter_addr = [](const Variable *v, bool load, uint32_t target = 0) {
        if (v->arg_index < CUDA_MAX_KERNEL_PARAMETERS - 1)
            buffer.fmt("    ld.param.u64 %%rd%u, [arg%u];\n", target, v->arg_index - 1);
        else
            buffer.fmt("    ldu.global.u64 %%rd%u, [%%rd2 + %u];\n",
                       target, (v->arg_index - (CUDA_MAX_KERNEL_PARAMETERS - 1)) * 8);

        if (v->size > 1 || !load)
            buffer.fmt("    mul.wide.u32 %%rd1, %%r0, %u;\n"
                       "    add.u64 %%rd%u, %%rd%u, %%rd1;\n",
                       var_type_size[(int) v->type], target, target);
    };

    const Device &device = state.devices[active_stream->device];

    buffer.put(".version 6.3\n");
    buffer.fmt(".target sm_%i\n", device.compute_capability);
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

    buffer.put(".entry enoki_^^^^^^^^^^^^^^^^(.param .u32 size,\n");
    for (uint32_t index = 1; index < kernel_args.size(); ++index)
        buffer.fmt("                              .param .u64 arg%u%s\n",
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

    buffer.fmt("\nL1: // Loop body (compute capability %i)\n",
               device.compute_capability);

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

            get_parameter_addr(v, true, v->direct_pointer ? v->reg_index : 0u);

            if (likely(!v->direct_pointer)) {
                if (likely(v->type != (uint32_t) VarType::Bool)) {
                    buffer.fmt("    %s.%s %s%u, [%%rd0];\n",
                           v->size == 1 ? "ldu.global" : "ld.global.cs",
                           var_type_name_ptx[(int) v->type],
                           var_type_prefix[(int) v->type],
                           v->reg_index);
                } else {
                    buffer.fmt("    %s.u8 %%w0, [%%rd0];\n",
                           v->size == 1 ? "ldu.global" : "ld.global.cs");
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

            get_parameter_addr(v, false);

            if (likely(v->type != (uint32_t) VarType::Bool)) {
                buffer.fmt("    st.global.cs.%s [%%rd0], %s%u;\n",
                       var_type_name_ptx[(int) v->type],
                       var_type_prefix[(int) v->type],
                       v->reg_index);
            } else {
                buffer.fmt("    selp.u16 %%w0, 1, 0, %%p%u;\n",
                           v->reg_index);
                buffer.put("    st.global.cs.u8 [%rd0], %w0;\n");
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
}

/// Replace generic LLVM intrinsics by more efficient hardware-specific ones
static void jit_llvm_rewrite_intrinsics() {
    const char *replacements[][4] = {
        { "minnum.v4f32",     "x86.sse.min.ps",        nullptr, nullptr },
        { "maxnum.v4f32",     "x86.sse.max.ps",        nullptr, nullptr },
        { "minnum.v8f32",     "x86.avx.min.ps.256",    nullptr, nullptr },
        { "maxnum.v8f32",     "x86.avx.max.ps.256",    nullptr, nullptr },
        { "minnum.v16f32",    "x86.avx512.min.ps.512", "i32 4", "i32"},
        { "maxnum.v16f32",    "x86.avx512.max.ps.512", "i32 4", "i32"}
    };

    intrinsics_buffer.clear();

    char *ptr = (char *) buffer.get();
    while (true) {
        char *ptr_next = strstr(ptr, "@llvm.");
        if (!ptr_next) {
            intrinsics_buffer.put(ptr);
            break;
        }

        bool is_declaration = false;
        char *line = ptr_next;
        while (line != ptr && *line != '\n')
            --line;
        if (line && strncmp(line + 1, "declare", 7) == 0)
            is_declaration = true;

        ptr_next += 6;
        ptr_next[-1] = '\0';
        intrinsics_buffer.put(ptr);
        intrinsics_buffer.putc('.');

        for (auto &o : replacements) {
            if (strncmp(ptr_next, o[0], strlen(o[0])) == 0) {
                intrinsics_buffer.put(o[1]);
                ptr_next += strlen(o[0]);

                char *paren = strchr(ptr_next, ')');
                const char *suffix = is_declaration ? o[3] : o[2];
                if (paren && suffix) {
                    *paren = '\0';
                    intrinsics_buffer.fmt("%s, %s)", ptr_next, suffix);
                    ptr_next = paren + 1;
                }
                break;
            }
        }

        ptr = ptr_next;
    }

    intrinsics_buffer.swap(buffer);
}

void jit_assemble_llvm(ScheduledGroup group, const char *suffix = "") {
    const int width = jit_llvm_vector_width;

    bool log_trace = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace;

    buffer.fmt("define void @enoki_^^^^^^^^^^^^^^^^%s(i64 %%start, "
               "i64 %%end, i8** %%ptrs) #0", suffix);
    if (width > 1)
        buffer.fmt(" alignstack(%u)", std::max(16u, width * (uint32_t) sizeof(float)));
    buffer.put(" {\nentry:");
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
    buffer.put("    br i1 %cond, label %done, label %loop, !llvm.loop !2\n");
    buffer.put("}\n\n");
}

void jit_assemble_llvm_suffix() {
    if (intrinsics_buffer.size() > 1) {
        buffer.put(intrinsics_buffer.get());
        jit_llvm_rewrite_intrinsics();
        buffer.putc('\n');
    }

    buffer.put("!0 = !{!0}\n");
    buffer.put("!1 = !{!1, !0}\n");
    buffer.put("!2 = !{!\"llvm.loop.unroll.disable\", !\"llvm.loop.vectorize.enable\", i1 0}\n\n");
    buffer.fmt("attributes #0 = { norecurse nounwind \"target-cpu\"=\"%s\"", jit_llvm_target_cpu);
    if (jit_llvm_target_features)
        buffer.fmt(" \"target-features\"=\"%s\"", jit_llvm_target_features);
    buffer.put(" }");
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
            size_t isize    = (size_t) var_type_size[(int) v->type],
                   var_size = (size_t) group.size * isize;

            void *data = jit_malloc(cuda ? AllocType::Device : AllocType::Host, var_size);

            // jit_malloc() may temporarily release the lock, variable pointer might have changed
            v = jit_var(index);

            v->data = data;
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
        cuda_check(cuMemcpyAsync((CUdeviceptr) args_extra_dev,
                                 (CUdeviceptr) args_extra_host,
                                 args_extra_size, active_stream->handle));

        kernel_args.push_back(args_extra_dev);
        jit_free(args_extra_host);

        /* Safe, because there won't be further allocations on the current
           stream until after this kernel has executed. */
        jit_free(args_extra_dev);
    }

    jit_log(Info, "jit_run(): launching kernel (n=%u, in=%u, out=%u, ops=%u) ..",
            group.size, n_args_in - 1, n_args_out, n_regs_total);

    intrinsics_buffer.clear();
    intrinsics_set.clear();
    buffer.clear();

    if (cuda) {
        jit_assemble_cuda(group, n_regs_total);
    } else {
        jit_assemble_llvm(group);

        /* Compile a separate scalar variant of the kernel if it uses
           scatter/gather intrinsics. This is needed to deal with the last
           packet, where not all lanes are guaranteed to be valid. */
        if (jit_llvm_vector_width != 1 &&
            (intrinsics_buffer.contains("@llvm.masked.scatter") ||
             intrinsics_buffer.contains("@llvm.masked.gather"))) {
            uint32_t vector_width = 1;
            std::swap(jit_llvm_vector_width, vector_width);
            jit_assemble_llvm(group, "_scalar");
            std::swap(jit_llvm_vector_width, vector_width);
        }

        jit_assemble_llvm_suffix();
    }

    /// Replace '^'s in 'enoki_^^^^^^^^' by hash code
    kernel_hash = hash_kernel(buffer.get());
    snprintf(kernel_name, 17, "%016llx", (unsigned long long) kernel_hash);

    char *offset = (char *) buffer.get();
    do {
        offset = strchr(offset, '^');
        if (!offset)
            break;
        memcpy(offset, kernel_name, 16);
    } while (true);

    jit_log(Debug, "%s", buffer.get());
}

void jit_run(Stream *stream, ScheduledGroup group) {
    bool llvm = stream == nullptr;
    int device_id = llvm ? -1 : stream->device;

    float codegen_time = timer();
    KernelKey kernel_key((char *) buffer.get(), device_id);
    size_t hash_code = KernelHash::compute_hash(kernel_hash, device_id);
    auto it = state.kernel_cache.find(kernel_key, hash_code);
    Kernel kernel;

    if (it == state.kernel_cache.end()) {
        bool cache_hit = jit_kernel_load(
            buffer.get(), buffer.size(), llvm, kernel_hash, kernel);

        if (!cache_hit) {
            if (llvm)
                jit_llvm_compile(buffer.get(), buffer.size(), kernel);
            else
                jit_cuda_compile(buffer.get(), buffer.size(), kernel);

            jit_kernel_write(buffer.get(), buffer.size(), llvm, kernel_hash, kernel);
        }

        if (llvm) {
            jit_llvm_disasm(kernel);
        } else {
            CUresult ret;
            /* Unlock while synchronizing */ {
                unlock_guard guard(state.mutex);
                ret = cuModuleLoadData(&kernel.cuda.cu_module, kernel.data);
            }
            if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
                jit_malloc_trim();
                /* Unlock while synchronizing */ {
                    unlock_guard guard(state.mutex);
                    ret = cuModuleLoadData(&kernel.cuda.cu_module, kernel.data);
                }
            }
            cuda_check(ret);

            // Locate the kernel entry point
            char kernel_name[23];
            snprintf(kernel_name, 23, "enoki_%016llx", (unsigned long long) kernel_hash);
            cuda_check(cuModuleGetFunction(&kernel.cuda.cu_func, kernel.cuda.cu_module,
                                           kernel_name));

            /// Determine a suitable thread count to maximize occupancy
            int unused, block_size;
            cuda_check(cuOccupancyMaxPotentialBlockSize(
                &unused, &block_size,
                kernel.cuda.cu_func, nullptr, 0, 0));
            kernel.cuda.block_size = (uint32_t) block_size;

            /// Enoki doesn't use shared memory at all, prefer to have more L1 cache.
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.cu_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.cu_func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                CU_SHAREDMEM_CARVEOUT_MAX_L1));

            free(kernel.data);
            kernel.data = nullptr;
        }

        float link_time = timer();
        jit_log(Info, "jit_run(): cache %s, codegen: %s, %s: %s, %s.",
                cache_hit ? "hit" : "miss",
                std::string(jit_time_string(codegen_time)).c_str(),
                cache_hit ? "load" : "build",
                std::string(jit_time_string(link_time)).c_str(),
                std::string(jit_mem_string(kernel.size)).c_str());

        kernel_key.str = (char *) malloc_check(buffer.size() + 1);
        memcpy(kernel_key.str, buffer.get(), buffer.size() + 1);
        state.kernel_cache.emplace(kernel_key, kernel);
    } else {
        kernel = it.value();
        jit_log(Debug, "jit_run(): cache hit, codegen: %s.",
                jit_time_string(codegen_time));
    }

    if (llvm) {
        uint32_t width = kernel.llvm.func != kernel.llvm.func_scalar
                             ? jit_llvm_vector_width : 1u,
                 rounded = group.size / width * width;

        if (likely(rounded > 0))
            kernel.llvm.func(0, rounded, kernel_args_extra.data());

        if (unlikely(rounded != group.size))
            kernel.llvm.func_scalar(rounded, group.size, kernel_args_extra.data());
    } else {
        size_t kernel_args_size = (size_t) kernel_args.size() * sizeof(uint64_t);

        void *config[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER,
            kernel_args.data(),
            CU_LAUNCH_PARAM_BUFFER_SIZE,
            &kernel_args_size,
            CU_LAUNCH_PARAM_END
        };

        uint32_t block_count, thread_count;
        const Device &device = state.devices[device_id];
        device.get_launch_config(&block_count, &thread_count, group.size,
                                 (uint32_t) kernel.cuda.block_size);

        cuda_check(cuLaunchKernel(kernel.cuda.cu_func, block_count, 1, 1, thread_count,
                                  1, 1, 0, active_stream->handle, nullptr, config));
    }
}

/// Evaluate all computation that is queued on the current device & stream
void jit_eval() {
    /* The function 'jit_eval()' cannot be executed concurrently. Temporarily
       release 'state.mutex' before acquiring 'state.eval_mutex'. */
    state.mutex.unlock();
    lock_guard guard(state.eval_mutex);
    state.mutex.lock();

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

        jit_run(stream, group);

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
        if (!v->stmt)
            continue;

        jit_cse_drop(index, v);

        if (unlikely(v->free_stmt))
            free(v->stmt);

        bool side_effect = v->side_effect;
        uint32_t dep[3], extra_dep = v->extra_dep;
        memcpy(dep, v->dep, sizeof(uint32_t) * 3);

        memset(v->dep, 0, sizeof(uint32_t) * 3);
        v->extra_dep = 0;
        v->stmt = nullptr;

        for (int j = 0; j < 3; ++j)
            jit_var_dec_ref_int(dep[j]);
        jit_var_dec_ref_ext(extra_dep);

        if (unlikely(side_effect)) {
            if (extra_dep) {
                Variable *v2 = jit_var(extra_dep);
                v2->pending_scatter = false;
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

