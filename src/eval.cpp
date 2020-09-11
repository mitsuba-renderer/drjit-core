/*
    src/eval.cpp -- Main computation graph evaluation routine

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "tbb.h"
#include "itt.h"
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
static Buffer intrinsics_buffer{1};

/// Temporary buffer for storing a single LLVM intrinsic (LLVM only)
static Buffer intrinsic_buffer_tmp{1};

/// LLVM: Does the kernel require the supplemental IR? (Used for 'scatter_add' atm.)
static bool jit_llvm_supplement = false;

#if defined(ENOKI_JIT_ENABLE_TBB)
std::vector<std::pair<uint32_t, uint32_t>> jit_llvm_scatter_add_variables;
#endif

// ====================================================================

/// Forward declaration
void jit_render_stmt_llvm_unroll(uint32_t index, Variable *v);

/// Recursively traverse the computation graph to find variables needed by a computation
static void jit_var_traverse(uint32_t size, uint32_t index, Variable *v) {
    if (!visited.emplace(size, index).second)
        return;

    if (likely(!v->direct_pointer && !v->data)) {
        struct Dep {
            uint32_t index;
            uint32_t tsize;
            Variable *v;
        } depv[4];

        memset(depv, 0, sizeof(depv));
        for (int i = 0; i < 4; ++i) {
            uint32_t dep_index = v->dep[i];
            if (dep_index == 0)
                break;
            Variable *v = jit_var(dep_index);
            depv[i].index = dep_index;
            depv[i].tsize = v->tsize;
            depv[i].v = v;
        }

        // Simple sorting network
        #define SWAP(i, j) \
            if (depv[i].tsize < depv[j].tsize) \
                std::swap(depv[i], depv[j]);
        SWAP(0, 1);
        SWAP(2, 3);
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(1, 2);

        #undef SWAP

        for (auto &dep: depv) {
            if (!dep.index)
                break;
            jit_var_traverse(size, dep.index, dep.v);
        }
    }

    v->output_flag = false;
    schedule.emplace_back(size, index);
}

/// Convert an IR template with '$' expressions into valid IR (PTX variant)
void jit_render_stmt_cuda(uint32_t index, Variable *v) {
    const char *s = v->stmt;
    char c;

    buffer.put("    ");

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
                case 'i': buffer.put("%r0"); continue;
                case 't': prefix_table = var_type_name_ptx;     break;
                case 'b': prefix_table = var_type_name_ptx_bin; break;
                case 's': prefix_table = var_type_size_str; break;
                case 'r': prefix_table = var_type_prefix; break;
                default:
                    jit_fail("jit_render_stmt_cuda(): encountered invalid \"$\" "
                             "expression (unknown type \"%c\") in \"%s\"!", type, v->stmt);
            }

            uint32_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 4))
                jit_fail("jit_render_stmt_cuda(%s): encountered invalid \"$\" "
                         "expression (argument out of bounds)!", v->stmt);

            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            if (unlikely(dep_id == 0))
                jit_fail("jit_render_stmt_cuda(%s): encountered invalid \"$\" "
                         "expression (dependency %u is missing)!", v->stmt, arg_id);

            Variable *dep = jit_var(dep_id);
            buffer.put(prefix_table[(int) dep->type]);

            if (type == 'r')
                buffer.put_uint32(dep->reg_index);
        }
    } while (c != '\0');

    buffer.put(";\n");
}

/// Convert an IR template with '$' expressions into valid IR (LLVM variant)
void jit_render_stmt_llvm(uint32_t index, Variable *v, const char *suffix = "") {
    const char *s = v->stmt;

    if (s[0] == '$' && s[1] >= '0' && s[1] <= '9') {
        uint32_t width = 1 << (s[1] - '0');
        if (width < jit_llvm_vector_width) {
            jit_render_stmt_llvm_unroll(index, v);
            return;
        } else {
            s += 2;
        }
    }

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
                case 'w': buffer.put_uint32(jit_llvm_vector_width); continue;
                case 'z': buffer.put("zeroinitializer"); continue;
                case 'S': continue;
                case 'l':
                case 't': prefix_table = var_type_name_llvm; break;
                case 'T': prefix_table = var_type_name_llvm_big; break;
                case 's': prefix_table = var_type_size_str; break;
                case 'o':
                case 'b': prefix_table = var_type_name_llvm_bin; break;
                case 'a': prefix_table = var_type_name_llvm_abbrev; break;
                case 'r': prefix_table = var_type_prefix; break;
                case 'O':
                    buffer.putc('<');
                    for (uint32_t i = 0; i < jit_llvm_vector_width; ++i) {
                        buffer.put("i1 1");
                        if (i + 1 < jit_llvm_vector_width)
                            buffer.put(", ");
                    }
                    buffer.putc('>');
                    continue;

                default:
                    jit_fail("jit_render_stmt_llvm(): encountered invalid \"$\" "
                             "expression (unknown type \"%c\")!", type);
            }

            uint32_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 4))
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
                    buffer.put(prefix);
                    buffer.put_uint32(dep->reg_index);
                    if (!dep->direct_pointer)
                        buffer.put(suffix);
                    break;

                case 'a':
                case 'b':
                case 's':
                case 't':
                case 'T':
                    buffer.put(prefix);
                    break;

                case 'l':
                    buffer.putc('<');
                    for (uint32_t i = 0; i < jit_llvm_vector_width; ++i) {
                        buffer.put(prefix);
                        buffer.putc(' ');
                        buffer.put_uint32(i);
                        if (is_float)
                            buffer.put(".0");
                        if (i + 1 < jit_llvm_vector_width)
                            buffer.put(", ");
                    }
                    buffer.putc('>');
                    break;

                case 'o':
                    buffer.putc('<');
                    for (uint32_t i = 0; i < jit_llvm_vector_width; ++i) {
                        buffer.put(prefix);
                        buffer.put(" -1");
                        if (i + 1 < jit_llvm_vector_width)
                            buffer.put(", ");
                    }

                    buffer.putc('>');
                    break;
            }
        }
    }

    buffer.putc('\n');
    s = v->stmt;

    // Check for intrinsics
    const char *ek = strstr(s, "@ek.");
    if (ek) {
        jit_llvm_supplement = true;
        return;
    }

    s = strstr(s, "call ");
    if (likely(!s))
        return;
    s += 5;

    uint32_t types = 0;
    for (int i = 0; i < 4; ++i) {
        if (!v->dep[i])
            break;
        types = types * 16 + jit_var(v->dep[i])->type;
    }

    intrinsic_buffer_tmp.clear();
    intrinsic_buffer_tmp.put("declare ");

    while ((c = *s++) != '\0') {
        if (c != '$') {
            intrinsic_buffer_tmp.putc(c);
        } else {
            const char **prefix_table = nullptr, type = *s++;
            bool stop = false;

            switch (type) {
                case 'z':
                case 'O': intrinsic_buffer_tmp.rewind(1); continue;
                case 't': prefix_table = var_type_name_llvm; break;
                case 'b': prefix_table = var_type_name_llvm_bin; break;
                case 'a': prefix_table = var_type_name_llvm_abbrev; break;
                case 'w': intrinsic_buffer_tmp.put_uint32(jit_llvm_vector_width); continue;
                case 'S': while (*s != ',' && *s != ')' && *s != '\0') { ++s; } continue;
                case 's':
                case 'r': s++; intrinsic_buffer_tmp.rewind(1); continue;
                case 'n': stop = true; break;
                case 'o':
                case 'l': s++; continue;
                default:
                    jit_fail("jit_render_stmt_llvm(): encountered invalid \"$\" "
                             "expression (unknown type \"%c\")!", type);
            }
            if (stop)
                break;

            uint32_t arg_id = *s++ - '0';
            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            Variable *dep = jit_var(dep_id);
            intrinsic_buffer_tmp.put(prefix_table[(int) dep->type]);
        }
    }
    intrinsic_buffer_tmp.putc('\n');

    if (!intrinsics_buffer.contains(intrinsic_buffer_tmp.get()))
        intrinsics_buffer.put(intrinsic_buffer_tmp.get());
}

/// Expand fixed-length LLVM intrinsics by calling them multiple times
void jit_render_stmt_llvm_unroll(uint32_t index, Variable *v) {
    const char *s = v->stmt;
    uint32_t width       = 1 << (s[1] - '0'),
             width_host  = jit_llvm_vector_width,
             last_level  = 0;

    /// Recursively partition dependencies into arrays of size 'width'
    for (uint32_t i = 0; i < 4; ++i) {
        if (v->dep[i] == 0)
            continue;
        Variable *dep = jit_var(v->dep[i]);
        if (dep->direct_pointer)
            continue;
        for (uint32_t l = 0; ; ++l) {
            uint32_t w = width_host / (2u << l);
            if (w < width)
                break;

            for (uint32_t j = 0; j < (1u << l); ++j) {
                for (uint32_t k = 0; k < 2; ++k) {
                    char source[64];
                    if (l == 0)
                        snprintf(source, sizeof(source), "%s%u",
                                 var_type_prefix[(int) dep->type], dep->reg_index);
                    else
                        snprintf(source, sizeof(source), "%s%u_unroll_%s%u_%u_%u",
                                 var_type_prefix[(int) dep->type], dep->reg_index,
                                 var_type_prefix[v->type] + 1, v->reg_index,
                                 l - 1, j);

                    buffer.fmt("    %s%u_unroll_%s%u_%u_%u = shufflevector <%u "
                               "x %s> %s, <%u x "
                               "%s> undef, <%u x i32> <",
                               var_type_prefix[(int) dep->type], dep->reg_index,
                               var_type_prefix[v->type] + 1, v->reg_index,
                               l, 2 * j + k, w * 2u,
                               var_type_name_llvm[(int) dep->type], source,
                               w * 2u, var_type_name_llvm[(int) dep->type], w);
                    for (uint32_t r = 0; r < w; ++r)
                        buffer.fmt("i32 %u%s", r + k*w, r + 1 < w ? ", " : "");
                    buffer.put(">\n");
                }
            }
            buffer.put("\n");

            last_level = l;
        }
    }

    jit_llvm_vector_width = width;

    for (uint32_t j = 0; j < width_host / width; ++j) {
        char suffix[64];
        snprintf(suffix, 64, "_unroll_%s%u_%u_%u",
                 var_type_prefix[v->type] + 1,
                 v->reg_index, last_level, j);
        jit_render_stmt_llvm(index, v, suffix);
    }
    buffer.putc('\n');

    jit_llvm_vector_width = width_host;

    // Stop here if the statement doesn't produce a return value
    if ((VarType) v->type == VarType::Invalid)
        return;

    // Recursively reassemble and output array of size 'width_host'
    for (uint32_t l = last_level; ; l /= 2) {
        uint32_t w = width_host / (2u << l);
        for (uint32_t j = 0; j < (1u << l); ++j) {
            char target[64];
            if (l == 0)
                snprintf(target, sizeof(target), "%s%u",
                         var_type_prefix[v->type], v->reg_index);
            else
                snprintf(target, sizeof(target), "%s%u_unroll_%s%u_%u_%u",
                         var_type_prefix[v->type], v->reg_index,
                         var_type_prefix[v->type] + 1, v->reg_index,
                         l - 1, j);

            buffer.fmt("    %s = shufflevector <%u "
                       "x %s> %s%u_unroll_%s%u_%u_%u, <%u x "
                       "%s> %s%u_unroll_%s%u_%u_%u, <%u x i32> <",
                       target,
                       w, var_type_name_llvm[v->type],
                       var_type_prefix[v->type], v->reg_index,
                       var_type_prefix[v->type] + 1, v->reg_index, l, 2*j,
                       w, var_type_name_llvm[v->type],
                       var_type_prefix[v->type], v->reg_index,
                       var_type_prefix[v->type] + 1, v->reg_index, l, 2*j+1,
                       w*2u);
            for (uint32_t r = 0; r < 2 * w; ++r)
                buffer.fmt("i32 %u%s", r, r + 1 < 2 * w ? ", " : "");
            buffer.put(">\n");
        }
        if (l == 0)
            break;
        buffer.put("\n");
    }
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
                       var_type_size[v->type], target, target);
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
                   index - 1, (index + 1 < (uint32_t) kernel_args.size()) ? "," : ") {");

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
        const char *label = log_trace ? jit_var_label(index) : nullptr;

        if (v->arg_type == ArgType::Input) {
            if (unlikely(log_trace)) {
                buffer.fmt("\n    // Load %s%u%s%s\n",
                           var_type_prefix[v->type],
                           v->reg_index, label ? ": " : "",
                           label ? label : "");
            }

            get_parameter_addr(v, true, v->direct_pointer ? v->reg_index : 0u);

            if (likely(!v->direct_pointer)) {
                if (likely(v->type != (uint32_t) VarType::Bool)) {
                    buffer.fmt("    %s.%s %s%u, [%%rd0];\n",
                           v->size == 1 ? "ldu.global" : "ld.global.cs",
                           var_type_name_ptx[v->type],
                           var_type_prefix[v->type],
                           v->reg_index);
                } else {
                    buffer.fmt("    %s.u8 %%w0, [%%rd0];\n",
                           v->size == 1 ? "ldu.global" : "ld.global.cs");
                    buffer.fmt("    setp.ne.u16 %s%u, %%w0, 0;\n",
                           var_type_prefix[v->type],
                           v->reg_index);
                }
            }
        } else {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Evaluate %s%u%s%s\n",
                           var_type_prefix[v->type], v->reg_index,
                           label ? ": " : "", label ? label : "");

            jit_render_stmt_cuda(index, v);
        }

        if (v->arg_type == ArgType::Output) {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Store %s%u%s%s\n",
                           var_type_prefix[v->type], v->reg_index,
                           label ? ": " : "", label ? label : "");

            get_parameter_addr(v, false);

            if (likely(v->type != (uint32_t) VarType::Bool)) {
                buffer.fmt("    st.global.cs.%s [%%rd0], %s%u;\n",
                       var_type_name_ptx[v->type],
                       var_type_prefix[v->type],
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
        uint32_t reg_id = v->reg_index, arg_id = v->arg_index - 1;
        const char *type = (VarType) v->type == VarType::Bool
                               ? "i8" : var_type_name_llvm[v->type];

        if (unlikely(log_trace))
            buffer.fmt("\n    ; Prepare argument %u\n", arg_id);

        buffer.fmt("    %%a%u_i = getelementptr inbounds i8*, i8** %%ptrs, i32 %u\n", arg_id, arg_id);

        if (likely(!v->direct_pointer)) {
            buffer.fmt("    %%a%u_p = load i8*, i8** %%a%u_i, align 8, !alias.scope !1\n", arg_id, arg_id);
            buffer.fmt("    %%a%u = bitcast i8* %%a%u_p to %s*\n", arg_id, arg_id, type);
            if (v->size == 1) {
                buffer.fmt("    %%a%u_s = load %s, %s* %%a%u, align %u, !alias.scope !1\n", arg_id,
                           type, type, arg_id, var_type_size[v->type]);
                if ((VarType) v->type == VarType::Bool)
                    buffer.fmt("    %%a%u_s1 = trunc i8 %%a%u_s to i1\n", arg_id, arg_id);
            }
        } else {
            buffer.fmt("    %%rd%u = load i8*, i8** %%a%u_i, align 8, !alias.scope !1\n", reg_id, arg_id);
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
        uint32_t align = v->unaligned ? 1 : (var_type_size[v->type] * width),
                 reg_id = v->reg_index, arg_id = v->arg_index - 1;
        const char *reg_prefix = var_type_prefix[v->type],
                   *type = (VarType) v->type == VarType::Bool
                               ? "i8" : var_type_name_llvm[v->type];
        const char *label = log_trace ? jit_var_label(index) : nullptr;
        uint32_t size = v->size;

        if (v->arg_type == ArgType::Input) {
            if (v->direct_pointer)
                continue;

            if (unlikely(log_trace))
                buffer.fmt("\n    ; Load %s%u%s%s\n",
                           reg_prefix, reg_id, label ? ": " : "",
                           label ? label : "");

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
                buffer.fmt("\n    ; Evaluate %s%u%s%s\n", reg_prefix, reg_id,
                           label ? ": " : "", label ? label : "");
            jit_render_stmt_llvm(index, v);
        }

        if (v->arg_type == ArgType::Output) {
            if (unlikely(log_trace))
                buffer.fmt("\n    ; Store %s%u%s%s\n", reg_prefix, reg_id,
                           label ? ": " : "", label ? label : "");
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
        buffer.putc('\n');
    }

    buffer.put("!0 = !{!0}\n");
    buffer.put("!1 = !{!1, !0}\n");
    buffer.put("!2 = !{!\"llvm.loop.unroll.disable\", !\"llvm.loop.vectorize.enable\", i1 0}\n\n");
    buffer.fmt("attributes #0 = { norecurse nounwind \"target-cpu\"=\"%s\" "
               "\"stack-probe-size\"=\"%u\"", jit_llvm_target_cpu, 1024*1024*1024);
    if (jit_llvm_target_features)
        buffer.fmt(" \"target-features\"=\"%s\"", jit_llvm_target_features);
    buffer.put(" }");
}

void jit_assemble(Stream *stream, ScheduledGroup group) {
    bool cuda = stream->cuda;

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

#if defined(ENOKI_JIT_ENABLE_TBB)
    jit_llvm_scatter_add_variables.clear();
#endif

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
                       var_type_prefix[v->type],
                       n_regs_total, index);

            const char *label = jit_var_label(index);
            if (label)
                buffer.fmt(" \"%s\"", label);
            if (v->size == 1)
                buffer.put(" [scalar]");
            if (v->direct_pointer)
                buffer.put(" [direct_pointer]");
            else if (v->data != nullptr)
                buffer.put(" [in]");
            else if (v->scatter)
                buffer.put(" [scat]");
            else if (v->output_flag && v->size == group.size)
                buffer.put(" [out]");

            jit_trace("%s", buffer.get());
        }

        if (v->data || v->direct_pointer) {
            v->arg_index = (uint16_t) (n_args_in + n_args_out);
            v->arg_type = ArgType::Input;
            n_args_in++;
            push = true;
        } else if (v->output_flag && !v->scatter && v->size == group.size) {
            size_t isize    = (size_t) var_type_size[v->type],
                   var_size = (size_t) group.size * isize;

            // Padding to support out-of-bounds accesses in LLVM gather operations
            if (cuda && isize < 4)
                isize += 4 - isize;

            AllocType alloc_type = AllocType::Device;
            if (!cuda) {
#if defined(ENOKI_JIT_ENABLE_TBB)
                alloc_type = AllocType::HostAsync;
#else
                alloc_type = AllocType::Host;
#endif
            }

            void *data = jit_malloc(alloc_type, var_size);

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

#if defined(ENOKI_JIT_ENABLE_TBB)
        /// LLVM: parallel scatter_add into the same array requires extra precautions
        if (unlikely(!cuda && v->scatter &&
                         (strstr(v->stmt, "ek.scatter_add") ||
                          strstr(v->stmt, "ek.masked_scatter_add")))) {
            Variable *base_ptr = jit_var(v->dep[0]);
            if (unlikely(!base_ptr->direct_pointer))
                jit_fail("jit_run(): invalid error while handling ek.scatter_add (1).");
            Variable *base = jit_var(base_ptr->dep[0]);
            if (unlikely(base->data != base_ptr->data))
                jit_fail("jit_run(): invalid error while handling ek.scatter_add (2).");

            std::pair<uint32_t, uint32_t> item(
                base_ptr->arg_index - 1, base_ptr->dep[0]);

            if (std::find(jit_llvm_scatter_add_variables.begin(),
                          jit_llvm_scatter_add_variables.end(),
                          item) == jit_llvm_scatter_add_variables.end())
                jit_llvm_scatter_add_variables.push_back(item);
        }
#endif

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

    jit_llvm_supplement = false;
    intrinsics_buffer.clear();
    buffer.clear();

    if (cuda) {
        jit_assemble_cuda(group, n_regs_total);
    } else {
        jit_assemble_llvm(group);

        /* Compile a separate scalar variant of the kernel if it uses
           scatter/gather intrinsics. This is needed to deal with the last
           packet, where not all lanes are guaranteed to be valid. */
        if (jit_llvm_vector_width != 1 &&
            (intrinsics_buffer.contains("@llvm.masked.load") ||
             intrinsics_buffer.contains("@llvm.masked.store") ||
             intrinsics_buffer.contains("@llvm.masked.scatter") ||
             intrinsics_buffer.contains("@llvm.masked.gather") ||
             jit_llvm_supplement)) {
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

    float codegen_time = timer();
    jit_log(Info,
            "jit_run(): launching kernel (n=%u, in=%u, out=%u, ops=%u, "
            "took %s) ..",
            group.size, n_args_in - 1, n_args_out, n_regs_total,
            jit_time_string(codegen_time));

    jit_log(Debug, "%s", buffer.get());
}

void jit_run(Stream *stream, ScheduledGroup group) {
    KernelKey kernel_key((char *) buffer.get(), stream->device);
    size_t hash_code = KernelHash::compute_hash(kernel_hash, stream->device);
    auto it = state.kernel_cache.find(kernel_key, hash_code);
    Kernel kernel;

    if (it == state.kernel_cache.end()) {
        bool cache_hit = jit_kernel_load(
            buffer.get(), (uint32_t) buffer.size(), stream->cuda, kernel_hash, kernel);

        if (!cache_hit) {
            if (stream->cuda)
                jit_cuda_compile(buffer.get(), buffer.size(), kernel);
            else
                jit_llvm_compile(buffer.get(), buffer.size(), kernel,
                                 jit_llvm_supplement);

            jit_kernel_write(buffer.get(), (uint32_t) buffer.size(), stream->cuda,
                             kernel_hash, kernel);
        }

        if (!stream->cuda) {
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
        jit_log(Info, "jit_run(): cache %s, %s: %s, %s.",
                cache_hit ? "hit" : "miss",
                cache_hit ? "load" : "build",
                std::string(jit_time_string(link_time)).c_str(),
                std::string(jit_mem_string(kernel.size)).c_str());

        kernel_key.str = (char *) malloc_check(buffer.size() + 1);
        memcpy(kernel_key.str, buffer.get(), buffer.size() + 1);
        state.kernel_cache.emplace(kernel_key, kernel);
        if (cache_hit)
            state.kernel_soft_misses++;
        else
            state.kernel_hard_misses++;
    } else {
        kernel = it.value();
        state.kernel_hits++;
    }
    state.kernel_launches++;

    if (stream->cuda) {
        size_t kernel_args_size = (size_t) kernel_args.size() * sizeof(uint64_t);

        void *config[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER,
            kernel_args.data(),
            CU_LAUNCH_PARAM_BUFFER_SIZE,
            &kernel_args_size,
            CU_LAUNCH_PARAM_END
        };

        uint32_t block_count, thread_count;
        const Device &device = state.devices[stream->device];
        device.get_launch_config(&block_count, &thread_count, group.size,
                                 (uint32_t) kernel.cuda.block_size);

        cuda_check(cuLaunchKernel(kernel.cuda.cu_func, block_count, 1, 1, thread_count,
                                  1, 1, 0, active_stream->handle, nullptr, config));
    } else {
        uint32_t width = kernel.llvm.func != kernel.llvm.func_scalar
                             ? jit_llvm_vector_width : 1u,
                 rounded = group.size / width * width;

        uint32_t packets = (rounded + jit_llvm_vector_width - 1) / jit_llvm_vector_width;
        jit_log(Trace, "jit_run(): processing %u packet%s and %u scalar entries",
                packets, packets == 1 ? "": "s", group.size - rounded);

#if defined(ENOKI_JIT_ENABLE_TBB)
#  if defined(ENOKI_ITTNOTIFY)
        const void *itt = kernel.llvm.itt;
#  else
        const void *itt = nullptr;
#  endif

        if (likely(rounded > 0))
            tbb_stream_enqueue_kernel(
                stream, kernel.llvm.func, 0, rounded,
                (uint32_t) kernel_args_extra.size(), kernel_args_extra.data(),
                stream->parallel_dispatch, itt);

        if (unlikely(rounded != group.size))
            tbb_stream_enqueue_kernel(
                stream, kernel.llvm.func_scalar, rounded, group.size,
                (uint32_t) kernel_args_extra.size(), kernel_args_extra.data(),
                stream->parallel_dispatch, itt);
#else
        unlock_guard guard(state.mutex);
        if (likely(rounded > 0))
            kernel.llvm.func(0, rounded, kernel_args_extra.data());

        if (unlikely(rounded != group.size))
            kernel.llvm.func_scalar(rounded, group.size, kernel_args_extra.data());
#endif
    }
}

static ProfilerRegion profiler_region_eval("jit_eval");

/// Evaluate all computation that is queued on the current device & stream
void jit_eval() {
    ProfilerPhase profiler(profiler_region_eval);

    /* The function 'jit_eval()' cannot be executed concurrently. Temporarily
       release 'state.mutex' before acquiring 'state.eval_mutex'. */
    state.mutex.unlock();
    lock_guard_t<std::mutex> guard(state.eval_mutex);
    state.mutex.lock();

    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise(
            "jit_eval(): you must invoke jitc_set_device() to choose a target "
            "device before evaluating expressions using the JIT compiler.");

    visited.clear();
    schedule.clear();

    // Collect variables that must be computed and their subtrees
    for (uint32_t index : stream->todo) {
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;

        Variable *v = &it.value();
        if (v->ref_count_ext == 0 || v->data != nullptr)
            continue;

        if (unlikely(v->cuda != stream->cuda))
            jit_raise("jit_eval(): attempted to evaluate variable %u "
                      "associated with the %s backend, while the %s backend "
                      "was selected via jitc_set_device()!",
                      index, v->cuda ? "CUDA" : "LLVM",
                      stream->cuda ? "CUDA" : "LLVM");

        jit_var_traverse(v->size, index, v);
        v->output_flag = true;
    }
    stream->todo.clear();

    if (schedule.empty())
        return;

    scoped_set_context_maybe guard2(stream->context);

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
    bool cuda_parallel_dispatch =
        stream->parallel_dispatch && stream->cuda && schedule_groups.size() > 1;

    if (!cuda_parallel_dispatch) {
        jit_log(Debug, "jit_eval(): begin.");
    } else {
        jit_log(Debug, "jit_eval(): begin (parallel dispatch to %zu streams).",
                schedule_groups.size());
        cuda_check(cuEventRecord(stream->event, stream->handle));
    }

    uint32_t group_idx = 1;
    for (ScheduledGroup &group : schedule_groups) {
        jit_assemble(stream, group);

        Stream *sub_stream = stream;
        if (cuda_parallel_dispatch) {
            uint32_t stream_index = 1000 * (stream->stream + 1) + group_idx++;
            jit_set_device(stream->device, stream_index);
            sub_stream = active_stream;
            cuda_check(cuStreamWaitEvent(sub_stream->handle, stream->event, 0));
            active_stream = stream;
        }

        jit_run(stream, group);

        if (cuda_parallel_dispatch) {
            cuda_check(cuEventRecord(sub_stream->event, sub_stream->handle));
            cuda_check(cuStreamWaitEvent(stream->handle, sub_stream->event, 0));
        }
    }

#if defined(ENOKI_JIT_ENABLE_TBB)
    if (!stream->cuda)
        tbb_stream_submit_kernel(stream);
#endif

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
        if (!(v->stmt && !v->direct_pointer && (v->data || v->scatter)))
            continue;

        jit_cse_drop(index, v);

        v->is_literal_one = v->is_literal_zero = false;
        if (unlikely(v->free_stmt))
            free(v->stmt);

        bool scatter = v->scatter;
        uint32_t dep[4];
        memcpy(dep, v->dep, sizeof(uint32_t) * 4);
        memset(v->dep, 0, sizeof(uint32_t) * 4);
        v->stmt = nullptr;

        if (unlikely(scatter)) {
            Variable *ptr = jit_var(dep[0]);
            if (unlikely(!ptr->direct_pointer))
                jit_fail("jit_eval(): invalid scatter target!");
            Variable *target = jit_var(ptr->dep[0]);
            target->pending_scatter = false;

            Variable *v2 = jit_var(index);
            if (unlikely(v2->ref_count_ext != 1 || v2->ref_count_int != 0))
                jit_fail("jit_eval(): invalid invalid reference for scatter operation");
            jit_var_dec_ref_ext(index, v2);
        }

        for (int j = 0; j < 4; ++j)
            jit_var_dec_ref_int(dep[j]);
    }

    jit_free_flush();
    jit_log(Debug, "jit_eval(): done.");
}
