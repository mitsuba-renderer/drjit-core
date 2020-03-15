#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"

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

/// Ordered list of variables that should be computed
static std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
static std::vector<ScheduledGroup> schedule_groups;

/// Auxiliary data structure needed to compute 'schedule_sizes' and 'schedule'
static tsl::robin_set<std::pair<uint32_t, uint32_t>, pair_hash> visited;

/// Input/output arguments of the kernel being evaluated
static std::vector<void *> kernel_args, kernel_args_extra;

/// Name of the last generated kernel
static char kernel_name[9];

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

void jit_render_stmt(uint32_t index, Variable *v) {
    const char *s = v->stmt;
    char c;

    buffer.put("    ");
    while ((c = *s++) != '\0') {
        if (c != '$') {
            buffer.putc(c);
        } else {
            const char **prefix_table = nullptr, type = *s++;
            switch (type) {
                case 't': prefix_table = var_type_name_ptx;     break;
                case 'b': prefix_table = var_type_name_ptx_bin; break;
                case 'r': prefix_table = var_type_register_ptx; break;
                default:
                    jit_fail("jit_render_stmt(): encountered invalid \"$\" "
                             "expression (unknown type)!");
            }

            uint32_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 3))
                jit_fail("jit_render_stmt(%s): encountered invalid \"$\" "
                         "expression (argument out of bounds)!", v->stmt);

            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            if (unlikely(dep_id == 0))
                jit_fail("jit_render_stmt(%s): encountered invalid \"$\" "
                         "expression (dependency %u is missing)!", v->stmt, arg_id);

            Variable *dep = jit_var(dep_id);
            buffer.put(prefix_table[(int) dep->type]);

            if (type == 'r')
                buffer.fmt("%u", dep->reg_index);
        }
    }

    buffer.put(";\n");
}

void jit_assemble(ScheduledGroup group) {
    uint32_t n_args_in    = 0,
             n_args_out   = 0,
             n_regs_total = 4; // The first 4 variables are reserved

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
                       var_type_register_ptx[(int) v->type],
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
            v->data = jit_malloc(AllocType::Device, var_size);
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
            if (kernel_args.size() < CUDA_MAX_KERNEL_PARAMETERS - 1)
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

    if (unlikely(!kernel_args_extra.empty())) {
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

    auto get_parameter_addr = [](const Variable *v, uint32_t target = 0) {
        if (v->arg_index < CUDA_MAX_KERNEL_PARAMETERS - 1)
            buffer.fmt("    ld.param.u64 %%rd%u, [arg%u];\n", target, v->arg_index - 1);
        else
            buffer.fmt("    ldu.global.u64 %%rd%u, [%%rd2 + %u];\n",
                       target, (v->arg_index - (CUDA_MAX_KERNEL_PARAMETERS - 1)) * 8);

        if (v->size > 1)
            buffer.fmt("    mul.wide.u32 %%rd1, %%r2, %u;\n"
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

    buffer.put(".visible .entry enoki_@@@@@@@@(.param .u32 size,\n");
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

    /// Replace '@'s in 'enoki_@@@@@@@@' by MD5 hash
    snprintf(kernel_name, 9, "%08x", crc32(0, buffer.get(), buffer.size()));
    memcpy((void *) strchr(buffer.get(), '@'), kernel_name, 8);

    bool log_trace = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace;


    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        uint32_t index = schedule[group_index].index;
        Variable *v = jit_var(index);

        if (v->arg_type == ArgType::Input) {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Load %s%u%s%s\n",
                           var_type_register_ptx[(int) v->type],
                           v->reg_index, v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");

            get_parameter_addr(v, v->direct_pointer ? v->reg_index : 0u);

            if (likely(!v->direct_pointer)) {
                if (likely(v->type != (uint32_t) VarType::Bool)) {
                    buffer.fmt("    %s.global.%s %s%u, [%%rd0];\n",
                           v->size == 1 ? "ldu" : "ld",
                           var_type_name_ptx[(int) v->type],
                           var_type_register_ptx[(int) v->type],
                           v->reg_index);
                } else {
                    buffer.fmt("    %s.global.u8 %%w0, [%%rd0];\n",
                           v->size == 1 ? "ldu" : "ld",
                           var_type_name_ptx[(int) v->type],
                           var_type_register_ptx[(int) v->type],
                           v->reg_index);
                    buffer.fmt("    setp.ne.u16 %s%u, %%w0, 0;",
                           var_type_register_ptx[(int) v->type],
                           v->reg_index);
                }
            }
        } else {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Evaluate %s%u%s%s\n",
                           var_type_register_ptx[(int) v->type],
                           v->reg_index,
                           v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");

            jit_render_stmt(index, v);
        }

        if (v->arg_type == ArgType::Output) {
            if (unlikely(log_trace))
                buffer.fmt("\n    // Store %s%u%s%s\n",
                           var_type_register_ptx[(int) v->type],
                           v->reg_index, v->has_label ? ": " : "",
                           v->has_label ? jit_var_label(index) : "");

            get_parameter_addr(v);

            if (likely(v->type != (uint32_t) VarType::Bool)) {
                buffer.fmt("    st.global.%s [%%rd0], %s%u;\n",
                       var_type_name_ptx[(int) v->type],
                       var_type_register_ptx[(int) v->type],
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

    jit_log(Debug, "%s", buffer.get());
}

void jit_run(ScheduledGroup group) {
    float codegen_time = timer();

    auto it = state.kernel_cache.find(buffer.get());
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
        bool cache_hit = strstr(info_log.get(), "ptxas info") != nullptr;
        jit_log(Debug, "Detailed linker output:\n%s", info_log.get());

        CUresult ret = cuModuleLoadData(&kernel.cu_module, link_output);
        if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
            jit_malloc_trim();
            ret = cuModuleLoadData(&kernel.cu_module, link_output);
        }
        cuda_check(ret);

        // Locate the kernel entry point
        std::string name = std::string("enoki_") + kernel_name;
        cuda_check(cuModuleGetFunction(&kernel.cu_func, kernel.cu_module,
                                       name.c_str()));

        /// Enoki doesn't use shared memory at all..
        cuda_check(cuFuncSetAttribute(
            kernel.cu_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
        cuda_check(cuFuncSetAttribute(
            kernel.cu_func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            CU_SHAREDMEM_CARVEOUT_MAX_L1));

        int reg_count;
        cuda_check(cuFuncGetAttribute(
            &reg_count, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel.cu_func));

        // Destroy the linker invocation
        cuda_check(cuLinkDestroy(link_state));

        char *str = (char *) malloc(buffer.size() + 1);
        memcpy(str, buffer.get(), buffer.size() + 1);

        cuda_check(cuOccupancyMaxPotentialBlockSize(
            &kernel.block_count, &kernel.thread_count,
            kernel.cu_func, nullptr, 0, 0));

        state.kernel_cache.emplace(str, kernel);

        jit_log(Debug,
                "jit_run(): cache %s, codegen: %s, %s: %s, %i registers, %i "
                "threads, %i blocks.",
                cache_hit ? "miss" : "hit",
                std::string(jit_time_string(codegen_time)).c_str(),
                cache_hit ? "link" : "load",
                std::string(jit_time_string(link_time)).c_str(), reg_count,
                kernel.thread_count, kernel.block_count);
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

    uint32_t thread_count = kernel.thread_count,
             block_count  = kernel.block_count;

    /// Reduce the number of blocks when processing a very small amount of data
    if (group.size <= thread_count) {
        block_count = 1;
        thread_count = group.size;
    } else if (group.size <= thread_count * block_count) {
        block_count = (group.size + thread_count - 1) / thread_count;
    }

    cuda_check(cuLaunchKernel(kernel.cu_func, block_count, 1, 1, thread_count,
                              1, 1, 0, active_stream->handle, nullptr, config));
}

/// Evaluate all computation that is queued on the current device & stream
void jit_eval() {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_eval(): device and stream must be set! (call "
                  "jit_device_set() beforehand)!");

    if (stream->todo.empty())
        return;

    visited.clear();
    schedule.clear();
    schedule_groups.clear();

    // Collect variables that must be computed and their subtrees
    for (uint32_t index : stream->todo)
        jit_var_traverse(jit_var_size(index), index);
    stream->todo.clear();

    // Group them from large to small sizes while preserving dependencies
    std::stable_sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            return a.size > b.size;
        });

    // Are there independent groups of work that could be dispatched in parallel?
    bool parallel_dispatch =
        state.parallel_dispatch &&
        schedule[0].size != schedule[schedule.size() - 1].size;

    if (!parallel_dispatch) {
        jit_log(Debug, "jit_eval(): begin.");
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
        jit_log(Debug, "jit_eval(): begin (parallel dispatch to %zu streams).",
                schedule.size());
        cuda_check(cuEventRecord(stream->event, stream->handle));
    }

    uint32_t stream_index = 1000 * stream->stream;
    for (ScheduledGroup &group : schedule_groups) {
        jit_assemble(group);

        Stream *sub_stream = stream;
        if (parallel_dispatch) {
            jit_device_set(stream->device, stream_index);
            sub_stream = active_stream;
            cuda_check(cuStreamWaitEvent(sub_stream->handle, stream->event, 0));
        }

        jit_run(group);

        if (parallel_dispatch) {
            cuda_check(cuEventRecord(sub_stream->event, sub_stream->handle));
            cuda_check(cuStreamWaitEvent(stream->handle, sub_stream->event, 0));
        }

        stream_index++;
    }

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
        bool side_effect = v->side_effect;
        uint32_t dep[3], extra_dep = v->extra_dep;
        memcpy(dep, v->dep, sizeof(uint32_t) * 3);

        v->side_effect = false;
        v->dirty = false;

        /* Don't bother with CSE for evaluated scalar variables to replace
           costly loads with faster arithmetic. */
        if (v->size == 1)
            jit_cse_drop(index, v);

        memset(v->dep, 0, sizeof(uint32_t) * 3);
        v->extra_dep = 0;
        for (int j = 0; j < 3; ++j)
            jit_var_int_ref_dec(dep[j]);
        jit_var_ext_ref_dec(extra_dep);

        if (side_effect)
            jit_var_ext_ref_dec(index);
    }

    jit_free_flush();
    jit_log(Debug, "jit_eval(): done.");
}

