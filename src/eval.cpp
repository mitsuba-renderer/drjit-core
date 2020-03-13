#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"

#define CUDA_MAX_KERNEL_PARAMETERS 4

// ====================================================================
//  The following data structures are temporarily used during program
//  generation. They are declared as global variables to enable memory
//  reuse across jit_eval() calls.
// ====================================================================

/// Ordered unique list containing sizes of variables to be computed
static std::vector<uint32_t> schedule_sizes;

/// Map variable size => ordered list of variables that should be computed
static tsl::robin_map<uint32_t, std::vector<uint32_t>> schedule;

/// Auxiliary data structure needed to compute 'schedule_sizes' and 'schedule'
static tsl::robin_set<std::pair<uint32_t, uint32_t>, pair_hash> visited;

/// Name of the last generated kernel
static char kernel_name[9];

/// Input/output arguments of the kernel being evaluated
static std::vector<void *> kernel_args, kernel_args_extra;

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

    schedule[size].push_back(index);
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
                             "expression (unknown type).");
            }

            uint8_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 3))
                jit_fail("jit_render_stmt(): encountered invalid \"$\" "
                         "expression (argument out of bounds!).");

            Variable *dep = jit_var(arg_id == 0 ? index : v->dep[arg_id - 1]);
            buffer.put(prefix_table[(int) dep->type]);

            if (type == 'r')
                buffer.fmt("%u", dep->reg_index);
        }
    }

    buffer.put(";\n");
}

void jit_assemble(uint32_t size) {
    const std::vector<uint32_t> &sched = schedule[size];
    uint32_t n_vars_in = 0, n_vars_out = 0, n_vars_total = 0;

    (void) timer();
    jit_log(Trace, "jit_assemble(size=%u): register map:", size);

    /// Push the size argument
    void *tmp = 0;
    memcpy(&tmp, &size, sizeof(uint32_t));
    kernel_args.clear();
    kernel_args_extra.clear();
    kernel_args.push_back(tmp);
    n_vars_in++;

    for (uint32_t index : sched) {
        Variable *v = jit_var(index);
        bool push = true;

        if (unlikely(v->ref_count_int == 0 && v->ref_count_ext == 0))
            jit_fail("jit_assemble(): schedule contains unreferenced variable %u!", index);
        else if (unlikely(v->size != 1 && v->size != size))
            jit_fail("jit_assemble(): schedule contains variable %u with incompatible size "
                     "(%u and %u)!", index, size, v->size);
        else if (unlikely(v->data == nullptr && !v->direct_pointer && v->stmt == nullptr))
            jit_fail("jit_assemble(): schedule contains variable %u with empty statement!", index);

        if (state.log_level >= LogLevel::Trace) {
            buffer.clear();
            buffer.fmt("   - %s%u -> %u",
                       var_type_register_ptx[(int) v->type],
                       n_vars_total, index);

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
            else if (v->ref_count_ext > 0 && v->size == size)
                buffer.put(" [out]");

            jit_log(Trace, "%s", buffer.get());
        }

        if (v->data || v->direct_pointer) {
            n_vars_in++;
            push = true;
            v->arg_index = (uint16_t) n_vars_total;
            v->arg_type = ArgType::Input;
        } else if (!v->side_effect && v->ref_count_ext > 0 && v->size == size) {
            size_t var_size = (size_t) size * (size_t) var_type_size[(int) v->type];
            v->data = jit_malloc(AllocType::Device, var_size);
            n_vars_out++;
            push = true;
            v->arg_index = (uint16_t) n_vars_total;
            v->arg_type = ArgType::Output;
            v->tsize = 1;
        } else {
            v->arg_index = (uint16_t) 0xffff;
            v->arg_type = ArgType::Register;
        }

        if (push) {
            if (kernel_args.size() < CUDA_MAX_KERNEL_PARAMETERS - 1)
                kernel_args.push_back(v->data);
            else
                kernel_args_extra.push_back(v->data);
        }

        v->reg_index  = n_vars_total++;
    }

    if (unlikely(n_vars_total > 0xFFFFFFu))
        jit_fail("jit_run(): The queued computation involves more than 16 "
                 "million variables, which overflowed an internal counter. "
                 "Even if Enoki could compile such a large program, it would "
                 "not run efficiently. Please periodically run jitc_eval() to "
                 "break down the computation into smaller chunks.");
    else if (unlikely(n_vars_in + n_vars_out > 0xFFFFu))
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

    jit_log(Debug, "jit_run(): launching kernel (n=%u, in=%u, out=%u, ops=%u) ..",
            size, n_vars_in, n_vars_out, n_vars_total);

    buffer.clear();
    buffer.put(".version 6.3\n");
    buffer.put(".target sm_61\n");
    buffer.put(".address_size 64\n");

    // auto param_ref = [&](int index) {
    //     if (index < CUDA_MAX_KERNEL_PARAMETERS - 1)
    //         buffer.fmt("[arg%u]", index);
    //     else
    //         buffer.fmt("[%%arg_extra + %u]", index * 8);
    // };

    buffer.put(".visible .entry enoki_@@@@@@@@(.param .u32 size,\n");
    for (uint32_t index = 1; index < kernel_args.size(); ++index)
        buffer.fmt("                               .param .u64 arg%u%s\n",
                   index - 1, (index + 1 < kernel_args.size()) ? "," : ") {");

    uint32_t n_vars_decl = std::max(3u, n_vars_total);
    buffer.fmt("    .reg.b8 %%b<%u>;\n", n_vars_decl);
    buffer.fmt("    .reg.b16 %%w<%u>;\n", n_vars_decl);
    buffer.fmt("    .reg.b32 %%r<%u>, %%size, %%index, %%step;\n", n_vars_decl);
    buffer.fmt("    .reg.b64 %%rd<%u>%s;\n", n_vars_decl,
               kernel_args_extra.empty() ? "" : ", %%arg_extra");
    buffer.fmt("    .reg.f32 %%f<%u>;\n", n_vars_decl);
    buffer.fmt("    .reg.f64 %%d<%u>;\n", n_vars_decl);
    buffer.fmt("    .reg.pred %%p<%u>, %%done;\n\n", n_vars_decl);
    buffer.put("    // Grid-stride loop setup\n");

    buffer.put("    ld.param.u32 %size, [size];\n");

    if (!kernel_args_extra.empty())
        buffer.fmt("    ld.param.u64 %%arg_extra, [arg_%u];\n",
                   CUDA_MAX_KERNEL_PARAMETERS - 1);

    buffer.put("    mov.u32 %r0, %ctaid.x;\n");
    buffer.put("    mov.u32 %r1, %ntid.x;\n");
    buffer.put("    mov.u32 %r2, %tid.x;\n");
    buffer.put("    mad.lo.u32 %index, %r0, %r1, %r2;\n");
    buffer.put("    setp.ge.u32 %done, %index, %size;\n");
    buffer.put("    @%done bra L0;\n");
    buffer.put("\n");
    buffer.put("    mov.u32 %r0, %nctaid.x;\n");
    buffer.put("    mul.lo.u32 %step, %r1, %r0;\n");
    buffer.put("\n");
    buffer.put("L1:\n");
    buffer.put("    // Loop body\n");;
    buffer.put("\n");

    /// Replace '@'s in 'enoki_@@@@@@@@' by MD5 hash
    snprintf(kernel_name, 9, "%08x", crc32(buffer.get(), buffer.size()));
    memcpy((void *) strchr(buffer.get(), '@'), kernel_name, 8);

    jit_log(Debug, "%s", buffer.get());

    for (uint32_t index : sched) {
        Variable *v = jit_var(index);

        if (v->arg_type == ArgType::Input) {
        }

        jit_render_stmt(index, v);

        if (v->arg_type == ArgType::Output) {
        }
    }

    buffer.put("    add.u32 %index, %index, %step;\n");
    buffer.put("    setp.ge.u32 %done, %index, %size;\n");
    buffer.put("    @!%done bra L1;\n");
    buffer.put("\n");
    buffer.put("L0:\n");
    buffer.put("    ret;\n");
    buffer.put("}");
}

void jit_run(uint32_t size) {
    float codegen_time = timer();

    auto it = state.kernels.find(buffer.get());
    Kernel kernel;

    if (it == state.kernels.end()) {
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

        state.kernels[str] = kernel;

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
    if (size <= thread_count) {
        block_count = 1;
        thread_count = size;
    } else if (size <= thread_count * block_count) {
        block_count = (size + thread_count - 1) / thread_count;
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
    schedule_sizes.clear();

    for (uint32_t index : stream->todo) {
        size_t size = jit_var_size(index);
        jit_var_traverse(size, index);
        schedule_sizes.push_back(size);
    }

    stream->todo.clear();

    std::sort(schedule_sizes.begin(), schedule_sizes.end(),
              std::greater<uint32_t>());

    bool parallel_dispatch = state.parallel_dispatch && schedule.size() > 1;

    if (!parallel_dispatch) {
        jit_log(Debug, "jit_eval(): begin.");
    } else {
        jit_log(Debug, "jit_eval(): begin (parallel dispatch to %zu streams).",
                schedule.size());
        cuda_check(cuEventRecord(stream->event, stream->handle));
    }

    uint32_t stream_index = 1000 * stream->stream;
    for (uint32_t size : schedule_sizes) {
        jit_assemble(size);

        Stream *sub_stream = stream;
        if (parallel_dispatch) {
            jit_device_set(stream->device, stream_index);
            sub_stream = active_stream;
            cuda_check(cuStreamWaitEvent(sub_stream->handle, stream->event, 0));
        }

        jit_run(size);

        if (parallel_dispatch) {
            cuda_check(cuEventRecord(sub_stream->event, sub_stream->handle));
            cuda_check(cuStreamWaitEvent(stream->handle, sub_stream->event, 0));
        }

        stream_index++;
    }

    jit_device_set(stream->device, stream->stream);

    /**
     * At this point, all variables and their dependencies are computed, which
     * means that we can remove internal edges between them. This in turn will
     * cause many of the variables to be garbage-collected.
     */
    jit_log(Debug, "jit_eval(): cleaning up..");

    for (uint32_t size : schedule_sizes) {
        const std::vector<uint32_t> &sched = schedule[size];

        for (uint32_t index : sched) {
            auto it = state.variables.find(index);
            if (it == state.variables.end())
                continue;

            Variable *v = &it.value();
            bool side_effect = v->side_effect;
            v->side_effect = false;
            v->dirty = false;

            /* Don't bother with CSE for evaluated scalar variables to replace
               costly loads with faster arithmetic. */
            if (size == 1)
                jit_cse_drop(index, v);

            // if (v->data != nullptr && v->stmt != nullptr) {
                uint32_t dep[3], extra_dep = v->extra_dep;
                memcpy(dep, v->dep, sizeof(uint32_t) * 3);
                memset(v->dep, 0, sizeof(uint32_t) * 3);
                v->extra_dep = 0;
                for (int j = 0; j < 3; ++j)
                    jit_var_int_ref_dec(dep[j]);
                jit_var_ext_ref_dec(extra_dep);
            // }

            if (side_effect)
                jit_var_ext_ref_dec(index);
        }
    }

    jit_free_flush();
    jit_log(Debug, "jit_eval(): done.");
}

/// Call jit_eval() only if the variable 'index' requires evaluation
void jit_eval_var(uint32_t index) {
    Variable *v = jit_var(index);
    if (v->data == nullptr || v->dirty)
        jit_eval();
}
