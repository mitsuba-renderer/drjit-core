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

/// Maps between Enoki variable indices and program variables
static tsl::robin_map<uint32_t, uint32_t> reg_map;

/// Name of the last generated kernel
static char kernel_name[9];

/// Input/output arguments of the kernel being evaluated
static std::vector<void *> kernel_args, kernel_args_extra;

// ====================================================================

static const char *cuda_register_type(VarType type) {
    switch (type) {
        case VarType::UInt8:    return "u8";
        case VarType::Int8:     return "s8";
        case VarType::UInt16:   return "u16";
        case VarType::Int16:    return "s16";
        case VarType::UInt32:   return "u32";
        case VarType::Int32:    return "s32";
        case VarType::Pointer:
        case VarType::UInt64:   return "u64";
        case VarType::Int64:    return "s64";
        case VarType::Float16:  return "f16";
        case VarType::Float32:  return "f32";
        case VarType::Float64:  return "f64";
        case VarType::Bool:     return "pred";
        default: jit_fail("cuda_register_type(): invalid type!");
    }
}

static const char *cuda_register_type_bin(VarType type) {
    switch (type) {
        case VarType::UInt8:
        case VarType::Int8:    return "b8";
        case VarType::UInt16:
        case VarType::Float16:
        case VarType::Int16:   return "b16";
        case VarType::Float32:
        case VarType::UInt32:
        case VarType::Int32:   return "b32";
        case VarType::Pointer:
        case VarType::Float64:
        case VarType::UInt64:
        case VarType::Int64:   return "b64";
        case VarType::Bool:    return "pred";
        default: jit_fail("cuda_register_type_bin(): invalid type!");
    }
}

static const char *cuda_register_name(VarType type) {
    switch (type) {
        case VarType::UInt8:
        case VarType::Int8:    return "%b";
        case VarType::UInt16:
        case VarType::Int16:   return "%w";
        case VarType::UInt32:
        case VarType::Int32:   return "%r";
        case VarType::Pointer:
        case VarType::UInt64:
        case VarType::Int64:   return "%rd";
        case VarType::Float32: return "%f";
        case VarType::Float64: return "%d";
        case VarType::Bool:    return "%p";
        default: jit_fail("cuda_register_name(): invalid type!");
    }
}

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

    for (auto [id, tsize] : ch)
        jit_var_traverse(size, id);

    schedule[size].push_back(index);
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

        if (state.log_level >= 4) {
            buffer.clear();
            buffer.fmt("   - %s%u -> %u",
                       cuda_register_name(v->type),
                       n_vars_total, index);

            if (v->label)
                buffer.fmt(" \"%s\"", v->label);
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
        } else if (!v->side_effect && v->ref_count_ext > 0 && v->size == size) {
            size_t var_size = (size_t) size * jit_type_size(v->type);
            v->data = jit_malloc(AllocType::Device, var_size);
            n_vars_out++;
            push = true;
        }

        if (push) {
            if (kernel_args.size() < CUDA_MAX_KERNEL_PARAMETERS - 1)
                kernel_args.push_back(v->data);
            else
                kernel_args_extra.push_back(v->data);
        }

        reg_map[index] = n_vars_total++;
    }

    if (!kernel_args_extra.empty()) {
        size_t args_extra_size = kernel_args_extra.size() * sizeof(uint64_t);
        void *args_extra_host = jit_malloc(AllocType::HostPinned, args_extra_size);
        void *args_extra_dev  = jit_malloc(AllocType::Device, args_extra_size);

        memcpy(args_extra_host, kernel_args_extra.data(), args_extra_size);
        cuda_check(cudaMemcpyAsync(args_extra_dev, args_extra_host, args_extra_size,
                                   cudaMemcpyHostToDevice, active_stream->handle));

        kernel_args.push_back(args_extra_dev);
        jit_free(args_extra_host);
        // Safe, because there won't be further allocations until after this kernel has executed.
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
    buffer.fmt("    .reg.b64 %%rd<%u>, %%arg_extra;\n", n_vars_decl);
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
    buffer.put("    add.u32 %index, %index, %step;\n");
    buffer.put("    setp.ge.u32 %done, %index, %size;\n");
    buffer.put("    @!%done bra L1;\n");
    buffer.put("\n");
    buffer.put("L0:\n");
    buffer.put("    ret;\n");
    buffer.put("}");

    /// Replace '@'s in 'enoki_@@@@@@@@' by MD5 hash
    snprintf(kernel_name, 9, "%08x", crc32(buffer.get(), buffer.size()));
    memcpy((void *) strchr(buffer.get(), '@'), kernel_name, 8);

    jit_log(Debug, "%s", buffer.get());

#if 0
    for (uint32_t index : sweep) {
        Variable &var = ctx[index];

        if (var.is_collected() || (var.cmd.empty() && var.data == nullptr && !var.direct_pointer))
            throw std::runtime_error(
                "CUDABackend: found invalid/expired variable " + std::to_string(index) + " in schedule! ");

        if (var.size != 1 && var.size != size)
            throw std::runtime_error(
                "CUDABackend: encountered arrays of incompatible size! (" +
                std::to_string(size) + " vs " + std::to_string(var.size) + ")");

        oss << std::endl;
        if (var.data || var.direct_pointer) {
            size_t index = ptrs.size();
            ptrs.push_back(var.data);

            oss << std::endl
                << "    // Load register " << cuda_register_name(var.type) << reg_map[index];
            if (!var.label.empty())
                oss << ": " << var.label;
            oss << std::endl;

            if (!var.direct_pointer) {
                oss << "    " << (parameter_direct ? "ld.param.u64 %rd8, " : "ld.global.u64 %rd8, ")
                    << param_ref(index) << ";" << std::endl;

                const char *load_instr = "ldu";
                if (var.size != 1) {
                    oss << "    mul.wide.u32 %rd9, %r2, " << cuda_register_size(var.type) << ";" << std::endl
                        << "    add.u64 %rd8, %rd8, %rd9;" << std::endl;
                    load_instr = "ld";
                }
                if (var.type != VarType::Bool) {
                    oss << "    " << load_instr << ".global." << cuda_register_type(var.type) << " "
                        << cuda_register_name(var.type) << reg_map[index] << ", [%rd8]"
                        << ";" << std::endl;
                } else {
                    oss << "    " << load_instr << ".global.u8 %w1, [%rd8];" << std::endl
                        << "    setp.ne.u16 " << cuda_register_name(var.type) << reg_map[index] << ", %w1, 0;";
                }
            } else {
                oss << "    " << (parameter_direct ? "ld.param.u64 " : "ldu.global.u64 ")
                    << cuda_register_name(var.type)
                    << reg_map[index] << ", " << param_ref(index) << ";";
            }

            n_in++;
        } else {
            if (!var.label.empty())
                oss << "    // Compute register "
                    << cuda_register_name(var.type) << reg_map[index] << ": "
                    << var.label << std::endl;
            cuda_render_cmd(oss, ctx, reg_map, index);
            n_arith++;

            if (var.side_effect) {
                n_out++;
                continue;
            }

            if (var.ref_count_ext == 0)
                continue;

            if (var.size != size)
                continue;

            size_t size_in_bytes =
                cuda_var_size(index) * cuda_register_size(var.type);

            var.data = cuda_malloc(size_in_bytes);
            var.subtree_size = 1;
#if !defined(NDEBUG)
            if (ctx.log_level >= 4)
                std::cerr << "cuda_eval(): allocated variable " << index
                          << " -> " << var.data << " (" << size_in_bytes
                          << " bytes)" << std::endl;
#endif
            size_t index = ptrs.size();
            ptrs.push_back(var.data);
            n_out++;

            oss << std::endl
                << "    // Store register " << cuda_register_name(var.type) << reg_map[index];
            if (!var.label.empty())
                oss << ": " << var.label;
            oss << std::endl
                << "    " << (parameter_direct ? "ld.param.u64 %rd8, " : "ld.global.u64 %rd8, ")
                << param_ref(index) << ";" << std::endl;
            if (var.size != 1) {
                oss << "    mul.wide.u32 %rd9, %r2, " << cuda_register_size(var.type) << ";" << std::endl
                    << "    add.u64 %rd8, %rd8, %rd9;" << std::endl;
            }
            if (var.type != VarType::Bool) {
                oss << "    st.global." << cuda_register_type(var.type) << " [%rd8], "
                    << cuda_register_name(var.type) << reg_map[index] << ";"
                    << std::endl;
            } else {
                oss << "    selp.u16 %w1, 1, 0, " << cuda_register_name(var.type)
                    << reg_map[index] << ";" << std::endl;
                oss << "    st.global.u8" << " [%rd8], %w1;" << std::endl;
            }
        }
    }

    oss << std::endl
        << "    add.u32     %r2, %r2, %r3;" << std::endl
        << "    setp.ge.u32 %p0, %r2, %r1;" << std::endl
        << "    @!%p0 bra L1;" << std::endl;
#endif
}

void jit_run(uint32_t size) {
    float codegen_time = timer();

    auto it = state.kernels.find(buffer.get());
    Kernel kernel;

    if (it == state.kernels.end()) {
        const uintptr_t log_size = 8192;
        std::unique_ptr<char[]> error_log(new char[log_size]),
                                 info_log(new char[log_size]);
        CUjit_option arg[5] = {
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
            jit_fail("Assembly dump:\n\n%s\n\njit_run(): linker error:\n\n%s",
                     buffer.get(), error_log.get());

        void *link_output = nullptr;
        size_t link_output_size = 0;
        cuda_check(cuLinkComplete(link_state, &link_output, &link_output_size));
        if (rt != CUDA_SUCCESS)
            jit_fail("Assembly dump:\n\n%s\n\njit_run(): linker error:\n\n%s",
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
        jit_fail("jit_var_copy_to_device(): device and stream must be set! "
                 "(call jit_device_set() beforehand)!");

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
        cuda_check(cudaEventRecord(stream->event, stream->handle));
    }

    uint32_t stream_index = 1000 * stream->stream;
    for (uint32_t size : schedule_sizes) {
        jit_assemble(size);

        Stream *sub_stream = stream;
        if (parallel_dispatch) {
            jit_device_set(stream->device, stream_index);
            sub_stream = active_stream;
            cuda_check(cudaStreamWaitEvent(sub_stream->handle, stream->event, 0));
        }

        jit_run(size);

        if (parallel_dispatch) {
            cuda_check(cudaEventRecord(sub_stream->event, sub_stream->handle));
            cuda_check(cudaStreamWaitEvent(stream->handle, sub_stream->event, 0));
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

            if (size == 1) {
                // Don't bother with CSE for evaluated scalar variables to replace
                // costly loads with faster arithmetic.
                jit_cse_drop(index, v);
            }

            // if (v->data != nullptr && v->stmt != nullptr) {
                uint32_t dep[3], extra_dep = v->extra_dep;
                memcpy(dep, v->dep, sizeof(uint32_t) * 3);
                memset(v->dep, 0, sizeof(uint32_t) * 3);
                v->extra_dep = 0;
                for (int j = 0; j < 3; ++j)
                    jit_var_dec_ref_int(dep[j]);
                jit_var_dec_ref_ext(extra_dep);
            // }

            if (side_effect)
                jit_var_dec_ref_ext(index);
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
