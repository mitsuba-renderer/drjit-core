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
#include "profiler.h"
#include "util.h"
#include "optix_api.h"

// ====================================================================
//  The following data structures are temporarily used during program
//  generation. They are declared as global variables to enable memory
//  reuse across jitc_eval() calls.
// ====================================================================

/// Are we recording an OptiX kernel?
bool uses_optix = false;

/// Ordered list of variables that should be computed
std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
std::vector<ScheduledGroup> schedule_groups;

/// Auxiliary data structure needed to compute 'schedule_sizes' and 'schedule'
static tsl::robin_set<std::pair<uint32_t, uint32_t>, pair_hash> visited;

/// Kernel parameter buffer and device copy
static std::vector<void *> kernel_params;
static uint8_t *kernel_params_global = nullptr;
static uint32_t kernel_param_count = 0;

/// Buffer containing global declarations
Buffer globals { 0 };

/// Ensure uniqueness of global declarations (intrinsics, virtual functions)
GlobalsSet globals_set;

/// Temporary scratch space for scheduled tasks (LLVM only)
static std::vector<Task *> scheduled_tasks;

/// Hash code of the last generated kernel
size_t kernel_hash = 0;

/// Name of the last generated kernel
static char kernel_name[17] { };

// ====================================================================

/// Recursively traverse the computation graph to find variables needed by a computation
static void jitc_var_traverse(uint32_t size, uint32_t index) {
    if (!visited.emplace(size, index).second)
        return;

    Variable *v = jitc_var(index);
    for (int i = 0; i < 4; ++i) {
        uint32_t index2 = v->dep[i];
        if (index2 == 0)
            break;
        jitc_var_traverse(size, index2);
    }

    if (unlikely(v->extra)) {
        auto it = state.extra.find(index);
        if (it == state.extra.end())
            jit_fail("jit_var_traverse(): could not find matching 'extra' record!");

        const Extra &extra = it->second;
        for (uint32_t i = 0; i < extra.dep_count; ++i) {
            uint32_t index2 = extra.dep[i];
            if (index2)
                jitc_var_traverse(size, index2);
        }
    }

    // If we're really visiting this variable the first time, no matter its size
    if (visited.emplace(0, index).second)
        v->output_flag = false;

    schedule.emplace_back(size, index);
}

void jitc_assemble(ThreadState *ts, ScheduledGroup group) {
    JitBackend backend = ts->backend;

    uses_optix = false;
    kernel_params.clear();

    uint32_t n_params_in      = 0,
             n_params_out     = 0,
             n_side_effects = 0,
             n_regs         = 0;

    if (backend == JitBackend::CUDA) {
        uintptr_t size = 0;
        memcpy(&size, &group.size, sizeof(uint32_t));
        kernel_params.push_back((void *) size);

        // The first 4 variables are reserved on the CUDA backend
        n_regs = 4;
    } else {
        // First 3 parameters reserved for: kernel ptr, size, ITT identifier
        for (int i = 0; i < 3; ++i)
            kernel_params.push_back(nullptr);
    }

    (void) timer();

    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        uint32_t index = schedule[group_index].index;
        Variable *v = jitc_var(index);

        // Some sanity checks
        if (unlikely((JitBackend) v->backend != backend))
            jitc_raise("jit_assemble(): variable scheduled in wrong ThreadState");
        if (unlikely(v->ref_count_int == 0 && v->ref_count_ext == 0))
            jitc_fail("jit_assemble(): schedule contains unreferenced variable %u!", index);
        if (unlikely(v->size != 1 && v->size != group.size))
            jitc_fail("jit_assemble(): schedule contains variable %u with incompatible size "
                     "(%u and %u)!", index, v->size, group.size);
        if (unlikely(!v->data && !v->literal && !v->stmt))
            jitc_fail("jit_assemble(): variable %u has no statement!", index);
        if (unlikely(v->literal && v->data))
            jitc_fail("jit_assemble(): variable is simultaneously literal and evaluated");
        if (unlikely(v->dirty))
            jitc_fail("jit_assemble(): dirty variable encountered!");

        if (v->data) {
            v->param_index = (uint16_t) kernel_params.size();
            v->param_type = ParamType::Input;
            kernel_params.push_back(v->data);
            n_params_in++;
        } else if (v->output_flag && v->size == group.size) {
            size_t isize = (size_t) var_type_size[v->type],
                   dsize = (size_t) group.size * isize;

            // Padding to support out-of-bounds accesses in LLVM gather operations
            if (backend == JitBackend::LLVM && isize < 4)
                dsize += 4 - isize;

            void *data =
                jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                        : AllocType::HostAsync,
                            dsize);

            // The addr. of 'v' can change (jitc_malloc() may release the lock)
            v = jitc_var(index);

            v->data = data;
            v->param_index = (uint16_t) kernel_params.size();
            v->param_type = ParamType::Output;
            kernel_params.push_back(data);
            n_params_out++;
        } else if (v->literal && (VarType) v->type == VarType::Pointer) {
            v->param_index = (uint16_t) kernel_params.size();
            v->param_type = ParamType::Input;
            kernel_params.push_back((void *) v->value);
            n_params_in++;
        } else {
            v->param_type = ParamType::Register;
            v->param_index = (uint16_t) 0xFFFF;
            n_side_effects += v->side_effect;
            uses_optix |= v->optix;
        }

        v->reg_index = n_regs++;
    }

    if (unlikely(n_regs > 0xFFFFFFu))
        jitc_fail("jit_run(): The queued computation involves more than 16 "
                 "million variables, which overflowed an internal counter. "
                 "Even if Enoki could compile such a lparame program, it would "
                 "not run efficiently. Please periodically run jit_eval() to "
                 "break down the computation into smaller chunks.");
    else if (unlikely(n_params_in + n_params_out > 0xFFFF))
        jitc_fail("jit_run(): The queued computation involves more than 16K "
                 "kernel parameters to pass input and output array "
                 "addresses, scalars, etc. Even if Enoki could compile "
                 "such a lparame program, it is unlikely that it would run "
                 "efficiently. Please periodically run jit_eval() to break "
                 "down the computation into smaller chunks.");

    kernel_param_count = (uint32_t) kernel_params.size();

    // Pass parameters through global memory if too large or using OptiX
    if (backend == JitBackend::CUDA &&
        (uses_optix || kernel_param_count > ENOKI_CUDA_ARG_LIMIT)) {
        size_t size = kernel_param_count * sizeof(void *);
        uint8_t *tmp = (uint8_t *) jitc_malloc(AllocType::HostPinned, size);
        kernel_params_global = (uint8_t *) jitc_malloc(AllocType::Device, size);
        memcpy(tmp, kernel_params.data(), size);
        jitc_memcpy_async(backend, kernel_params_global, tmp, size);
        jitc_free(tmp);
        kernel_params.clear();
        kernel_params.push_back(kernel_params_global);
    }

    bool trace = std::max(state.log_level_stderr, state.log_level_callback) >=
                 LogLevel::Trace;

    if (unlikely(trace)) {
        buffer.clear();
        for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
            uint32_t index = schedule[group_index].index;
            Variable *v = jitc_var(index);

            buffer.fmt("   - %s%u -> %u: ", var_type_prefix[v->type],
                       v->reg_index, index);

            const char *label = jitc_var_label(index);
            if (label)
                buffer.fmt("label=\"%s\", ", label);
            if (v->param_type == ParamType::Input)
                buffer.fmt("in, offset=%u, ", v->param_index);
            if (v->param_type == ParamType::Output)
                buffer.fmt("out, offset=%u, ", v->param_index);
            if (v->literal)
                buffer.put("literal, ");
            if (v->size == 1 && v->param_type != ParamType::Output)
                buffer.put("scalar, ");
            if (v->side_effect)
                buffer.put("side effects, ");
            buffer.rewind(2);
            buffer.putc('\n');
        }
        jitc_trace("jit_assemble(size=%u): register map:\n%s",
                  group.size, buffer.get());
    }

    globals.clear();
    globals_set.clear();
    buffer.clear();
    if (backend == JitBackend::CUDA)
        jitc_assemble_cuda(ts, group, n_regs, kernel_param_count);
    else
        jitc_assemble_llvm(ts, group);

    // Replace '^'s in 'enoki_^^^^^^^^' by a hash code
    kernel_hash = hash_kernel(buffer.get());
    snprintf(kernel_name, 17, "%016llx", (unsigned long long) kernel_hash);
    const char *name_start = strchr(buffer.get(), '^');
    if (unlikely(!name_start))
        jitc_fail("jit_eval(): could not find kernel name!");
    memcpy((char *) name_start, kernel_name, 16);

    if (unlikely(trace))
        jitc_trace("%s", buffer.get());

    float codegen_time = timer();
    jitc_log(Info,
            "  -> launching %016llx (%sn=%u, in=%u, out=%u, ops=%u, jit=%s):",
            (unsigned long long) kernel_hash, uses_optix ? "via OptiX, " : "",
            group.size, n_params_in, n_params_out + n_side_effects, n_regs,
            jitc_time_string(codegen_time));
}

Task *jitc_run(ThreadState *ts, ScheduledGroup group) {
    KernelKey kernel_key((char *) buffer.get(), ts->device);
    auto it = state.kernel_cache.find(
        kernel_key, KernelHash::compute_hash(kernel_hash, ts->device));
    Kernel kernel;

    if (it == state.kernel_cache.end()) {
        bool cache_hit = false;

        if (!uses_optix)
            cache_hit = jitc_kernel_load(buffer.get(), (uint32_t) buffer.size(),
                                         ts->backend, kernel_hash, kernel);

        if (!cache_hit) {
            if (ts->backend == JitBackend::CUDA) {
                if (!uses_optix) {
                    jitc_cuda_compile(buffer.get(), buffer.size(), kernel);
                } else {
#if defined(ENOKI_JIT_ENABLE_OPTIX)
                    jitc_optix_compile(ts, buffer.get(), buffer.size(), kernel, kernel_hash);
#else
                    jitc_fail("jit_run(): OptiX support was not enabled in Enoki-JIT.");
#endif
                }
            } else {
                jitc_llvm_compile(buffer.get(), buffer.size(), kernel);
            }

            if (kernel.data)
                jitc_kernel_write(buffer.get(), (uint32_t) buffer.size(),
                                  ts->backend, kernel_hash, kernel);
        }

        if (ts->backend == JitBackend::LLVM) {
            jitc_llvm_disasm(kernel);
        } else if (!uses_optix) {
            CUresult ret;
            /* Unlock while synchronizing */ {
                unlock_guard guard(state.mutex);
                ret = cuModuleLoadData(&kernel.cuda.mod, kernel.data);
            }
            if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
                jitc_malloc_trim();
                /* Unlock while synchronizing */ {
                    unlock_guard guard(state.mutex);
                    ret = cuModuleLoadData(&kernel.cuda.mod, kernel.data);
                }
            }
            cuda_check(ret);

            // Locate the kernel entry point
            char kernel_name[23];
            snprintf(kernel_name, 23, "enoki_%016llx", (unsigned long long) kernel_hash);
            cuda_check(cuModuleGetFunction(&kernel.cuda.func, kernel.cuda.mod,
                                           kernel_name));

            // Determine a suitable thread count to maximize occupancy
            int unused, block_size;
            cuda_check(cuOccupancyMaxPotentialBlockSize(
                &unused, &block_size,
                kernel.cuda.func, nullptr, 0, 0));
            kernel.cuda.block_size = (uint32_t) block_size;

            // Enoki doesn't use shared memory at all, prefer to have more L1 cache.
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                CU_SHAREDMEM_CARVEOUT_MAX_L1));

            free(kernel.data);
            kernel.data = nullptr;
        }

        float link_time = timer();
        jitc_log(Info, "     cache %s, %s: %s, %s.",
                cache_hit ? "hit" : "miss",
                cache_hit ? "load" : "build",
                std::string(jitc_time_string(link_time)).c_str(),
                std::string(jitc_mem_string(kernel.size)).c_str());

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

    if (ts->backend == JitBackend::CUDA) {
#if defined(ENOKI_JIT_ENABLE_OPTIX)
        if (unlikely(uses_optix)) {
            jitc_optix_launch(ts, kernel, group.size, kernel_params_global,
                              kernel_param_count);
            return nullptr;
        }
#endif

        if (!uses_optix) {
            size_t buffer_size = kernel_params.size() * sizeof(void *);

            void *config[] = {
                CU_LAUNCH_PARAM_BUFFER_POINTER,
                kernel_params.data(),
                CU_LAUNCH_PARAM_BUFFER_SIZE,
                &buffer_size,
                CU_LAUNCH_PARAM_END
            };

            uint32_t block_count, thread_count, size = group.size;
            const Device &device = state.devices[ts->device];
            device.get_launch_config(&block_count, &thread_count, size,
                                     (uint32_t) kernel.cuda.block_size);

            cuda_check(cuLaunchKernel(kernel.cuda.func, block_count, 1, 1,
                                      thread_count, 1, 1, 0, ts->stream,
                                      nullptr, config));
        }
    } else {
        uint32_t packets =
            (group.size + jitc_llvm_vector_width - 1) / jitc_llvm_vector_width;

        auto callback = [](uint32_t index, void *ptr) {
            void **params = (void **) ptr;
            LLVMKernelFunction kernel = (LLVMKernelFunction) params[0];
            uint32_t size       = (uint32_t) (uintptr_t) params[1],
                     block_size = (uint32_t) ((uintptr_t) params[1] >> 32),
                     start      = index * block_size,
                     end        = std::min(start + block_size, size);

#if defined(ENOKI_JIT_ENABLE_ITTNOTIFY)
            // Signal start of kernel
            __itt_task_begin(enoki_domain, __itt_null, __itt_null,
                             (__itt_string_handle *) params[2]);
#endif
            // Perform the main computation
            kernel(start, end, params);

#if defined(ENOKI_JIT_ENABLE_ITTNOTIFY)
            // Signal termination of kernel
            __itt_task_end(enoki_domain);
#endif
        };

        uint32_t block_size = ENOKI_POOL_BLOCK_SIZE,
                 blocks = (group.size + block_size - 1) / block_size;

        kernel_params[0] = (void *) kernel.llvm.func;
        kernel_params[1] = (void *) ((((uintptr_t) block_size) << 32) + (uintptr_t) group.size);

#if defined(ENOKI_JIT_ENABLE_ITTNOTIFY)
        kernel_params[2] = kernel.llvm.itt;
#endif

        jitc_log(Trace, "jit_run(): scheduling %u packet%s in %u block%s ..", packets,
                packets == 1 ? "" : "s", blocks, blocks == 1 ? "" : "s");

        return task_submit_dep(
            nullptr, &ts->task, 1, blocks,
            callback,
            kernel_params.data(),
            kernel_params.size() * sizeof(void *),
            nullptr
        );
    }

    return nullptr;
}

static ProfilerRegion profiler_region_eval("jit_eval");

/// Evaluate all computation that is queued on the given ThreadState
void jitc_eval(ThreadState *ts) {
    if (!ts || (ts->scheduled.empty() && ts->side_effects.empty()))
        return;

    uint32_t flags = jitc_flags();
    if (unlikely(flags & (uint32_t) JitFlag::RecordingLoop))
        jitc_raise(
            "jit_eval(): Enoki is currently recording a loop. In such "
            "cases, you are not allowed to run operations that trigger a "
            "kernel evaluation via jit_eval(). Set a breakpoint on "
            "jit_raise() to find the offending code in your program.");

    if (unlikely(flags & (uint32_t) JitFlag::RecordingVCall))
        jitc_raise(
            "jit_eval(): Enoki is currently recording a virtual function call. "
            "In such cases, you are not allowed to run operations that trigger "
            "a kernel evaluation via jit_eval(). Set a breakpoint on "
            "jit_raise() to find the offending code in your program.");

    ProfilerPhase profiler(profiler_region_eval);

    /* The function 'jitc_eval()' modifies several global data structures
       and should never be executed concurrently. However, there are a few
       places where it needs to temporarily release the main lock as part of
       its work, which is dangerous because another thread could use that
       opportunity to enter 'jitc_eval()' and cause corruption. The following
       therefore temporarily unlocks 'state.mutex' and then locks a separate
       mutex 'state.eval_mutex' specifically guarding these data structures */

    state.mutex.unlock();
    lock_guard guard(state.eval_mutex);
    state.mutex.lock();

    visited.clear();
    schedule.clear();

    // Collect variables that must be computed and their subtrees
    for (auto &source : { ts->scheduled, ts->side_effects }) {
        for (uint32_t index : source) {
            auto it = state.variables.find(index);
            if (it == state.variables.end())
                continue;

            Variable *v = &it.value();

            // Skip variables that aren't externally referenced or already evaluated
            if (v->ref_count_ext == 0 || v->data || v->literal)
                continue;

            jitc_var_traverse(v->size, index);
            v->output_flag = (VarType) v->type != VarType::Void;
        }
    }

    ts->scheduled.clear();
    ts->side_effects.clear();

    if (schedule.empty())
        return;

    // Group them from lparame to small sizes while preserving dependencies
    std::stable_sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            return a.size > b.size;
        });

    // Partition into groups of matching size
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

    jitc_log(Info, "jit_eval(): launching %zu kernel%s.",
            schedule_groups.size(),
            schedule_groups.size() == 1 ? "" : "s");

    // Are there independent groups of work that could be dispatched in parallel?
    scoped_set_context_maybe guard2(ts->context);

    scheduled_tasks.clear();
    for (ScheduledGroup &group : schedule_groups) {
        jitc_assemble(ts, group);

        scheduled_tasks.push_back(jitc_run(ts, group));

        if (ts->backend == JitBackend::CUDA) {
            jitc_free(kernel_params_global);
            kernel_params_global = nullptr;
        }
    }

    if (ts->backend == JitBackend::LLVM) {
        if (scheduled_tasks.size() == 1) {
            task_release(ts->task);
            ts->task = scheduled_tasks[0];
        } else {
            if (unlikely(scheduled_tasks.empty()))
                jitc_fail("jit_eval(): no tasks generated!");

            // Insert a barrier task
            Task *new_task = task_submit_dep(nullptr, scheduled_tasks.data(),
                                             scheduled_tasks.size());
            task_release(ts->task);
            for (Task *t : scheduled_tasks)
                task_release(t);
            ts->task = new_task;
        }
    }

    /* At this point, all variables and their dependencies are computed, which
       means that we can remove internal edges between them. This in turn will
       cause many of the variables to be garbage-collected. */
    jitc_log(Debug, "jit_eval(): cleaning up..");

    for (ScheduledVariable sv : schedule) {
        uint32_t index = sv.index;

        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;

        Variable *v = &it.value();
        if (!(v->output_flag || v->side_effect))
            continue;

        if (v->literal) {
            jitc_var_eval_literal(index, v);
            continue;
        }

        jitc_cse_drop(index, v);

        if (v->literal) {
            v->literal = 0;
            v->value = 0;
        } else {
            if (v->free_stmt)
                free(v->stmt);
            v->stmt = nullptr;
        }

        uint32_t dep[4];
        memcpy(dep, v->dep, sizeof(uint32_t) * 4);
        memset(v->dep, 0, sizeof(uint32_t) * 4);
        v->stmt = nullptr;

        if (v->side_effect) {
            if (dep[0]) {
                Variable *ptr = jitc_var(dep[0]);
                if ((VarType) ptr->type == VarType::Pointer)
                    jitc_var(ptr->dep[3])->dirty = false;
            }
            jitc_var_dec_ref_ext(index);
        }

        for (int j = 0; j < 4; ++j)
            jitc_var_dec_ref_int(dep[j]);
    }

    jitc_free_flush(ts);
    jitc_log(Info, "jit_eval(): done.");
}

