/*
    src/eval.cpp -- Main computation graph evaluation routine

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "profile.h"
#include "util.h"
#include "optix.h"
#include "loop.h"
#include "call.h"
#include "trace.h"
#include <tsl/robin_set.h>

// ====================================================================
//  The following data structures are temporarily used during program
//  generation. They are declared as global variables to enable memory
//  reuse across jitc_eval() calls. Access to them is protected by the
//  central Dr.Jit mutex.
// ====================================================================

/// Ordered list of variables that should be computed
std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
std::vector<ScheduledGroup> schedule_groups;

struct VisitedKey {
    uint32_t size;
    uint32_t index;
    uint32_t depth;
    VisitedKey(uint32_t size, uint32_t index, uint32_t depth)
        : size(size), index(index), depth(depth) { }

    bool operator==(VisitedKey k) const {
        return k.size == size && k.index == index && k.depth == depth;
    }
};

struct VisitedKeyHash {
    size_t operator()(VisitedKey k) const {
        return (size_t) fmix64((uint64_t(k.size ^ k.depth) << 32) |  k.index);
    }
};

/// Auxiliary data structure needed to compute 'schedule' and 'schedule_groups'
static tsl::robin_set<VisitedKey, VisitedKeyHash> visited;

/// Kernel parameter buffer and device copy
static std::vector<void *> kernel_params;
static uint8_t *kernel_params_global = nullptr;
static uint32_t kernel_param_count = 0;

/// Ensure uniqueness of globals/callables arrays
GlobalsMap globals_map;

/// StringBuffer for global definitions (intrinsics, callables, etc.)
StringBuffer globals { 1000 };

/// Temporary scratch space for scheduled tasks (LLVM only)
static std::vector<Task *> scheduled_tasks;

/// Hash code of the last generated kernel
XXH128_hash_t kernel_hash { 0, 0 };

/// Name of the last generated kernel
char kernel_name[52 /* strlen("__direct_callable__") + 32 + 1 */] { };

// Total number of operations used across the entire kernel (including functions)
static uint32_t n_ops_total = 0;

/// Are we recording an OptiX kernel?
bool uses_optix = false;

/// Size and alignment of auxiliary buffer needed by virtual function calls
int32_t alloca_size = -1;
int32_t alloca_align = -1;

/// Number of tentative callables that were assembled in the kernel being compiled
uint32_t callable_count = 0;

/// Number of unique callables in the kernel being compiled
uint32_t callable_count_unique = 0;

/// Specifies the nesting level of virtual calls being compiled
uint32_t callable_depth = 0;

/// Information about the kernel launch to go in the kernel launch history
KernelHistoryEntry kernel_history_entry;

/// List of enqueued callbacks (bound checks, async dr.print statements, etc.)
static std::vector<uint32_t> eval_callbacks;

/// Temporary todo list needed to correctly process loops in jitc_var_traverse()
static std::vector<VisitedKey> visit_later;

// ====================================================================

// Don't perform scatters, whose output buffer is found to be unreferenced
bool jitc_var_maybe_suppress_scatter(uint32_t index, Variable *v, uint32_t depth) {
    Variable *target = jitc_var(v->dep[0]);
    Variable *target_ptr = jitc_var(target->dep[3]);
    if (target_ptr->ref_count != 0 || depth != 0)
        return false;

    jitc_log(Debug, "jit_eval(): eliding scatter r%u, whose output is unreferenced.", index);
    if (callable_depth == 0)
        jitc_var_dec_ref(index, v);
    return true;
}


/// Recursively traverse the computation graph to find variables needed by a computation
static void jitc_var_traverse(uint32_t size, uint32_t index, uint32_t depth = 0) {
    if (!visited.emplace(size, index, depth).second)
        return;

    Variable *v = jitc_var(index);
    switch ((VarKind) v->kind) {
        case VarKind::Scatter:
            if (jitc_var_maybe_suppress_scatter(index, v, depth))
                return;
            break;

        case VarKind::LoopPhi: {
                LoopData *loop = (LoopData *) jitc_var(v->dep[0])->data;
                jitc_var_traverse(size, loop->outer_in[v->literal], depth);
                visit_later.emplace_back(
                    size, loop->inner_out[v->literal], depth);
            }
            break;

        case VarKind::LoopOutput: {
                LoopData *loop = (LoopData *) jitc_var(v->dep[0])->data;
                jitc_var_traverse(size, loop->inner_out[v->literal], depth);
                jitc_var_traverse(size, loop->outer_in[v->literal], depth);
                jitc_var_traverse(size, loop->inner_in[v->literal], depth);
            }
            break;

        case VarKind::CallInput: {
                if (depth == 0) {
                    schedule.emplace_back(size, v->scope, index);
                    jitc_var_inc_ref(index, v);
                    return;
                } else {
                    jitc_var_traverse(size, v->dep[0], depth - 1);
                    return;
                }
            };

        case VarKind::Call: {
                CallData *call = (CallData *) v->data;
                if (unlikely(!call->optimize)) {
                    for (uint32_t i = 0; i < call->n_in; ++i)
                        jitc_var_traverse(size, call->outer_in[i], depth);
                    for (uint32_t i = 0; i < call->n_inst; ++i)
                        for (uint32_t j = 0; j < call->n_out; ++j)
                            jitc_var_traverse(size, call->inner_out[j + i*call->n_out], depth + 1);
                }
                for (uint32_t i : call->side_effects)
                    jitc_var_traverse(size, i, depth + 1);
            };
            break;

        case VarKind::CallOutput: {
                CallData *call = (CallData *) jitc_var(v->dep[0])->data;
                for (uint32_t i = 0; i < call->n_inst; ++i)
                    jitc_var_traverse(size, call->inner_out[v->literal + i * call->n_out], depth + 1);

            }
            break;

        case VarKind::TraceRay: {
                TraceData *call = (TraceData *) v->data;
                for (uint32_t i: call->indices)
                    jitc_var_traverse(size, i, depth);

            }
            break;

        default:
            break;
    }

    for (int i = 0; i < 4; ++i) {
        uint32_t index2 = v->dep[i];
        if (index2 == 0)
            break;
        jitc_var_traverse(size, index2, depth);
    }

    if (depth == 0) {
        // If we're visiting this variable the first time regardless of size
        if (visited.emplace(0, index, depth).second)
            v->output_flag = false;
        schedule.emplace_back(size, v->scope, index);
        jitc_var_inc_ref(index, v);
    }
}

void jitc_assemble(ThreadState *ts, ScheduledGroup group) {
    JitBackend backend = ts->backend;

    kernel_params.clear();
    globals.clear();
    globals_map.clear();
    alloca_size = alloca_align = -1;
    callable_count = 0;
    callable_count_unique = 0;
    kernel_history_entry = { };

#if defined(DRJIT_ENABLE_OPTIX)
    uses_optix = ts->backend == JitBackend::CUDA &&
                 jit_flag(JitFlag::ForceOptiX);
#endif

    uint32_t n_params_in    = 0,
             n_params_out   = 0,
             n_side_effects = 0,
             n_regs         = 0;

    if (backend == JitBackend::CUDA) {
        uintptr_t size = 0;
        memcpy(&size, &group.size, sizeof(uint32_t));
        kernel_params.push_back((void *) size);

        // The first 3 variables are reserved on the CUDA backend
        n_regs = 4;
    } else {
        // First 3 parameters reserved for: kernel ptr, size, ITT identifier
        for (int i = 0; i < 3; ++i)
            kernel_params.push_back(nullptr);
        n_regs = 1;
    }

    (void) timer();

    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        ScheduledVariable &sv = schedule[group_index];
        uint32_t index = sv.index;
        Variable *v = jitc_var(index);
        v->ssa_f32_cast = 0;

        // Some sanity checks
        if (unlikely((JitBackend) v->backend != backend))
            jitc_raise("jit_assemble(): variable r%u scheduled in wrong ThreadState", index);
        if (unlikely(v->ref_count == 0))
            jitc_fail("jit_assemble(): schedule contains unreferenced variable r%u!", index);
        if (unlikely(v->size != 1 && v->size != group.size))
            jitc_fail("jit_assemble(): schedule contains variable r%u with incompatible size "
                     "(%u and %u)!", index, v->size, group.size);
        if (unlikely(v->is_dirty()))
            jitc_fail("jit_assemble(): dirty variable r%u encountered!", index);

        v->param_offset = (uint32_t) kernel_params.size() * sizeof(void *);
        v->reg_index = n_regs++;

        if (v->is_evaluated()) {
            n_params_in++;
            v->param_type = ParamType::Input;
            kernel_params.push_back(v->data);
        } else if (v->output_flag && v->size == group.size) {
            n_params_out++;
            v->param_type = ParamType::Output;

            size_t isize = (size_t) type_size[v->type],
                   dsize = (size_t) group.size * isize;

            // Padding to support out-of-bounds accesses in LLVM gather operations
            if (backend == JitBackend::LLVM && isize < 4)
                dsize += 4 - isize;

            sv.data = jitc_malloc(
                backend == JitBackend::CUDA ? AllocType::Device
                                            : AllocType::HostAsync,
                dsize); // Note: unsafe to access 'v' after jitc_malloc().

            kernel_params.push_back(sv.data);
        } else if (v->is_literal() && (VarType) v->type == VarType::Pointer) {
            n_params_in++;
            v->param_type = ParamType::Input;
            kernel_params.push_back((void *) v->literal);
        } else {
            n_side_effects += (uint32_t) v->side_effect;
            v->param_type = ParamType::Register;
            v->param_offset = 0xFFFF;

            #if defined(DRJIT_ENABLE_OPTIX)
                uses_optix |= v->optix;
            #endif
        }
    }

    if (unlikely(n_regs > 0xFFFFF))
        jitc_log(Warn,
                 "jit_run(): The generated kernel uses a more than 1 million "
                 "variables (%u) and will likely not run efficiently. Consider "
                 "periodically running jit_eval() to break the computation "
                 "into smaller chunks.", n_regs);

    if (unlikely(kernel_params.size() > 0xFFFF))
        jitc_log(Warn,
                 "jit_run(): The generated kernel accesses more than 64K "
                 "arrays (%zu) and will likely not run efficiently. Consider "
                 "periodically running jit_eval() to break the computation "
                 "into smaller chunks.", kernel_params.size());

    kernel_param_count = (uint32_t) kernel_params.size();
    n_ops_total = n_regs;

    // Pass parameters through global memory if too large or using OptiX
    if (backend == JitBackend::CUDA &&
        (uses_optix || kernel_param_count > DRJIT_CUDA_ARG_LIMIT)) {
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

            buffer.fmt("   - %s%u -> r%u: ", type_prefix[v->type],
                       v->reg_index, index);

            const char *label = jitc_var_label(index);
            if (label)
                buffer.fmt("label=\"%s\", ", label);
            if (v->param_type == ParamType::Input)
                buffer.fmt("in, offset=%u, ", v->param_offset);
            if (v->param_type == ParamType::Output)
                buffer.fmt("out, offset=%u, ", v->param_offset);
            if (v->is_literal())
                buffer.put("literal, ");
            if (v->size == 1 && v->param_type != ParamType::Output)
                buffer.put("scalar, ");
            if (v->side_effect)
                buffer.put("side effects, ");
            buffer.rewind_to(buffer.size() - 2);
            buffer.put('\n');
        }
        jitc_trace("jit_assemble(size=%u): register map:\n%s",
                  group.size, buffer.get());
    }

    buffer.clear();
    if (backend == JitBackend::CUDA)
        jitc_cuda_assemble(ts, group, n_regs, kernel_param_count);
    else
        jitc_llvm_assemble(ts, group);

    // Replace '^'s in '__raygen__^^^..' or 'drjit_^^^..' with hash
    kernel_hash = hash_kernel(buffer.get());

    size_t hash_offset = strchr(buffer.get(), '^') - buffer.get(),
           end_offset = buffer.size(),
           prefix_len = uses_optix ? 10 : 6;

    buffer.rewind_to(hash_offset);
    buffer.put_q64_unchecked(kernel_hash.high64);
    buffer.put_q64_unchecked(kernel_hash.low64);
    buffer.rewind_to(end_offset);
    memset(kernel_name, 0, sizeof(kernel_name));
    memcpy(kernel_name, buffer.get() + hash_offset - prefix_len,
           prefix_len + 32);

    if (unlikely(trace || (jitc_flags() & (uint32_t) JitFlag::PrintIR))) {
        buffer.put('\n');
        if (state.log_callback)
            state.log_callback(LogLevel::Info, buffer.get());
        else
            fputs(buffer.get(), stderr);
        buffer.rewind_to(buffer.size() - 1);
    }

    float codegen_time = timer();

    if (n_side_effects)
        jitc_log(
            Info, "  -> launching %016llx (%sn=%u, in=%u, out=%u, se=%u, ops=%u, jit=%s):",
            (unsigned long long) kernel_hash.high64,
            uses_optix ? "via OptiX, " : "", group.size, n_params_in,
            n_params_out, n_side_effects, n_ops_total, jitc_time_string(codegen_time));
    else
        jitc_log(
            Info, "  -> launching %016llx (%sn=%u, in=%u, out=%u, ops=%u, jit=%s):",
            (unsigned long long) kernel_hash.high64,
            uses_optix ? "via OptiX, " : "", group.size, n_params_in,
            n_params_out, n_ops_total, jitc_time_string(codegen_time));

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        kernel_history_entry.backend = backend;
        kernel_history_entry.type = KernelType::JIT;
        kernel_history_entry.hash[0] = kernel_hash.low64;
        kernel_history_entry.hash[1] = kernel_hash.high64;
        kernel_history_entry.ir = (char *) malloc_check(buffer.size() + 1);
        memcpy(kernel_history_entry.ir, buffer.get(), buffer.size() + 1);
        kernel_history_entry.uses_optix = uses_optix;
        kernel_history_entry.size = group.size;
        kernel_history_entry.input_count = n_params_in;
        kernel_history_entry.output_count = n_params_out + n_side_effects;
        kernel_history_entry.operation_count = n_ops_total;
        kernel_history_entry.codegen_time = codegen_time * 1e-3f;
    }
}

static ProfilerRegion profiler_region_backend_compile("jit_eval: compiling");

Task *jitc_run(ThreadState *ts, ScheduledGroup group) {
    uint64_t flags = 0;

#if defined(DRJIT_ENABLE_OPTIX)
    if (uses_optix) {
        const OptixPipelineCompileOptions &pco = ts->optix_pipeline->compile_options;
        flags =
            ((uint64_t) pco.numAttributeValues << 0)      + // 4 bit
            ((uint64_t) pco.numPayloadValues   << 4)      + // 4 bit
            ((uint64_t) pco.usesMotionBlur     << 8)      + // 1 bit
            ((uint64_t) pco.traversableGraphFlags  << 9)  + // 16 bit
            ((uint64_t) pco.usesPrimitiveTypeFlags << 25);  // 32 bit
    }
#endif

    KernelKey kernel_key((char *) buffer.get(), ts->device, flags);
    auto it = state.kernel_cache.find(
        kernel_key,
        KernelHash::compute_hash(kernel_hash.high64, ts->device, flags));
    Kernel kernel;
    memset(&kernel, 0, sizeof(Kernel)); // quench uninitialized variable warning on MSVC

    if (it == state.kernel_cache.end()) {
        bool cache_hit = false;

        if (ts->backend == JitBackend::CUDA) {
			ProfilerPhase profiler(profiler_region_backend_compile);
			if (!uses_optix) {
				kernel.size = 1; // dummy size value to distinguish between OptiX and CUDA kernels
				kernel.data = nullptr;
				std::tie(kernel.cuda.mod, cache_hit) = jitc_cuda_compile(buffer.get());
			} else {
				#if defined(DRJIT_ENABLE_OPTIX)
					cache_hit = jitc_optix_compile(
						ts, buffer.get(), buffer.size(), kernel_name, kernel);
				#endif
			}
		} else {
            cache_hit = jitc_kernel_load(buffer.get(), (uint32_t) buffer.size(),
                                         ts->backend, kernel_hash, kernel);

			if (!cache_hit) {
				ProfilerPhase profiler(profiler_region_backend_compile);
				jitc_llvm_compile(kernel);
                jitc_kernel_write(buffer.get(), (uint32_t) buffer.size(),
                                  ts->backend, kernel_hash, kernel);
				jitc_llvm_disasm(kernel);
			}
		}

        if (ts->backend == JitBackend::CUDA && !uses_optix) {
            // Locate the kernel entry point
            size_t offset = buffer.size();
            buffer.fmt_cuda(2, "drjit_$Q$Q", kernel_hash.high64,
                            kernel_hash.low64);
            cuda_check(cuModuleGetFunction(&kernel.cuda.func, kernel.cuda.mod,
                                           buffer.get() + offset));
            buffer.rewind_to(offset);

            // Determine a suitable thread count to maximize occupancy
            int unused, block_size;
            cuda_check(cuOccupancyMaxPotentialBlockSize(
                &unused, &block_size,
                kernel.cuda.func, nullptr, 0, 0));
            kernel.cuda.block_size = (uint32_t) block_size;

            // DrJit doesn't use shared memory at all, prefer to have more L1 cache.
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                CU_SHAREDMEM_CARVEOUT_MAX_L1));
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

        if (unlikely(jit_flag(JitFlag::KernelHistory))) {
            kernel_history_entry.cache_disk = cache_hit;
            kernel_history_entry.cache_hit = cache_hit;
            if (!cache_hit)
                kernel_history_entry.backend_time = link_time * 1e-3f;
        }
    } else {
        kernel_history_entry.cache_hit = true;
        kernel = it.value();
        state.kernel_hits++;
    }
    state.kernel_launches++;

    if (unlikely(jit_flag(JitFlag::KernelHistory) &&
                 ts->backend == JitBackend::CUDA)) {
        auto &e = kernel_history_entry;
        cuda_check(cuEventCreate((CUevent *) &e.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &e.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) e.event_start, ts->stream));
    }

    Task* ret_task = nullptr;
    ret_task = ts->launch(kernel, group.size, &kernel_params,
                          kernel_param_count, kernel_params_global);

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        if (ts->backend == JitBackend::CUDA) {
            cuda_check(cuEventRecord((CUevent) kernel_history_entry.event_end,
                                     ts->stream));
        } else {
            task_retain(ret_task);
            kernel_history_entry.task = ret_task;
        }

        state.kernel_history.append(kernel_history_entry);
    }

    return ret_task;
}

static ProfilerRegion profiler_region_eval("jit_eval");

// Forward declaration
void jitc_eval_impl(ThreadState *ts);

/// Evaluate all computation that is queued on the given ThreadState
void jitc_eval(ThreadState *ts) {
    if (!ts || (ts->scheduled.empty() && ts->side_effects.empty()))
        return;

    ProfilerPhase profiler(profiler_region_eval);

    /* The function 'jitc_eval()' modifies several global data structures
       and should never be executed concurrently. However, there are a few
       places where it needs to temporarily release the main lock as part of
       its work, which is dangerous because another thread could use that
       opportunity to enter 'jitc_eval()' and cause corruption. The following
       therefore temporarily unlocks 'state.lock' and then locks a separate
       lock 'state.eval_lock' specifically guarding these data structures */

    {
        lock_release(state.lock);
        lock_guard guard(state.eval_lock);
        lock_acquire(state.lock);
        jitc_eval_impl(ts);
    }

    if (unlikely(!eval_callbacks.empty())) {
        std::vector<uint32_t> cb;
        eval_callbacks.swap(cb);

        jitc_log(Trace, "jit_eval(): running %zu callbacks ..", cb.size());
        for (uint32_t index: cb) {
            VariableExtra &extra = state.extra[jitc_var(index)->extra];
            if (extra.callback_internal) {
                extra.callback(index, 0, extra.callback_data);
            } else {
                unlock_guard guard_2(state.lock);
                extra.callback(index, 0, extra.callback_data);
            }
            jitc_var_dec_ref(index);
        }
        cb.clear();
        if (eval_callbacks.empty())
            cb.swap(eval_callbacks);
    }

    jitc_log(Info, "jit_eval(): done.");
}

void jitc_eval_impl(ThreadState *ts) {
    visited.clear();
    visit_later.clear();
    schedule.clear();

    for (WeakRef wr: ts->scheduled) {
        // Skip variables that expired, or which we already evaluated
        Variable *v = jitc_var(wr);
        if (!v || v->is_evaluated())
            continue;
        jitc_var_traverse(v->size, wr.index);
        v->output_flag = true;
    }

    ts->scheduled.clear();

    for (uint32_t index: ts->side_effects)
        jitc_var_traverse(jitc_var(index)->size, index);

    ts->side_effects.clear();

    // Should not be replaced by a range-based for loop,
    // as the traversal may append further items
    for (size_t i = 0; i < visit_later.size(); ++i) {
        VisitedKey vk = visit_later[i];
        jitc_var_traverse(vk.size, vk.index, vk.depth);
    }

    if (schedule.empty())
        return;

    // Order variables into groups of matching size
    std::stable_sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            if (a.size > b.size)
                return true;
            else if (a.size < b.size)
                return false;
            else if (a.scope > b.scope)
                return false;
            else if (a.scope < b.scope)
                return true;
            else
                return false;
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
            task_release(jitc_task);
            jitc_task = scheduled_tasks[0];
        } else {
            jitc_assert(!scheduled_tasks.empty(),
                        "jit_eval(): no tasks generated!");

            // Insert a barrier task
            Task *new_task = task_submit_dep(nullptr, scheduled_tasks.data(),
                                             (uint32_t) scheduled_tasks.size());
            task_release(jitc_task);
            for (Task *t : scheduled_tasks)
                task_release(t);
            jitc_task = new_task;
        }
    }

    /* Variables and their dependencies are now computed, hence internal edges
       between them can be removed. This will cause many variables to expire. */
    jitc_log(Debug, "jit_eval(): cleaning up..");

    for (ScheduledVariable sv : schedule) {
        uint32_t index = sv.index;
        Variable *v = jitc_var(index);
        v->reg_index = 0;

        if (unlikely(v->extra)) {
            VariableExtra &extra = state.extra[v->extra];
            if (extra.callback) {
                eval_callbacks.push_back(index);
                jitc_var_inc_ref(index, v);
            }
        }

        if (!(v->output_flag || v->side_effect)) {
            jitc_var_dec_ref(index, v);
            continue;
        }

        jitc_assert(!v->is_literal(),
                   "jit_eval(): internal error: did not expect a literal "
                   "constant variable here!");

        jitc_lvn_drop(index, v);

        if (v->output_flag && v->size == sv.size) {
            v->kind = (uint32_t) VarKind::Evaluated;
            v->data = sv.data;
            v->output_flag = false;
            v->consumed = false;
        }

        uint32_t dep[4], side_effect = v->side_effect;
        memcpy(dep, v->dep, sizeof(uint32_t) * 4);
        memset(v->dep, 0, sizeof(uint32_t) * 4);
        v->side_effect = false;

        jitc_var_dec_ref(index, v);

        if (side_effect)
            jitc_var_dec_ref(index, v);

        for (int j = 0; j < 4; ++j)
            jitc_var_dec_ref(dep[j]);
    }
}

static ProfilerRegion profiler_region_assemble_func("jit_assemble_func");

XXH128_hash_t jitc_assemble_func(const CallData *call, uint32_t inst,
                                 uint32_t in_size, uint32_t in_align,
                                 uint32_t out_size, uint32_t out_align) {
    ProfilerPhase profiler(profiler_region_assemble_func);

    visited.clear();
    visit_later.clear();
    schedule.clear();
    callable_depth++;

    const std::vector<uint32_t> &check = call->checkpoints;

    const size_t n_out = call->n_out,
                 n_se = check.empty() ? 0 : (check[inst + 1] - check[inst]);

    const uint32_t *out = call->inner_out.data() + n_out * inst,
                   *se = call->side_effects.empty()
                             ? nullptr
                             : (call->side_effects.data() + check[inst]);

    for (size_t i = 0; i < n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        jitc_var_traverse(1, out[i]);
    }

    for (uint32_t i = 0; i < n_se; ++i)
        jitc_var_traverse(1, se[i]);

    // Should not be replaced by a range-based for loop,
    // as the traversal may append further items
    for (size_t i = 0; i < visit_later.size(); ++i) {
        VisitedKey vk = visit_later[i];
        jitc_var_traverse(vk.size, vk.index, vk.depth);
    }

    std::stable_sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            return a.scope < b.scope;
        });

    uint32_t n_regs = call->backend == JitBackend::CUDA ? 4 : 1;
    for (ScheduledVariable &sv : schedule) {
        Variable *v = jitc_var(sv.index);
        v->reg_index = n_regs++;
    }

    size_t kernel_offset = buffer.size();

    switch (call->backend) {
        case JitBackend::LLVM:
            jitc_llvm_assemble_func(call, inst);
            break;

        case JitBackend::CUDA:
            jitc_cuda_assemble_func(call, inst, in_size, in_align, out_size,
                                    out_align, n_regs);
            break;

        default:
            jitc_fail("jitc_assemble_func(): unsupported backend!");
    }
    callable_depth--;

    size_t kernel_length = buffer.size() - kernel_offset;

    if (jit_flag(JitFlag::MergeFunctions)) {
        kernel_hash = XXH128(buffer.get() + kernel_offset, kernel_length, 0);
    } else {
        kernel_hash.low64 = callable_count;
        kernel_hash.high64 = 0;
    }

    if (globals_map.emplace(GlobalKey(kernel_hash, true),
                            GlobalValue(globals.size(), kernel_length)).second) {
        // Replace '^'s in 'func_^^^..' or '__direct_callable__^^^..' with hash
        size_t hash_offset = strchr(buffer.get() + kernel_offset, '^') - buffer.get(),
               end_offset = buffer.size();

        buffer.rewind_to(hash_offset);
        buffer.put_q64_unchecked(kernel_hash.high64);
        buffer.put_q64_unchecked(kernel_hash.low64);
        buffer.rewind_to(end_offset);

        n_ops_total += n_regs;
        callable_count_unique++;
        globals.put(buffer.get() + kernel_offset, kernel_length);
    }

    callable_count++;

    buffer.rewind_to(kernel_offset);

    for (ScheduledVariable &sv: schedule) {
        Variable *v = jitc_var(sv.index);
        if (unlikely(v->extra)) {
            VariableExtra &extra = state.extra[v->extra];
            if (extra.callback) {
                eval_callbacks.push_back(sv.index);
                jitc_var_inc_ref(sv.index, v);
            }
        }
        v->reg_index = 0;
        v->output_flag = false;
        jitc_var_dec_ref(sv.index, v);
    }

    return kernel_hash;
}

/// Register a global declaration that will be included in the final program
void jitc_register_global(const char *str) {
    size_t length = strlen(str);
    if (globals_map.emplace(GlobalKey(XXH128(str, length, 0), false),
                            GlobalValue(globals.size(), length)).second)
        globals.put(str, length);
}
