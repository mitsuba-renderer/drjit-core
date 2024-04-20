/*
    src/call.cpp -- Code generation for virtual function calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "registry.h"
#include "util.h"
#include "op.h"
#include "profile.h"
#include "loop.h"
#include "trace.h"
#include "call.h"
#include <set>

static std::vector<CallData *> calls_assembled;

extern void jitc_var_call_analyze(CallData *call, uint32_t inst_id,
                                  uint32_t index, uint32_t &data_offset);

/// Weave a virtual function call into the computation graph
void jitc_var_call(const char *name, uint32_t self, uint32_t mask_,
                   uint32_t n_inst, const uint32_t *inst_id, uint32_t n_in,
                   const uint32_t *in, uint32_t n_inner_out,
                   const uint32_t *inner_out, const uint32_t *checkpoints,
                   uint32_t *out) {

    const uint32_t checkpoint_mask = 0x7fffffff;

#if defined(DRJIT_ENABLE_NVTX) || defined(DRJIT_ENABLE_ITTNOTIFY)
    std::string profile_name = std::string("jit_var_call: ") + name;
    ProfilerRegion profiler_region(profile_name.c_str());
    ProfilerPhase profiler(profiler_region);
#endif

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    if (n_inst == 0)
        jitc_raise("jit_var_call('%s'): must have at least one instance!", name);

    if (n_inner_out % n_inst != 0)
        jitc_raise("jit_var_call('%s'): list of all output indices must be a "
                   "multiple of the instance count (n_inner_out=%u, n_inst=%u)!",
                   name, n_inner_out, n_inst);

    uint32_t n_out = n_inner_out / n_inst, size = 0;
    bool symbolic = false, dirty = false;

    JitBackend backend;
    /* Check 'self' */ {
        const Variable *self_v = jitc_var(self);
        size = self_v->size;
        symbolic |= (bool) self_v->symbolic;
        dirty |= self_v->is_dirty();
        backend = (JitBackend) self_v->backend;
        if ((VarType) self_v->type != VarType::UInt32)
            jitc_raise("jit_var_call(): 'self' argument must be an unsigned "
                       "32-bit integer array");
    }

    /* Check 'mask' */ {
        const Variable *mask_v = jitc_var(mask_);
        size = std::max(size, mask_v->size);
        symbolic |= (bool) mask_v->symbolic;
        dirty |= (bool) mask_v->is_dirty();
        if ((VarType) mask_v->type != VarType::Bool)
            jitc_raise(
                "jit_var_call(): 'mask' argument must be a boolean array");
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if ((VarKind) v->kind == VarKind::CallInput) {
            if (!v->dep[0])
                jitc_raise("jit_var_call(): symbolic variable r%u does not "
                           "reference another input!", in[i]);
            Variable *v2 = jitc_var(v->dep[0]);
            symbolic |= (bool) v2->symbolic;
            dirty |= v2->is_dirty();
            size = std::max(size, v2->size);
        } else if (!v->is_literal()) {
            jitc_raise("jit_var_call(): input variable r%u must either be a "
                       "literal or symbolic wrapper around another variable!", in[i]);
        }
        if (v->size != 1)
            jitc_raise(
                "jit_var_call(): size of input variable r%u is %u (must be 1)!",
                in[i], v->size);
    }

    for (uint32_t i = 0; i < n_inner_out; ++i) {
        const Variable *v0 = jitc_var(inner_out[i % n_out]),
                       *v1 = jitc_var(inner_out[i]);

        if (v0->type != v1->type)
            jitc_raise("jit_var_call(): output types don't match between instances.");

        if (v1->size != 1)
            jitc_raise("jit_var_call(): size of output variable r%u is %u "
                       "(must be 1)!", inner_out[i], v1->size);
    }

    ThreadState *ts = thread_state(backend);
    if (ts->side_effects_symbolic.size() !=
        (checkpoints[n_inst] & checkpoint_mask))
        jitc_raise("jitc_var_call(): side effect queue doesn't have the "
                   "expected size!");

    if (dirty) {
        jitc_eval(ts);

        dirty = jitc_var(self)->is_dirty() || jitc_var(mask_)->is_dirty();
        for (uint32_t i = 0; i < n_in; ++i) {
            const Variable *v = jitc_var(in[i]);
            if ((VarKind) v->kind == VarKind::CallInput)
                dirty |= jitc_var(v->dep[0])->is_dirty();
        }

        if (unlikely(dirty))
            jitc_raise("jit_var_call(): inputs remain dirty after evaluation!");
    }

    // =====================================================
    // 3. Apply any masks on the stack, ignore self==NULL
    // =====================================================

    uint32_t flags = jitc_flags();
    bool optimize = flags & (uint32_t) JitFlag::OptimizeCalls,
         debug = flags & (uint32_t) JitFlag::Debug;

    Ref mask;
    {
        uint32_t zero = 0;
        Ref null_instance = steal(jitc_var_literal(backend, VarType::UInt32, &zero, size, 0)),
            is_non_null   = steal(jitc_var_neq(self, null_instance)),
            mask_2        = steal(jitc_var_and(mask_, is_non_null));

        mask = steal(jitc_var_mask_apply(mask_2, size));

        if (debug)
            mask = steal(jitc_var_check_bounds(BoundsCheckType::Call, self,
                                               mask, n_inst + 1));
    }

    // =====================================================
    // 3. Stash information about inputs and outputs
    // =====================================================
    //
    std::unique_ptr<CallData> call(new CallData());
    call->backend = backend;
    call->name = strdup(name);
    call->n_in = n_in;
    call->n_out = n_out;
    call->n_inst = n_inst;
    call->optimize = optimize;
    call->inst_id = std::vector<uint32_t>(inst_id, inst_id + n_inst);
    call->inst_hash.resize(n_inst);
    call->outer_in.reserve(n_in);
    call->inner_in.reserve(n_in);
    call->outer_out.reserve(n_out);
    call->inner_out.reserve(n_inner_out);
    call->out_offset.resize(n_out, 0);

    // Move recorded side effects & offsets into 'call'
    call->checkpoints.reserve(n_inst + 1);
    for (uint32_t i = 0; i < n_inst; ++i)
        call->checkpoints.push_back(checkpoints[i] - checkpoints[0]);
    call->checkpoints.push_back(checkpoints[n_inst] - checkpoints[0]);

    call->side_effects = std::vector<uint32_t>(
        ts->side_effects_symbolic.begin() + (checkpoints[0] & checkpoint_mask),
        ts->side_effects_symbolic.begin() + (checkpoints[n_inst] & checkpoint_mask));
    ts->side_effects_symbolic.resize(checkpoints[0] & checkpoint_mask);

    // Collect inputs
    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t index = in[i];
        Variable *v = jitc_var(index);
        jitc_var_inc_ref(index, v);

        call->inner_in.push_back(index);
        if (!v->is_literal())
            index = v->dep[0];
        call->outer_in.push_back(index);
    }

    // =====================================================
    // 4. Collect evaluated data accessed by the instances
    // =====================================================

    call->data_offset.reserve(n_inst);

    // Collect accesses to evaluated variables/pointers
    uint32_t data_size = 0, inst_id_max = 0;
    for (uint32_t i = 0; i < n_inst; ++i) {
        if (unlikely(checkpoints[i] > checkpoints[i + 1]))
            jitc_raise("jitc_var_call(): values in 'checkpoints' are not "
                       "monotonically increasing!");

        uint32_t id = inst_id[i];
        call->data_offset.push_back(data_size);

        for (uint32_t j = 0; j < n_out; ++j)
            jitc_var_call_analyze(call.get(), i, inner_out[j + i * n_out],
                                  data_size);

        for (uint32_t j = checkpoints[i]; j != checkpoints[i + 1]; ++j)
            jitc_var_call_analyze(call.get(), i,
                                  call->side_effects[j - checkpoints[0]],
                                  data_size);

        // Restore to full alignment
        data_size = (data_size + 7) / 8 * 8;
        inst_id_max = std::max(id, inst_id_max);
    }

    // Allocate memory + wrapper variables for call offset and data arrays
    call->offset_size = (inst_id_max + 1) * sizeof(uint64_t);

    AllocType at =
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync;
    call->offset = (uint64_t *) jitc_malloc(at, call->offset_size);
    uint8_t *data_d = (uint8_t *) jitc_malloc(at, data_size);

    Ref data_buf, data_v,
        offset_buf = steal(jitc_var_mem_map(
            backend, VarType::UInt64, call->offset, inst_id_max + 1, 1)),
        offset_v =
            steal(jitc_var_pointer(backend, call->offset, offset_buf, 0));

    char temp[128];
    snprintf(temp, sizeof(temp), "Call: %s [offsets]", name);
    jitc_var_set_label(offset_buf, temp);

    if (data_size) {
        data_buf = steal(
            jitc_var_mem_map(backend, VarType::UInt8, data_d, data_size, 1));
        snprintf(temp, sizeof(temp), "Call: %s [data]", name);
        jitc_var_set_label(data_buf, temp);

        data_v = steal(jitc_var_pointer(backend, data_d, data_buf, 0));

        AggregationEntry *agg = nullptr;
        size_t agg_size = sizeof(AggregationEntry) * call->data_map.size();

        if (backend == JitBackend::CUDA)
            agg = (AggregationEntry *) jitc_malloc(AllocType::HostPinned, agg_size);
        else
            agg = (AggregationEntry *) malloc_check(agg_size);

        AggregationEntry *p = agg;

        for (auto kv : call->data_map) {
            uint32_t index = (uint32_t) kv.first, offset = kv.second;
            if (offset == (uint32_t) -1)
                continue;

            const Variable *v = jitc_var(index);
            bool is_pointer = (VarType) v->type == VarType::Pointer;
            p->offset = offset;
            p->size = is_pointer ? 8 : -(int) type_size[v->type];
            p->src = is_pointer ? (const void *) v->literal : v->data;
            p++;
        }

        std::sort(agg, p,
                  [](const AggregationEntry &a, const AggregationEntry &b) {
                      return a.offset < b.offset;
                  });

        jitc_aggregate(backend, data_d, agg, (uint32_t) (p - agg));
    } else {
        call->data_map.clear();
    }

    // =====================================================
    // 5. Create special variable representing the call op.
    // =====================================================

    Ref call_v;

    if (data_size)
        call_v = steal(jitc_var_new_node_4(
            backend, VarKind::Call, VarType::Void, size, symbolic, self,
            jitc_var(self), mask, jitc_var(mask), offset_v, jitc_var(offset_v),
            data_v, jitc_var(data_v)));
    else
        call_v = steal(
            jitc_var_new_node_3(backend, VarKind::Call, VarType::Void, size,
                                symbolic, self, jitc_var(self), mask,
                                jitc_var(mask), offset_v, jitc_var(offset_v)));

    call->id = call_v;
    {
        Variable *call_var = jitc_var(call_v);
        call_var->optix = call->use_optix;
        call_var->data = call.get();
    }

    // =====================================================
    // 6. Create output variables
    // =====================================================

    // Optimize calling conventions by reordering inputs
    uint32_t n_devirt = 0;

    call->inner_out.resize(n_inner_out, 0);

    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = inner_out[i];
        const Variable *v = jitc_var(index);

        bool uniform = true;
        for (uint32_t j = 0; j < n_inst; ++j) {
            uint32_t index2 = inner_out[j * n_out + i];
            uniform &= index == index2;
            jitc_var_inc_ref(index2);
            call->inner_out[j * n_out + i] = index2;
        }

        // Devirtualize uniform scalar literals when optimizations are enabled
        if (uniform && optimize && v->is_literal()) {
            out[i] = jitc_var_and(index, mask);
            n_devirt++;
            call->outer_out.push_back(WeakRef());
        } else {

            uint32_t out_index =
                jitc_var_new_node_1(backend, VarKind::CallOutput, (VarType) v->type,
                                    size, symbolic, call_v, jitc_var(call_v), i);
            Variable *v2 = jitc_var(out_index);
            v2->literal = i;

            snprintf(temp, sizeof(temp), "Call: %s [out %u]", name, i);
            jitc_var_set_label(out_index, temp);

            call->outer_out.emplace_back(out_index, v2->counter);
            out[i] = out_index;
        }
    }

    uint32_t se_count = checkpoints[n_inst] - checkpoints[0];

    jitc_log(InfoSym,
             "jit_var_call(r%u, self=r%u): call (\"%s\") with %u instance%s, %u "
             "input%s, %u output%s (%u devirtualized), %u side effect%s, %u "
             "byte%s of call data, %u elements%s%s", (uint32_t) call_v, self, name, n_inst,
             n_inst == 1 ? "" : "s", n_in, n_in == 1 ? "" : "s", n_out,
             n_out == 1 ? "" : "s", n_devirt, se_count, se_count == 1 ? "" : "s",
             data_size, data_size == 1 ? "" : "s", size,
             (n_devirt == n_out && se_count == 0) ? " (optimized away)" : "",
             symbolic ? " ([symbolic])" : "");


    // =====================================================
    // 8. Install code generation and deallocation callbacks
    // =====================================================

    snprintf(temp, sizeof(temp), "Call: %s", name);
    jitc_var_set_label(call_v, temp);

    Ref se_v;
    if (se_count) {
        /* The call has side effects. Create a dummy variable to
           ensure that they are evaluated */
        se_v = steal(jitc_var_new_node_1(backend, VarKind::Nop, VarType::Void,
                                         size, symbolic, call_v,
                                         jitc_var(call_v)));

        snprintf(temp, sizeof(temp), "Call: %s [side effects]", name);
        jitc_var_set_label(se_v, temp);
    }

    jitc_var_set_callback(
        call_v,
        [](uint32_t, int free, void *p) {
            if (free)
                delete (CallData *) p;
        },
        call.release(), true);

    jitc_var_mark_side_effect(se_v.release());
}

static ProfilerRegion profiler_region_call_assemble("jit_var_call_assemble");

/// Data structure to sort function arguments/return values in order of
/// decreasing size and increasing index
struct PermKey {
    uint32_t size;
    uint32_t index;
    PermKey(uint32_t size, uint32_t index) : size(size), index(index) { }

    bool operator<(const PermKey &k) const {
        if (size > k.size)
            return true;
        else if (size < k.size)
            return false;
        return index < k.index;
    }
};

static std::vector<PermKey> call_perm;

/// Called when Dr.Jit compiles a function call, specifically the 'Call' IR node
void jitc_var_call_assemble(CallData *call, uint32_t call_reg,
                            uint32_t self_reg, uint32_t mask_reg,
                            uint32_t offset_reg, uint32_t data_reg) {

    ProfilerPhase profiler(profiler_region_call_assemble);

    // =====================================================
    // 1. Need to backup state before we can JIT recursively
    // =====================================================

    struct JitBackupRecord {
        ScheduledVariable sv;
        uint32_t param_type : 2;
        uint32_t output_flag : 1;
        uint32_t reg_index;
        uint32_t param_offset;
    };

    std::vector<JitBackupRecord> backup;
    backup.reserve(schedule.size());

    for (const ScheduledVariable &sv : schedule) {
        const Variable *v = jitc_var(sv.index);
        backup.push_back(JitBackupRecord{ sv, v->param_type, v->output_flag,
                                          v->reg_index, v->param_offset });
    }

    // =========================================================
    // 2. Determine size and alignment of parameter and return
    //    value buffers. Unreferenced return values present an
    //    opportunity for dead code elimination.
    // =========================================================

    bool optimize = jitc_flags() & (uint32_t) JitFlag::OptimizeCalls;

    uint32_t n_inst = call->n_inst,
             n_in = call->n_in,
             n_out = call->n_out,
             n_in_active = 0,
             n_out_active = 0,
             in_size = 0, in_size_all = 0, in_align = 1,
             out_size = 0, out_size_all = 0, out_align = 1;

    call_perm.clear();
    call_perm.reserve(std::max(n_in, n_out));
    for (uint32_t i = 0; i < n_in; ++i)
        call_perm.emplace_back(type_size[jitc_var(call->inner_in[i])->type], i);
    std::sort(call_perm.begin(), call_perm.end());

    for (auto [size, i] : call_perm) {
        Variable *v = jitc_var(call->outer_in[i]);
        in_size_all += size;

        if (optimize && !v->reg_index)
            continue;

        v->param_offset = in_size;
        in_size += size;
        in_align = std::max(size, in_align);
        n_in_active++;
    }

    call_perm.clear();
    for (uint32_t i = 0; i < n_out; ++i)
        call_perm.emplace_back(type_size[jitc_var(call->inner_out[i])->type], i);
    std::sort(call_perm.begin(), call_perm.end());

    if (call->backend == JitBackend::LLVM) {
        // In LLVM mode, the same buffer is used for inputs *and* outputs
        out_size = in_size;
        out_align = in_align;

        // Ensure alignment given that we're appending to the input buffer
        if (!call_perm.empty()) {
            const uint32_t size = call_perm[0].size;
            out_size = (out_size + size - 1) / size * size;
        }
    }

    for (PermKey k : call_perm) {
        out_size_all += k.size;

        uint32_t param_offset = (uint32_t) -1;
        Variable *v = jitc_var(call->outer_out[k.index]);
        if ((v && v->reg_index) || !optimize) {
            param_offset = out_size;
            out_size += k.size;
            out_align = std::max(k.size, out_align);
            n_out_active++;
        }
        call->out_offset[k.index] = (uint32_t) param_offset;
    }

    // =====================================================
    // 3. Compile code for all instances and collapse
    // =====================================================

    using CallablesSet = std::set<XXH128_hash_t, XXH128Cmp>;
    CallablesSet callables_set;

    int32_t alloca_size_backup = alloca_size;
    int32_t alloca_align_backup = alloca_align;
    alloca_size = alloca_align = -1;

    for (size_t i = 0; i < n_inst; ++i) {
        XXH128_hash_t hash =
            jitc_assemble_func(call, (uint32_t) i, in_size, in_align, out_size, out_align);
        call->inst_hash[i] = hash;
        callables_set.insert(hash);
    }

    alloca_size = alloca_size_backup;
    alloca_align = alloca_align_backup;

    if (call->backend == JitBackend::LLVM)
        jitc_var_call_assemble_llvm(call, call_reg, self_reg, mask_reg,
                                    offset_reg, data_reg, out_size, out_align);
    else
        jitc_var_call_assemble_cuda(call, call_reg, self_reg, mask_reg,
                                    offset_reg, data_reg, in_size, in_align,
                                    out_size, out_align);

    // =====================================================
    // 4. Restore previously backed-up JIT state
    // =====================================================

    schedule.clear();
    for (const JitBackupRecord &b : backup) {
        Variable *v = jitc_var(b.sv.index);
        v->param_type = b.param_type;
        v->output_flag = b.output_flag;
        v->reg_index = b.reg_index;
        v->param_offset = b.param_offset;
        schedule.push_back(b.sv);
    }

    // Undo previous change (for more sensible debug out put about buffer sizes)
    if (call->backend == JitBackend::LLVM)
        out_size -= in_size;

    jitc_log(InfoSym,
             "jit_var_call_assemble(): call (\"%s\") to %zu/%u functions, "
             "passing %u/%u inputs (%u/%u bytes), %u/%u outputs (%u/%u bytes), "
             "%zu side effects",
             call->name.c_str(), callables_set.size(), call->n_inst,
             n_in_active, n_in, in_size, in_size_all, n_out_active, n_out,
             out_size, out_size_all, call->side_effects.size());

    jitc_var_inc_ref(call->id);
    calls_assembled.push_back(call);
    call->clear_side_effects();
}

/// Collect scalar / pointer variables referenced by a computation
void jitc_var_call_analyze(CallData *call, uint32_t inst_id, uint32_t index,
                           uint32_t &data_offset) {
    uint64_t key = (uint64_t) index + (((uint64_t) inst_id) << 32);
    auto it_and_status = call->data_map.emplace(key, (uint32_t) -1);
    if (!it_and_status.second)
        return;

    const Variable *v = jitc_var(index);

    if (v->optix)
        call->use_optix |= true;

    VarKind kind = (VarKind) v->kind;

    if (kind == VarKind::CallSelf) {
        call->use_self = true;
    } else if (kind == VarKind::Counter) {
        call->use_index = true;
    } else if (kind == VarKind::ThreadIndex) {
        call->use_thread_id = true;
    } else if (kind == VarKind::CallInput) {
        return;
    } else if (kind == VarKind::Call) {
        CallData *call2 = (CallData *) v->data;
        call->use_self  |= call2->use_self;
        call->use_optix |= call2->use_optix;
        call->use_index |= call2->use_index;
        call->use_thread_id |= call2->use_thread_id;
    } else if (kind == VarKind::LoopCond) {
        LoopData *loop = (LoopData *) jitc_var(v->dep[0])->data;
        for (uint32_t index_2: loop->inner_out)
            jitc_var_call_analyze(call, inst_id, index_2, data_offset);
    } else if (kind == VarKind::TraceRay) {
        TraceData *td = (TraceData *) v->data;
        for (uint32_t index_2: td->indices)
            jitc_var_call_analyze(call, inst_id, index_2, data_offset);
    } else if (v->is_evaluated() || (VarType) v->type == VarType::Pointer) {
        uint32_t tsize = type_size[v->type],
                 offset = (data_offset + tsize - 1) / tsize * tsize;
        it_and_status.first.value() = offset;
        data_offset = offset + tsize;

        if (v->size != 1)
            jitc_raise(
                "jit_var_call(): the virtual function call associated with "
                "instance %u accesses an evaluated variable r%u of type "
                "%s and size %u. However, only *scalar* (size == 1) "
                "evaluated variables can be accessed while recording "
                "virtual function calls",
                inst_id, index, type_name[v->type], v->size);
    }

    for (int i = 0; i < 4; ++i) {
        uint32_t index_2 = v->dep[i];
        if (!index_2)
            break;

        jitc_var_call_analyze(call, inst_id, index_2, data_offset);
    }
}

void jitc_call_upload(ThreadState *ts) {
    for (CallData *call : calls_assembled) {
        uint64_t *data;
        if (ts->backend == JitBackend::CUDA)
            data = (uint64_t *) jitc_malloc(AllocType::HostPinned, call->offset_size);
        else
            data = (uint64_t *) malloc_check(call->offset_size);

        memset(data, 0, call->offset_size);

        for (uint32_t i = 0; i < call->n_inst; ++i) {
            auto it = globals_map.find(GlobalKey(call->inst_hash[i], true));
            if (it == globals_map.end())
                jitc_fail("jitc_call_upload(): could not find callable!");

            // high part: instance data offset, low part: callable index
            data[call->inst_id[i]] =
                (((uint64_t) call->data_offset[i]) << 32) |
                it->second.callable_index;
        }

        jitc_memcpy_async(ts->backend, call->offset, data, call->offset_size);

        // Free call offset table asynchronously
        if (call->backend == JitBackend::CUDA) {
            jitc_free(data);
        } else {
            Task *new_task = task_submit_dep(
                nullptr, &jitc_task, 1, 1,
                [](uint32_t, void *payload) { free(*((void **) payload)); },
                &data, sizeof(void *), nullptr, 1);
            task_release(jitc_task);
            jitc_task = new_task;
        }
    }

    for (CallData *call : calls_assembled)
        jitc_var_dec_ref(call->id);
    calls_assembled.clear();
}

// Compute a permutation to reorder an array of registered pointers
CallBucket *jitc_var_call_reduce(JitBackend backend, const char *domain,
                                 uint32_t index, uint32_t *bucket_count_inout) {

    struct CallReduceRecord {
        CallBucket *buckets;
        uint32_t bucket_count;

        ~CallReduceRecord() {
            for (uint32_t i = 0; i < bucket_count; ++i)
                jitc_var_dec_ref(buckets[i].index);
            jitc_free(buckets);
        }
    };

    VariableExtra *extra = jitc_var_extra(jitc_var(index));
    CallReduceRecord *rec = (CallReduceRecord *) extra->callback_data;
    if (rec) {
        *bucket_count_inout = rec->bucket_count;
        return rec->buckets;
    }

    uint32_t bucket_count;
    if (domain)
        bucket_count = jitc_registry_id_bound(backend, domain);
    else
        bucket_count = *bucket_count_inout;

    if (unlikely(bucket_count == 0)) {
        *bucket_count_inout = 0;
        return nullptr;
    }

    bucket_count++;

    // Ensure input index array is fully evaluated
    void *self = nullptr;
    Ref index_ptr = steal(jitc_var_data(index, true, &self));

    uint32_t size = jitc_var(index)->size;

    if (domain)
        jitc_log(InfoSym, "jit_var_call_reduce(r%u, domain=\"%s\")", index, domain);
    else
        jitc_log(InfoSym, "jit_var_call_reduce(r%u)", index);

    if (jitc_flags() & (uint32_t) JitFlag::Debug) {
        Ref max_idx_v = steal(jitc_var_reduce(backend, VarType::UInt32, ReduceOp::Max, index));
        uint32_t max_idx = 0;
        jitc_var_read(max_idx_v, 0, &max_idx);
        if (max_idx >= bucket_count)
            jitc_raise("jit_var_call_reduce(): out-of-bounds callable ID %u "
                       "(must be < %u).", max_idx, bucket_count);
    }

    size_t perm_size    = (size_t) size * (size_t) sizeof(uint32_t),
           offsets_size = (size_t(bucket_count) * 4 + 1) * sizeof(uint32_t);

    if (backend == JitBackend::LLVM)
        perm_size += jitc_llvm_vector_width * sizeof(uint32_t);

    uint8_t *offsets = (uint8_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::HostPinned : AllocType::Host, offsets_size);
    uint32_t *perm = (uint32_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync, perm_size);

    // Compute permutation
    uint32_t unique_count = jitc_mkperm(backend, (const uint32_t *) self, size,
                                        bucket_count, perm, (uint32_t *) offsets),
             unique_count_out = unique_count;

    // Register permutation variable with JIT backend and transfer ownership
    uint32_t perm_var = jitc_var_mem_map(backend, VarType::UInt32, perm, size, 1);

    Variable v2;
    v2.kind = (uint32_t) VarKind::Evaluated;
    v2.type = (uint32_t) VarType::UInt32;
    v2.backend = (uint32_t) backend;
    v2.dep[3] = perm_var;
    v2.retain_data = true;
    v2.unaligned = 1;

    struct InputBucket {
        uint32_t id, offset, size, unused;
    };

    InputBucket *input_buckets = (InputBucket *) offsets;

    std::sort(
        input_buckets,
        input_buckets + unique_count,
        [](const InputBucket &b1, const InputBucket &b2) {
            return b1.size > b2.size;
        }
    );

    for (uint32_t i = 0; i < unique_count; ++i) {
        InputBucket bucket = input_buckets[i];

        // Create variable for permutation subrange
        v2.data = perm + bucket.offset;
        v2.size = bucket.size;

        jitc_var_inc_ref(perm_var);

        uint32_t index2 = jitc_var_new(v2);

        CallBucket bucket_out;
        if (domain)
            bucket_out.ptr = jitc_registry_ptr(backend, domain, bucket.id);
        else
            bucket_out.ptr = nullptr;

        bucket_out.index = index2;
        bucket_out.id = bucket.id;

        memcpy(input_buckets + i, &bucket_out, sizeof(CallBucket));

        jitc_trace("jit_var_call_reduce(): registered variable %u: bucket %u "
                   "(" DRJIT_PTR ") of size %u.", index2, bucket_out.id,
                   (uintptr_t) bucket_out.ptr, bucket.size);
    }

    jitc_var_dec_ref(perm_var);

    jitc_var_set_callback(
        index,
        [](uint32_t, int free, void *p) {
            if (free)
                delete (CallReduceRecord *) p;
        },
        new CallReduceRecord{ (CallBucket *) offsets, unique_count_out }, true);

    *bucket_count_inout = unique_count_out;
    return (CallBucket *) offsets;
}
