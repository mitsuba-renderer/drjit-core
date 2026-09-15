/*
    src/isect.cpp -- Custom intersection functions for ray tracing backends

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "isect.h"
#include "call.h"
#include "eval.h"
#include "io.h"
#include "log.h"
#include "malloc.h"
#include "unit.h"
#include "util.h"
#include "var.h"
#include <algorithm>

#if defined(DRJIT_ENABLE_CUDA)
#  include "cuda.h"
#endif
#if defined(DRJIT_ENABLE_OPTIX)
#  include "optix.h"
#  include "optix_api.h"
#endif

/// Live bindings
static std::vector<IsectBinding *> isect_bindings;

/// Recording session of the calling thread (see jit_isect_begin). 'record'
/// is the symbolic side-effect queue length at the start with the previous
/// 'SymbolicScope' flag in the top bit, as jit_record_begin() encodes it.
struct IsectSession {
    bool active = false;
    JitBackend backend;
    uint32_t record;
    uint32_t in[9];
};

static thread_local IsectSession isect_session;

static_assert(offsetof(IsectBinding, record) == 0,
              "The binding record must be the first member");

IsectFunc::~IsectFunc() { delete call; }

static bool operator==(const XXH128_hash_t &a, const XXH128_hash_t &b) {
    return a.low64 == b.low64 && a.high64 == b.high64;
}

// ============================================================================
//  Recording
// ============================================================================

/// Free callback of the handle variable
static void jitc_isect_func_free(uint32_t, int free, void *data) {
    if (!free)
        return;
    IsectFunc *f = (IsectFunc *) data;
    jitc_log(Debug, "jit_isect: releasing function \"%s\"", f->call->name.c_str());
    delete f;
}

static IsectFunc *jitc_isect_func(uint32_t index) {
    Variable *v = jitc_var(index);
    VariableExtra *extra = v->extra ? &state.extra[v->extra] : nullptr;
    if ((VarType) v->type != VarType::Void || !extra ||
        extra->callback != jitc_isect_func_free)
        jitc_raise("jit_isect_bind(): r%u is not an intersection function "
                   "handle!", index);
    return (IsectFunc *) extra->callback_data;
}

void jitc_isect_begin(JitBackend backend, VarType float_type, uint32_t *in) {
    IsectSession &s = isect_session;

    if (s.active)
        jitc_raise("jit_isect_begin(): a recording session is already active "
                   "on this thread!");
    if (backend != JitBackend::LLVM && backend != JitBackend::CUDA)
        jitc_raise("jit_isect_begin(): intersection functions are only "
                   "supported on the LLVM and CUDA backends!");
    if (float_type != VarType::Float32 && float_type != VarType::Float64)
        jitc_raise("jit_isect_begin(): the ray inputs must be single or "
                   "double precision floating point arrays!");

    ThreadState *ts = thread_state(backend);

    // Value numbering across the recording boundary would let the body pick
    // up expressions of the surrounding computation, and vice versa
    jitc_new_scope(backend);

    // Symbolic recording session with the call mask on the mask stack, which
    // masks memory accesses of inactive lanes (mirrors jit_record_begin,
    // which cannot be reused here since the lock is already held)
    ts->record_stack.push_back("jit_isect");
    s.record = (uint32_t) ts->side_effects_symbolic.size();
    if (jitc_flag(JitFlag::SymbolicScope))
        s.record |= 0x80000000u;
    jitc_set_flag(JitFlag::SymbolicScope, true);

    uint32_t mask = jitc_var_call_mask(backend);
    jitc_var_mask_push(backend, mask);
    jitc_var_dec_ref(mask);

    // The session keeps its own reference so that an input the body does
    // not use stays valid until jit_isect_end()
    for (uint32_t i = 0; i < 9; ++i) {
        uint32_t index = jitc_var_placeholder(
            backend, i < 8 ? float_type : VarType::UInt32);
        jitc_var_inc_ref(index);
        s.in[i] = in[i] = index;
    }

    s.backend = backend;
    s.active = true;
}

/**
 * Run the capture analysis of the recorded graph, exactly like an indirect
 * call with one instance: evaluated size-1 variables and pointer literals
 * reachable from the outputs become slots of the data block, and the analysis
 * recurses through the remaining (symbolic or lazily-computed) nodes to reach
 * them. The analysis runs once, when the function is created, so the slot
 * layout is fixed for the function's lifetime.
 */
static void jitc_isect_analyze(IsectFunc *f) {
    CallData *call = f->call;
    JitBackend backend = f->backend;
    ThreadState *ts = thread_state(backend);

    jitc_visit_new_gen();
    for (uint32_t j = 0; j < call->n_out; ++j)
        jitc_var_call_analyze(call, 0, call->inner_out[j]);

    uint32_t data_size = 0, align, llvm_pkt_cap;
    jitc_call_layout_params(ts, align, llvm_pkt_cap);

    std::vector<CallData::CaptureSlot> reordered;
    jitc_call_layout_instance(call, 0, (uint32_t) call->slots.size(), backend,
                              align, llvm_pkt_cap, data_size, reordered);
    call->data_size = align_up(data_size, align);
}

/**
 * Render the body once, outside of any kernel, to obtain its hash. The
 * hash is what identifies the function's unit in a kernel, so bindings can
 * be resolved against kernels that were compiled before the function was
 * recorded (frozen function replay after a scene update).
 */
static void jitc_isect_hash(IsectFunc *f) {
    // The assembler state (buffer, unit builders, schedule, ...) is global
    // and only ever touched under 'eval_lock'
    lock_release(state.lock);
    lock_guard guard(state.eval_lock);
    lock_acquire(state.lock);

    // A standalone assembly outside of jitc_eval() must leave the kernel
    // assembler state untouched. The global 'schedule' holds the caller's
    // pending roots, some of which may already be freed, so it is preserved
    // opaquely (swapped out and back) rather than through ScopedScheduleBackup,
    // which dereferences every entry. jitc_assemble_func() clears 'schedule'
    // at entry and exit, so it operates on a clean slate.
    struct ScopedAssembler {
        std::vector<ScheduledVariable> schedule_backup;
        bool uses_optix_backup;

        ScopedAssembler(JitBackend backend) : uses_optix_backup(uses_optix) {
            schedule_backup.swap(schedule);
            jitc_unit_reset();
            buffer.clear();
            uses_optix = backend == JitBackend::CUDA;
        }

        ~ScopedAssembler() {
            uses_optix = uses_optix_backup;
            jitc_unit_reset();
            buffer.clear();
            schedule_backup.swap(schedule);
        }
    } assembler(f->backend);

    ScopedAllocaBackup alloca_backup;
    f->hash = jitc_assemble_func(f->call, 0, 0, 0, 0, 0);
}

uint32_t jitc_isect_end(const char *name, const uint32_t *out) {
    IsectSession &s = isect_session;
    const uint32_t n_in = 9, n_out = 4;

    if (!s.active)
        jitc_raise("jit_isect_end(): no recording session is active on this "
                   "thread!");
    if (!name)
        name = "isect";

    s.active = false;
    JitBackend backend = s.backend;
    ThreadState *ts = thread_state(backend);

    // Close the session (mirrors jit_record_end with cleanup)
    jitc_var_mask_pop(backend);
    uint32_t checkpoint = s.record & 0x7fffffffu;
    std::vector<uint32_t> &se = ts->side_effects_symbolic;
    bool side_effects = se.size() != checkpoint;
    ts->record_stack.pop_back();
    jitc_set_flag(JitFlag::SymbolicScope, (s.record & 0x80000000u) != 0);
    while (checkpoint < se.size()) {
        jitc_var_dec_ref(se.back());
        se.pop_back();
    }
    jitc_new_scope(backend);

    struct ReleaseInputs {
        const uint32_t *in;
        ~ReleaseInputs() {
            for (uint32_t i = 0; i < 9; ++i)
                jitc_var_dec_ref(in[i]);
        }
    } release_inputs { s.in };

    if (!out)
        return 0;

    if (side_effects)
        jitc_raise("jit_isect_end(\"%s\"): the recorded computation performs "
                   "a side effect (e.g. a scatter), which is not allowed "
                   "inside an intersection function.", name);

    VarType float_type = (VarType) jitc_var(s.in[0])->type;
    VarType out_types[n_out] = { VarType::Bool, float_type, VarType::UInt32,
                                 VarType::UInt32 };
    for (uint32_t j = 0; j < n_out; ++j) {
        if (!out[j])
            jitc_raise("jit_isect_end(\"%s\"): output %u is uninitialized!",
                       name, j);
        const Variable *v = jitc_var(out[j]);
        if ((VarType) v->type != out_types[j])
            jitc_raise("jit_isect_end(\"%s\"): output %u (r%u) has type %s, "
                       "expected %s!", name, j, out[j], type_name[v->type],
                       type_name[(int) out_types[j]]);
        if ((JitBackend) v->backend != backend)
            jitc_raise("jit_isect_end(\"%s\"): output %u (r%u) belongs to "
                       "a different backend!", name, j, out[j]);
        if (v->size != 1)
            jitc_raise("jit_isect_end(\"%s\"): output %u (r%u) has size %u "
                       "(must be 1)!", name, j, out[j], v->size);
    }

    // The body is a single-instance call without a call site. Besides the
    // graph, jitc_assemble_func() reads 'checkpoints' (no side effects) and
    // 'out_offset' (every output is used).
    std::unique_ptr<IsectFunc> f(new IsectFunc());
    f->backend = backend;
    f->call = new CallData();

    CallData *call = f->call;
    call->backend = backend;
    call->name = name;
    call->optimize = true;
    call->isect = true;
    call->n_in = n_in;
    call->n_out = n_out;
    call->n_inst = 1;
    call->checkpoints = { 0, 0 };
    call->out_offset.assign(n_out, 0);

    for (uint32_t i = 0; i < n_in; ++i) {
        jitc_var_inc_ref(s.in[i]);
        call->inner_in.push_back(s.in[i]);
    }

    for (uint32_t j = 0; j < n_out; ++j) {
        jitc_var_inc_ref(out[j]);
        call->inner_out.push_back(out[j]);
    }

    jitc_isect_analyze(f.get());

    const char *what = nullptr;
    if (call->use_nested)
        what = "a nested function call";
    else if (call->use_outer)
        what = "an input of an enclosing symbolic call";
    else if (call->use_self)
        what = "the 'self' pointer of an enclosing call";
    else if (call->use_trace)
        what = "a ray tracing operation";
    else if (call->use_index)
        what = "the array index (dr.arange / dr.counter)";
    else if (call->use_thread_id)
        what = "the thread index";

    if (what)
        jitc_raise("jit_isect_end(\"%s\"): the recorded computation uses "
                   "%s, which is not supported inside an intersection "
                   "function.", name, what);

    jitc_isect_hash(f.get());

    uint32_t index = jitc_var_new_node_0(backend, VarKind::Nop, VarType::Void,
                                         1, 0, (uintptr_t) f.get());
    jitc_var_set_callback(index, jitc_isect_func_free, f.get(), true);
    f->id = index;

    jitc_log(InfoSym, "jit_isect_end(\"%s\"): function r%u with %zu data "
             "slot%s (%u bytes), hash %016llx%016llx", name, index,
             call->slots.size(), call->slots.size() == 1 ? "" : "s",
             call->data_size, (unsigned long long) f->hash.high64,
             (unsigned long long) f->hash.low64);

    f.release();
    return index;
}

// ============================================================================
//  Bindings
// ============================================================================

static void jitc_isect_binding_free(IsectBinding *b) {
    jitc_free(b->record.data);
    if (b->sbt_var)
        jitc_var_dec_ref(b->sbt_var);
    jitc_var_dec_ref(b->func->id);
    jitc_free(b);
}

/// Allocate and fill a binding's data block from the captured variables
static void jitc_isect_fill(IsectBinding *b) {
    IsectFunc *f = b->func;
    CallData *call = f->call;
    JitBackend backend = f->backend;

    uint32_t size = call->data_size,
             n = (uint32_t) call->slots.size();
    if (!size)
        return;

    // Pending scatters into captured variables must land before the copy
    bool dirty = false;
    for (const CallData::CaptureSlot &slot : call->slots) {
        const Variable *v = jitc_var(slot.ref);
        dirty |= v && v->is_dirty();
    }
    if (dirty)
        jitc_eval(thread_state(backend));

    void *block = jitc_malloc(backend, size, /*shared=*/backend == JitBackend::LLVM);
    b->record.data = block;

    if (!n)
        return;

    AggregationEntry *agg = (AggregationEntry *) jitc_malloc(
        backend, sizeof(AggregationEntry) * n, /*shared=*/true);
    AggregationEntry *p = agg;

    for (const CallData::CaptureSlot &slot : call->slots) {
        const Variable *v = jitc_var(slot.ref);
        if (!v)
            jitc_fail("jit_isect_bind(): captured variable r%u no longer "
                      "exists!", slot.ref.index);
        jitc_call_capture_entry(p++, v, slot.offset);
    }

    jitc_aggregate(backend, block, agg, n);
    jitc_free(agg);
}

JitIsectBinding *jitc_isect_bind(uint32_t func, uintptr_t scene,
                                 uint32_t record_index, uint32_t flags,
                                 void *user) {
    IsectFunc *f = jitc_isect_func(func);
    JitBackend backend = f->backend;
    if (!scene)
        jitc_raise("jit_isect_bind(): the scene key must be nonzero!");

    // On CUDA, the key is the shader binding table that the trace operations
    // reference, and the launch recovers it from the thread state
    uint32_t sbt_var = 0;
    if (backend == JitBackend::CUDA) {
        Variable *sbt = jitc_var((uint32_t) scene);
        if ((VarType) sbt->type != VarType::Void)
            jitc_raise("jit_isect_bind(): the scene key must be the index of "
                       "the shader binding table variable!");
        sbt_var = (uint32_t) scene;
        scene = (uintptr_t) sbt->literal;
        jitc_var_inc_ref(sbt_var, sbt);
    }

    // The binding (and the data block, see jitc_isect_fill) may still be
    // read by kernels in flight when it is released. A 'shared' allocation
    // is freed only once those kernels have completed.
    IsectBinding *b = (IsectBinding *) jitc_malloc(backend, sizeof(IsectBinding),
                                                   /*shared=*/true);
    memset(b, 0, sizeof(IsectBinding));
    b->record.flags = flags;
    b->record.user = user;
    b->func = f;
    b->scene = scene;
    b->sbt_var = sbt_var;
    b->record_index = record_index;
    jitc_var_inc_ref(f->id);

    try {
        jitc_isect_fill(b);
    } catch (...) {
        jitc_isect_binding_free(b);
        throw;
    }

    isect_bindings.push_back(b);
    jitc_log(Debug, "jit_isect_bind(): bound function \"%s\" to scene "
             DRJIT_PTR, f->call->name.c_str(), scene);
    return &b->record;
}

void jitc_isect_unbind(JitIsectBinding *record) {
    IsectBinding *b = (IsectBinding *) record;
    auto it = std::find(isect_bindings.begin(), isect_bindings.end(), b);
    if (it == isect_bindings.end())
        jitc_raise("jit_isect_unbind(): unknown binding " DRJIT_PTR "!",
                   (uintptr_t) record);
    *it = isect_bindings.back();
    isect_bindings.pop_back();
    jitc_isect_binding_free(b);
}

// ============================================================================
//  Kernel assembly and launch
// ============================================================================

void jitc_isect_assemble_scene(JitBackend backend, uintptr_t scene) {
    for (IsectBinding *b : isect_bindings) {
        IsectFunc *f = b->func;
        if (f->backend != backend || (scene && b->scene != scene) ||
            jitc_unit_callable_known(f->hash))
            continue;

        ScopedScheduleBackup schedule_backup;
        ScopedAllocaBackup alloca_backup;
        XXH128_hash_t hash = jitc_assemble_func(f->call, 0, 0, 0, 0, 0);

        if (!(hash == f->hash))
            jitc_fail("jit_isect: the intersection function \"%s\" rendered "
                      "to a different body (%016llx%016llx) than when it was "
                      "recorded (%016llx%016llx). The code generator state "
                      "must not change between the two.",
                      f->call->name.c_str(),
                      (unsigned long long) hash.high64,
                      (unsigned long long) hash.low64,
                      (unsigned long long) f->hash.high64,
                      (unsigned long long) f->hash.low64);
    }
}

/// Find the kernel's unit for a body hash, or null
static const KernelIsectUnit *jitc_isect_find_unit(const Kernel &kernel,
                                                   XXH128_hash_t hash) {
    for (uint32_t i = 0; i < kernel.isect_count; ++i)
        if (kernel.isect[i].hash == hash)
            return &kernel.isect[i];
    return nullptr;
}

void jitc_isect_launch(JitBackend backend, const Kernel &kernel,
                       void *optix_sbt, void *stream) {
#if !defined(DRJIT_ENABLE_OPTIX)
    (void) optix_sbt; (void) stream;
#endif

    for (IsectBinding *b : isect_bindings) {
        if (b->func->backend != backend)
            continue;

        const KernelIsectUnit *u = jitc_isect_find_unit(kernel, b->func->hash);
        if (!u)
            continue;

        if (backend == JitBackend::LLVM) {
            b->record.code = u->handle;
        }
#if defined(DRJIT_ENABLE_OPTIX)
        else if (backend == JitBackend::CUDA &&
                 b->scene == (uintptr_t) optix_sbt && b->optix_pg != u->handle) {
            // Point the launched table's hit record at the unit's program
            // group and at the data block. The table base is read here since
            // the application may reallocate it between launches; kernels in
            // flight keep the old record until the queued copies land.
            const OptixShaderBindingTable *sbt =
                (const OptixShaderBindingTable *) b->scene;
            uint8_t *rec = (uint8_t *) sbt->hitgroupRecordBase +
                           (size_t) b->record_index * sbt->hitgroupRecordStrideInBytes;

            jitc_optix_check(optixSbtRecordPackHeader(
                (OptixProgramGroup) u->handle, b->optix_header));
            cuda_check(cuMemcpyAsync(rec, b->optix_header,
                                     sizeof(b->optix_header),
                                     (CUstream) stream));
            cuda_check(cuMemcpyAsync(rec + OPTIX_SBT_RECORD_HEADER_SIZE +
                                         sizeof(uint32_t) * 2,
                                     &b->record.data, sizeof(void *),
                                     (CUstream) stream));
            b->optix_pg = u->handle;
        }
#endif
    }
}

bool jitc_isect_check(JitBackend backend, const Kernel &kernel) {
    for (uint32_t i = 0; i < kernel.isect_count; ++i) {
        bool found = false;
        for (IsectBinding *b : isect_bindings) {
            if (b->func->backend == backend &&
                b->func->hash == kernel.isect[i].hash) {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    return true;
}

void jitc_isect_invalidate() {
    for (IsectBinding *b : isect_bindings) {
        b->record.code = nullptr;
        b->optix_pg = nullptr;
    }
}

void jitc_isect_shutdown() {
    size_t n_bindings = isect_bindings.size();

    for (IsectBinding *b : isect_bindings)
        jitc_isect_binding_free(b);

    if (n_bindings && state.leak_warnings)
        jitc_log(Warn, "jit_shutdown(): leaked %zu intersection binding%s!",
                 n_bindings, n_bindings == 1 ? "" : "s");

    isect_bindings.clear();
}
