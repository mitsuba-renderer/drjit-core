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

#if defined(DRJIT_ENABLE_CUDA)
#  include "cuda.h"
#endif
#if defined(DRJIT_ENABLE_OPTIX)
#  include "optix.h"
#  include "optix_api.h"
#endif
#if defined(DRJIT_ENABLE_METAL)
#  include "metal.h"
#endif

std::vector<JitIsectBindingExt *> isect_bindings;
bool isect_motion = false;

/// Select the intersection function variant while rendering
struct ScopedIsectMotion {
    bool backup = isect_motion;
    ScopedIsectMotion(bool motion) { isect_motion = motion; }
    ~ScopedIsectMotion() { isect_motion = backup; }
};

static_assert(offsetof(JitIsectBindingExt, record) == 0,
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

/// Recover the function from its handle variable
static IsectFunc *jitc_isect_func(uint32_t index) {
    return (IsectFunc *) state.extra[jitc_var(index)->extra].callback_data;
}

void jitc_isect_begin(JitBackend backend, VarType float_type, uint32_t *in) {
    if (float_type != VarType::Float32 && float_type != VarType::Float64)
        jitc_raise("jit_isect_begin(): the ray inputs must be single or "
                   "double precision floating point arrays!");

    ThreadState *ts = thread_state(backend);

    // Value numbering across the recording boundary would let the body pick
    // up expressions of the surrounding computation, and vice versa
    jitc_new_scope(backend);

    // Symbolic recording session with the call mask on the mask stack, which
    // masks memory accesses of inactive lanes
    ts->isect_record = jitc_record_begin(backend, "jit_isect");

    uint32_t mask = jitc_var_call_mask(backend);
    jitc_var_mask_push(backend, mask);
    jitc_var_dec_ref(mask);

    // The session keeps its own reference so that an input the body does
    // not use stays valid until jit_isect_end()
    for (uint32_t i = 0; i < 9; ++i) {
        uint32_t index = jitc_var_placeholder(
            backend, i < 8 ? float_type : VarType::UInt32);
        jitc_var_inc_ref(index);
        ts->isect_in[i] = in[i] = index;
    }
}

/**
 * Run the capture analysis of an indirect call on the recorded graph. The
 * evaluated variables and pointer literals that the body reads become slots
 * of the data block, laid out once here for the function's lifetime.
 */
static void jitc_isect_analyze(IsectFunc *f) {
    CallData *call = f->call;
    JitBackend backend = call->backend;
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
 * Render the body once, outside of any kernel, to obtain its hash. The hash
 * identifies the function's unit in every kernel, including kernels that
 * were compiled before the function was recorded (frozen function replay
 * after a scene update). jitc_isect_assemble_scene() refreshes it.
 */
static XXH128_hash_t jitc_isect_hash(IsectFunc *f, bool motion) {
    // The assembler state (buffer, unit builders, schedule, ...) is global
    // and only ever touched under 'eval_lock'
    lock_release(state.lock);
    lock_guard guard(state.eval_lock);
    lock_acquire(state.lock);

    // Outside of jitc_eval(), the schedule is empty and the remaining
    // assembler state is reset at the start of the next kernel. The guards
    // release the schedule of the body once it is rendered.
    ScopedScheduleBackup schedule_backup;
    ScopedAllocaBackup alloca_backup;
    jitc_unit_reset();
    buffer.clear();

    struct ScopedOptixFlag {
        bool backup = uses_optix;
        ~ScopedOptixFlag() { uses_optix = backup; }
    } optix_flag;
    uses_optix = f->call->backend == JitBackend::CUDA;

    ScopedIsectMotion motion_scope(motion);
    return jitc_assemble_func(f->call, 0, 0, 0, 0, 0);
}

uint32_t jitc_isect_end(JitBackend backend, const char *name,
                        const uint32_t *out) {
    const uint32_t n_in = 9, n_out = 4;
    ThreadState *ts = thread_state(backend);
    const uint32_t *in = ts->isect_in;

    // Close the session, discarding any side effects that were queued
    jitc_var_mask_pop(backend);
    bool side_effects = ts->side_effects_symbolic.size() !=
                        (ts->isect_record & 0x7fffffffu);
    jitc_record_end(backend, ts->isect_record, /*cleanup=*/1);
    jitc_new_scope(backend);

    struct ReleaseInputs {
        const uint32_t *in;
        ~ReleaseInputs() {
            for (uint32_t i = 0; i < 9; ++i)
                jitc_var_dec_ref(in[i]);
        }
    } release_inputs { in };

    if (!out)
        return 0;

    if (side_effects)
        jitc_raise("jit_isect_end(\"%s\"): the recorded computation performs "
                   "a side effect (e.g. a scatter), which is not allowed "
                   "inside an intersection function.", name);

    VarType float_type = (VarType) jitc_var(in[0])->type;
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
        jitc_var_inc_ref(in[i]);
        call->inner_in.push_back(in[i]);
    }

    for (uint32_t j = 0; j < n_out; ++j) {
        jitc_var_inc_ref(out[j]);
        call->inner_out.push_back(out[j]);
    }

    jitc_isect_analyze(f.get());

    const char *what = nullptr;
    if (call->use_nested)
        what = "a nested function call";
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

    f->hash[0] = jitc_isect_hash(f.get(), false);

    uint32_t index = jitc_var_new_node_0(backend, VarKind::Nop, VarType::Void,
                                         1, 0, (uintptr_t) f.get());
    jitc_var_set_callback(index, jitc_isect_func_free, f.get(), true);
    f->id = index;

    jitc_log(InfoSym, "jit_isect_end(\"%s\"): function r%u with %zu data "
             "slot%s (%u bytes), hash %016llx%016llx", name, index,
             call->slots.size(), call->slots.size() == 1 ? "" : "s",
             call->data_size, (unsigned long long) f->hash[0].high64,
             (unsigned long long) f->hash[0].low64);

    f.release();
    return index;
}

// ============================================================================
//  Bindings
// ============================================================================

static void jitc_isect_binding_free(JitIsectBindingExt *b) {
    jitc_free(b->record.data);
    if (b->scene_var)
        jitc_var_dec_ref(b->scene_var);
    jitc_var_dec_ref(b->func->id);
    jitc_free(b);
}

/// Allocate and fill a binding's data block from the captured variables
static void jitc_isect_fill(JitIsectBindingExt *b) {
    IsectFunc *f = b->func;
    CallData *call = f->call;
    JitBackend backend = call->backend;

    uint32_t size = call->data_size,
             n = (uint32_t) call->slots.size();
    if (!size)
        return;

    // Pending scatters into captured variables must land before the copy
    bool dirty = false;
    for (const CallData::CaptureSlot &slot : call->slots)
        dirty |= jitc_var(slot.ref)->is_dirty();
    if (dirty)
        jitc_eval(thread_state(backend));

    void *block = jitc_malloc(backend, size, /*shared=*/backend == JitBackend::LLVM);
    b->record.data = block;

    if (!n)
        return;

    AggregationEntry *agg = (AggregationEntry *) jitc_malloc(
        backend, sizeof(AggregationEntry) * n, /*shared=*/true);
    AggregationEntry *p = agg;

    for (const CallData::CaptureSlot &slot : call->slots)
        jitc_call_capture_entry(p++, jitc_var(slot.ref), slot.offset);

    jitc_aggregate(backend, block, agg, n);
    jitc_free(agg);
}

JitIsectBinding *jitc_isect_bind(uint32_t func, uintptr_t scene,
                                 uint32_t record_index, void *user) {
    IsectFunc *f = jitc_isect_func(func);
    JitBackend backend = f->call->backend;

    // On the GPU backends, the key is the variable that the trace operations
    // reference (the shader binding table on CUDA, the scene on Metal)
    uint32_t scene_var = 0;
    bool motion = false;
    if (backend == JitBackend::CUDA) {
        Variable *sbt = jitc_var((uint32_t) scene);
        scene_var = (uint32_t) scene;
        scene = (uintptr_t) sbt->literal;
        jitc_var_inc_ref(scene_var, sbt);
    }
#if defined(DRJIT_ENABLE_METAL)
    else if (backend == JitBackend::Metal) {
        // The scene's cleanup callback releases its bindings, so the binding
        // must not keep the scene variable alive
        MetalScene *s = jitc_metal_get_scene((uint32_t) scene);
        if (record_index >= s->isect_count)
            jitc_raise("jit_isect_bind(): table entry %u is out of bounds "
                       "(the scene has %u entries)!", record_index,
                       s->isect_count);
        scene = (uintptr_t) s;

        // The intersection functions of a scene with instance motion
        // carry a matching tag, which the function's static variant lacks
        motion = (s->geometry_types_mask & 0x10u) != 0;
        if (motion && !f->hash[1].low64 && !f->hash[1].high64)
            f->hash[1] = jitc_isect_hash(f, true);
    }
#endif

    // The binding (and the data block, see jitc_isect_fill) may still be
    // read by kernels in flight when it is released. A 'shared' allocation
    // is freed only once those kernels have completed.
    JitIsectBindingExt *b = (JitIsectBindingExt *) jitc_malloc(
        backend, sizeof(JitIsectBindingExt), /*shared=*/true);
    memset(b, 0, sizeof(JitIsectBindingExt));
    b->record.user = user;
    b->func = f;
    b->scene = scene;
    b->scene_var = scene_var;
    b->record_index = record_index;
    b->motion = motion;
    jitc_var_inc_ref(f->id);

    try {
        jitc_isect_fill(b);
    } catch (...) {
        jitc_isect_binding_free(b);
        throw;
    }

#if defined(DRJIT_ENABLE_METAL)
    // The intersection function reads its data block through this table
    if (backend == JitBackend::Metal) {
        MetalScene *s = (MetalScene *) scene;
        s->isect_table[record_index] = (uint64_t) (uintptr_t) b->record.data;
        jitc_metal_scene_release_ifts(s);
    }
#endif

    b->list_index = (uint32_t) isect_bindings.size();
    isect_bindings.push_back(b);
    jitc_log(Debug, "jit_isect_bind(): bound function \"%s\" to scene "
             DRJIT_PTR, f->call->name.c_str(), scene);
    return &b->record;
}

void jitc_isect_unbind(JitIsectBinding *record) {
    JitIsectBindingExt *b = (JitIsectBindingExt *) record;
    uint32_t i = b->list_index;
    if (i >= isect_bindings.size() || isect_bindings[i] != b)
        jitc_raise("jit_isect_unbind(): unknown binding " DRJIT_PTR "!",
                   (uintptr_t) record);
    isect_bindings[i] = isect_bindings.back();
    isect_bindings[i]->list_index = i;
    isect_bindings.pop_back();

#if defined(DRJIT_ENABLE_METAL)
    if (b->func->call->backend == JitBackend::Metal)
        jitc_metal_scene_release_ifts((MetalScene *) b->scene);
#endif

    jitc_isect_binding_free(b);
}

// ============================================================================
//  Kernel assembly and launch
// ============================================================================

void jitc_isect_assemble_scene(JitBackend backend, uintptr_t scene) {
    for (JitIsectBindingExt *b : isect_bindings) {
        IsectFunc *f = b->func;
        XXH128_hash_t &known = f->hash[b->motion];
        if (f->call->backend != backend || (scene && b->scene != scene) ||
            jitc_unit_callable_known(known))
            continue;

        ScopedScheduleBackup schedule_backup;
        ScopedAllocaBackup alloca_backup;
        ScopedIsectMotion motion_scope(b->motion);
        XXH128_hash_t hash = jitc_assemble_func(f->call, 0, 0, 0, 0, 0);

        // The body renders differently when code generation options (JIT
        // flags, log level) changed since the recording. Kernels compiled
        // before the change no longer resolve the function, later ones use
        // the new hash.
        if (!(hash == known)) {
            jitc_log(Debug, "jit_isect: the body of \"%s\" changed to "
                     "%016llx%016llx", f->call->name.c_str(),
                     (unsigned long long) hash.high64,
                     (unsigned long long) hash.low64);
            known = hash;
        }
    }
}

const KernelIsectUnit *jitc_isect_find_unit(const Kernel &kernel,
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

    for (JitIsectBindingExt *b : isect_bindings) {
        if (b->func->call->backend != backend)
            continue;

        const KernelIsectUnit *u = jitc_isect_find_unit(kernel, b->func->hash[0]);
        if (!u)
            continue;

        if (backend == JitBackend::LLVM)
            b->record.code = u->handle;
#if defined(DRJIT_ENABLE_OPTIX)
        else if (backend == JitBackend::CUDA &&
                 b->scene == (uintptr_t) optix_sbt && b->optix_pg != u->handle) {
            // Write the header of the unit's program group and the data
            // block pointer into the binding's hit record. The table base
            // is read at every launch because the application may
            // reallocate the table between launches.
            const OptixShaderBindingTable *sbt =
                (const OptixShaderBindingTable *) b->scene;
            uint8_t *rec = (uint8_t *) sbt->hitgroupRecordBase +
                           (size_t) b->record_index * sbt->hitgroupRecordStrideInBytes;

            // The copy stages pageable host memory before returning
            alignas(16) uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
            jitc_optix_check(optixSbtRecordPackHeader(
                (OptixProgramGroup) u->handle, header));
            cuda_check(cuMemcpyAsync(rec, header, sizeof(header),
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

void jitc_isect_invalidate() {
    for (JitIsectBindingExt *b : isect_bindings) {
        b->record.code = nullptr;
        b->optix_pg = nullptr;
    }
}

void jitc_isect_shutdown() {
    size_t n_bindings = isect_bindings.size();

    for (JitIsectBindingExt *b : isect_bindings)
        jitc_isect_binding_free(b);

    if (n_bindings && state.leak_warnings)
        jitc_log(Warn, "jit_shutdown(): leaked %zu intersection binding%s!",
                 n_bindings, n_bindings == 1 ? "" : "s");

    isect_bindings.clear();
}
