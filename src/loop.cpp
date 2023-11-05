#include "internal.h"
#include "var.h"
#include "loop.h"
#include "log.h"
#include "eval.h"
#include "op.h"

uint32_t jitc_var_loop_start(const char *name, size_t n_indices, uint32_t *indices) {
    JitBackend backend = JitBackend::None;
    bool symbolic = false, dirty = false;
    uint32_t size = 0;

    // A few sanity checks
    if (!n_indices)
        jitc_raise("jit_var_loop_start(): attempted to record a symbolic loop "
                   "without state variables.");

    for (size_t i = 0; i < n_indices; ++i) {
        uint32_t index = indices[i];
        if (!index)
            jitc_raise("jit_var_loop_start(): loop state variable %zu is "
                       "uninitialized (i.e., it has size 0).", i);

        const Variable *v2 = jitc_var(index);
        if (i == 0) {
            backend = (JitBackend) v2->backend;
            size = v2->size;
            symbolic = v2->symbolic;
            dirty = v2->is_dirty();
        } else {
            if ((JitBackend) v2->backend != backend)
                jitc_raise(
                    "jit_var_loop_start(): the loop state involves variables with "
                    "different Dr.Jit backends, which is not permitted.");

            if (v2->size != size && size != 1 && v2->size != 1)
                jitc_raise("jit_var_loop_init(): loop state variable %zu (r%u) has "
                           "an initial shape (size %u) that is incompatible with "
                           "that of the loop (size %u).", i, index, v2->size, size);

            symbolic |= v2->symbolic;
            dirty |= v2->is_dirty();
            size = std::max(size, v2->size);
        }
    }

    // Ensure side effects are fully processed
    if (dirty) {
        jitc_eval(thread_state(backend));
        dirty = false;
        for (size_t i = 0; i < n_indices; ++i)
            dirty |= jitc_var(indices[i])->is_dirty();
        if (dirty)
            jitc_raise("jit_var_loop_start(): inputs remain dirty after evaluation!");
    }

    Ref loop_start;
    {
        Variable v;
        v.kind = (uint32_t) VarKind::LoopStart;
        v.type = (uint32_t) VarType::Void;
        v.size = size;
        v.backend = (uint32_t) backend;
        v.symbolic = 1;
        v.extra = 1;

        jitc_new_scope(backend);
        loop_start = steal(jitc_var_new(v, false));
        jitc_new_scope(backend);

    }

    if (!name)
        name = "unnamed";

    std::unique_ptr<LoopData> ld(new LoopData(name, loop_start, n_indices, symbolic));
    state.extra[loop_start].callback_data = ld.get();
    loop_start.release();

    Variable v_phi;
    v_phi.kind = (uint32_t) VarKind::LoopPhi;
    v_phi.backend = (uint32_t) backend;
    v_phi.symbolic = 1;
    v_phi.size = size;
    v_phi.dep[0] = ld->loop_start;

    for (size_t i = 0; i < n_indices; ++i) {
        uint32_t index = indices[i];
        Variable *v2 = jitc_var(index);
        jitc_var_inc_ref(index, v2);
        ld->outer_inputs.push_back(index);

        v_phi.type = v2->type;
        v_phi.literal = (uint64_t) i;
        jitc_var_inc_ref(ld->loop_start);
        uint32_t index_new = jitc_var_new(v_phi, false);
        ld->inner_inputs.push_back(index_new);
        jitc_var_inc_ref(index_new);
        indices[i] = index_new;
    }

    jitc_new_scope(backend);

    // Construct a dummy variable that keeps 'ld' alive until the loop is fully constructed
    Variable v;
    v.kind = (uint32_t) VarKind::Nop;
    v.type = (uint32_t) VarType::Void;
    v.size = 1;
    v.backend = (uint32_t) backend;
    v.extra = 1;
    Ref loop_holder = steal(jitc_var_new(v, false));

    Extra &e = state.extra[loop_holder];
    e.callback = [](uint32_t, int free, void *p) {
        if (free)
            delete (LoopData *) p;
    };
    e.callback_internal = true;
    e.callback_data = ld.release();

    return loop_holder.release();
}

uint32_t jitc_var_loop_cond(uint32_t loop, uint32_t active) {
    LoopData *ld = (LoopData *) state.extra[loop].callback_data;

    Variable *loop_start_v = jitc_var(ld->loop_start),
             *active_v = jitc_var(active);

    if ((VarType) active_v->type != VarType::Bool)
        jitc_raise("jit_var_loop_cond(): loop condition must be a boolean variable");
    if (!active_v->symbolic)
        jitc_raise("jit_var_loop_cond(): loop condition does not depend on any of the loop variables");

    Variable v;
    v.kind = (uint32_t) VarKind::LoopCond;
    v.type = (uint32_t) VarType::Void;
    v.size = std::max(loop_start_v->size, active_v->size);
    v.backend = active_v->backend;
    v.dep[0] = ld->loop_start;
    v.dep[1] = active;
    v.symbolic = 1;
    jitc_var_inc_ref(ld->loop_start, loop_start_v);
    jitc_var_inc_ref(active, active_v);

    JitBackend backend = (JitBackend) active_v->backend;
    jitc_new_scope(backend);
    uint32_t cond = jitc_var_new(v, false);
    jitc_new_scope(backend);
    return cond;
}

bool jitc_var_loop_end(uint32_t loop, uint32_t cond, uint32_t *indices) {
    LoopData *ld = (LoopData *) state.extra[loop].callback_data;

    JitBackend backend;
    uint32_t size;

    if ((jitc_flags() & (uint32_t) JitFlag::OptimizeLoops) && !ld->retry) {
        size_t n_eliminated = 0;
        for (size_t i = 0; i < ld->size; ++i) {
            bool eliminate = false;

            if (indices[i] == ld->inner_inputs[i]) {
                eliminate = true;
            } else {
                const Variable *v1 = jitc_var(ld->outer_inputs[i]),
                               *v2 = jitc_var(indices[i]);

                eliminate = v1->is_literal() && v2->is_literal() &&
                            v1->literal == v2->literal;
            }

            if (eliminate) {
                jitc_var_inc_ref(ld->outer_inputs[i]);
                jitc_var_dec_ref(ld->inner_inputs[i]);
                ld->inner_inputs[i] = ld->outer_inputs[i];
                n_eliminated++;
            }
        }

        if (n_eliminated > 0) {
            for (size_t i = 0; i < ld->size; ++i)
                indices[i] = ld->inner_inputs[i];
            jitc_log(Debug,
                     "jit_var_loop(r%u): re-recording to eliminate %zu/%zu constant "
                     "loop state variables.", ld->loop_start, n_eliminated, ld->size);
            ld->retry = true;
            return false;
        }
    }

    {
        Variable *cond_v = jitc_var(cond);
        size = cond_v->size;
        backend = (JitBackend) cond_v->backend;

        uint32_t active = cond_v->dep[1];
        for (size_t i = 0; i < ld->size; ++i) {
            uint32_t index = indices[i];
            if (!index)
                jitc_raise(
                    "jit_var_loop_end(): loop state variable %zu has become "
                    "uninitialized (i.e., it now has size 0)", i);

            Variable *v2 = jitc_var(index);

            if (v2->size != size && size != 1 && v2->size != 1)
                jitc_raise(
                    "jit_var_loop_init(): loop state variable %zu (r%u) has "
                    "a final shape (size %u) that is incompatible with "
                    "that of the loop (size %u).",
                    i, index, v2->size, size);

            size = std::max(v2->size, size);

            if (ld->inner_inputs[i] != ld->outer_inputs[i]) {
                ld->inner_outputs.push_back(
                    jitc_var_select(active, index, ld->inner_inputs[i]));
            } else {
                jitc_var_inc_ref(ld->inner_inputs[i]);
                ld->inner_outputs.push_back(ld->inner_inputs[i]);
            }
        }
    }

    Variable v;
    v.kind = (uint32_t) VarKind::LoopEnd;
    v.type = (uint32_t) VarType::Void;
    v.backend = (uint32_t) backend;
    v.size = size;
    v.dep[0] = ld->loop_start;
    v.dep[1] = cond;
    v.symbolic = 1;
    v.extra = 1;
    jitc_var_inc_ref(ld->loop_start);
    jitc_var_inc_ref(cond);

    jitc_new_scope(backend);
    Ref loop_end = steal(jitc_var_new(v, false));
    jitc_new_scope(backend);

    Variable v_phi;
    v_phi.kind = (uint32_t) VarKind::LoopResult;
    v_phi.backend = (uint32_t) backend;
    v_phi.symbolic = ld->symbolic;
    v_phi.size = size;
    v_phi.dep[0] = ld->loop_start;
    v_phi.dep[1] = loop_end;

    size_t state_vars_size = 0,
           state_vars_actual = 0,
           state_vars_actual_size = 0;
    for (size_t i = 0; i < ld->size; ++i) {
        uint32_t index = indices[i], index_new;

        if (ld->inner_inputs[i] != ld->outer_inputs[i]) {
            const Variable *v2 = jitc_var(index);
            v_phi.literal = (uint64_t) i;
            v_phi.type = v2->type;
            jitc_var_inc_ref(ld->loop_start);
            jitc_var_inc_ref(loop_end);
            index_new = jitc_var_new(v_phi, false);
            state_vars_actual++;
            state_vars_actual_size += type_size[v2->type];
        } else {
            index_new = ld->outer_inputs[i];
            jitc_var_inc_ref(index_new);
        }

        state_vars_size += type_size[jitc_var(index_new)->type];
        indices[i] = index_new;
        ld->outer_outputs.push_back(
            WeakRef(index_new, jitc_var(index_new)->counter));
    }

    // Transfer ownership of the LoopData instance
    {
        Extra &e1 = state.extra[loop_end]; // keep order, don't merge on same line
        Extra &e2 = state.extra[(uint32_t) loop];
        std::swap(e1.callback, e2.callback);
        std::swap(e1.callback_data, e2.callback_data);
        std::swap(e1.callback_internal, e2.callback_internal);
    }

    size_t se_count = 0;
    jitc_log(InfoSym,
             "jit_var_loop(loop_start=r%u, loop_cond=r%u, loop_end=r%u): "
             "symbolic loop (\"%s\") with %zu/%zu state variable%s (%zu/%zu "
             "bytes), %zu side effect%s, array size %u%s",
             ld->loop_start, cond, (uint32_t) loop_end, ld->name.c_str(),
             state_vars_actual, ld->size, ld->size == 1 ? "" : "s",
             state_vars_actual_size, state_vars_size, se_count,
             se_count == 1 ? "" : "s", size, ld->symbolic ? " [symbolic]" : "");

    return true;
}

