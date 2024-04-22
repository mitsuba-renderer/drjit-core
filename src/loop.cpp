#include "internal.h"
#include "var.h"
#include "loop.h"
#include "log.h"
#include "eval.h"
#include "op.h"

uint32_t jitc_var_loop_start(const char *name, bool symbolic, size_t n_indices, uint32_t *indices) {
    JitBackend backend = JitBackend::None;
    bool dirty = false;

    // A few sanity checks
    if (!n_indices)
        jitc_raise("jit_var_loop_start(): attempted to record a symbolic loop "
                   "without state variables");

    for (size_t i = 0; i < n_indices; ++i) {
        uint32_t index = indices[i];
        if (!index)
            jitc_raise("jit_var_loop_start(): loop state variable %zu is "
                       "uninitialized (i.e., it has size 0)", i);

        const Variable *v2 = jitc_var(index);
        if (i == 0) {
            backend = (JitBackend) v2->backend;
            dirty = v2->is_dirty();
        } else {
            if ((JitBackend) v2->backend != backend)
                jitc_raise(
                    "jit_var_loop_start(): the loop state involves variables with "
                    "different Dr.Jit backends, which is not permitted");

            dirty |= v2->is_dirty();
        }
    }

    // Ensure side effects are fully processed
    if (dirty) {
        jitc_eval(thread_state(backend));
        for (size_t i = 0; i < n_indices; ++i) {
            if (jitc_var(indices[i])->is_dirty())
                jitc_raise_dirty_error(indices[i]);
        }
    }

    Ref loop_start;
    {
        Variable v;
        v.kind = (uint32_t) VarKind::LoopStart;
        v.type = (uint32_t) VarType::Void;
        v.size = 1;
        v.backend = (uint32_t) backend;
        v.symbolic = symbolic;

        jitc_new_scope(backend);
        loop_start = steal(jitc_var_new(v, true));
        jitc_new_scope(backend);
    }

    if (!name)
        name = "unnamed";

    std::unique_ptr<LoopData> ld(new LoopData(name, loop_start, n_indices, symbolic));
    jitc_var(loop_start)->data = ld.get();
    loop_start.release();

    Variable v_phi;
    v_phi.kind = (uint32_t) VarKind::LoopPhi;
    v_phi.backend = (uint32_t) backend;
    v_phi.symbolic = 1;
    v_phi.dep[0] = ld->loop_start;

    for (size_t i = 0; i < n_indices; ++i) {
        uint32_t index = indices[i];
        Variable *v2 = jitc_var(index);
        jitc_var_inc_ref(index, v2);
        jitc_var_inc_ref(index, v2);
        ld->outer_in.push_back(index);

        v_phi.type = v2->type;
        v_phi.literal = (uint64_t) i;
        v_phi.size = v2->size;
        v_phi.dep[3] = index;
        jitc_var_inc_ref(ld->loop_start);
        uint32_t index_new = jitc_var_new(v_phi, true);
        ld->inner_in.push_back(index_new);
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
    v.data = ld.get();

    Ref loop_holder = steal(jitc_var_new(v, true));
    jitc_var_set_callback(
        loop_holder,
        [](uint32_t, int free, void *p) {
            if (free)
                delete (LoopData *) p;
        },
        ld.release(), true);

    return loop_holder.release();
}

uint32_t jitc_var_loop_cond(uint32_t loop, uint32_t active) {
    LoopData *ld = (LoopData *) jitc_var(loop)->data;

    Variable *loop_start_v = jitc_var(ld->loop_start),
             *active_v = jitc_var(active);

    if ((VarType) active_v->type != VarType::Bool)
        jitc_raise("jit_var_loop_cond(): condition must be a boolean variable");

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
    uint32_t cond = jitc_var_new(v, true);
    jitc_new_scope(backend);
    return cond;
}

bool jitc_var_loop_end(uint32_t loop, uint32_t cond, uint32_t *indices, uint32_t checkpoint) {
    LoopData *ld = (LoopData *) jitc_var(loop)->data;
    bool optimize = jitc_flags() & (uint32_t) JitFlag::OptimizeLoops;

    checkpoint &= 0x7fffffff; // ignore top bit, which jit_record_start() uses as a flag

    // Determine the size of variables that are processed by this loop
    uint32_t size = jitc_var(cond)->size;
    for (size_t i = 0; i < ld->size; ++i) {
        // Ignore loop-invariant state variables
        if (indices[i] == ld->inner_in[i])
            continue;

        const Variable *v1 = jitc_var(ld->outer_in[i]),
                       *v2 = jitc_var(indices[i]);

        // Ignore variables that are the target of side effects from the loop state
        if (v2->is_dirty())
            continue;

        size = std::max(size, std::max(v1->size, v2->size));
    }

    if (!ld->retry) {
        size_t n_eliminated = 0;
        for (size_t i = 0; i < ld->size; ++i) {
            const Variable *v1 = jitc_var(ld->outer_in[i]),
                           *v2 = jitc_var(indices[i]);

            bool eliminate = false;
            if (v2->is_dirty()) {
                // Remove variables that are the target of side effects from the loop state
                eliminate = true;
            } else if (indices[i] == ld->inner_in[i]) {
                // Remove loop-invariant state variables. Do this always when optimizations are
                // turned on. Otherwise, only do it when they aren't compatible with the loop shape.
                eliminate = optimize || v2->size != size;
            } else {
                // Remove loop-invariant literal constants.
                eliminate = optimize && v1->is_literal() &&
                             v2->is_literal() && v1->literal == v2->literal;
            }

            if (eliminate) {
                jitc_trace("jit_var_loop(r%u): eliminating redundant loop "
                           "variable r%u.",
                           ld->loop_start, ld->outer_in[i]);
                jitc_var_inc_ref(ld->outer_in[i]);
                jitc_var_dec_ref(ld->inner_in[i]);
                ld->inner_in[i] = ld->outer_in[i];
                n_eliminated++;
            }
        }

        if (n_eliminated > 0) {
            jitc_log(Debug,
                     "jit_var_loop(r%u): re-recording loop \"%s\" to eliminate "
                     "%zu/%zu redundant loop state variables.",
                     ld->loop_start, ld->name.c_str(), n_eliminated, ld->size);
            for (size_t i = 0; i < ld->size; ++i) {
                indices[i] = ld->inner_in[i];
                jitc_var_inc_ref(indices[i]);
            }

            std::vector<uint32_t> &se_list =
                thread_state(jitc_var(cond)->backend)->side_effects_symbolic;
            if (checkpoint > se_list.size())
                jitc_fail("jit_var_loop(): invalid 'checkpoint' parameter!");

            while (checkpoint < se_list.size()) {
                uint32_t index = se_list.back();
                se_list.pop_back();
                jitc_log(Debug, "jit_var_loop(): deleting side effect r%u", index);
                jitc_var_dec_ref(index);
            }

            ld->retry = true;
            return false;
        }
    }

    JitBackend backend;
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

            const Variable *v1 = jitc_var(ld->inner_in[i]);
            Variable *v2 = jitc_var(index);

            uint32_t new_index;
            if (ld->inner_in[i] != ld->outer_in[i]) {
                if (v2->size != size && size != 1 && v2->size != 1)
                    jitc_raise(
                        "jit_var_loop_end(): loop state variable %zu (r%u) has "
                        "a final shape (size %u) that is incompatible with "
                        "that of the loop (size %u)",
                        i, index, v2->size, size);

                size = std::max(v2->size, size);

                if (backend == JitBackend::LLVM) {
                    new_index = jitc_var_select(active, index, ld->inner_in[i]);
                } else {
                    new_index = index;
                    jitc_var_inc_ref(index);
                }
            } else if (v2->is_dirty()) {
                jitc_var_inc_ref(index);
                new_index = index;
            } else {
                if (index != ld->inner_in[i] &&
                    !(v2->is_literal() && v1->is_literal() && v1->literal == v2->literal))
                    jitc_raise(
                        "jit_var_loop_end(): loop state variable %zu (r%u) was "
                        "presumed to be constant, but it changed (to r%u) when "
                        "re-recording the loop a second time",
                        i, index, ld->inner_in[i]);
                jitc_var_inc_ref(ld->inner_in[i]);
                new_index = ld->inner_in[i];
            }
            ld->inner_out.push_back(new_index);
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
    jitc_var_inc_ref(ld->loop_start);
    jitc_var_inc_ref(cond);

    jitc_new_scope(backend);
    Ref loop_end = steal(jitc_var_new(v, true));
    jitc_new_scope(backend);

    Variable v_phi;
    v_phi.kind = (uint32_t) VarKind::LoopOutput;
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

        if (ld->inner_in[i] != ld->outer_in[i]) {
            const Variable *v2 = jitc_var(index);
            v_phi.literal = (uint64_t) i;
            v_phi.type = v2->type;
            jitc_var_inc_ref(ld->loop_start);
            jitc_var_inc_ref(loop_end);
            state_vars_actual_size += type_size[v2->type];
            state_vars_actual++;
            index_new = jitc_var_new(v_phi, true);
        } else {
            index_new = ld->inner_out[i];
            jitc_var_inc_ref(index_new);
        }

        state_vars_size += type_size[jitc_var(index_new)->type];
        indices[i] = index_new;
        ld->outer_out.emplace_back(index_new, jitc_var(index_new)->counter);
    }

    std::vector<uint32_t> &se_list =
        thread_state(backend)->side_effects_symbolic;

    if (checkpoint > se_list.size())
        jitc_fail("jit_var_loop(): invalid 'checkpoint' parameter");

    Ref se = borrow(loop_end);
    uint32_t se_count = 0;

    while (checkpoint < se_list.size()) {
        uint32_t se_index = se_list.back();
        se_list.pop_back();
        jitc_var(se_index)->side_effect = false;

        Variable v_se;
        v_se.kind = (uint32_t) VarKind::Nop;
        v_se.type = (uint32_t) VarType::Void;
        v_se.backend = (uint32_t) backend;
        v_se.symbolic = ld->symbolic;
        v_se.size = size;
        v_se.dep[0] = se_index;
        v_se.dep[1] = se;
        jitc_var_inc_ref(se);
        se = steal(jitc_var_new(v_se, true));
        se_count++;
    }

    if (se != loop_end)
        jitc_var_mark_side_effect(se.release());

    // Transfer ownership of the LoopData instance
    {
        VariableExtra *e1 = jitc_var_extra(jitc_var(loop_end));
        VariableExtra *e2 = jitc_var_extra(jitc_var(loop));
        std::swap(e1->callback, e2->callback);
        std::swap(e1->callback_data, e2->callback_data);
        std::swap(e1->callback_internal, e2->callback_internal);
    }

    jitc_log(InfoSym,
             "jit_var_loop(loop_start=r%u, loop_cond=r%u, loop_end=r%u): "
             "created a loop (\"%s\") with %zu/%zu state variable%s (%zu/%zu "
             "bytes), %u side effect%s, array size %u.%s",
             ld->loop_start, cond, (uint32_t) loop_end, ld->name.c_str(),
             state_vars_actual, ld->size, ld->size == 1 ? "" : "s",
             state_vars_actual_size, state_vars_size, se_count,
             se_count == 1 ? "" : "s", size, ld->symbolic ? " [symbolic]" : "");

    return true;
}

