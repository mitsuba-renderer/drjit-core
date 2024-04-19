#include "internal.h"
#include "var.h"
#include "cond.h"
#include "log.h"
#include "eval.h"
#include <vector>

uint32_t jitc_var_cond_start(const char *name, bool symbolic, uint32_t cond_t, uint32_t cond_f) {
    Variable *cond_t_v = jitc_var(cond_t),
             *cond_f_v = jitc_var(cond_f);
    if (cond_t_v->is_dirty() || cond_f_v->is_dirty()) {
        jitc_eval(thread_state(cond_t_v->backend));
        cond_t_v = jitc_var(cond_t);
        cond_f_v = jitc_var(cond_f);
        if (cond_t_v->is_dirty() || cond_f_v->is_dirty())
            jitc_raise("jit_var_cond_start(): variable remains dirty following evaluation!");
    }

    Variable v2;
    v2.kind = (uint32_t) VarKind::CondStart;
    v2.type = (uint32_t) VarType::Void;
    v2.size = cond_t_v->size;
    v2.backend = cond_t_v->backend;
    v2.symbolic = symbolic;
    v2.dep[0] = cond_t;
    v2.dep[1] = cond_f;
    jitc_var_inc_ref(cond_t, cond_t_v);
    jitc_var_inc_ref(cond_f, cond_f_v);

    std::unique_ptr<CondData> cd(new CondData());
    cd->name = name;
    cd->se_offset = (uint32_t) thread_state(cond_t_v->backend)->side_effects_symbolic.size();

    v2.data = cd.get();

    jitc_new_scope((JitBackend) cond_t_v->backend);
    uint32_t index = jitc_var_new(v2);

    cond_t_v = jitc_var(cond_t);
    jitc_new_scope((JitBackend) cond_t_v->backend);

    jitc_var_set_callback(
        index,
        [](uint32_t, int free, void *p) {
            if (free)
                delete (CondData *) p;
        },
        cd.release(), true);

    return index;
}

uint32_t jitc_var_cond_append(uint32_t index, const uint32_t *rv, size_t count) {
    const Variable *v = jitc_var(index);
    CondData *cd = (CondData *) v->data;
    JitBackend backend = (JitBackend) v->backend;

    Variable v2;
    v2.kind = (uint32_t) cd->labels[0] ? VarKind::CondEnd : VarKind::CondMid;
    v2.type = (uint32_t) VarType::Void;
    v2.size = v->size;
    v2.backend = (uint32_t) backend;
    v2.dep[0] = index;
    v2.dep[1] = cd->labels[0];
    jitc_var_inc_ref(index);
    jitc_var_inc_ref(cd->labels[0]);
    jitc_new_scope(backend);
    uint32_t index_2 = jitc_var_new(v2);
    jitc_new_scope(backend);

    std::vector<uint32_t> &se = thread_state(backend)->side_effects_symbolic;
    uint32_t se_count = (uint32_t) se.size() - cd->se_offset;

    if (!cd->labels[0]) {
        cd->labels[0] = index_2;
        cd->indices_t = std::vector<uint32_t>(rv, rv + count);
        for (uint32_t index_t: cd->indices_t)
            jitc_var_inc_ref(index_t);
        cd->se_t = std::vector<uint32_t>(se.end() - se_count, se.end());
        se.resize(se.size() - se_count);
    } else if (!cd->labels[1]) {
        cd->labels[1] = index_2;
        if (count != cd->indices_t.size())
            jitc_raise("jitc_var_cond_append(): inconsistent number of return values!");
        cd->indices_f = std::vector<uint32_t>(rv, rv + count);
        for (uint32_t index_f: cd->indices_f)
            jitc_var_inc_ref(index_f);
        cd->se_f = std::vector<uint32_t>(se.end() - se_count, se.end());
        se.resize(se.size() - se_count);
    } else {
        jitc_raise("jitc_var_cond_append(): internal error!");
    }
    return index_2;
}

void jitc_var_cond_end(uint32_t index, uint32_t *rv_out) {
    const Variable *v = jitc_var(index);
    CondData *cd = (CondData *) v->data;
    uint32_t pred = cd->labels[1];
    JitBackend backend = (JitBackend) v->backend;
    uint32_t size = v->size;
    uint32_t i_cond = v->dep[0];
    bool symbolic = v->symbolic;

    Variable v2;
    v2.kind = (uint32_t) VarKind::CondOutput;
    v2.size = size;
    v2.backend = (uint32_t) backend;
    cd->indices_out.reserve(cd->indices_f.size());

    size_t variable_count_actual = 0,
           storage = 0,
           storage_actual = 0;

    for (size_t i = 0; i < cd->indices_t.size(); ++i) {
        uint32_t i_t = cd->indices_t[i], i_f = cd->indices_f[i];
        if (i_t == 0 || i_f == 0)
            jitc_raise("jit_var_cond(): return variable %zu is "
                       "uninitialized/partially initialized (r%u, r%u).",
                       i, i_t, i_f);

        Variable *v_t = jitc_var(i_t),
                 *v_f = jitc_var(i_f);
        VarType vt = (VarType) v_t->type;

        storage += type_size[(int) vt];
        if (i_t == i_f || v_f->is_dirty() ||
            (v_f->is_literal() && v_t->is_literal() && v_f->literal == v_t->literal)) {
            jitc_var_inc_ref(i_f, v_f);
            rv_out[i] = i_f;
            cd->indices_out.emplace_back(0, 0);
            continue;
        }

        uint32_t l1 = size, l2 = v_t->size, l3 = v_f->size,
                 lm = std::max(std::max(l1, l2), l3);

        if ((l1 != lm && l1 != 1) || (l2 != lm && l2 != 1) ||
            (l3 != lm && l3 != 1)) {
            for (size_t j = 0; j < i; ++j)
                jitc_var_dec_ref(rv_out[j]);
            jitc_raise("jit_var_cond(): operands r%u, r%u, and r%u have "
                       "incompatible sizes! (%u, %u, %u)",
                       i_cond, i_t, i_f, l1, l2, l3);
        }

        v2.dep[0] = i_t;
        v2.dep[1] = i_f;
        v2.dep[2] = pred;
        v2.type = (uint32_t) vt;
        jitc_var_inc_ref(i_t, v_t);
        jitc_var_inc_ref(i_f, v_f);
        jitc_var_inc_ref(pred);
        uint32_t index_2 = jitc_var_new(v2);
        cd->indices_out.emplace_back(
            index_2,
            jitc_var(index_2)->counter
        );
        rv_out[i] = index_2;
        variable_count_actual++;
        storage_actual += type_size[(int) vt];
    }

    size_t side_effect_count = cd->se_t.size() + cd->se_f.size();
    if (side_effect_count > 0) {
        auto traverse_se = [backend, size, pred, symbolic](std::vector<uint32_t> &list) {
            for (uint32_t index: list) {
                Variable *v2 = jitc_var(index);
                if (size != v2->size && v2->size != 1 && size != 1)
                    jitc_raise(
                        "jitc_var_cond_end(): size of side effect (%u) is "
                        "incompatible with size of loop condition (%u).",
                        v2->size, size);
                Variable v_se;
                v_se.kind = (uint32_t) VarKind::Nop;
                v_se.type = (uint32_t) VarType::Void;
                v_se.backend = (uint32_t) backend;
                v_se.symbolic = (uint32_t) symbolic;
                v_se.size = std::max(size, v2->size);
                v_se.dep[0] = index; // steal ref
                v_se.dep[1] = pred;
                v2->side_effect = false;
                jitc_var_inc_ref(pred);
                jitc_var_mark_side_effect(jitc_var_new(v_se));
            }
            list.clear();
        };

        traverse_se(cd->se_t);
        traverse_se(cd->se_f);
    }

    jitc_log(
        InfoSym,
        "jit_var_cond(cond=r%u): created a conditional statement (\"%s\") "
        "with %zu/%zu state variable%s (%zu/%zu bytes), %zu side effect%s.",
        index, cd->name.c_str(), variable_count_actual, cd->indices_t.size(),
        variable_count_actual == 1 ? "" : "s", storage, storage_actual,
        side_effect_count, side_effect_count == 1 ? "" : "s");
}
