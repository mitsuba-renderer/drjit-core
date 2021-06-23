#include "internal.h"
#include "var.h"
#include "log.h"
#include "eval.h"
#include "op.h"
#include <tsl/robin_set.h>

struct Loop {
    // A descriptive name
    char *name = nullptr;
    // Backend targeted by this loop
    JitBackend backend;
    // Variable index of loop start node
    uint32_t start = 0;
    // Variable index of loop end node
    uint32_t end = 0;
    /// Variable index of loop condition
    uint32_t cond = 0;
    /// Number of side effects
    uint32_t se_count = 0;
    /// Storage size in bytes for all variables before simplification
    uint32_t storage_size_initial = 0;
    /// Input variables before loop
    std::vector<uint32_t> in;
    /// Input variables before branch condition
    std::vector<uint32_t> in_cond;
    /// Input variables before loop body
    std::vector<uint32_t> in_body;
    /// Output placeholder variables within loop
    std::vector<uint32_t> out_body;
    /// Output variables after loop
    std::vector<uint32_t> out;
    /// Is the loop currently being simplified?
    bool simplify_flag = false;
};

// Forward declarations
static void jitc_var_loop_assemble_start(const Variable *v, const Extra &extra);
static void jitc_var_loop_assemble_cond(const Variable *v, const Extra &extra);
static void jitc_var_loop_assemble_end(const Variable *v, const Extra &extra);
static void jitc_var_loop_simplify(Loop *loop, uint32_t cause);

uint32_t jitc_var_loop_init(uint32_t **indices, uint32_t n_indices) {
    if (n_indices == 0)
        jitc_raise("jit_var_loop_init(): no loop state variables specified!");

    // Determine the variable size
    JitBackend backend = (JitBackend) jitc_var(*indices[0])->backend;
    uint32_t size = 0;

    for (size_t i = 0; i < n_indices; ++i) {
        uint32_t vsize = jitc_var(*indices[i])->size;
        if (size != 0 && vsize != 1 && vsize != size)
            jitc_raise("jit_var_loop_init(): loop state variables have an "
                       "inconsistent size (%u vs %u)!", vsize, size);
        if (vsize > size)
            size = vsize;
    }

    Variable v;
    v.size = size;
    v.placeholder = 1;
    v.backend = (uint32_t) backend;

    // Copy loop state before entering loop (CUDA)
    if (backend == JitBackend::CUDA) {
        v.stmt = (char *) "mov.$t0 $r0, $r1";

        for (size_t i = 0; i < n_indices; ++i) {
            uint32_t &index = *indices[i];
            v.dep[0] = index;
            v.type = jitc_var(index)->type;
            index = jitc_var_new(v, true);
        }
    }

    // Create a special node indicating the loop start
    Ref result = steal(jitc_var_new_stmt(backend, VarType::Void, "", 1, 0, nullptr));
    jitc_var(result)->size = size;

    // Create Phi nodes (LLVM)
    if (backend == JitBackend::LLVM) {
        v.stmt = (char *) "$r0 = phi <$w x $t0> [ $r0_final, %l_$i2_tail ], "
                          "[ $r1, %l_$i2_start ]";
        v.dep[1] = result;

        for (size_t i = 0; i < n_indices; ++i) {
            uint32_t &index = *indices[i];
            v.dep[0] = index;
            v.type = jitc_var(index)->type;
            index = jitc_var_new(v, true);
        }
    }

    jitc_log(InfoSym, "jit_var_loop_init(n_indices=%u, size=%u): r%u",
             n_indices, size, (uint32_t) result);

    return result.release();
}

uint32_t jitc_var_loop(const char *name, uint32_t loop_start, uint32_t loop_cond,
                       uint32_t n, const uint32_t *in, const uint32_t *out_body,
                       uint32_t se_offset, uint32_t *out, int check_invariant,
                       uint8_t *invariant) {
    JitBackend backend = (JitBackend) jitc_var(loop_start)->backend;
    ThreadState *ts = thread_state(backend);

    if (n == 0)
        jitc_raise("jit_var_loop(): must have at least one loop variable!");

    uint32_t se_count = (uint32_t) ts->side_effects.size() - se_offset;
    std::unique_ptr<Loop> loop(new Loop());
    loop->backend = backend;
    loop->in.reserve(n);
    loop->in_body.reserve(n);
    loop->in_cond.reserve(n);
    loop->out_body.reserve(n);
    loop->out.reserve(n);
    loop->name = strdup(name);
    loop->se_count = se_count;
    loop->cond = jitc_var(loop_cond)->dep[0];
    loop->start = loop_start;

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    bool optimize = jitc_flags() & (uint32_t) JitFlag::LoopOptimize;
    bool placeholder = false, dirty = false;
    uint32_t size = 1, n_invariant_provided = 0, n_invariant_detected = 0;
    char temp[256];

    {
        const Variable *v = jitc_var(loop->cond);
        if ((VarType) v->type != VarType::Bool)
            jitc_raise("jit_var_loop(): loop condition must be a boolean variable");
        if (!v->placeholder)
            jitc_raise("jit_var_loop(): loop condition does not depend on any of the loop variables");
        size = v->size;
        dirty = v->ref_count_se;
    }

    for (uint32_t i = 0; i < n; ++i) {
        // ============= Input side =============
        uint32_t index_1 = in[i];
        Variable *v1 = jitc_var(index_1);
        loop->storage_size_initial += type_size[v1->type];

        if (invariant[i]) {
            loop->in_body.push_back(0);
            loop->in_cond.push_back(0);
            loop->in.push_back(0);
            loop->out_body.push_back(0);
            n_invariant_provided++;
            continue;
        }

        if (!v1->placeholder || !v1->dep[0])
            jitc_raise("jit_var_loop(): input %u (r%u) must be a placeholder "
                       "variable (1)", i, index_1);
        uint32_t index_2 = v1->dep[0];
        Variable *v2 = jitc_var(index_2);
        if (!v2->placeholder || !v2->dep[0])
            jitc_raise("jit_var_loop(): input %u (r%u) must be a placeholder "
                       "variable (2)", i, index_2);
        uint32_t index_3 = v2->dep[0];
        Variable *v3 = jitc_var(index_3);

        if (v1->size != v2->size || v2->size != v3->size)
            jitc_raise("jit_var_loop(): size inconsistency for input %u (r%u, r%u, r%u)!",
                       i, index_1, index_2, index_2);

        loop->in_body.push_back(index_1);
        loop->in_cond.push_back(index_2);
        loop->in.push_back(index_3);
        size = std::max(v3->size, size);
        placeholder |= v3->placeholder;
        dirty |= v3->ref_count_se;

        // ============= Output side =============
        uint32_t index_o = out_body[i];
        const Variable *vo = jitc_var(index_o);
        size = std::max(vo->size, size);
        loop->out_body.push_back(index_o);

        // ============= Optimizations =============
        if (check_invariant && optimize) {
            bool eq_literal =
                     v3->literal && vo->literal && v3->value == vo->value,
                 unchanged = index_o == index_1,
                 is_invariant = eq_literal || unchanged;

            invariant[i] = is_invariant;
            n_invariant_detected += is_invariant;
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        auto it_in  = state.variables.find(loop->in_body[i]),
             it_out = state.variables.find(loop->out_body[i]);

        if (it_in != state.variables.end()) {
            const Variable &v = it_in->second;
            if (unlikely(v.size != 1 && v.size != size))
                jitc_raise("jit_var_loop(): loop input variable %u has an "
                           "incompatible size!", i);
        }

        if (it_out != state.variables.end()) {
            const Variable &v = it_out->second;
            if (unlikely(v.size != 1 && v.size != size))
                jitc_raise("jit_var_loop(): loop output variable %u has an "
                           "incompatible size!", i);
        }
    }

    temp[0] = '\0';

    if (n_invariant_detected && n_invariant_provided)
        jitc_fail("jit_var_loop(): internal error while detecting loop-invariant variables!");

    if (n_invariant_detected)
        snprintf(temp, sizeof(temp),
                 ", %u loop-invariant variables detected, recording loop once "
                 "more to optimize further..", n_invariant_detected);
    else if (n_invariant_provided)
        snprintf(temp, sizeof(temp), ", %u loop-invariant variables eliminated",
                 n_invariant_provided);

    jitc_log(InfoSym,
             "jit_var_loop(cond=r%u): loop (\"%s\") with %u loop variable%s, "
             "%u side effect%s, %u elements%s%s",
             loop->cond, name, n, n == 1 ? "" : "s", se_count,
             se_count == 1 ? "" : "s", size, temp,
             placeholder ? " (part of a recorded computation)" : "");

    if (dirty) {
        if (jit_flag(JitFlag::Recording))
            jitc_raise("jit_var_loop(): referenced a dirty variable while "
                       "JitFlag::Recording is active!");

        jitc_eval(ts);
        dirty = jitc_var(loop->cond)->ref_count_se;

        for (uint32_t i = 0; i < n; ++i) {
            if (invariant[i])
                continue;
            Variable *v1 = jitc_var(in[i]),
                     *v2 = jitc_var(v1->dep[0]),
                     *v3 = jitc_var(v2->dep[0]);
            dirty |= v3->ref_count_se;
        }

        if (unlikely(dirty))
            jitc_raise(
                "jit_var_loop(): inputs remain dirty after evaluation!");
    }

    if (n_invariant_detected)
        return 0;

    // ============= Label variables =============
    for (uint32_t i = 0; i < n; ++i) {
        if (!loop->in_body[i])
            continue;

        const char *label = jitc_var_label(loop->in[i]);

        snprintf(temp, sizeof(temp), "%s%sLoop (%s) [in %u, body]",
                 label ? label : "", label ? ", " : "", name, i);
        jitc_var_set_label(loop->in_body[i], temp);
        snprintf(temp, sizeof(temp), "%s%sLoop (%s) [in %u, cond]",
                 label ? label : "", label ? ", " : "", name, i);
        jitc_var_set_label(loop->in_cond[i], temp);
    }

    // =====================================================
    // 2. Configure loop start insertion point
    // =====================================================

    {
        Variable *v = jitc_var(loop_start);
        v->extra = 1;
        v->size = size;
        state.extra[loop_start].assemble = jitc_var_loop_assemble_start;

        snprintf(temp, sizeof(temp), "Loop (%s) [start]", name);
        jitc_var_set_label(loop_start, temp);
    }

    // =====================================================
    // 3. Configure loop branch insertion point
    // =====================================================

    {
        Variable *v = jitc_var(loop_cond);
        v->extra = 1;
        v->size = size;
        Extra &e = state.extra[loop_cond];
        e.assemble = jitc_var_loop_assemble_cond;
        e.callback_data = loop.get();

        snprintf(temp, sizeof(temp), "Loop (%s) [cond]", name);
        jitc_var_set_label(loop_cond, temp);
    }

    // =====================================================
    // 4. Create variable representing the end of the loop
    // =====================================================

    uint32_t loop_end_dep[2] = { loop_start, loop_cond };

    Ref loop_end = steal(
        jitc_var_new_stmt(backend, VarType::Void, "", 1, 2, loop_end_dep));
    loop->end = loop_end;
    {
        Variable *v = jitc_var(loop_end);
        v->extra = 1;
        v->size = size;
        Extra &e = state.extra[loop_end];
        e.n_dep = 2*n + se_count;
        e.dep = (uint32_t *) malloc((2 * n + se_count) * sizeof(uint32_t));
        memcpy(e.dep, loop->out_body.data(), n * sizeof(uint32_t));
        memcpy(e.dep + n, loop->in_cond.data(), n * sizeof(uint32_t));
        for (uint32_t i = 0; i < n; ++i) {
            jitc_var_inc_ref_int(loop->out_body[i]);
            jitc_var_inc_ref_int(loop->in_cond[i]);
        }
        e.assemble = jitc_var_loop_assemble_end;
        e.callback_data = loop.get();

        auto &se = ts->side_effects;
        for (uint32_t i = 0; i < se_count; ++i) {
            uint32_t index = se[se.size() - se_count + i];
            // The 'loop_end' node should depend on this side effect
            jitc_var_inc_ref_int(index);
            state.extra[loop_end].dep[2 * n + i] = index;

            // This side effect should depend on 'loop_branch'
            jitc_var(index)->extra = 1;
            Extra &e2 = state.extra[index];
            uint32_t dep_size_2 = (e2.n_dep + 1) * sizeof(uint32_t);
            uint32_t *tmp = (uint32_t *) malloc(dep_size_2);
            if (e2.n_dep)
                memcpy(tmp, e2.dep, dep_size_2);
            tmp[e2.n_dep] = loop_cond;
            jitc_var_inc_ref_int(loop_cond);
            e2.n_dep++;
            free(e2.dep);
            e2.dep = tmp;
        }

        se.resize(se_offset);

        snprintf(temp, sizeof(temp), "Loop (%s) [end]", name);
        jitc_var_set_label(loop_end, temp);
    }

    // =====================================================
    // 5. Create variable representing side effects
    // =====================================================

    Ref loop_se;
    if (se_count) {
        uint32_t loop_se_dep[1] = { loop_end };
        loop_se = steal(
            jitc_var_new_stmt(backend, VarType::Void, "", 1, 1, loop_se_dep));
        Variable *v = jitc_var(loop_se);
        v->size = size;
        snprintf(temp, sizeof(temp), "Loop (%s) [side effects]", name);
        jitc_var_set_label(loop_se, temp);
    }

    // =====================================================
    // 6. Create output variables
    // =====================================================

    auto var_callback = [](uint32_t index, int free, void *ptr) {
        if (!ptr)
            return;

        Loop *loop_2 = (Loop *) ptr;
        if (!free) {
            // Disable callback
            state.extra[index].callback_data = nullptr;
        }

        // An output variable is no longer referenced. Find out which one.
        uint32_t n2 = loop_2->out.size();
        uint32_t offset = (uint32_t) -1;
        for (uint32_t i = 0; i < n2; ++i) {
            if (loop_2->out[i] == index)
                offset = i;
        }

        if (offset == (uint32_t) -1)
            jitc_fail("jit_var_loop(): expired output variable %u could not "
                      "be located!", index);
        loop_2->out[offset] = 0;

        /* When this output variable is removed, it may also enable some
           simplification within the loop. */
        jitc_var_loop_simplify(loop_2, offset);
    };

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t index = loop->out_body[i];
        if (!index) {
            // Loop-invariant variable
            out[i] = in[i];
            loop->out.push_back(0);
            jitc_var_inc_ref_ext(out[i]);
            continue;
        }
        const char *label = jitc_var_label(loop->in_body[i]);
        const char *delim = strrchr(label, '[');
        if (unlikely(!label))
            jitc_fail("jit_var_loop(): internal error while creating output label");

        const Variable *v = jitc_var(index);
        Variable v2;
        if (backend == JitBackend::CUDA)
            v2.stmt = (char *) "mov.$t0 $r0, $r1";
        else
            v2.stmt = (char *) "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>";
        v2.size = size;
        v2.placeholder = placeholder;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.dep[0] = loop->in_cond[i];
        v2.dep[1] = loop_end;
        v2.extra = 1;
        jitc_var_inc_ref_int(loop->in_cond[i]);
        jitc_var_inc_ref_int(loop_end);
        uint32_t index_2 = jitc_var_new(v2, true);
        loop->out.push_back(index_2);
        out[i] = index_2;

        snprintf(temp, sizeof(temp), "%.*s[out %u]", (int) (delim - label), label, i);
        jitc_var_set_label(index_2, temp);

        if (optimize) {
            Extra &e = state.extra[index_2];
            e.callback = var_callback;
            e.callback_data = loop.get();
            e.callback_internal = true;
        }
    }

    {
        Extra &e1 = state.extra[loop_start];
        e1.callback = [](uint32_t, int free_var, void *ptr) {
            if (free_var && ptr) {
                Loop *loop_2 = (Loop *) ptr;
                free(loop_2->name);
                delete loop_2;
            }
        };
        e1.callback_internal = true;
        e1.callback_data = loop.release();
    }

    return loop_se.release();
}

static void jitc_var_loop_dfs(tsl::robin_set<uint32_t, UInt32Hasher> &set, uint32_t index) {
    if (!set.insert(index).second)
        return;
    // jitc_trace("jitc_var_dfs(r%u)", index);

    const Variable *v = jitc_var(index);
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index_2 = v->dep[i];
        if (!index_2)
            break;
        jitc_var_loop_dfs(set, index_2);
    }

    if (unlikely(v->extra)) {
        auto it = state.extra.find(index);
        if (it == state.extra.end())
            jitc_fail("jit_var_loop_dfs(): could not find matching 'extra' record!");

        const Extra &extra = it->second;
        for (uint32_t i = 0; i < extra.n_dep; ++i) {
            uint32_t index_2 = extra.dep[i];
            if (index_2)
                jitc_var_loop_dfs(set, index_2);
        }
    }
}

static void jitc_var_loop_simplify(Loop *loop, uint32_t cause) {
    if (loop->simplify_flag)
        return;
    loop->simplify_flag = true;

    const uint32_t n = loop->in.size();
    uint32_t n_freed = 0, n_rounds = 0;
    tsl::robin_set<uint32_t, UInt32Hasher> visited;

    jitc_trace("jit_var_loop_simplify(): loop output %u freed, simplifying..",
               cause);

    bool progress;
    while (true) {
        n_rounds++;
        progress = false;
        visited.clear();

        for (uint32_t i = 0; i < n; ++i)
            visited.insert(loop->in[i]);

        // Find all inputs that are reachable from the outputs that are still alive
        for (uint32_t i = 0; i < n; ++i) {
            if (!loop->out[i] || !loop->out_body[i])
                continue;
            // jitc_trace("jit_var_loop_simplify(): DFS from %u (r%u)", i, loop->out_body[i]);
            jitc_var_loop_dfs(visited, loop->out_body[i]);
        }

        // jitc_trace("jit_var_loop_simplify(): DFS from loop condition (r%u)", loop->cond);
        jitc_var_loop_dfs(visited, loop->cond);

        // Find all inputs that are reachable from the side effects
        Extra &e = state.extra[loop->end];
        for (uint32_t i = 2*n; i < e.n_dep; ++i) {
            if (!e.dep[i])
                continue;
            // jitc_trace("jit_var_loop_simplify(): DFS from side effect %u (r%u)", i-2*n, e.dep[i]);
            jitc_var_loop_dfs(visited, e.dep[i]);
        }

        /// Propagate until no further changes
        bool again;
        do {
            again = false;
            for (uint32_t i = 0; i < n; ++i) {
                if (loop->in_cond[i] &&
                    visited.find(loop->in_cond[i]) != visited.end() &&
                    visited.find(loop->out_body[i]) == visited.end()) {
                    jitc_var_loop_dfs(visited, loop->out_body[i]);
                    again = true;
                }
            }
        } while (again);

        /// Remove loop variables that are never referenced
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t index = loop->in_cond[i];
            if (index == 0 || visited.find(index) != visited.end())
                continue;
            n_freed++;

            jitc_trace("jit_var_loop_simplify(): freeing unreferenced loop "
                       "variable %u (r%u -> r%u -> r%u)", i, loop->in[i],
                       loop->in_cond[i], loop->in_body[i]);
            progress = true;

            Extra &e_end = state.extra[loop->end];
            if (unlikely(e_end.dep[n + i] != loop->in_cond[i]))
                jitc_fail("jit_var_loop_simplify: internal error (3)");
            if (unlikely(e_end.dep[i] != loop->out_body[i]))
                jitc_fail("jit_var_loop_simplify: internal error (2)");
            e_end.dep[i] = e_end.dep[n + i] = 0;

            jitc_var_dec_ref_int(loop->in_cond[i]);
            jitc_var_dec_ref_int(loop->out_body[i]);

            loop->in[i] = loop->in_cond[i] = loop->in_body[i] = loop->out_body[i] = 0;
        }

        if (!progress)
            break;
    }
    loop->simplify_flag = false;
    jitc_trace("jit_var_loop_simplify(): done, freed %u loop variables in %u rounds.",
               n_freed, n_rounds);
}

static void jitc_var_loop_assemble_start(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index;

    if (loop->backend == JitBackend::LLVM) {
        buffer.fmt("    br label %%l_%u_start\n", loop_reg);
        buffer.fmt("\nl_%u_start:\n", loop_reg);
        buffer.fmt("    br label %%l_%u_cond\n", loop_reg);
    }

    std::pair<uint32_t, uint32_t> result{ 0, 0 };

    buffer.fmt("\nl_%u_cond: %s Loop (%s)\n", loop_reg,
               loop->backend == JitBackend::CUDA ? "//" : ";",
               loop->name);

    jitc_log(InfoSym,
             "jit_var_loop_assemble(): loop (\"%s\") with %u/%u loop "
             "variable%s (%u/%u bytes), %u side effect%s",
             loop->name, result.first, (uint32_t) loop->in.size(),
             result.first == 1 ? "" : "s", result.second, loop->storage_size_initial,
             (uint32_t) loop->se_count, loop->se_count == 1 ? "" : "s");
}

static void jitc_var_loop_assemble_cond(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index,
             mask_reg = jitc_var(loop->cond)->reg_index;

    if (loop->backend == JitBackend::CUDA) {
        buffer.fmt("    @!%%p%u bra l_%u_done;\n", mask_reg, loop_reg);
    } else {
        uint32_t width = jitc_llvm_vector_width;
        char global[128];
        snprintf(
            global, sizeof(global),
            "declare i1 @llvm.experimental.vector.reduce.or.v%ui1(<%u x i1>)\n",
            width, width);
        jitc_register_global(global);

        buffer.fmt("    %%p%u = call i1 @llvm.experimental.vector.reduce.or.v%ui1(<%u x i1> %%p%u)\n"
                   "    br i1 %%p%u, label %%l_%u_body, label %%l_%u_done\n",
                   loop_reg, width, width, mask_reg, loop_reg, loop_reg, loop_reg);
    }

    buffer.fmt("\nl_%u_body:\n", loop_reg);
}

static void jitc_var_loop_assemble_end(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index,
             mask_reg = jitc_var(loop->cond)->reg_index;

    if (loop->backend == JitBackend::LLVM)
        buffer.fmt("    br label %%l_%u_tail\n"
                   "\nl_%u_tail:\n", loop_reg, loop_reg);

    uint32_t width = jitc_llvm_vector_width;
    for (size_t i = 0; i < loop->in_body.size(); ++i) {
        auto it_in = state.variables.find(loop->in_cond[i]),
             it_out = state.variables.find(loop->out_body[i]);

        if (it_in == state.variables.end())
            continue;
        else if (it_out == state.variables.end())
            jitc_fail("jit_var_loop_assemble_end(): internal error!");

        const Variable *v_in = &it_in->second,
                       *v_out = &it_out->second;
        uint32_t vti = it_in->second.type;

        if (loop->backend == JitBackend::LLVM)
            buffer.fmt("    %s%u_final = select <%u x i1> %%p%u, <%u x %s> %s%u, "
                       "<%u x %s> %s%u\n",
                       type_prefix[vti], v_in->reg_index, width, mask_reg, width,
                       type_name_llvm[vti], type_prefix[vti], v_out->reg_index,
                       width, type_name_llvm[vti], type_prefix[vti],
                       v_in->reg_index);
        else
            buffer.fmt("    mov.%s %s%u, %s%u;\n", type_name_ptx[vti],
                       type_prefix[vti], v_in->reg_index, type_prefix[vti],
                       v_out->reg_index);
    }

    if (loop->backend == JitBackend::CUDA)
        buffer.fmt("    bra l_%u_cond;\n", loop_reg);
    else
        buffer.fmt("    br label %%l_%u_cond;\n", loop_reg);

    buffer.fmt("\nl_%u_done:\n", loop_reg);

    if (loop->se_count) {
        uint32_t *dep = state.extra[loop->end].dep;
        uint32_t n = (uint32_t) loop->in.size();
        for (uint32_t i = 0; i < loop->se_count; ++i) {
            uint32_t &index = dep[2*n + i];
            if (!jitc_var(index)->side_effect)
                jitc_fail("jitc_var_loop_assemble(): internal error (3)");
            jitc_var_dec_ref_int(index);
            index = 0;
        }
        loop->se_count = 0;
    }
}
