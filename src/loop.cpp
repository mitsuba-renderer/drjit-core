#include "internal.h"
#include "var.h"
#include "log.h"
#include "eval.h"
#include <tsl/robin_set.h>

struct Loop {
    // A descriptive name
    const char *name = nullptr;
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
};

// Forward declarations
static void jitc_var_loop_assemble_start(const Variable *v, const Extra &extra);
static void jitc_var_loop_assemble_cond(const Variable *v, const Extra &extra);
static void jitc_var_loop_assemble_end(const Variable *v, const Extra &extra);
static void jitc_var_loop_simplify(Loop *loop, uint32_t cause);

void jitc_var_loop(const char *name, uint32_t cond, uint32_t n,
                   const uint32_t *in, const uint32_t *out_body,
                   uint32_t se_offset, uint32_t *out,
                   int check_invariant, uint8_t *invariant) {
    const Variable *cond_v = jitc_var(cond);
    JitBackend backend = (JitBackend) cond_v->backend;
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
    loop->cond = cond;
    loop->name = name;
    loop->se_count = se_count;

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    bool optimize = jitc_flags() & (uint32_t) JitFlag::LoopOptimize;
    bool placeholder = false;
    uint32_t size = 1, n_invariant_provided = 0, n_invariant_detected = 0;
    char temp[128];

    {
        const Variable *v = jitc_var(cond);
        if ((VarType) v->type != VarType::Bool)
            jitc_raise("jit_var_loop(): 'cond' must be a boolean variable");
        if (!v->placeholder)
            jitc_raise("jit_var_loop(): 'cond' must be a placeholder variable");
        size = v->size;
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
            loop->out.push_back(0);
            n_invariant_provided++;
            continue;
        }

        if (!v1->placeholder || !v1->placeholder_iface || !v1->dep[0])
            jitc_raise("jit_var_loop(): inputs must be placeholder variables (1)");
        uint32_t index_2 = v1->dep[0];
        Variable *v2 = jitc_var(index_2);
        if (!v2->placeholder || !v2->placeholder_iface || !v2->dep[0])
            jitc_raise("jit_var_loop(): inputs must be placeholder variables (2)");
        uint32_t index_3 = v2->dep[0];
        Variable *v3 = jitc_var(index_3);

        loop->in_body.push_back(index_1);
        loop->in_cond.push_back(index_2);
        loop->in.push_back(index_3);
        size = std::max(v1->size, size);
        placeholder |= v3->placeholder;

        // ============= Output side =============
        uint32_t index_o = out_body[i];
        const Variable *vo = jitc_var(index_o);
        if (!vo->literal && !vo->placeholder)
            jitc_raise("jit_var_loop(): outputs must be placeholder or literal variables");
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

    jitc_log(Info,
             "jit_var_loop(cond=r%u): loop (\"%s\") with %u loop variable%s, %u side effect%s%s",
             cond, name, n, n == 1 ? "" : "s", se_count, se_count == 1 ? "" : "s", temp);

    if (n_invariant_detected)
        return;

    // =====================================================
    // 2. Create variable representing the start of the loop
    // =====================================================

    Ref loop_start =
        steal(jitc_var_new_stmt(backend, VarType::Void, "", 1, 0, nullptr));
    loop->start = loop_start;

    {
        size_t dep_size = n * sizeof(uint32_t);
        snprintf(temp, sizeof(temp), "Loop: %s [start]", name);
        Variable *v = jitc_var(loop_start);
        v->extra = 1;
        v->size = size;
        Extra &e = state.extra[loop_start];
        e.label = strdup(temp);
        e.n_dep = n;
        e.dep = (uint32_t *) malloc(dep_size);
        memcpy(e.dep, loop->in.data(), dep_size);
        for (uint32_t i = 0; i < n; ++i)
            jitc_var_inc_ref_int(loop->in[i]);
        e.assemble = jitc_var_loop_assemble_start;
    }

    // =====================================================
    // 3. Create variable representing the branch condition
    // =====================================================

    uint32_t loop_cond_dep[2] = { cond, loop_start };
    Ref loop_cond = steal(
        jitc_var_new_stmt(backend, VarType::Void, "", 1, 2, loop_cond_dep));
    {
        snprintf(temp, sizeof(temp), "Loop: %s [branch]", name);
        Variable *v = jitc_var(loop_cond);
        v->extra = 1;
        v->size = size;
        Extra &e = state.extra[loop_cond];
        e.label = strdup(temp);
        e.assemble = jitc_var_loop_assemble_cond;
        e.callback_data = loop.get();
    }

    // =====================================================
    // 4. Add depencencies to placeholders to ensure order
    // =====================================================

    for (uint32_t i = 0; i < n; ++i) {
        if (!loop->in_body[i])
            continue;
        Variable *v_body = jitc_var(loop->in_body[i]),
                 *v_cond = jitc_var(loop->in_cond[i]);
        v_body->dep[1] = loop_cond;
        v_cond->dep[1] = loop_start;
        v_body->stmt = (char *) "";
        v_cond->stmt = (char *) "";
        jitc_var_inc_ref_int(loop_start);
        jitc_var_inc_ref_int(loop_cond);

        snprintf(temp, sizeof(temp), "Loop: %s [in %u]", name, i);
        jitc_var_set_label(loop->in[i], temp, false);
        snprintf(temp, sizeof(temp), "Loop: %s [in %u, cond.]", name, i);
        jitc_var_set_label(loop->in_cond[i], temp, false);
        snprintf(temp, sizeof(temp), "Loop: %s [in %u, body]", name, i);
        jitc_var_set_label(loop->in_body[i], temp, false);
    }

    // =====================================================
    // 5. Create variable representing the end of the loop
    // =====================================================

    uint32_t loop_end_dep[2] = { loop_start, loop_cond };

    Ref loop_end = steal(
        jitc_var_new_stmt(backend, VarType::Void, "", 1, 2, loop_end_dep));
    loop->end = loop_end;
    {
        snprintf(temp, sizeof(temp), "Loop: %s [end]", name);
        Variable *v = jitc_var(loop_end);
        v->extra = 1;
        v->size = size;
        Extra &e = state.extra[loop_end];
        e.label = strdup(temp);
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
    }

    // =====================================================
    // 6. Create variable representing side effects
    // =====================================================

    Ref loop_se;
    if (se_count) {
        uint32_t loop_se_dep[1] = { loop_end };
        loop_se = steal(
            jitc_var_new_stmt(backend, VarType::Void, "", 1, 1, loop_se_dep));
        {
            snprintf(temp, sizeof(temp), "Loop: %s [side effects]", name);
            Variable *v = jitc_var(loop_se);
            v->side_effect = 1;
            v->size = size;
        }
        ts->side_effects.push_back(loop_se.release());
    }

    // =====================================================
    // 7. Create output variables
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
            jitc_var_inc_ref_ext(out[i]);
            continue;
        }

        snprintf(temp, sizeof(temp), "Loop: %s [out %u]", name, i);

        const Variable *v = jitc_var(index);
        Variable v2;
        if (backend == JitBackend::CUDA)
            v2.stmt = (char *) "mov.$t0 $r0, $r1";
        else
            v2.stmt = (char *) "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>";
        v2.size = size;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.dep[0] = loop->in_cond[i];
        v2.dep[1] = loop_end;
        v2.extra = 1;
        jitc_var_inc_ref_int(loop->in_cond[i]);
        jitc_var_inc_ref_int(loop_end);
        uint32_t index_2 = jitc_var_new(v2, true);
        Extra &e = state.extra[index_2];
        e.label = strdup(temp);
        loop->out.push_back(index_2);
        out[i] = index_2;

        if (optimize) {
            e.callback = var_callback;
            e.callback_data = loop.get();
            e.callback_internal = true;
        }
    }

    {
        Extra &e1 = state.extra[loop_start];
        e1.callback = [](uint32_t, int free, void *ptr) {
            if (free)
                delete (Loop *) ptr;
        };
        e1.callback_internal = true;
        e1.callback_data = loop.release();
    }
}

static void jitc_var_loop_dfs(tsl::robin_set<uint32_t> &set, uint32_t index) {
    if (!set.insert(index).second)
        return;

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
            uint32_t index2 = extra.dep[i];
            if (index2)
                jitc_var_loop_dfs(set, index2);
        }
    }
}

static void jitc_var_loop_simplify(Loop *loop, uint32_t cause) {
    uint32_t n = loop->in.size();
    tsl::robin_set<uint32_t> visited;

    jitc_trace("jit_var_loop_simplify(): output %u freed, simplifying..", cause);

    // Find all inputs that are reachable from the outputs that are still alive
    for (uint32_t i = 0; i < n; ++i) {
        if (!loop->out[i] || !loop->out_body[i])
            continue;
        jitc_var_loop_dfs(visited, loop->out_body[i]);
    }

    // Find all inputs that are reachable from the side effects
    Extra &e = state.extra[loop->end];
    for (uint32_t i = 2*n; i < e.n_dep; ++i) {
        if (e.dep[i])
            jitc_var_loop_dfs(visited, e.dep[i]);
    }

    /// Remove loop variables that are never referenced
    uint32_t n_freed = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t index = loop->in_body[i];
        if (index == 0 || visited.find(index) != visited.end())
            continue;
        n_freed++;

        jitc_trace(
            "jit_var_loop_simplify(): freeing unreferenced loop variable %u", i);

        Extra &e_start = state.extra[loop->start];
        Extra &e_end = state.extra[loop->end];
        if (unlikely(e_start.dep[i] != loop->in[i]))
            jitc_fail("jit_var_loop_simplify: internal error (1)");
        if (unlikely(e_end.dep[n + i] != loop->in_cond[i]))
            jitc_fail("jit_var_loop_simplify: internal error (3)");
        if (unlikely(e_end.dep[i] != loop->out_body[i]))
            jitc_fail("jit_var_loop_simplify: internal error (2)");
        e_end.dep[i] = e_end.dep[n + i] = e_start.dep[i] = 0;

        jitc_var_dec_ref_int(loop->in[i]);
        jitc_var_dec_ref_int(loop->in_cond[i]);
        jitc_var_dec_ref_int(loop->out_body[i]);

        loop->in[i] = loop->in_cond[i] = loop->out_body[i] = 0;
    }

    jitc_trace("jit_var_loop_simplify(): done, freed %u loop variables.",
               n_freed);
}

static std::pair<uint32_t, uint32_t>
jitc_var_loop_copy(JitBackend backend, const std::vector<uint32_t> &dst,
                   const std::vector<uint32_t> &src) {
    uint32_t count = 0, size = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        auto it_src = state.variables.find(src[i]),
             it_dst = state.variables.find(dst[i]);
        if (it_src == state.variables.end() || it_dst == state.variables.end())
            continue;
        const Variable *v_src = &it_src->second, *v_dst = &it_dst->second;
        uint32_t vti = it_src->second.type;

        if (unlikely(backend == JitBackend::CUDA &&
                     (v_src->reg_index == 0 || v_dst->reg_index == 0)))
            jitc_fail("jit_var_loop_copy(): internal error (can't move r%u <- "
                      "r%u as one of them hasn't been assigned a register)!",
                      dst[i], src[i]);

        if (backend == JitBackend::CUDA)
            buffer.fmt("    mov.%s %s%u, %s%u;\n", type_name_ptx[vti],
                       type_prefix[vti], v_dst->reg_index, type_prefix[vti],
                       v_src->reg_index);
        else
            buffer.fmt("    %s%u = bitcast <%u x %s> %s%u to <%u x %s>\n",
                       type_prefix[vti], v_dst->reg_index, jitc_llvm_vector_width,
                       type_name_llvm[vti], type_prefix[vti], v_src->reg_index,
                       jitc_llvm_vector_width, type_name_llvm[vti]);
        count++;
        size += type_size[vti];
    }

    return { count, size };
}

static std::pair<uint32_t, uint32_t>
jitc_var_loop_phi_llvm(uint32_t loop_reg, const std::vector<uint32_t> &in,
                       const std::vector<uint32_t> &in_cond,
                       const std::vector<uint32_t> &out_body) {
    uint32_t count = 0, size = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        auto it_in = state.variables.find(in[i]),
             it_in_cond = state.variables.find(in_cond[i]),
             it_out_body = state.variables.find(out_body[i]);

        if (it_in_cond == state.variables.end() &&
            it_out_body == state.variables.end() &&
            it_in == state.variables.end())
            continue;

        if (it_in_cond == state.variables.end() ||
            it_out_body == state.variables.end() ||
            it_in == state.variables.end())
            jitc_fail("jitc_var_loop_phi_llvm(): internal error!");

        const Variable *v_in = &it_in->second,
                       *v_in_cond = &it_in_cond->second,
                       *v_out_body = &it_out_body->second;

        uint32_t vti = it_in->second.type;
        buffer.fmt("    %s%u = phi <%u x %s> [ %s%u, %%l_%u_start ], [ %s%u_final, "
                   "%%l_%u_tail ]\n",
                   type_prefix[vti], v_in_cond->reg_index, jitc_llvm_vector_width,
                   type_name_llvm[vti], type_prefix[vti], v_in->reg_index,
                   loop_reg, type_prefix[vti], v_out_body->reg_index, loop_reg);

        count++;
        size += type_size[vti];
    }

    return { count, size };
}

static void jitc_var_loop_select_llvm(uint32_t mask_reg,
                                      const std::vector<uint32_t> &out_body,
                                      const std::vector<uint32_t> &in_body,
                                      const std::vector<uint32_t> &in) {
    uint32_t width = jitc_llvm_vector_width;
    for (size_t i = 0; i < in_body.size(); ++i) {
        auto it_in = state.variables.find(in_body[i]),
             it_out = state.variables.find(out_body[i]);

        if (it_in == state.variables.end() &&
            it_out == state.variables.end())
            continue;

        if (it_in == state.variables.end()) {
            it_in = state.variables.find(in[i]);
            if (it_in == state.variables.end())
                jitc_fail("jit_var_loop_select_llvm(): internal error!");
        }

        const Variable *v_in = &it_in->second,
                       *v_out = &it_out->second;
        uint32_t vti = it_in->second.type;

        buffer.fmt("    %s%u_final = select <%u x i1> %%p%u, <%u x %s> %s%u, "
                   "<%u x %s> %s%u\n",
                   type_prefix[vti], v_out->reg_index, width, mask_reg, width,
                   type_name_llvm[vti], type_prefix[vti], v_out->reg_index,
                   width, type_name_llvm[vti], type_prefix[vti],
                   v_in->reg_index);
    }
}

static void jitc_var_loop_assemble_start(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index;
    if (loop->backend == JitBackend::LLVM)
        buffer.fmt("    br label %%l_%u_start\n", loop_reg);

    buffer.fmt("\nl_%u_start:\n", loop_reg);

    std::pair<uint32_t, uint32_t> result{ 0, 0 };

    if (loop->backend == JitBackend::CUDA)
        result = jitc_var_loop_copy(loop->backend, loop->in_cond, loop->in);
    else
        buffer.fmt("    br label %%l_%u_cond\n", loop_reg);

    buffer.fmt("\nl_%u_cond: %s Loop \"%s\"\n", loop_reg,
               loop->backend == JitBackend::CUDA ? "//" : ";",
               loop->name);

    if (loop->backend == JitBackend::LLVM)
        result = jitc_var_loop_phi_llvm(loop_reg, loop->in, loop->in_cond,
                                        loop->out_body);

    jitc_log(Info,
             "jit_var_loop_assemble(): loop (\"%s\") with %u/%u loop "
             "variable%s (%u/%u bytes), %u side effect%s",
             loop->name, result.first, (uint32_t) loop->in.size(),
             result.first == 1 ? "" : "s", result.second, loop->storage_size_initial,
             (uint32_t) loop->se_count, loop->se_count == 1 ? "" : "s");
}

static void jitc_var_loop_assemble_cond(const Variable *v, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index,
             mask_reg = jitc_var(v->dep[0])->reg_index;

    if (loop->backend == JitBackend::CUDA) {
        buffer.fmt("    @!%%p%u bra l_%u_done;\n", mask_reg, loop_reg);
    } else {
        uint32_t width = jitc_llvm_vector_width;
        char global[128];
        snprintf(
            global, sizeof(global),
            "declare i1 @llvm.experimental.vector.reduce.or.v%ui1(<%u x i1>)\n",
            width, width);

        auto global_hash = XXH128(global, strlen(global), 0);
        if (globals_map.emplace(global_hash, globals_map.size()).second)
            globals.push_back(global);

        buffer.fmt("    %%p%u = call i1 @llvm.experimental.vector.reduce.or.v%ui1(<%u x i1> %%p%u)\n"
                   "    br i1 %%p%u, label %%l_%u_body, label %%l_%u_done\n",
                   loop_reg, width, width, mask_reg, loop_reg, loop_reg, loop_reg);
    }

    buffer.fmt("\nl_%u_body:\n", loop_reg);
    (void) jitc_var_loop_copy(loop->backend, loop->in_body, loop->in_cond);

    buffer.putc('\n');
}

static void jitc_var_loop_assemble_end(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index;
    buffer.putc('\n');

    if (loop->backend == JitBackend::LLVM)
        buffer.fmt("    br label %%l_%u_tail\n"
                   "\nl_%u_tail:\n", loop_reg, loop_reg);

    if (loop->backend == JitBackend::CUDA)
        (void) jitc_var_loop_copy(loop->backend, loop->in_cond, loop->out_body);
    else
        (void) jitc_var_loop_select_llvm(jitc_var(loop->cond)->reg_index,
                                         loop->out_body, loop->in_body, loop->in);

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
