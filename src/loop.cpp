#include "internal.h"
#include "var.h"
#include "log.h"

struct Loop {
    // A descriptive name
    const char *name = nullptr;
    // Variable index of loop start node
    uint32_t start = 0;
    // Variable index of loop end node
    uint32_t end = 0;
    /// Variable index of loop condition
    uint32_t cond = 0;
    /// Number of side effects
    uint32_t se_count = 0;
    /// Storage size in bytes for all variables before simplification
    uint32_t storage_size = 0;
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

static uint32_t jitc_refcount(uint32_t index) {
    auto it = state.variables.find(index);
    if (it == state.variables.end())
        return 0;
    return it->second.ref_count_int + it->second.ref_count_ext;
}

void jitc_var_loop(const char *name, uint32_t cond, uint32_t n,
                   const uint32_t *in, const uint32_t *out_body,
                   uint32_t se_offset, uint32_t *out) {
    const Variable *cond_v = jitc_var(cond);
    JitBackend backend = (JitBackend) cond_v->backend;
    ThreadState *ts = thread_state(backend);

    if (n == 0)
        jitc_raise("jit_var_loop(): must have at least one loop variable!");

    uint32_t se_count = (uint32_t) ts->side_effects.size() - se_offset;
    std::unique_ptr<Loop> loop(new Loop());
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
    uint32_t size = 1;
    bool placeholder = false;
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
        loop->storage_size += type_size[v1->type];

        // ============= Output side =============
        uint32_t index_o = out_body[i];
        const Variable *vo = jitc_var(index_o);
        size = std::max(vo->size, size);
        loop->out_body.push_back(index_o);

        // ============= Optimizations =============

        // if (invariant) {
        //     bool eq_literal =
        //         v3->literal && vo->literal && v3->value == vo->value;
        //     bool unchanged = out_body[i] == in[i];
        //
        //     invariant[i] = optimize && (eq_literal || unchanged);
        // }
    }

    for (uint32_t i = 0; i < n; ++i) {
        const Variable *vi = jitc_var(loop->in_body[i]),
                       *vo = jitc_var(loop->out_body[i]);
        if ((vi->size != 1 && vi->size != size) ||
            (vo->size != 1 && vo->size != size))
            jitc_raise(
                "jit_var_loop(): input/output arrays have incompatible size!");
    }

    jitc_log(Info,
             "jit_var_loop(cond=r%u): loop (\"%s\") with %u loop variable%s, %u side effect%s",
             cond, name, n, n == 1 ? "" : "s", se_count, se_count == 1 ? "" : "s");

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
        Variable *v_body   = jitc_var(loop->in_body[i]),
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
        e.n_dep = n + se_count;
        e.dep = (uint32_t *) malloc((n + se_count) * sizeof(uint32_t));
        memcpy(e.dep, loop->out_body.data(), n * sizeof(uint32_t));
        for (uint32_t i = 0; i < n; ++i)
            jitc_var_inc_ref_int(loop->out_body[i]);
        e.assemble = jitc_var_loop_assemble_end;
        e.callback_data = loop.get();

        auto &se = ts->side_effects;
        for (uint32_t i = 0; i < se_count; ++i) {
            uint32_t index = se[se.size() - se_count + i];
            // The 'loop_end' node should depend on this side effect
            jitc_var_inc_ref_int(index);
            state.extra[loop_end].dep[n + i] = index;

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

        if (jitc_refcount(loop_2->in_cond[offset]) > 2 ||
            jitc_refcount(loop_2->in_body[offset]) > 1)
            return; // still needed

        // Inform recursive computation graphs via reference counting
        Extra &e1 = state.extra[loop_2->end];
        if (unlikely(e1.dep[offset] != loop_2->out_body[offset]))
            jitc_fail("jit_var_loop(): internal error (1)");
        jitc_var_dec_ref_int(loop_2->out_body[offset]);
        e1.dep[offset] = 0;
        loop_2->out_body[offset] = 0;

        // Check if any input parameters became irrelevant
        for (uint32_t i = 0; i < n2; ++i) {
            if (!loop_2->in[i] || jitc_refcount(loop_2->in_cond[i]) > 0)
                continue;

            Extra &e2 = state.extra[loop_2->start];
            if (unlikely(e2.dep[i] != loop_2->in[i]))
                jitc_fail("jit_var_loop(): internal error (2)");

            jitc_var_dec_ref_int(loop_2->in[i]);
            e2.dep[i] = 0;
            loop_2->in[i] = 0;
            loop_2->in_cond[i] = 0;
            loop_2->in_body[i] = 0;
        }
    };

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t index = loop->out_body[i];
        snprintf(temp, sizeof(temp), "Loop: %s [out %u]", name, i);

        const Variable *v = jitc_var(index);
        Variable v2;
        v2.stmt = (char *) "mov.$t0 $r0, $r1";
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

static std::pair<uint32_t, uint32_t>
jitc_var_loop_move(const std::vector<uint32_t> &dst,
                   const std::vector<uint32_t> &src) {
    uint32_t count = 0, size = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        auto it_src = state.variables.find(src[i]),
             it_dst = state.variables.find(dst[i]);

        if (it_src == state.variables.end() || it_dst == state.variables.end())
            continue;

        uint32_t vti = it_src->second.type;

        buffer.fmt("    mov.%s %s%u, %s%u;\n", type_name_ptx[vti],
                   type_prefix[vti], it_dst->second.reg_index, type_prefix[vti],
                   it_src->second.reg_index);
        count++;
        size += type_size[vti];
    }

    return { count, size };
}

static void jitc_var_loop_assemble_start(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index;

    buffer.fmt("\nl_%u_start:\n", loop_reg);
    auto result = jitc_var_loop_move(loop->in_cond, loop->in);
    buffer.fmt("\nl_%u_cond:\n", loop_reg);

    jitc_log(Info,
             "jit_var_loop_assemble(): loop (\"%s\") with %u/%u loop "
             "variable%s (%u/%u bytes), %u side effect%s",
             loop->name, result.first, (uint32_t) loop->in.size(),
             result.first == 1 ? "" : "s", result.second, loop->storage_size,
             (uint32_t) loop->se_count, loop->se_count == 1 ? "" : "s");
}

static void jitc_var_loop_assemble_cond(const Variable *v, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index,
             mask_reg = jitc_var(v->dep[0])->reg_index;

    buffer.fmt("    @!%%p%u bra l_%u_done;\n", mask_reg, loop_reg);
    buffer.fmt("\nl_%u_body:\n", loop_reg);
    (void) jitc_var_loop_move(loop->in_body, loop->in_cond);
    buffer.putc('\n');
}

static void jitc_var_loop_assemble_end(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->start)->reg_index;
    buffer.putc('\n');
    (void) jitc_var_loop_move(loop->in_cond, loop->out_body);
    buffer.fmt("    bra l_%u_cond;\n", loop_reg);
    buffer.fmt("\nl_%u_done:\n", loop_reg);

    if (loop->se_count) {
        uint32_t *dep = state.extra[loop->end].dep;
        uint32_t n = (uint32_t) loop->in.size();
        for (uint32_t i = 0; i < loop->se_count; ++i) {
            uint32_t &index = dep[n + i];
            if (!jitc_var(index)->side_effect)
                jit_fail("jitc_var_loop_assemble(): internal error (3)");
            jitc_var_dec_ref_int(index);
            index = 0;
        }
        loop->se_count = 0;
    }
}
