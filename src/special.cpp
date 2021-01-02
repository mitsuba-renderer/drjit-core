#include "internal.h"
#include "log.h"
#include "var.h"

struct VCall {
    uint32_t n_inst;
    std::vector<uint32_t> in;
    std::vector<uint32_t> out_all;
    std::vector<uint32_t> n_se;
    std::vector<uint32_t> out;
    std::vector<uint32_t> se;
};

static void jitc_var_vcall_assemble_func(VCall *vcall, uint32_t index) {

}

static void jitc_var_vcall_assemble(const Variable *v, const Extra &extra) {
    VCall *vcall = (VCall *) extra.payload;
    const Variable *self = jitc_var(v->dep[0]);
    uint32_t reg_index = v->reg_index,
             n_inst = vcall->n_inst;

    for (uint32_t i = 0; i < vcall->n_inst; ++i)
        jitc_var_vcall_assemble_func(vcall, i);

    // Address of 'v', 'extra' may have changed, don't access

    uint32_t in_size = 0, in_align = 1,
             out_size = 0, out_align = 1;

    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        uint32_t size = var_type_size[it->second.type];
        in_size += size;
        in_align = std::max(size, in_align);
    }

    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t size = var_type_size[v2->type];
        out_size += size;
        out_align = std::max(size, out_align);
    }

    buffer.put("    {\n");

    for (uint32_t i = 0; i < vcall->n_inst + 1; ++i) {
        if (i == 0)
            buffer.put("        proto: .callprototype (");
        else
            buffer.put("        .func (");

        if (out_size)
            buffer.fmt(".param .align %u .b8 out[%u]", out_align, out_size);
        if (i == 0)
            buffer.put(") _ (.reg.u64 extra");
        else
            buffer.fmt(") asdf_%u (.reg.u64 extra", i);
        if (in_size)
            buffer.fmt(", .param .align %u .b8 in[%u]", in_align, in_size);
        buffer.put(");\n");
    }

    buffer.fmt("        .global .u64 calltbl_%u[] = { asdf_1, asdf_2 };\n", reg_index);
    buffer.fmt("        .reg.u64 %%target;\n"
               "        mov.u64 %%target, calltbl_%u;\n"
               "        mad.wide.u32 %%target, %s%u, 8, %%target;\n\n",
               reg_index, var_type_prefix[self->type], self->reg_index);

    if (out_size)
        buffer.fmt("        .param .align %u .b8 out[%u];\n", out_align, out_size);
    if (in_size)
        buffer.fmt("        .param .align %u .b8 in[%u];\n", in_align, in_size);

    uint32_t offset = 0;
    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = jitc_var(it->second.dep[0]);
        uint32_t size = var_type_size[v2->type];
        buffer.fmt("        st.param.%s [in+%u], %s%u;\n",
                   var_type_name_ptx[v2->type], offset,
                   var_type_prefix[v2->type], v2->reg_index);

        offset += size;
    }
    buffer.fmt("        call (%s), %%target, (0, %s), proto;\n",
               out_size ? "out" : "", in_size ? "in" : "");

    offset = 0;
    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t size = var_type_size[v2->type];
        buffer.fmt("        ld.param.%s %s%u, [out+%u];\n",
                   var_type_name_ptx[v2->type], var_type_prefix[v2->type],
                   v2->reg_index, offset);

        offset += size;
    }
    buffer.put("    }\n");
}

void jitc_var_vcall(const char *domain, uint32_t self, uint32_t n_inst,
                    uint32_t n_in, const uint32_t *in, uint32_t n_out_all,
                    const uint32_t *out_all, const uint32_t *n_se,
                    uint32_t *out) {

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    if (n_inst == 0)
        jitc_raise("jit_var_vcall(): must have at least one instance!");

    if (n_out_all % n_inst != 0)
        jitc_raise("jit_var_vcall(): list of all output indices must be a "
                   "multiple of the instance count!");

    uint32_t n_out = n_out_all / n_inst, n_in_placeholder = 0, size = 0;

    JitBackend backend;
    /* Check 'self' */ {
        const Variable *self_v = jitc_var(self);
        size = self_v->size;
        backend = (JitBackend) self_v->backend;
        if ((VarType) self_v->type != VarType::UInt32)
            jitc_raise("jit_var_vcall(): 'self' argument must be of type "
                       "UInt32 (was: %s)", var_type_name[self_v->type]);
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->placeholder)
            n_in_placeholder++;
        else if (!v->literal)
            jitc_raise("jit_var_vcall(): inputs must either be literal or "
                       "placeholder variables!");
        size = std::max(size, v->size);
    }

    for (uint32_t i = 0; i < n_out_all; ++i) {
        const Variable *v = jitc_var(out_all[i]),
                       *v0 = jitc_var(out_all[i % n_out]);
        size = std::max(size, v->size);
        if (v->type != v0->type)
            jitc_raise(
                "jit_var_vcall(): output types don't match between instances!");
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise(
                "jit_var_vcall(): input/output arrays have incompatible size!");
    }

    for (uint32_t i = 0; i < n_out_all; ++i) {
        const Variable *v = jitc_var(out_all[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise(
                "jit_var_vcall(): input/output arrays have incompatible size!");
    }

    jitc_log(Info, "jit_var_vcall(n_inst=%u, in=%u, out=%u, se=%u)", n_inst,
             n_in, n_out, n_se[n_inst - 1] - n_se[0]);

    // =====================================================
    // 2. Stash information about inputs and outputs
    // =====================================================

    std::unique_ptr<VCall> vcall(new VCall());
    vcall->n_inst = n_inst;
    vcall->in = std::vector<uint32_t>(in, in + n_in);
    vcall->out_all = std::vector<uint32_t>(out_all, out_all + n_out_all);
    vcall->n_se = std::vector<uint32_t>(n_se, n_se + n_inst);
    vcall->out.resize(n_out, 0);

    // Optimize calling conventions by reordering from largest -> smallest
    auto comp = [](uint32_t i1, uint32_t i2) {
        return var_type_size[jitc_var(i1)->type] >
               var_type_size[jitc_var(i2)->type];
    };

    std::sort(vcall->in.begin(), vcall->in.begin() + n_in, comp);
    for (uint32_t i = 0; i < n_inst; ++i)
        std::sort(vcall->out_all.begin() + (i + 0) * n_out,
                  vcall->out_all.begin() + (i + 1) * n_out, comp);

    char temp[128];
    snprintf(temp, sizeof(temp), "Indirect call: %s", domain);
    uint32_t special =
        jitc_var_new_stmt(backend, VarType::Void, temp, 0, 1, &self);

    // Associate extra record with this variable
    Extra *extra = &state.extra[special];
    Variable *v = jitc_var(special);
    v->extra = 1;
    v->size = size;

    // Register dependencies
    extra->dep_count = n_in_placeholder;
    extra->dep = (uint32_t *) malloc(sizeof(uint32_t) * n_in_placeholder);
    uint32_t *p = extra->dep;

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (!v->placeholder)
            continue;
        uint32_t index = v->dep[0];
        jitc_var_inc_ref_int(index);
        *p++ = index;
    }

    for (uint32_t i = 0; i < n_out_all; ++i)
        jitc_var_inc_ref_int(out_all[i]);

    extra->free_callback = [](void *ptr) { delete (VCall *) ptr; };
    extra->assemble = jitc_var_vcall_assemble;

    for (uint32_t i = 0; i < n_out; ++i) {
        const Variable *v = jitc_var(out_all[i]);
        out[i] = vcall->out[i] =
            jitc_var_new_stmt(backend, (VarType) v->type, "", 0, 1, &special);
    }

    state.extra[special].payload = vcall.release();
}
