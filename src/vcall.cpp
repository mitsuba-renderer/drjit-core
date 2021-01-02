/*
    src/vcall.cpp -- Code generation for virtual function calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"

/// Encodes information about a virtual function call
struct VCall {
    JitBackend backend;

    /// Number of instances
    uint32_t n_inst;

    /// Input variables at call site
    std::vector<uint32_t> in;

    /// Input placeholder variables
    std::vector<uint32_t> in_nested;

    /// Output variables at call site
    std::vector<uint32_t> out;

    /// Output variables *per instance*
    std::vector<uint32_t> out_nested;

    /// Number of side effects *per instance*
    std::vector<uint32_t> n_se;

    /// Compressed side effect index list
    std::vector<uint32_t> se;
};

// Forward declarations
static void jitc_var_vcall_assemble(uint32_t self_reg, VCall *v);

static XXH128_hash_t jitc_var_vcall_assemble_func(VCall *vcall,
                                                  uint32_t instance_id,
                                                  uint32_t in_size,
                                                  uint32_t in_align,
                                                  uint32_t out_size,
                                                  uint32_t out_align);

// Weave a virtual function call into the computation graph
void jitc_var_vcall(const char *domain, uint32_t self, uint32_t n_inst,
                    uint32_t n_in, const uint32_t *in, uint32_t n_out_nested,
                    const uint32_t *out_nested, const uint32_t *n_se,
                    uint32_t *out) {

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    if (n_inst == 0)
        jitc_raise("jit_var_vcall(): must have at least one instance!");

    if (n_out_nested % n_inst != 0)
        jitc_raise("jit_var_vcall(): list of all output indices must be a "
                   "multiple of the instance count!");

    uint32_t n_out = n_out_nested / n_inst, size = 0;

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
        if (v->placeholder) {
            if (!v->dep[3])
                jitc_raise("jit_var_vcall(): placeholder variable does not "
                           "reference another input!");
        } else if (!v->literal) {
            jitc_raise("jit_var_vcall(): inputs must either be literal or "
                       "placeholder variables!");
        }
        size = std::max(size, v->size);
    }

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v = jitc_var(out_nested[i]),
                       *v0 = jitc_var(out_nested[i % n_out]);
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

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v = jitc_var(out_nested[i]);
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
    vcall->backend = backend;
    vcall->n_inst = n_inst;
    vcall->in_nested.reserve(n_in);
    vcall->out.reserve(n_out);
    vcall->out_nested = std::vector<uint32_t>(out_nested, out_nested + n_out_nested);
    vcall->n_se = std::vector<uint32_t>(n_se, n_se + n_inst);

    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t index = in[i];
        Variable *v = jitc_var(index);
        if (!v->placeholder)
            continue;
        vcall->in.push_back(v->dep[3]);
        vcall->in_nested.push_back(index);
        // steal this reference
        v->dep[3] = 0;
    }

    // =====================================================
    // 3. Create special variable encoding the function call
    // =====================================================

    uint32_t special =
        jitc_var_new_stmt(backend, VarType::Void, "", 0, 1, &self);

    // Associate 'extra' record with this variable
    Variable *v = jitc_var(special);
    Extra *extra = &state.extra[special];
    v->extra = 1;
    v->size = size;
    size_t dep_size = vcall->in.size() * sizeof(uint32_t);
    extra->n_dep = (uint32_t) vcall->in.size();
    extra->dep = (uint32_t *) malloc(dep_size);
    memcpy(extra->dep, vcall->in.data(), dep_size);

    char temp[128];
    snprintf(temp, sizeof(temp), "VCall: %s", domain);
    extra->label = strdup(temp);

    // =====================================================
    // 4. Register deallocation and codegen callbacks
    // =====================================================

    for (auto index: vcall->out_nested)
        jitc_var_inc_ref_int(index);

    extra->free_callback = [](void *ptr) {
        VCall *vcall_2 = (VCall *) ptr;
        for (uint32_t index: vcall_2->out_nested)
            jitc_var_dec_ref_int(index);
        delete (VCall *) vcall_2;
    };

    extra->assemble = [](const Variable *v, const Extra &extra) {
        jitc_var_vcall_assemble(jitc_var(v->dep[0])->reg_index,
                                (VCall *) extra.payload);
    };

    // =====================================================
    // 5. Create output variables
    // =====================================================

    for (uint32_t i = 0; i < n_out; ++i) {
        const Variable *v = jitc_var(out_nested[i]);
        Variable v2;
        v2.stmt = (char *) "";
        v2.size = size;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.dep[0] = special;
        jitc_var_inc_ref_int(special);
        uint32_t index = jitc_var_new(v2, true);
        vcall->out.push_back(index);
        snprintf(temp, sizeof(temp), "VCall: %s [out %u]", domain, i);
        jitc_var_set_label(index, temp);
        out[i] = index;
    }

    // =====================================================
    // 6. Optimize calling conventions by reordering args
    // =====================================================

    auto comp = [](uint32_t i1, uint32_t i2) {
        return var_type_size[jitc_var(i1)->type] >
               var_type_size[jitc_var(i2)->type];
    };

    std::sort(vcall->in.begin(), vcall->in.end(), comp);
    std::sort(vcall->out.begin(), vcall->out.end(), comp);

    std::sort(vcall->in_nested.begin(), vcall->in_nested.end(), comp);
    for (uint32_t i = 0; i < n_inst; ++i)
        std::sort(vcall->out_nested.begin() + (i + 0) * n_out,
                  vcall->out_nested.begin() + (i + 1) * n_out, comp);

    state.extra[special].payload = vcall.release();
}

/// Called by the JIT compiler when compiling
static void jitc_var_vcall_assemble(uint32_t self_reg,
                                    VCall *vcall) {
    // =====================================================
    // 1. Need to backup state before we can JIT recursively
    // =====================================================

    struct JitBackupRecord {
        ScheduledVariable sv;
        uint32_t output_flag : 1;
        uint32_t param_type : 2;
        uint32_t param_offset;
        uint32_t reg_index;
    };


    std::vector<JitBackupRecord> backup;
    backup.reserve(schedule.size());
    for (const ScheduledVariable &sv : schedule) {
        const Variable *v = jitc_var(sv.index);
        backup.push_back(JitBackupRecord{ sv, v->output_flag, v->param_type,
                                          v->param_offset, v->reg_index });
    }

    // =====================================================
    // 2. Determine calling conventions
    // =====================================================

    uint32_t in_size = 0, in_align = 1,
             out_size = 0, out_align = 1;

    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        Variable *v2 = &it.value();
        uint32_t size = var_type_size[v2->type];
        v2->param_offset = in_size;
        in_size += size;
        in_align = std::max(size, in_align);
    }

    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        Variable *v2 = &it.value();
        uint32_t size = var_type_size[v2->type];
        v2->param_offset = out_size;
        out_size += size;
        out_align = std::max(size, out_align);
    }

    // =====================================================
    // 3. Compile code for all instances and collapse
    // =====================================================

    std::vector<XXH128_hash_t> func_id(vcall->n_inst);
    for (uint32_t i = 0; i < vcall->n_inst; ++i)
        func_id[i] = jitc_var_vcall_assemble_func(vcall, i, in_size, in_align,
                                                  out_size, out_align);

    // =====================================================
    // 4. Insert call prototypes
    // =====================================================

    buffer.put("    {\n");

    for (uint32_t i = 0; i < vcall->n_inst + 1; ++i) {
        if (i == 0)
            buffer.put("        proto: .callprototype (");
        else
            buffer.put("        .visible .func (");

        if (out_size)
            buffer.fmt(".param .align %u .b8 result[%u]", out_align, out_size);
        if (i == 0)
            buffer.put(") _(.reg .u64 extra");
        else
            buffer.fmt(") func_%016llx%016llx(.reg.u64 extra",
                       (unsigned long long) func_id[i - 1].high64,
                       (unsigned long long) func_id[i - 1].low64);
        if (in_size)
            buffer.fmt(", .param .align %u .b8 params[%u]", in_align, in_size);
        buffer.put(");\n");
    }

    // =====================================================
    // 5. Insert call table and lookup sequence
    // =====================================================

    buffer.put("        .global .u64 tbl[] = {\n");
    for (uint32_t i = 0; i < vcall->n_inst; ++i) {
        buffer.fmt("            func_%016llx%016llx%s",
                   (unsigned long long) func_id[i].high64,
                   (unsigned long long) func_id[i].low64,
                   i + 1 < vcall->n_inst ? ",\n" : "\n");
    }
    buffer.fmt("        };\n\n"
               "        setp.ne.u32 %%p3, %%r%u, 0;\n"
               "        @%%p3 ld.global.u64 %%rd3, tbl[%%r%u + (-1)];\n",
               self_reg, self_reg);

    // =====================================================
    // 6. Insert the actual call
    // =====================================================

    // Special handling for predicates
    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        if ((VarType) v2->type != VarType::Bool)
            continue;
        buffer.fmt("        selp.u16 %%w%u, 1, 0, %%p%u\n",
                   v2->reg_index, v2->reg_index);
    }

    buffer.put("\n        {\n");
    if (out_size)
        buffer.fmt("            .param .align %u .b8 out[%u];\n", out_align, out_size);
    if (in_size)
        buffer.fmt("            .param .align %u .b8 in[%u];\n", in_align, in_size);

    uint32_t offset = 0;
    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t size = var_type_size[v2->type];

        const char *tname = var_type_name_ptx[v2->type],
                   *prefix = var_type_prefix[v2->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v2->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        buffer.fmt("            st.param.%s [in+%u], %s%u;\n", tname, offset,
                   prefix, v2->reg_index);

        offset += size;
    }

    buffer.fmt("            @%%p3 call (%s), %%rd3, (%%rd2%s), proto;\n",
               out_size ? "out" : "", in_size ? ", in" : "");

    offset = 0;
    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t size = var_type_size[v2->type];

        const char *tname = var_type_name_ptx[v2->type],
                   *prefix = var_type_prefix[v2->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v2->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        buffer.fmt("            ld.param.%s %s%u, [out+%u];\n",
                   tname, prefix, v2->reg_index, offset);
        offset += size;
    }

    buffer.put("        }\n\n");

    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        if ((VarType) v2->type == VarType::Bool) {
            // Special handling for predicates
            buffer.fmt("        setp.ne.u16 %%p%u, %%w%u, 0;\n",
                       v2->reg_index, v2->reg_index);
        }
        buffer.fmt("        @!%%p3 mov.%s %s%u, 0;\n",
                   var_type_name_ptx_bin[v2->type],
                   var_type_prefix[v2->type], v2->reg_index);
    }

    buffer.put("    }\n");

    // =====================================================
    // 7. Restore previously backed-up JIT state
    // =====================================================

    schedule.clear();
    for (const JitBackupRecord &b : backup) {
        Variable *v = jitc_var(b.sv.index);
        v->output_flag = b.output_flag;
        v->param_type = b.param_type;
        v->param_offset = b.param_offset;
        v->reg_index = b.reg_index;
        schedule.push_back(b.sv);
    }
}

static XXH128_hash_t jitc_var_vcall_assemble_func(VCall *vcall,
                                                  uint32_t instance_id,
                                                  uint32_t in_size,
                                                  uint32_t in_align,
                                                  uint32_t out_size,
                                                  uint32_t out_align) {
    ThreadState *ts = thread_state(vcall->backend);

    // =====================================================
    // 1. Transfer calling conventions to instance
    // =====================================================

    for (size_t i = 0; i < vcall->in.size(); ++i) {
        auto it1 = state.variables.find(vcall->in_nested[i]);
        auto it2 = state.variables.find(vcall->in[i]);

        if ((it1 == state.variables.end()) != (it1 == state.variables.end()))
            jit_fail("jitc_var_vcall_assemble_func(): internal error!");
        else if (it1 == state.variables.end())
            continue;

        it1.value().param_offset = it2.value().param_offset;
    }

    uint32_t out_offset = vcall->out.size() * instance_id;
    for (size_t i = 0; i < vcall->out.size(); ++i) {
        uint32_t index = vcall->out_nested[i + out_offset];
        auto it1 = state.variables.find(index);
        auto it2 = state.variables.find(vcall->out[i]);

        if ((it1 == state.variables.end()) != (it1 == state.variables.end()))
            jit_fail("jitc_var_vcall_assemble_func(): internal error!");
        else if (it1 == state.variables.end())
            continue;

        it1.value().param_offset = it2.value().param_offset;
        ts->scheduled.push_back(index);
    }

    // =====================================================
    // 2. Go!!!
    // =====================================================

    return jitc_assemble_func(ts, in_size, in_align, out_size, out_align);
}
