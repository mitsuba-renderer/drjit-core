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

    /// Implement call via indirect branch?
    bool branch;

    /// ID of call variable
    uint32_t id;

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

    ~VCall() {
        for (uint32_t index : out_nested)
            jitc_var_dec_ref_ext(index);
    }
};

// Forward declarations
static void jitc_var_vcall_assemble(uint32_t self_reg, VCall *v);

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
                       "UInt32 (was: %s)", type_name[self_v->type]);
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->placeholder) {
            if (!v->dep[0])
                jitc_raise("jit_var_vcall(): placeholder variable does not "
                           "reference another input!");
        } else if (!v->literal) {
            jitc_raise("jit_var_vcall(): inputs must either be literal or "
                       "placeholder variables!");
        }
        size = std::max(size, v->size);
    }

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v  = jitc_var(out_nested[i]),
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

    // =====================================================
    // 2. Create special variable encoding the function call
    // =====================================================

    Ref special = steal(jitc_var_new_stmt(backend, VarType::Void, "", 0, 1, &self));

    // =====================================================
    // 3. Stash information about inputs and outputs
    // =====================================================

    std::unique_ptr<VCall> vcall(new VCall());
    vcall->backend = backend;
    vcall->branch = jitc_flags() & (uint32_t) JitFlag::VCallBranch;
    vcall->id = special;
    vcall->n_inst = n_inst;
    vcall->in.reserve(n_in);
    vcall->in_nested.reserve(n_in);
    vcall->out.reserve(n_out);
    vcall->out_nested.reserve(n_out_nested);
    vcall->n_se = std::vector<uint32_t>(n_se, n_se + n_inst);
    std::vector<bool> coherent(n_out, true);

    // Reference the nested computation until cleanup by the callback below
    for (uint32_t i = 0; i < n_inst; ++i) {
        for (uint32_t j = 0; j < n_out; ++j) {
            uint32_t index = out_nested[i * n_out + j];
            coherent[j] = coherent[j] && (index == out_nested[j]);
            jitc_var_inc_ref_ext(index);
            vcall->out_nested.push_back(index);
        }
    }

    uint32_t n_devirt = 0;

    bool optimize = jitc_flags() & (uint32_t) JitFlag::VCallOptimize;
    if (optimize) {
        for (uint32_t j = 0; j < n_out; ++j) {
            if (!coherent[j])
                continue;
            for (uint32_t i = 0; i < n_inst; ++i) {
                uint32_t &index = vcall->out_nested[i * n_out + j];
                jitc_var_dec_ref_ext(index);
                index = 0;
            }
            uint32_t index = out_nested[j];
            out[j] = jitc_var_resize(index, size);
            n_devirt++;
        }
    }

    jitc_log(Info,
             "jit_var_vcall(r%u): %u instances, %u inputs, %u outputs, %u "
             "devirtualized, %u side effects", self,
             n_inst, n_in, n_out, n_devirt, n_se[n_inst - 1] - n_se[0]);

    // =====================================================
    // 4. Create output variables
    // =====================================================

    auto var_callback = [](uint32_t index, int free, void *ptr) {
        if (!ptr)
            return;

        VCall *vcall_2 = (VCall *) ptr;
        if (!free) {
            // Disable callback
            state.extra[index].payload = nullptr;
        }

        // An output variable is no longer referenced. Find out which one.
        uint32_t n_out_2 = vcall_2->out.size();
        uint32_t offset = (uint32_t) -1;
        for (uint32_t i = 0; i < n_out_2; ++i) {
            if (vcall_2->out[i] == index)
                offset = i;
        }

        if (offset == (uint32_t) -1)
            jitc_fail("jit_var_vcall(): expired output variable %u could not "
                      "be located!", index);

        // Inform recursive computation graphs via reference counting
        for (uint32_t j = 0; j < vcall_2->n_inst; ++j) {
            uint32_t &index_2 = vcall_2->out_nested[n_out_2 * j + offset];
            jitc_var_dec_ref_ext(index_2);
            index_2 = 0;
        }
        vcall_2->out[offset] = 0;

        // Check if any input parameters became irrelevant
        for (uint32_t i = 0; i < vcall_2->in.size(); ++i) {
            uint32_t index_2 = vcall_2->in_nested[i];
            if (index_2 &&
                state.variables.find(index_2) == state.variables.end()) {
                jitc_var_dec_ref_int(vcall_2->in[i]);
                vcall_2->in[i] = 0;
                vcall_2->in_nested[i] = 0;
                state.extra[vcall_2->id].dep[i] = 0;
            }
        }
    };

    char temp[128];
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i];
        if (!index) {
            vcall->out.push_back(0);
            continue;
        }

        snprintf(temp, sizeof(temp), "VCall: %s [out %u]", domain, i);

        const Variable *v = jitc_var(index);
        Variable v2;
        v2.stmt = (char *) "";
        v2.size = size;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.dep[0] = special;
        v2.extra = 1;
        jitc_var_inc_ref_int(special);
        uint32_t index_2 = jitc_var_new(v2, true);
        Extra &extra = state.extra[index_2];
        if (optimize) {
            extra.payload = vcall.get();
            extra.callback = var_callback;
            extra.callback_internal = true;
        }
        extra.label = strdup(temp);
        vcall->out.push_back(index_2);
        out[i] = index_2;
    }

    special.reset();

    // =====================================================
    // 5. Optimize calling conventions by reordering args
    // =====================================================

    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t index = in[i];
        Variable *v = jitc_var(index);
        if (!v->placeholder)
            continue;

        // Ignore unreferenced inputs
        if (optimize && v->ref_count_int == 0)
            continue;

        vcall->in_nested.push_back(index);

        uint32_t &index_2 = v->dep[0];
        vcall->in.push_back(index_2);
        jitc_var_inc_ref_int(index_2);
    }

    auto comp = [](uint32_t i0, uint32_t i1) {
        int s0 = i0 ? type_size[jitc_var(i0)->type] : 0;
        int s1 = i1 ? type_size[jitc_var(i1)->type] : 0;
        return s0 > s1;
    };

    std::sort(vcall->in.begin(), vcall->in.end(), comp);
    std::sort(vcall->out.begin(), vcall->out.end(), comp);
    std::sort(vcall->in_nested.begin(), vcall->in_nested.end(), comp);
    for (uint32_t i = 0; i < n_inst; ++i)
        std::sort(vcall->out_nested.begin() + (i + 0) * n_out,
                  vcall->out_nested.begin() + (i + 1) * n_out, comp);

    // =====================================================
    // 6. Install callbacks for call variable
    // =====================================================

    snprintf(temp, sizeof(temp), "VCall: %s", domain);
    size_t dep_size = vcall->in.size() * sizeof(uint32_t);

    Variable *v_special = jitc_var(vcall->id);
    Extra *e_special = &state.extra[vcall->id];
    v_special->extra = 1;
    v_special->size = size;
    e_special->label = strdup(temp);
    e_special->n_dep = (uint32_t) vcall->in.size();
    e_special->dep = (uint32_t *) malloc(dep_size);

    /// Steal input dependencies from placeholder arguments
    memcpy(e_special->dep, vcall->in.data(), dep_size);

    e_special->payload = vcall.release();
    e_special->callback = [](uint32_t, int free, void *ptr) {
        if (free)
            delete (VCall *) ptr;
    };
    e_special->callback_internal = true;

    e_special->assemble = [](const Variable *v, const Extra &extra) {
        jitc_var_vcall_assemble(jitc_var(v->dep[0])->reg_index,
                                (VCall *) extra.payload);
    };
}

/// Called by the JIT compiler when compiling
static void jitc_var_vcall_assemble(uint32_t self_reg,
                                    VCall *vcall) {
    // =====================================================
    // 1. Need to backup state before we can JIT recursively
    // =====================================================

    struct JitBackupRecord {
        ScheduledVariable sv;
        uint32_t param_type : 2;
        uint32_t output_flag : 1;
        uint32_t reg_index;
        uint32_t param_offset;
    };

    std::vector<JitBackupRecord> backup;
    backup.reserve(schedule.size());
    for (const ScheduledVariable &sv : schedule) {
        const Variable *v = jitc_var(sv.index);
        backup.push_back(JitBackupRecord{ sv, v->param_type, v->output_flag,
                                          v->reg_index, v->param_offset });
    }

    // =====================================================
    // 2. Determine calling conventions
    // =====================================================

    uint32_t in_size = 0, in_align = 1,
             out_size = 0, out_align = 1,
             n_in = vcall->in.size(),
             n_out = vcall->out.size(),
             n_in_active = 0,
             n_out_active = 0,
             extra_size = 0;

    for (uint32_t i = 0; i < n_in; ++i) {
        auto it = state.variables.find(vcall->in[i]);
        if (it == state.variables.end())
            continue;
        uint32_t size = type_size[it.value().type],
                 offset = in_size;
        Variable *v = &it.value();
        in_size += size;
        in_align = std::max(size, in_align);
        n_in_active++;

        // Transfer to instances
        auto it2 = state.variables.find(vcall->in_nested[i]);
        if (it2 == state.variables.end())
            continue;
        Variable *v2 = &it2.value();
        v2->param_offset = offset;
        v2->reg_index = v->reg_index;
    }

    for (uint32_t i = 0; i < n_out; ++i) {
        auto it = state.variables.find(vcall->out_nested[i]);
        if (it == state.variables.end())
            continue;
        Variable *v = &it.value();
        uint32_t size = type_size[v->type];
        out_size += size;
        out_align = std::max(size, out_align);
        n_out_active++;
    }

    // =====================================================
    // 3. Compile code for all instances and collapse
    // =====================================================

    char ret_label[32];
    snprintf(ret_label, sizeof(ret_label), "l_%u_done",
             jitc_var(vcall->id)->reg_index);
    size_t globals_offset = globals.size();

    std::vector<XXH128_hash_t> func_id(vcall->n_inst);
    ThreadState *ts = thread_state(vcall->backend);
    for (uint32_t i = 0; i < vcall->n_inst; ++i)
        func_id[i] = jitc_assemble_func(
            ts, in_size, in_align, out_size, out_align, extra_size, n_in,
            vcall->in.data(), n_out, vcall->out.data(),
            vcall->out_nested.data() + n_out * i,
            vcall->branch ? ret_label : nullptr);

    // =====================================================
    // 4. Restore previously backed-up JIT state
    // =====================================================

    schedule.clear();
    for (const JitBackupRecord &b : backup) {
        Variable *v = jitc_var(b.sv.index);
        v->param_type = b.param_type;
        v->output_flag = b.output_flag;
        v->reg_index = b.reg_index;
        v->param_offset = b.param_offset;
        schedule.push_back(b.sv);
    }

    // =====================================================
    // 5. Insert call prototypes
    // =====================================================

    buffer.put("    {\n");
    if (!vcall->branch) {
        buffer.put("        proto: .callprototype");
        if (out_size)
            buffer.fmt(" (.param .align %u .b8 result[%u])", out_align, out_size);
        buffer.put(" _(");
        if (extra_size) {
            buffer.put(".reg .u64 extra");
            if (extra_size)
                buffer.put(", ");
        }
        if (in_size)
            buffer.fmt(".param .align %u .b8 params[%u]", in_align, in_size);
        buffer.put(");\n");
    }

    // =====================================================
    // 6. Insert call table and lookup sequence
    // =====================================================

    if (vcall->branch)
        buffer.put("        bt: .branchtargets\n");
    else
        buffer.put("        .global .u64 tbl[] = {\n");

    for (uint32_t i = 0; i < vcall->n_inst; ++i) {
        buffer.fmt("            %s_%016llx%016llx%s",
                   vcall->branch ? "l" : "func",
                   (unsigned long long) func_id[i].high64,
                   (unsigned long long) func_id[i].low64,
                   i + 1 < vcall->n_inst ? ",\n" : "");
    }

    if (vcall->branch) {
        buffer.fmt(";\n\n"
                   "        setp.ne.u32 %%p3, %%r%u, 0;\n"
                   "        sub.u32 %%r3, %%r%u, 1;\n"
                   "        @%%p3 brx.idx %%r3, bt;\n"
                   , self_reg, self_reg);
    } else {
        buffer.fmt(" };\n\n"
                   "        setp.ne.u32 %%p3, %%r%u, 0;\n"
                   "        @%%p3 ld.global.u64 %%rd2, tbl[%%r%u + (-1)];\n",
                   self_reg, self_reg);
    }

    // =====================================================
    // 7. Insert the actual call
    // =====================================================

    if (!vcall->branch) {
        // Special handling for predicates
        for (uint32_t in : vcall->in) {
            auto it = state.variables.find(in);
            if (it == state.variables.end())
                continue;
            const Variable *v2 = &it->second;

            if ((VarType) v2->type != VarType::Bool)
                continue;

            buffer.fmt("        selp.u16 %%w%u, 1, 0, %%p%u;\n",
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
            uint32_t size = type_size[v2->type];

            const char *tname = type_name_ptx[v2->type],
                       *prefix = type_prefix[v2->type];

            // Special handling for predicates (pass via u8)
            if ((VarType) v2->type == VarType::Bool) {
                tname = "u8";
                prefix = "%w";
            }

            buffer.fmt("            st.param.%s [in+%u], %s%u;\n", tname, offset,
                       prefix, v2->reg_index);

            offset += size;
        }

        buffer.fmt("            @%%p3 call %s%%rd2, (%s%s%s), proto;\n",
                   out_size ? "(out), " : "",
                   extra_size ? "%rd3" : "",
                   extra_size && in_size ? ", " : "",
                   in_size ? "in" : "");

        offset = 0;
        for (uint32_t i = 0; i < n_out; ++i) {
            uint32_t index = vcall->out_nested[i],
                     index_2 = vcall->out[i];
            auto it = state.variables.find(index);
            if (it == state.variables.end())
                continue;
            uint32_t size = type_size[it->second.type],
                     load_offset = offset;
            offset += size;

            // Skip if outer access expired
            auto it2 = state.variables.find(index_2);
            if (it2 == state.variables.end())
                continue;

            const Variable *v2 = &it2.value();

            const char *tname = type_name_ptx[v2->type],
                       *prefix = type_prefix[v2->type];

            // Special handling for predicates (pass via u8)
            if ((VarType) v2->type == VarType::Bool) {
                tname = "u8";
                prefix = "%w";
            }

            buffer.fmt("            ld.param.%s %s%u, [out+%u];\n",
                       tname, prefix, v2->reg_index, load_offset);
        }

        buffer.put("        }\n\n");

        for (uint32_t out : vcall->out) {
            auto it = state.variables.find(out);
            if (it == state.variables.end())
                continue;
            const Variable *v2 = &it->second;
            if ((VarType) v2->type != VarType::Bool)
                continue;

            // Special handling for predicates
            buffer.fmt("        setp.ne.u16 %%p%u, %%w%u, 0;\n",
                       v2->reg_index, v2->reg_index);
        }
    }

    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        buffer.fmt("        @!%%p3 mov.%s %s%u, 0;\n",
                   type_name_ptx_bin[v2->type],
                   type_prefix[v2->type], v2->reg_index);
    }

    if (vcall->branch)
        buffer.fmt("        bra %s;\n", ret_label);

    buffer.put("    }\n");

    if (vcall->branch) {
        buffer.putc('\n');
        for (uint32_t i = globals_offset; i < globals.size(); ++i)
            buffer.put(globals[i].c_str(), globals[i].length());
        buffer.fmt("%s:\n", ret_label);
        globals.resize(globals_offset);
    }

    std::sort(
        func_id.begin(), func_id.end(), [](XXH128_hash_t a, XXH128_hash_t b) {
            return std::tie(a.high64, a.low64) < std::tie(b.high64, b.low64);
        });

    size_t n_unique = std::unique(
        func_id.begin(), func_id.end(), [](XXH128_hash_t a, XXH128_hash_t b) {
            return std::tie(a.high64, a.low64) == std::tie(b.high64, b.low64);
        }) - func_id.begin();

    jitc_log(Info,
             "jit_var_vcall_assemble(): indirect %s to %zu/%zu instances, "
             "passing %u/%u inputs (%u bytes), %u/%u outputs (%u bytes)",
             vcall->branch ? "branch" : "call", n_unique, func_id.size(),
             n_in_active, n_in, in_size, n_out_active, n_out, out_size);

}
