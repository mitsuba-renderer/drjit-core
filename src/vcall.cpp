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
#include "registry.h"
#include "util.h"

/// Encodes information about a virtual function call
struct VCall {
    JitBackend backend;

    /// A descriptive name
    char *name = nullptr;

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

    /// Per-instance offsets into side effects list
    std::vector<uint32_t> se_offset;
    /// Compressed side effect index list
    std::vector<uint32_t> se;

    /// Mapping from variable index to offset into call data
    tsl::robin_map<uint64_t, uint32_t> data_map;
    std::vector<uint32_t> data_offsets;

    /// Storage in bytes for inputs/outputs before simplifications
    uint32_t in_count_initial = 0;
    uint32_t in_size_initial = 0;
    uint32_t out_size_initial = 0;

    ~VCall() {
        for (uint32_t index : out_nested)
            jitc_var_dec_ref_ext(index);
        clear_side_effects();
        free(name);
    }

    void clear_side_effects() {
        if (se_offset.empty() || se_offset.back() == se_offset.front())
            return;
        for (uint32_t index : se)
            jitc_var_dec_ref_ext(index);
        se.clear();
        std::fill(se_offset.begin(), se_offset.end(), 0);
    }
};

// Forward declarations
static void jitc_var_vcall_assemble(VCall *vcall, uint32_t self_reg,
                                    uint32_t offset_reg, uint32_t data_reg);
static void jitc_var_vcall_assemble_cuda(
    VCall *vcall, const std::vector<XXH128_hash_t> &func_id, uint32_t vcall_reg,
    uint32_t self_reg, uint32_t offset_reg, uint32_t data_reg,
    uint32_t n_out, uint32_t in_size, uint32_t in_align,
    uint32_t out_size, uint32_t out_align, const char *ret_label,
    uint32_t vcall_table_offset);

static void jitc_var_vcall_assemble_llvm(
    VCall *vcall, uint32_t vcall_reg, uint32_t self_reg, uint32_t offset_reg,
    uint32_t data_reg, uint32_t n_out, uint32_t in_size, uint32_t in_align,
    uint32_t out_size, uint32_t out_align, uint32_t vcall_table_offset);

static void
jitc_var_vcall_collect_data(tsl::robin_map<uint64_t, uint32_t> &data_map,
                            uint32_t &data_offset, uint32_t inst_id,
                            uint32_t index);

// Weave a virtual function call into the computation graph
void jitc_var_vcall(const char *name, uint32_t self, uint32_t n_inst,
                    uint32_t n_in, const uint32_t *in, uint32_t n_out_nested,
                    const uint32_t *out_nested, const uint32_t *se_offset,
                    uint32_t *out) {

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    if (n_inst == 0)
        jitc_raise("jit_var_vcall(): must have at least one instance!");

    if (n_out_nested % n_inst != 0)
        jitc_raise("jit_var_vcall(): list of all output indices must be a "
                   "multiple of the instance count!");

    uint32_t n_out = n_out_nested / n_inst, size = 0,
             in_size_initial = 0, out_size_initial = 0;

    bool placeholder = false;

    JitBackend backend;
    /* Check 'self' */ {
        const Variable *self_v = jitc_var(self);
        size = self_v->size;
        placeholder |= self_v->placeholder;
        backend = (JitBackend) self_v->backend;
        if ((VarType) self_v->type != VarType::UInt32)
            jitc_raise("jit_var_vcall(): 'self' argument must be of type "
                       "UInt32 (was: %s)", type_name[self_v->type]);
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->placeholder_iface) {
            if (!v->dep[0])
                jitc_raise("jit_var_vcall(): placeholder variable r%u does not "
                           "reference another input!", in[i]);
            placeholder |= jitc_var(v->dep[0])->placeholder;
        } else if (!v->literal) {
            jitc_raise("jit_var_vcall(): input variable r%u must either be a "
                       "literal or placeholder wrapping another variable!", in[i]);
        }
        size = std::max(size, v->size);
        in_size_initial += type_size[v->type];
    }

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v  = jitc_var(out_nested[i]),
                       *v0 = jitc_var(out_nested[i % n_out]);
        size = std::max(size, v->size);
        if (v->type != v0->type)
            jitc_raise(
                "jit_var_vcall(): output types don't match between instances!");

        if (i < n_out)
            out_size_initial += type_size[v->type];
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

    ThreadState *ts = thread_state(backend);
    if (ts->side_effects.size() != se_offset[n_inst])
        jitc_raise("jitc_var_vcall(): side effect queue doesn't have the "
                   "expected size!");

    // =====================================================
    // 2. Allocate call data
    // =====================================================

    tsl::robin_map<uint64_t, uint32_t> data_map;
    std::vector<uint32_t> data_offsets;
    uint32_t data_size = 0;

    // Collect accesses to evaluated variables/pointers
    for (uint32_t i = 0; i < n_inst; ++i) {
        data_offsets.push_back(data_size);

        for (uint32_t j = 0; j < n_out; ++j)
            jitc_var_vcall_collect_data(data_map, data_size, i,
                                        out_nested[j + i * n_out]);

        for (uint32_t j = se_offset[i]; j != se_offset[i + 1]; ++j)
            jitc_var_vcall_collect_data(data_map, data_size, i,
                                        ts->side_effects[j]);

        // Restore to full alignment
        data_size = (data_size + 7) / 8 * 8;
    }

    Ref data_v, data_offsets_v;

    if (data_size) {
        void *data_offsets_d =
            jitc_malloc(backend == JitBackend::CUDA ? AllocType::HostPinned
                                                    : AllocType::Host,
                        n_inst * sizeof(uint32_t));
        memcpy(data_offsets_d, data_offsets.data(), n_inst * sizeof(uint32_t));

        if (backend == JitBackend::CUDA)
            data_offsets_d = jitc_malloc_migrate(data_offsets_d, AllocType::Device, 1);

        uint8_t *data_d = (uint8_t *) jitc_malloc(backend == JitBackend::CUDA
                                                      ? AllocType::Device
                                                      : AllocType::HostAsync,
                                                  data_size);

        for (auto kv : data_map) {
            uint32_t index = (uint32_t) kv.first, offset = kv.second;
            if (offset == (uint32_t) -1)
                continue;

            const Variable *v = jitc_var(index);
            if ((VarType) v->type == VarType::Pointer)
                jitc_poke(backend, data_d + offset, &v->value, sizeof(void *));
            else
                jitc_memcpy_async(backend, data_d + offset, v->data,
                                  type_size[v->type]);
        }

        Ref data_offsets_holder = steal(jitc_var_mem_map(
                backend, VarType::UInt32, data_offsets_d, n_inst, 1)),
            data_holder = steal(jitc_var_mem_map(
                backend, VarType::UInt8, data_d, data_size, 1));

        data_offsets_v = steal(jitc_var_new_pointer(backend, data_offsets_d,
                                                    data_offsets_holder, 0));
        data_v = steal(jitc_var_new_pointer(backend, data_d, data_holder, 0));
    } else {
        data_map.clear();
    }

    // =====================================================
    // 3. Create special variable encoding the function call
    // =====================================================

    uint32_t deps[3] = { self, data_offsets_v, data_v };
    Ref special = steal(jitc_var_new_stmt(backend, VarType::Void, "", 0,
                                          data_size ? 3 : 1, deps));

    // =====================================================
    // 4. Stash information about inputs and outputs
    // =====================================================

    std::unique_ptr<VCall> vcall(new VCall());
    vcall->backend = backend;
    vcall->name = strdup(name);
    vcall->branch = jitc_flags() & (uint32_t) JitFlag::VCallBranch;
    vcall->id = special;
    vcall->n_inst = n_inst;
    vcall->in.reserve(n_in);
    vcall->in_nested.reserve(n_in);
    vcall->out.reserve(n_out);
    vcall->out_nested.reserve(n_out_nested);
    vcall->in_size_initial = in_size_initial;
    vcall->out_size_initial = out_size_initial;
    vcall->in_count_initial = n_in;
    vcall->se_offset.reserve(n_inst + 1);
    vcall->se = std::vector<uint32_t>(
        ts->side_effects.begin() + se_offset[0],
        ts->side_effects.begin() + se_offset[n_inst]);
    vcall->data_map = std::move(data_map);
    vcall->data_offsets = std::move(data_offsets);
    ts->side_effects.resize(se_offset[0]);

    uint32_t se_count = se_offset[n_inst] - se_offset[0];
    if (se_count) {
        /* The call has side effects. Preserve this information via a dummy
           variable marked as a side effect, which references the call. */
        uint32_t special_id = special;
        uint32_t dummy =
            jitc_var_new_stmt(backend, VarType::Void, "", 1, 1, &special_id);
        jitc_var_mark_side_effect(dummy, 0);
    }

    std::vector<bool> coherent(n_out, true);
    for (uint32_t i = 0; i < n_inst; ++i) {
        for (uint32_t j = 0; j < n_out; ++j) {
            uint32_t index = out_nested[i * n_out + j];
            coherent[j] = coherent[j] && (index == out_nested[j]);

            /* Hold a reference to the nested computation until the cleanup
               callback callback later below is invoked. */
            jitc_var_inc_ref_ext(index);
            vcall->out_nested.push_back(index);
        }
        vcall->se_offset.push_back(se_offset[i] - se_offset[0]);
    }
    vcall->se_offset.push_back(se_offset[n_inst] - se_offset[0]);


    uint32_t n_devirt = 0;

    bool optimize = jitc_flags() & (uint32_t) JitFlag::VCallOptimize;
    if (optimize) {
        for (uint32_t j = 0; j < n_out; ++j) {
            if (!coherent[j])
                continue;

            out[j] = jitc_var_copy(out_nested[j]);
            Variable *v2 = jitc_var(out[j]);
            v2->placeholder = placeholder;
            v2->size = size;
            n_devirt++;

            for (uint32_t i = 0; i < n_inst; ++i) {
                uint32_t &index = vcall->out_nested[i * n_out + j];
                jitc_var_dec_ref_ext(index);
                index = 0;
            }
        }
    }

    jitc_log(Info,
             "jit_var_vcall(self=r%u): call (\"%s\") with %u instance%s, %u "
             "input%s, %u output%s (%u devirtualized), %u side effect%s, %u "
             "byte%s of call data",
             self, name, n_inst, n_inst == 1 ? "" : "s", n_in,
             n_in == 1 ? "" : "s", n_out, n_out == 1 ? "" : "s", n_devirt,
             se_count, se_count == 1 ? "" : "s", data_size,
             data_size == 1 ? "" : "s");

    // =====================================================
    // 5. Create output variables
    // =====================================================

    auto var_callback = [](uint32_t index, int free, void *ptr) {
        if (!ptr)
            return;

        VCall *vcall_2 = (VCall *) ptr;
        if (!free) {
            // Disable callback
            state.extra[index].callback_data = nullptr;
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
                Extra *e = &state.extra[vcall_2->id];
                if (unlikely(e->dep[i] != vcall_2->in[i]))
                    jitc_fail("jit_var_vcall(): internal error! (1)");
                jitc_var_dec_ref_int(vcall_2->in[i]);
                if (state.extra.find(vcall_2->id) == state.extra.end())
                    jit_fail("jit_var_vcall(): internal error! (2)");
                e = &state.extra[vcall_2->id]; // may have changed
                e->dep[i] = 0;
                vcall_2->in[i] = 0;
                vcall_2->in_nested[i] = 0;
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

        snprintf(temp, sizeof(temp), "VCall: %s [out %u]", name, i);

        const Variable *v = jitc_var(index);
        Variable v2;
        v2.stmt = (char *) "";
        v2.size = size;
        v2.placeholder = placeholder;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.dep[0] = special;
        v2.extra = 1;
        jitc_var_inc_ref_int(special);
        uint32_t index_2 = jitc_var_new(v2, true);
        Extra &extra = state.extra[index_2];
        if (optimize) {
            extra.callback = var_callback;
            extra.callback_data = vcall.get();
            extra.callback_internal = true;
        }
        extra.label = strdup(temp);
        vcall->out.push_back(index_2);
        out[i] = index_2;
    }

    // =====================================================
    // 6. Optimize calling conventions by reordering args
    // =====================================================

    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t index = in[i];
        Variable *v = jitc_var(index);
        if (!v->placeholder_iface)
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
    // 7. Install code generation and deallocation callbacks
    // =====================================================

    snprintf(temp, sizeof(temp), "VCall: %s", name);
    size_t dep_size = vcall->in.size() * sizeof(uint32_t);

    Variable *v_special = jitc_var(vcall->id);
    Extra *e_special = &state.extra[vcall->id];
    v_special->extra = 1;
    v_special->size = size;
    e_special->label = strdup(temp);
    e_special->n_dep = (uint32_t) vcall->in.size();
    e_special->dep = (uint32_t *) malloc(dep_size);

    // Steal input dependencies from placeholder arguments
    memcpy(e_special->dep, vcall->in.data(), dep_size);

    e_special->callback = [](uint32_t, int free, void *ptr) {
        if (free)
            delete (VCall *) ptr;
    };
    e_special->callback_internal = true;
    e_special->callback_data = vcall.release();

    e_special->assemble = [](const Variable *v, const Extra &extra) {
        const Variable *self_2 = jitc_var(v->dep[0]);
        uint32_t self_reg = self_2->reg_index,
                 offset_reg = 0, data_reg = 0;

        if (v->dep[1]) {
            offset_reg = jitc_var(v->dep[1])->reg_index;
            data_reg = jitc_var(v->dep[2])->reg_index;
        }

        jitc_var_vcall_assemble((VCall *) extra.callback_data, self_reg,
                                offset_reg, data_reg);
    };

    special.reset();
}

/// Called by the JIT compiler when compiling a virtual function call
static void jitc_var_vcall_assemble(VCall *vcall,
                                    uint32_t self_reg,
                                    uint32_t offset_reg,
                                    uint32_t data_reg) {
    uint32_t vcall_reg = jitc_var(vcall->id)->reg_index;

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
    // 2. Determine calling conventions (input/output size,
    //    and alignment). Less state may need to be passed
    //    compared to when the vcall was first created.
    // =====================================================

    uint32_t in_size = 0, in_align = 1,
             out_size = 0, out_align = 1,
             n_in = vcall->in.size(),
             n_out = vcall->out.size(),
             n_in_active = 0,
             n_out_active = 0;

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

        // Transfer parameter offset to instances
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
    snprintf(ret_label, sizeof(ret_label), "l%u_done", vcall_reg);
    size_t globals_offset = globals.size();

    uint32_t vcall_table_offset = (uint32_t) vcall_table.size();

    std::vector<XXH128_hash_t> func_id(vcall->n_inst);

    ThreadState *ts = thread_state(vcall->backend);
    for (uint32_t i = 0; i < vcall->n_inst; ++i) {
        func_id[i] = jitc_assemble_func(
            ts, i, in_size, in_align, out_size, out_align,
            vcall->data_offsets[i], vcall->data_map, n_in, vcall->in.data(),
            n_out, vcall->out.data(), vcall->out_nested.data() + n_out * i,
            vcall->se_offset[i + 1] - vcall->se_offset[i],
            vcall->se.data() + vcall->se_offset[i],
            vcall->branch ? ret_label : nullptr);
        if ((uses_optix && !vcall->branch) || vcall->backend == JitBackend::LLVM) {
            auto it = globals_map.find(func_id[i]);
            if (unlikely(it == globals_map.end()))
                jitc_fail("jit_vcall_assemble(): could not find generated function in globals map!");
            vcall_table.push_back(it->second);
        }
    }

    vcall->clear_side_effects();

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

    if (vcall->backend == JitBackend::CUDA)
        jitc_var_vcall_assemble_cuda(vcall, func_id, vcall_reg, self_reg,
                                     offset_reg, data_reg, n_out, in_size,
                                     in_align, out_size, out_align, ret_label,
                                     vcall_table_offset);
    else
        jitc_var_vcall_assemble_llvm(vcall, vcall_reg, self_reg,
                                     offset_reg, data_reg, n_out, in_size,
                                     in_align, out_size, out_align,
                                     vcall_table_offset);

    if (vcall->branch) {
        buffer.putc('\n');
        for (uint32_t i = globals_offset; i < globals.size(); ++i)
            buffer.put(globals[i].c_str(), globals[i].length());
        globals.resize(globals_offset);

        buffer.fmt("\n%s:\n", ret_label);
        if (data_reg && data_reg_global)
            buffer.fmt("    mov.u64 %%data, %%u%u;\n",
                       vcall->id);
        data_reg_global = 1;
    }

    std::sort(
        func_id.begin(), func_id.end(), [](XXH128_hash_t a, XXH128_hash_t b) {
            return std::tie(a.high64, a.low64) < std::tie(b.high64, b.low64);
        });

    size_t n_unique = std::unique(
        func_id.begin(), func_id.end(), [](XXH128_hash_t a, XXH128_hash_t b) {
            return std::tie(a.high64, a.low64) == std::tie(b.high64, b.low64);
        }) - func_id.begin();

    jitc_log(
        Info,
        "jit_var_vcall_assemble(): indirect %s (\"%s\") to %zu/%zu instances, "
        "passing %u/%u inputs (%u/%u bytes), %u/%u outputs (%u/%u bytes)",
        vcall->branch ? "branch" : "call", vcall->name, n_unique,
        func_id.size(), n_in_active, vcall->in_count_initial, in_size,
        vcall->in_size_initial, n_out_active, n_out, out_size,
        vcall->out_size_initial);
}

/// Virtual function call code generation -- CUDA/PTX-specific bits
static void jitc_var_vcall_assemble_cuda(
    VCall *vcall, const std::vector<XXH128_hash_t> &func_id, uint32_t vcall_reg,
    uint32_t self_reg, uint32_t offset_reg, uint32_t data_reg, uint32_t n_out,
    uint32_t in_size, uint32_t in_align, uint32_t out_size, uint32_t out_align,
    const char *ret_label, uint32_t vcall_table_offset) {


    buffer.fmt("    %s VCall (%s)\n",
               vcall->backend == JitBackend::CUDA ? "//" : ";",
               vcall->name);

    // Extra field for call data (only if needed)
    if (vcall->branch && data_reg) {
        if (!data_reg_global) {
            buffer.put("    .reg.u64 %data;\n");
        } else {
            buffer.fmt("    .reg.u64 %%u%u;\n"
                       "    mov.u64 %%u%u, %%data;\n",
                       vcall->id, vcall->id);
        }
    }

    // =====================================================
    // 1. Determine offset into instance-specific call data
    // =====================================================

    buffer.fmt("    {\n"
            "        setp.ne.u32 %%p3, %%r%u, 0;\n", self_reg);
    if (!uses_optix || data_reg)
        buffer.fmt("        sub.u32 %%r3, %%r%u, 1;\n", self_reg);

    if (data_reg)
        buffer.fmt("        mad.wide.u32 %%rd2, %%r3, 4, %%rd%u;\n"
                   "        @%%p3 ld.global.u32 %%rd2, [%%rd2];\n"
                   "        add.u64 %s, %%rd2, %%rd%u;\n",
                   offset_reg, vcall->branch ? "%data" : "%rd2", data_reg);

    buffer.putc('\n');

    // =====================================================
    // 2. Generate function table and look up the target
    // =====================================================

    if (!uses_optix) {
        /* On CUDA, we use 'brx.idx' (indirect branch) or 'call' (indirect call).
           These require different ways of specifying the possible targets. */
        if (vcall->branch)
            buffer.put("        bt: .branchtargets\n");
        else
            buffer.put("        .global .u64 tbl[] = {\n");

        for (uint32_t i = 0; i < vcall->n_inst; ++i) {
            buffer.fmt("            %s%016llx%016llx%s",
                       vcall->branch ? "l_" : "func_",
                       (unsigned long long) func_id[i].high64,
                       (unsigned long long) func_id[i].low64,
                       i + 1 < vcall->n_inst ? ",\n" : "");
        }

        if (vcall->branch)
            buffer.put(";\n\n");
        else
            buffer.put(" };\n\n");

        if (vcall->branch)
            buffer.put("        @%p3 brx.idx %r3, bt;\n");
        else
            buffer.put("        @%p3 ld.global.u64 %rd3, tbl[%r3];\n");
    } else {
        /* Indirect branches or calls are not allowed in the OptiX
           environment. Use a 'direct callable' or a sequence of direct
           branches. */
        for (uint32_t i = 0; i < vcall->n_inst; ++i)
            buffer.fmt("        // Target %u: %s%016llx%016llx\n", i,
                       vcall->branch ? "l_" : "__direct_callable__",
                       (unsigned long long) func_id[i].high64,
                       (unsigned long long) func_id[i].low64);

        if (vcall->branch) {
            auto branch_i = [&func_id, vcall_reg](uint32_t i) {
                if (i == 0)
                    buffer.fmt("bra l%u_masked;\n", vcall_reg);
                else
                    buffer.fmt("bra l_%016llx%016llx;\n",
                               (unsigned long long) func_id[i - 1].high64,
                               (unsigned long long) func_id[i - 1].low64);
            };

            uint32_t n = vcall->n_inst + 1;

            if (n < 8) {
                // If there are just a few instances, do a few comparisons + jumps
                for (uint32_t i = 0; i < n; ++i) {
                    buffer.fmt("        setp.eq.u32 %%p3, %%r%u, %u;\n"
                               "        @%%p3 ", self_reg, i);
                    branch_i(i);
                }
            } else {
                /* Otherwise, generate code for an explicit binary search that
                   will use O(log n) conditional jumps to reach the target. */
                uint32_t levels = log2i_ceil(n);
                for (uint32_t l = 0; l < levels; ++l) {
                    for (uint32_t i = 0; i < (1 << l); ++i) {
                        uint32_t start = (i + 0) << (levels - l),
                                 end = (i + 1) << (levels - l),
                                 mid = (start + end) / 2;

                        if (start >= n)
                            break;
                        buffer.fmt("l%u_%u_%u:\n", vcall_reg, l, i);

                        if (mid < n) {
                            buffer.fmt("        setp.ge.u32 %%p3, %%r%u, %u;\n"
                                       "        @%%p3 ", self_reg, mid);
                            if (l + 1 < levels)
                                buffer.fmt("bra l%u_%u_%u;\n", vcall_reg, l + 1, 2 * i + 1);
                            else
                                branch_i(2 * i + 1);
                        }
                        if (l + 1 < levels) {
                            buffer.fmt("        bra l%u_%u_%u;\n", vcall_reg, l + 1, 2 * i);
                        } else {
                            buffer.put("        ");
                            branch_i(2 * i);
                        }
                    }
                }
            }

            buffer.fmt("\nl%u_masked:\n", vcall_reg);
        } else {
            buffer.fmt("        add.s32 %%r3, %%r%u, %i;\n"
                       "        call (%%rd3), _optix_call_direct_callable, (%%r3);\n",
                       self_reg, (int) vcall_table_offset - 1);
        }
    }

    // =====================================================
    // 3. If doing an indirect call, insert it here
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

        // Call prototype
        buffer.put("            proto: .callprototype");
        if (out_size)
            buffer.fmt(" (.param .align %u .b8 result[%u])", out_align, out_size);
        buffer.put(" _(");
        if (data_reg) {
            buffer.put(".reg .u64 data");
            if (in_size)
                buffer.put(", ");
        }
        if (in_size)
            buffer.fmt(".param .align %u .b8 params[%u]", in_align, in_size);
        buffer.put(");\n");

        // Input/output parameter arrays
        if (out_size)
            buffer.fmt("            .param .align %u .b8 out[%u];\n", out_align, out_size);
        if (in_size)
            buffer.fmt("            .param .align %u .b8 in[%u];\n", in_align, in_size);

        // =====================================================
        // 4. Pass the input arguments
        // =====================================================

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

        buffer.fmt("            @%%p3 call %s%%rd3, (%s%s%s), proto;\n",
                   out_size ? "(out), " : "", data_reg ? "%rd2" : "",
                   data_reg && in_size ? ", " : "", in_size ? "in" : "");

        // =====================================================
        // 5. Read back the output arguments
        // =====================================================

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

    // =====================================================
    // 6. Set return value(s) to zero if the call was masked
    // =====================================================

    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        buffer.fmt("        %smov.%s %s%u, 0;\n",
                   vcall->branch? "" : "@!%p3 ",
                   type_name_ptx_bin[v2->type],
                   type_prefix[v2->type], v2->reg_index);
    }

    if (vcall->branch)
        buffer.fmt("        bra %s;\n", ret_label);

    buffer.put("    }\n");
}

/// Virtual function call code generation -- LLVM IR-specific bits
static void jitc_var_vcall_assemble_llvm(VCall *vcall, uint32_t vcall_reg,
                                         uint32_t self_reg, uint32_t offset_reg,
                                         uint32_t data_reg, uint32_t n_out,
                                         uint32_t in_size, uint32_t in_align,
                                         uint32_t out_size, uint32_t out_align,
                                         uint32_t vcall_table_offset) {

    // =====================================================
    // 1. Insert parameter passing allocations into prelude
    // =====================================================

    int width = jitc_llvm_vector_width;
    char tmp[256];
    snprintf(tmp, sizeof(tmp),
             "\n    %%r%u_in = alloca i8, i32 %u, align %u\n"
             "    %%r%u_out = alloca i8, i32 %u, align %u\n"
             "    %%r%u_table = load i8**, i8*** @vcall_table\n",
             vcall_reg, in_size * width, in_align * width, vcall_reg,
             out_size * width, out_align * width, vcall_reg);

    size_t alloca_size = strlen(tmp);
    buffer.putc(' ', alloca_size); // ensure there is enough space
    char *function_start = (char *) strchr(strrchr(buffer.get(), '{'), ':') + 1;
    memmove(function_start + alloca_size, function_start, strlen(function_start) - alloca_size);
    memcpy(function_start, tmp, alloca_size);

    auto register_global = [](const char *s) {
        XXH128_hash_t hash = XXH128(s, strlen(s), 0);
        if (globals_map.emplace(hash, globals_map.size()).second)
            globals.push_back(s);
    };

    // =====================================================
    // 2. Declare a few intrinsics that we will use
    // =====================================================

    snprintf(tmp, sizeof(tmp),
             "declare i32 @llvm.experimental.vector.reduce.umax.v%ui32(<%u x i32>)\n\n",
             width, width);
    register_global(tmp);
    register_global("@vcall_table = internal constant i8** null\n\n");
    if (out_size) {
        snprintf(tmp, sizeof(tmp), "declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)\n\n");
        register_global(tmp);
    }

    buffer.fmt("    br label %%l%u_start\n"
               "\nl%u_start:\n", vcall_reg, vcall_reg);

    // =====================================================
    // 3. Pass the input arguments
    // =====================================================

    // register_global("declare void @llvm.debugtrap()\n\n");
    // buffer.put("    call void @llvm.debugtrap()\n");

    uint32_t offset = 0;
    for (uint32_t i = 0; i < (uint32_t) vcall->in.size(); ++i) {
        uint32_t index = vcall->in[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t vti = v2->type;
        const VarType vt = (VarType) vti;
        uint32_t size = type_size[vti];

        const char *prefix = type_prefix[vti],
                   *tname = vt == VarType::Bool
                            ? "i8" : type_name_llvm[vti];

        if (vt == VarType::Bool)
            buffer.fmt("    %s%u_zext = zext <%u x i1> %s%u to <%u x i8>\n",
                       prefix, v2->reg_index, width, prefix, v2->reg_index, width);

        buffer.fmt(
            "    %%r%u_in_%u_0 = getelementptr inbounds i8, i8* %%r%u_in, i64 %u\n"
            "    %%r%u_in_%u_1 = bitcast i8* %%r%u_in_%u_0 to <%u x %s> *\n"
            "    store <%u x %s> %s%u%s, <%u x %s>* %%r%u_in_%u_1, align %u\n",
            vcall_reg, i, vcall_reg, offset,
            vcall_reg, i, vcall_reg, i, width, tname,
            width, tname, prefix, v2->reg_index, vt == VarType::Bool ? "_zext" : "", width, tname, vcall_reg, i, size * width
        );

        offset += size * width;
    }

    if (out_size) /// Zero-initialize memory region containing outputs
        buffer.fmt("    call void @llvm.memset.p0i8.i32(i8* %%r%u_out, i8 0, "
                   "i32 %u, i1 0)\n", vcall_reg, out_size * width);

    if (data_reg) {
        buffer.fmt("        do something with %%rd%u\n", offset_reg);
        // buffer.fmt("        mad.wide.u32 %%rd2, %%r3, 4, %%rd%u;\n"
        //            "        @%%p3 ld.global.u32 %%rd2, [%%rd2];\n"
        //            "        add.u64 %s, %%rd2, %%rd%u;\n",
        //            offset_reg, vcall->branch ? "%data" : "%rd2", data_reg);
    }

    // =====================================================
    // 4. Perform one call to each unique instance
    // =====================================================

    buffer.fmt("    br label %%l%u_check\n", vcall_reg);

    buffer.fmt("\nl%u_check:\n"
               "    %%r%u_self = phi <%u x i32> [ %%r%u, %%l%u_start ], [ %%r%u_self_next, %%l%u_call ]\n",
               vcall_reg, vcall_reg, width, self_reg, vcall_reg, vcall_reg, vcall_reg);
    buffer.fmt("    %%r%u_next = call i32 @llvm.experimental.vector.reduce.umax.v%ui32(<%u x i32> %%r%u_self)\n", vcall_reg, width, width, vcall_reg);
    buffer.fmt("    %%r%u_valid = icmp ne i32 %%r%u_next, 0\n"
               "    br i1 %%r%u_valid, label %%l%u_call, label %%l%u_end\n",
               vcall_reg, vcall_reg, vcall_reg, vcall_reg, vcall_reg);

    buffer.fmt("\nl%u_call:\n"
               "    %%r%u_bcast_0 = insertelement <%u x i32> undef, i32 %%r%u_next, i32 0\n"
               "    %%r%u_bcast = shufflevector <%u x i32> %%r%u_bcast_0, <%u x i32> undef, <%u x i32> zeroinitializer\n"
               "    %%r%u_active = icmp eq <%u x i32> %%r%u_self, %%r%u_bcast\n"
               "    %%r%u_self_next = select <%u x i1> %%r%u_active, <%u x i32> zeroinitializer, <%u x i32> %%r%u_self\n"
               "    %%r%u_table_index = add i32 %%r%u_next, %i\n"
               "    %%r%u_func_0 = getelementptr inbounds i8*, i8** %%r%u_table, i32 %%r%u_table_index\n"
               "    %%r%u_func_1 = load i8*, i8** %%r%u_func_0\n"
               "    %%r%u_func = bitcast i8* %%r%u_func_1 to void (<%u x i1>, i8*, i8*, i8*)*\n"
               "    call void %%r%u_func(<%i x i1> %%r%u_active, i8* %%r%u_in, i8* %%r%u_out, i8* null)\n"
               "    br label %%l%u_check\n"
               "\nl%u_end:\n",
               vcall_reg,
               vcall_reg, width, vcall_reg, // bcast_0
               vcall_reg, width, vcall_reg, width, width, // bcast
               vcall_reg, width, vcall_reg, vcall_reg, // active
               vcall_reg, width, vcall_reg, width, width, vcall_reg, // self_next
               vcall_reg, vcall_reg, (int) vcall_table_offset - 1, // table_index
               vcall_reg, vcall_reg, vcall_reg, // func_0
               vcall_reg, vcall_reg, // func_1
               vcall_reg, vcall_reg, width, // _func
               vcall_reg, width, vcall_reg, vcall_reg, vcall_reg, // call
               vcall_reg, // br
               vcall_reg);

    // =====================================================
    // 5. Read back the output arguments
    // =====================================================

    offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i],
                 index_2 = vcall->out[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        uint32_t size = type_size[it->second.type],
                 load_offset = offset;
        offset += size * width;

        // Skip if outer access expired
        auto it2 = state.variables.find(index_2);
        if (it2 == state.variables.end())
            continue;

        const Variable *v2 = &it2.value();

        uint32_t vti = v2->type;
        const VarType vt = (VarType) vti;

        const char *prefix = type_prefix[vti],
                   *tname = vt == VarType::Bool
                            ? "i8" : type_name_llvm[vti];

        buffer.fmt(
            "    %%r%u_out_%u_0 = getelementptr inbounds i8, i8* %%r%u_out, i64 %u\n"
            "    %%r%u_out_%u_1 = bitcast i8* %%r%u_out_%u_0 to <%u x %s> *\n"
            "    %s%u%s = load <%u x %s>, <%u x %s>* %%r%u_out_%u_1, align %u\n",
            vcall_reg, i, vcall_reg, load_offset, vcall_reg, i, vcall_reg, i,
            width, tname,
            prefix, v2->reg_index, vt == VarType::Bool ? "_0" : "", width, tname, width, tname,
            vcall_reg, i, size * width);

            if (vt == VarType::Bool)
                buffer.fmt("    %s%u = trunc <%u x i8> %s%u_0 to <%u x i1>\n",
                           prefix, v2->reg_index, width, prefix, v2->reg_index, width);
    }

    buffer.fmt("    br label %%l%u_done\n"
               "\nl%u_done:\n", vcall_reg, vcall_reg);
}

/// Collect scalar / pointer variables referenced by a computation
void jitc_var_vcall_collect_data(tsl::robin_map<uint64_t, uint32_t> &data_map,
                                 uint32_t &data_offset, uint32_t inst_id,
                                 uint32_t index) {
    uint64_t key = (uint64_t) index + (((uint64_t) inst_id) << 32);
    if (data_map.find(key) != data_map.end())
        return;

    const Variable *v = jitc_var(index);

    if (v->placeholder_iface) {
        return;
    } else if (v->data || (VarType) v->type == VarType::Pointer) {
        uint32_t tsize = type_size[v->type];
        data_offset = (data_offset + tsize - 1) / tsize * tsize;
        data_map.emplace(key, data_offset);
        data_offset += tsize;

        if (v->size != 1)
            jitc_raise(
                "jit_var_vcall(): the virtual function call associated with "
                "instance %u accesses an evaluated variable r%u of type "
                "%s and size %u. However, only *scalar* (size == 1) "
                "evaluated variables can be accessed while recording "
                "virtual function calls",
                inst_id, index, type_name[v->type], v->size);
    } else {
        data_map.emplace(key, (uint32_t) -1);
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t index_2 = v->dep[i];
            if (!index_2)
                break;

            jitc_var_vcall_collect_data(data_map, data_offset,
                                        inst_id, index_2);
        }
        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_var_vcall_collect_data(): could not find "
                          "matching 'extra' record!");

            const Extra &extra = it->second;
            for (uint32_t i = 0; i < extra.n_dep; ++i) {
                uint32_t index_2 = extra.dep[i];
                jitc_var_vcall_collect_data(data_map, data_offset,
                                            inst_id, index_2);
            }
        }

    }
}

// Compute a permutation to reorder an array of registered pointers
VCallBucket *jitc_var_vcall_reduce(JitBackend backend, const char *domain,
                                   uint32_t index, uint32_t *bucket_count_out) {
    auto it = state.extra.find(index);
    if (it != state.extra.end()) {
        auto &v = it.value();
        if (v.vcall_bucket_count) {
            *bucket_count_out = v.vcall_bucket_count;
            return v.vcall_buckets;
        }
    }

    uint32_t bucket_count = jitc_registry_get_max(domain) + 1;
    if (unlikely(bucket_count == 1)) {
        *bucket_count_out = 0;
        return nullptr;
    }

    jitc_var_eval(index);
    Variable *v = jitc_var(index);
    const void *ptr = v->data;
    uint32_t size = v->size;

    jitc_log(Debug, "jit_vcall(%u, domain=\"%s\")", index, domain);

    size_t perm_size    = (size_t) size * (size_t) sizeof(uint32_t),
           offsets_size = (size_t(bucket_count) * 4 + 1) * sizeof(uint32_t);

    uint32_t *offsets = (uint32_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::HostPinned : AllocType::Host, offsets_size);
    uint32_t *perm = (uint32_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync, perm_size);

    // Compute permutation
    uint32_t unique_count = jitc_mkperm(backend, (const uint32_t *) ptr, size,
                                        bucket_count, perm, offsets),
             unique_count_out = unique_count;

    // Register permutation variable with JIT backend and transfer ownership
    uint32_t perm_var = jitc_var_mem_map(backend, VarType::UInt32, perm, size, 1);

    Variable v2;
    v2.type = (uint32_t) VarType::UInt32;
    v2.backend = (uint32_t) backend;
    v2.dep[3] = perm_var;
    v2.retain_data = true;
    v2.unaligned = 1;

    uint32_t *offsets_out = offsets;

    for (uint32_t i = 0; i < unique_count; ++i) {
        uint32_t bucket_id     = offsets[i * 4 + 0],
                 bucket_offset = offsets[i * 4 + 1],
                 bucket_size   = offsets[i * 4 + 2];

        /// Crete variable for permutation subrange
        v2.data = perm + bucket_offset;
        v2.size = bucket_size;

        jitc_var_inc_ref_int(perm_var);

        uint32_t index = jitc_var_new(v2);

        void *ptr = jitc_registry_get_ptr(domain, bucket_id);
        memcpy(offsets_out, &ptr, sizeof(void *));
        memcpy(offsets_out + 2, &index, sizeof(uint32_t));
        offsets_out += 4;

        jitc_trace("jit_vcall(): registered variable %u: bucket %u (" ENOKI_PTR
                  ") of size %u.", index, bucket_id, (uintptr_t) ptr, bucket_size);
    }

    jitc_var_dec_ref_ext(perm_var);

    *bucket_count_out = unique_count_out;

    jitc_var(index)->extra = true;
    Extra &extra = state.extra[index];
    extra.vcall_bucket_count = unique_count_out;
    extra.vcall_buckets = (VCallBucket *) offsets;
    return extra.vcall_buckets;
}
