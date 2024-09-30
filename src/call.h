#pragma once

#include "internal.h"

/// Encodes information about a virtual function call
struct CallData {
    JitBackend backend;

    /// A descriptive name
    std::string name;

    /// Were optimizations enabled when this functions was traced?
    bool optimize;

    /// ID of the variable representing the call
    uint32_t id;

    /// Number of inputs and outputs
    uint32_t n_in, n_out;

    /// Number of instances/callables targeted by the call. Some of them may be
    /// merged later on.
    uint32_t n_inst;

    /// Array of size 'n_inst' with all instance IDs that might be referenced
    std::vector<uint32_t> inst_id;

    /// Input variables at call site
    std::vector<uint32_t> outer_in;
    /// Input variables. CallData holds a reference to these.
    std::vector<uint32_t> inner_in;
    /// Output variables *per instance*. CallData holds a reference to these.
    std::vector<uint32_t> inner_out;
    /// Output variables at call site
    std::vector<WeakRef> outer_out;
    /// Offset in return value buffer
    std::vector<uint32_t> out_offset;

    /// Per-instance offsets into the 'side_effects' list below
    std::vector<uint32_t> checkpoints;
    /// Compressed side effect index list
    std::vector<uint32_t> side_effects;

    /// Array of size 'n_inst' storing hashes of compiled callables
    std::vector<XXH128_hash_t> inst_hash;

    /// Mapping from variable index to offset into call data
    tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> data_map;
    std::vector<uint32_t> data_offset;

    uint64_t *offset = nullptr;
    size_t offset_size = 0;

    /// Does this call contain a 'CallSelf' variable?
    bool use_self = false;
    /// Does this call contain a 'Counter' variable?
    bool use_index = false;
    /// Does this call contain a 'ThreadIndex' variable?
    bool use_thread_id = false;
    /// Does this call contain OptiX operations?
    bool use_optix = false;

    ~CallData() {
        for (uint32_t index : inner_in)
            jitc_var_dec_ref(index);
        for (uint32_t index : inner_out)
            jitc_var_dec_ref(index);
        clear_side_effects();
    }

    void clear_side_effects() {
        if (side_effects.empty())
            return;
        for (uint32_t index : side_effects)
            jitc_var_dec_ref(index);
        side_effects.clear();
        checkpoints.clear();
    }
};

extern std::vector<CallData *> calls_assembled;

extern uint32_t jitc_var_loop_init(uint32_t *indices, uint32_t n_indices);

extern void jitc_var_call(const char *domain, bool symbolic, uint32_t self,
                          uint32_t mask, uint32_t n_inst, uint32_t max_inst_id,
                          const uint32_t *inst_id, uint32_t n_in,
                          const uint32_t *in, uint32_t n_inner_out,
                          const uint32_t *inner_out,
                          const uint32_t *checkpoints, uint32_t *out);

extern void jitc_call_upload(ThreadState *ts);

extern CallBucket *jitc_var_call_reduce(JitBackend backend, const char *domain,
                                        uint32_t index,
                                        uint32_t *bucket_count_out);

extern void jitc_var_call_assemble(CallData *call, uint32_t call_reg,
                                   uint32_t self_reg, uint32_t mask_reg,
                                   uint32_t offset_reg, uint32_t data_reg);

extern void jitc_var_call_assemble_llvm(CallData *call, uint32_t call_reg,
                                        uint32_t self_reg, uint32_t mask_reg,
                                        uint32_t offset_reg, uint32_t data_reg,
                                        uint32_t buf_size, uint32_t buf_align);

extern void jitc_var_call_assemble_cuda(CallData *call, uint32_t call_reg,
                                        uint32_t self_reg, uint32_t mask_reg,
                                        uint32_t offset_reg, uint32_t data_reg,
                                        uint32_t in_size, uint32_t in_align,
                                        uint32_t out_size, uint32_t out_align);
