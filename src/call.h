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
    /// Is input ``i`` actually passed to the callable (i.e. not pruned as a
    /// literal, single-use, or unreferenced argument)?
    std::vector<uint8_t> in_active;

    /// Per-instance offsets into the 'side_effects' list below
    std::vector<uint32_t> checkpoints;
    /// Compressed side effect index list
    std::vector<uint32_t> side_effects;

    /// Array of size 'n_inst' storing hashes of compiled callables
    std::vector<XXH128_hash_t> inst_hash;

    /// Encodes how one capture slot participates in the call-data layout.
    class SizeBucket {
    public:
        SizeBucket() = default;

        /// Classify a captured variable by storage size and packet-load eligibility.
        explicit SizeBucket(const Variable *v);

        /// Bucket ID: 0..3 for coalesceable 8/4/2/1-byte
        /// slots, 4..7 for uncoalesced slots of the same sizes.
        uint32_t id() const { return m_value; }

        /// Low two bits encoding the 8/4/2/1-byte size class.
        uint32_t size_class() const { return m_value & 3u; }

        /// Does this bucket belong to the packet-loadable prefix?
        bool coalesceable() const { return m_value < 4; }

        /// Storage size in bytes represented by this bucket.
        uint32_t size() const { return size_from_id(m_value); }

        /// Return the byte size represented by a size class or full bucket ID.
        static uint32_t size_from_id(uint32_t bucket_id) {
            return 8u >> (bucket_id & 3u);
        }

    private:
        static uint32_t size_class_from_size(uint32_t size) {
            return size == 8 ? 0u : size == 4 ? 1u : size == 2 ? 2u : 3u;
        }

        uint8_t m_value = 0;
    };

    /// A single evaluated/pointer value captured into the per-instance closure
    struct CaptureSlot {
        WeakRef     ref;    ///< weak reference to the variable (index + generation)
        uint32_t    offset; ///< byte offset after layout within this call's data block

        /// Size bucket assigned when the slot is captured. Keeping this on the
        /// slot avoids a temporary per-instance bucket vector and repeated
        /// coalescing classification in the layout/codegen hot path.
        SizeBucket bucket;
    };

    /// Capture slots, concatenated per instance (traversal-ordered while being
    /// collected, then offset-ascending after each instance is laid out).
    std::vector<CaptureSlot> slots;

    /// Slot and byte layout of a single callable instance's captured data.
    ///
    /// ``slots[slot_start, slot_start+slot_count)`` contains this instance's
    /// captured values sorted by increasing byte offset. The first part of that
    /// slot range contains fields that may be loaded via packet operations,
    /// grouped into the four size classes 8/4/2/1 bytes. Fields that must be
    /// loaded individually (for example opaque Metal resource handles) follow in
    /// uncoalesced buckets.
    struct InstanceLayout {
        struct Bucket {
            uint32_t slot_start = 0; ///< first slot in this size class
            uint32_t slot_count = 0; ///< number of slots in this size class

            uint32_t slot_end() const { return slot_start + slot_count; }
        };

        uint32_t slot_start = 0;     ///< first slot belonging to this instance
        uint32_t slot_count = 0;     ///< number of slots belonging to it
        uint32_t data_offset = 0;    ///< byte offset of the instance data block

        /// Relative byte offset just past all coalesceable fields. CUDA/Metal
        /// use this to bound vectorized loads over the contiguous coalesceable
        /// byte range. LLVM uses the size-class buckets below, since its packet
        /// loads are homogeneous.
        uint32_t coalesce_end = 0;
        uint32_t coalesce_count = 0; ///< number of coalesceable slots

        /// Coalesceable slots by size class: 0=8B, 1=4B, 2=2B, 3=1B.
        Bucket bucket[4];

        uint32_t slot_end() const { return slot_start + slot_count; }
    };

    /// Per-instance call-data layout, size 'n_inst'.
    std::vector<InstanceLayout> instance_layout;

    /// Total size (in bytes) of this call's backend-aligned data block
    uint32_t data_size = 0;
    /// Number of offset-table slots (= max instance id + 1)
    uint32_t offset_count = 0;

    /// Byte offset of this call's offset table within the fused buffer's offset
    /// region. Assigned once per call during code generation (see
    /// jitc_var_call_assemble).
    uint32_t offset_base = 0;
    /// Byte offset of this call's data block within the fused buffer's data
    /// region. Assigned once per call during code generation.
    uint32_t data_base = 0;

    /// Does this call contain a 'CallSelf' variable?
    bool use_self = false;
    /// Does this call contain a nested 'Call' node? Such callables receive the
    /// base pointer so the nested dispatch can reach its own buffer slice.
    bool use_nested = false;
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

/// This data structure maintains the running call data state during kernel
/// compilation. See the top of call.h for details on this.
struct CallBufferState {
    /// Combined size (bytes) of all offset tables / all data blocks so far.
    uint32_t fused_offset_size = 0;
    uint32_t fused_data_size = 0;

    /// Base pointer variable (0 if the kernel performs no calls) and the
    /// mem-mapped variable owning its device allocation.
    uint32_t base_v = 0;
    uint32_t base_src = 0;

    /// Register holding the base pointer, and its slot index in 'kernel_params'.
    uint32_t base_reg = 0;
    uint32_t base_param_index = 0;

    // Projection of a CallData::CaptureSlot: a strong reference plus a byte
    // offset into the data region following the offset tables.
    struct CaptureSlotRef {
        Ref      src;     ///< captured source variable
        uint32_t offset;  ///< byte offset within the fused data region
    };

    std::vector<CaptureSlotRef> data_entries;

    void reset() {
        fused_offset_size = fused_data_size = 0;
        base_v = base_src = base_reg = base_param_index = 0;
        data_entries.clear();
    }
};

extern CallBufferState call_buffer;
extern std::vector<CallData *> calls_assembled;

extern void jitc_call_bind_slots(const CallData *call, uint32_t inst);

extern uint32_t jitc_call_slot_rel_offset(const CallData *call, uint32_t inst,
                                          const Variable *v, uint32_t index);

/// Minimum utilization (in percent) for a packet transfer to be worthwhile.
static constexpr uint32_t jitc_packet_min_util = 75;

/// Pick the widest LLVM call-data packet gather count. Returns 1 when
/// packetization is not useful, so callers can fall back to scalar gathers.
inline uint32_t jitc_call_pick_llvm_packet_count(uint32_t remaining,
                                                 uint32_t max_count) {
    for (uint32_t cand = max_count; cand >= 2; cand /= 2) {
        uint32_t useful = remaining < cand ? remaining : cand;
        if (useful * 100 >= cand * jitc_packet_min_util)
            return cand;
    }
    return 1;
}

/// Pick the widest naturally aligned CUDA/Metal load width over a contiguous
/// byte range. Returns 4 when no vector load stays sufficiently utilized.
inline uint32_t jitc_call_pick_word_chunk_size(uint32_t offset,
                                               uint32_t remaining,
                                               uint32_t max_chunk) {
    for (uint32_t cand = max_chunk; cand >= 8; cand /= 2) {
        uint32_t useful = remaining < cand ? remaining : cand;
        if ((offset % cand) == 0 && useful * 100 >= cand * jitc_packet_min_util)
            return cand;
    }
    return 4;
}

extern uint32_t jitc_var_loop_init(uint32_t *indices, uint32_t n_indices);

extern void jitc_call_upload(ThreadState *ts);

extern CallBucket *jitc_var_call_reduce(JitBackend backend, const char *variant,
                                        const char *domain, uint32_t index,
                                        uint32_t *bucket_count_out);

extern void jitc_var_call_assemble(CallData *call, uint32_t call_reg,
                                   uint32_t self_reg, uint32_t mask_reg);

extern void jitc_var_call_assemble_llvm(CallData *call, uint32_t call_reg,
                                        uint32_t self_reg, uint32_t mask_reg,
                                        uint32_t buf_size, uint32_t buf_align);

extern void jitc_var_call_assemble_cuda(CallData *call, uint32_t call_reg,
                                        uint32_t self_reg, uint32_t mask_reg,
                                        uint32_t in_size, uint32_t in_align,
                                        uint32_t out_size, uint32_t out_align);

extern void jitc_var_call_assemble_metal(CallData *call, uint32_t call_reg,
                                         uint32_t self_reg, uint32_t mask_reg,
                                         uint32_t in_size, uint32_t in_align,
                                         uint32_t out_size, uint32_t out_align);

extern void jitc_var_call(const char *domain, bool symbolic, uint32_t self,
                          uint32_t mask, uint32_t n_inst, uint32_t max_inst_id,
                          const uint32_t *inst_id, uint32_t n_in,
                          const uint32_t *in, uint32_t n_inner_out,
                          const uint32_t *inner_out,
                          const uint32_t *checkpoints, uint32_t *out);

extern void jitc_var_call_analyze(CallData *call, uint32_t inst_id,
                                  uint32_t index);
