/*
    src/call.cpp -- Code generation for virtual function calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    ==========================================================

    This file implements the compilation of indirect ("virtual") function calls.
    An indirect call dispatches each array element to one of several callables,
    selected at runtime by a per-element instance index ('self'). Dr.Jit traces
    every callable once, emits it as a separate function in the kernel, and
    generates dispatch code that routes each lane to the matching callable.

    A callable cannot reach into the surrounding computation directly. Instead,
    any evaluated scalars and buffer/resource pointers it references are
    captured into a per-instance "call data" block that the callable reads at
    runtime. Entries in this block are denoted "slots". Each instance owns
    a contiguous region in the call data block storing its slots.

    Hoisting this data out of the callable body is also what makes callables
    mergeable: instances whose IR differs only in captured values compile to the
    same callable and are deduplicated. This shrinks the number of compiled
    functions (reducing compilation time) and routes more lanes through the same
    callable (improving SIMD/warp coherence).

    All indirect calls in a kernel share a single device buffer laid out as

        [ offset tables | data blocks ]

    The dispatch indexes the offset table by 'self' to obtain the callable index
    and the absolute offset of that instance's data block, from which it reads
    the captured values.

    Slots within an instance's datablock are arranged and padded so that data
    can be efficiently loaded using wide packet memory transactions. Some fields
    are excluded from this (e.g., opaque resource handles on Metal) because the
    compilation model forbids reading them in this way.
*/

#include <set>
#include <string.h>

#include "call.h"
#include "eval.h"
#include "internal.h"
#include "llvm.h"
#include "log.h"
#include "loop.h"
#include "op.h"
#include "profile.h"
#include "registry.h"
#include "tex.h"
#include "trace.h"
#include "util.h"
#include "var.h"
#include "coop_vec.h"

std::vector<CallData *> calls_assembled;

/// This global field stores the running call data state during compilation
CallBufferState call_buffer;

/// Scratch buffer holding one call's offset table. Used in jitc_call_update().
static std::vector<uint64_t> offset_entries;

/// Widest call-data packet gather the LLVM backend may issue, in elements.
static uint32_t llvm_call_data_packet_cap() {
    return std::max(1u, std::min(8u, jitc_llvm_vector_width));
}

/// Classify a call-data variable into a size bucket
CallData::SizeBucket::SizeBucket(const Variable *v) {
    bool coalesceable = true;

#if defined(DRJIT_ENABLE_METAL)
    if ((JitBackend) v->backend == JitBackend::Metal) {
        VarType vt = (VarType) v->type;

        // Opaque resource handles cannot be loaded through call-data packets.
        if (vt == VarType::Pointer && v->resource_kind() != ResourceKind::Buffer)
            coalesceable = false;

        // Metal currently miscompiles 1-byte fields when reconstructed from
        // packet words, so leave them in uncoalesced buckets.
        if (vt == VarType::Bool || vt == VarType::Int8 || vt == VarType::UInt8)
            coalesceable = false;
    }
#endif

    m_value = (uint8_t) ((coalesceable ? 0u : 4u) +
                         size_class_from_size(type_size[v->type]));
}

/// Byte alignment of each per-instance data block, so that concatenated blocks
/// stay aligned for the widest packet load this backend/device can issue.
static uint32_t call_data_align(const ThreadState *ts, bool uses_optix_) {
#if !defined(DRJIT_ENABLE_CUDA)
    (void) uses_optix_;
#endif

    switch ((JitBackend) ts->backend) {
#if defined(DRJIT_ENABLE_CUDA)
        case JitBackend::CUDA: // 256-bit (32B) needs CC >= 12.0, else 128-bit
            return jitc_cuda_supports_256bit(ts, uses_optix_) ? 32u : 16u;
#endif
#if defined(DRJIT_ENABLE_METAL)
        case JitBackend::Metal:
            return 16u;
#endif
        case JitBackend::LLVM: // packet cap x widest field (8B)
            return llvm_call_data_packet_cap() * 8u;

        default:
            return 8u;
    }
}

// Walk through all slots of instance ``inst`` and populate the ``param_offset``
// member with the logical variable offset within the data block.
void jitc_call_bind_slots(const CallData *call, uint32_t inst) {
    const CallData::InstanceLayout &layout = call->instance_layout[inst];
    for (uint32_t k = layout.slot_start; k < layout.slot_end(); ++k) {
        if (Variable *v = jitc_var(call->slots[k].ref))
            v->param_offset = k;
    }
}

/// Given an opaque variable ``v`` in ``call``, check if the variable has a
/// capture slot. Return its rel. position in the data block in that case or
/// fail. This is useful to catch situations where new IR nodes haven't been
/// correctly wired up in the recursive traversal in call.cpp or eval.cpp.
uint32_t jitc_call_slot_rel_offset(const CallData *call, uint32_t inst,
                                   const Variable *v, uint32_t index) {
    const CallData::InstanceLayout &layout = call->instance_layout[inst];
    uint32_t k = v->param_offset;
    if (likely(k >= layout.slot_start &&
               k <  layout.slot_end() &&
               jitc_var(call->slots[k].ref) == v))
        return call->slots[k].offset - layout.data_offset;

    if (v->is_evaluated())
        jitc_fail("jitc_call_slot_rel_offset(): variable r%u is referenced by a "
                  "recorded function call. However, it was evaluated between the "
                  "recording step and code generation (which is happening now). "
                  "This is not allowed.", index);
    else
        jitc_fail("jitc_call_slot_rel_offset(): could not find call-data slot for "
                  "variable r%u in function %s", index, call->name.c_str());
}

/**
 * \brief Map an instance's captured call data to a concrete byte layout.
 *
 * Call data slots are originally in traversal order and still lack a memory
 * layout. This function turns the slot subrange ``[lo, hi)`` into the physical
 * layout used by the call-data buffer. It performs the following steps:
 *
 *   1. Group slots by their size bucket: 8/4/2/1-byte packet-loadable
 *      values, followed by uncoalesced values of the same sizes.
 *
 *   2. Place packet-loadable buckets first. The generated LLVM IR
 *      packet-loads each size bucket separately, while CUDA/Metal issue
 *      vectorized loads over the contiguous coalesceable byte range.
 *
 *   3. Uncoalesced fields follow in naturally aligned scalar slots.
 *
 * The accompanying InstanceLayout record stashes information for later use
 * during code generation: the slot range, data-block base offset,
 * coalesceable byte prefix, and the four coalesceable size buckets.
 * ``reordered`` is a caller-provided scratch buffer.
 */
static void jitc_call_layout_instance(CallData *call, uint32_t lo, uint32_t hi,
                                      JitBackend backend, uint32_t alignment,
                                      uint32_t llvm_pkt_cap, uint32_t &data_size,
                                      std::vector<CallData::CaptureSlot> &reordered) {
    // Histogram by SizeBucket ID: 0..3 are packet-loadable 8/4/2/1-byte
    // buckets, 4..7 are uncoalesced buckets of the same sizes.
    uint32_t bucket_count[8] = { },
             max_field_size = 1u,
             coalesceable_bytes = 0;

    for (uint32_t k = lo; k < hi; ++k) {
        CallData::SizeBucket bucket = call->slots[k].bucket;
        uint32_t bucket_id = bucket.id(),
                 size      = bucket.size();
        bucket_count[bucket_id]++;
        max_field_size = std::max(max_field_size, size);
        if (bucket.coalesceable())
            coalesceable_bytes += size;
    }

    // LLVM loads loads buckets individually. The code below computes the
    // alignment that each size bucket will need to satisfy the widest packet
    // load instruction that the backend may emit. CUDA/Metal vector loads start
    // from the beginning of the block (with alignment ``block_align``), hence
    // the individual buckets only require natural alignment.
    uint32_t bucket_align[4];
    for (uint32_t c = 0; c < 4; ++c) {
        uint32_t nelems = 1;
        if (backend == JitBackend::LLVM)
            nelems = jitc_call_pick_llvm_packet_count(bucket_count[c], llvm_pkt_cap);
        bucket_align[c] = nelems * CallData::SizeBucket::size_from_id(c);
    }

    // Pick the block's start alignment so the coalesceable region can be read back
    // with the widest access the backend will issue, but never below the widest
    // field. LLVM must satisfy the strictest bucket; CUDA/Metal load the
    // contiguous coalesceable byte range in 32-bit chunks, widened when the
    // word-rounded range fills a vector load to >= 75% utilization
    // (jitc_call_pick_word_chunk_size).
    uint32_t block_align = max_field_size;
    if (backend == JitBackend::LLVM) {
        for (uint32_t c = 0; c < 4; ++c)
            if (bucket_count[c])
                block_align = std::max(block_align, bucket_align[c]);
    } else {
        uint32_t region_bytes = (coalesceable_bytes + 3u) & ~3u;
        block_align = std::max(max_field_size,
                               jitc_call_pick_word_chunk_size(0, region_bytes, alignment));
    }

    uint32_t base = align_up(data_size, block_align);

    CallData::InstanceLayout layout;
    layout.slot_start  = lo;
    layout.slot_count  = hi - lo;
    layout.data_offset = base;

    uint32_t bucket_order[4] = { 0, 1, 2, 3 };
    if (backend == JitBackend::LLVM) {
        // Insertion-sort the packet-loadable bucket IDs by descending packet
        // alignment to avoid padding between buckets. CUDA/Metal don't need this.
        for (uint32_t i = 1; i < 4; ++i) {
            uint32_t c = bucket_order[i], j = i;
            while (j > 0 && bucket_align[bucket_order[j - 1]] < bucket_align[c]) {
                bucket_order[j] = bucket_order[j - 1];
                --j;
            }
            bucket_order[j] = c;
        }
    }

    uint32_t bucket_slot[8] = { },   // next output slot per bucket
             bucket_offset[8] = { }, // next byte offset per bucket
             next_slot = 0,          // next free slot in ``reordered``
             next_offset = base;     // next free byte offset in the data block

    // Assign each non-empty bucket a start index in ``reordered`` and an aligned
    // byte offset. Keep slots sorted by ascending offset so CUDA/Metal can emit
    // vectorized loads over the coalesceable prefix.
    auto place_bucket = [&](uint32_t bucket_id, uint32_t align) {
        if (!bucket_count[bucket_id])
            return;

        next_offset              = align_up(next_offset, align);
        bucket_slot[bucket_id]   = next_slot;
        bucket_offset[bucket_id] = next_offset;

        uint32_t size  = CallData::SizeBucket::size_from_id(bucket_id),
                 bytes = bucket_count[bucket_id] * size;

        if (bucket_id < 4) {
            layout.bucket[bucket_id] = { lo + next_slot, bucket_count[bucket_id] };
            layout.coalesce_end =
                std::max(layout.coalesce_end, next_offset - base + bytes);
            layout.coalesce_count += bucket_count[bucket_id];
        }

        next_slot += bucket_count[bucket_id];
        next_offset += bytes;
    };

    for (uint32_t i = 0; i < 4; ++i)
        place_bucket(bucket_order[i], bucket_align[bucket_order[i]]);
    for (uint32_t c = 0; c < 4; ++c)
        place_bucket(4u + c, CallData::SizeBucket::size_from_id(c));

    // Scatter each slot into its bucket, preserving the traversal order within
    reordered.resize(hi - lo);
    for (uint32_t k = lo; k < hi; ++k) {
        const CallData::CaptureSlot &slot = call->slots[k];
        uint32_t bucket_id = slot.bucket.id();
        reordered[bucket_slot[bucket_id]++] =
            { slot.ref, bucket_offset[bucket_id], slot.bucket };
        bucket_offset[bucket_id] += slot.bucket.size();
    }

    std::copy(reordered.begin(), reordered.end(), call->slots.begin() + lo);
    call->instance_layout.push_back(layout);
    data_size = next_offset;
}

/// Weave a virtual function call into the computation graph
void jitc_var_call(const char *name, bool symbolic, uint32_t self,
                   uint32_t mask_, uint32_t n_inst, uint32_t max_inst_id,
                   const uint32_t *inst_id, uint32_t n_in, const uint32_t *in,
                   uint32_t n_inner_out, const uint32_t *inner_out,
                   const uint32_t *checkpoints, uint32_t *out) {

    const uint32_t checkpoint_mask = 0x7fffffff;

#if defined(DRJIT_ENABLE_NVTX) || defined(DRJIT_ENABLE_ITTNOTIFY)
    std::string profile_name = std::string("jit_var_call: ") + name;
    ProfilerRegion profiler_region(profile_name.c_str());
    ProfilerPhase profiler(profiler_region);
#endif

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    if (n_inst == 0)
        jitc_raise("jit_var_call('%s'): must have at least one instance!", name);

    if (n_inner_out % n_inst != 0)
        jitc_raise("jit_var_call('%s'): list of all output indices must be a "
                   "multiple of the instance count (n_inner_out=%u, n_inst=%u)!",
                   name, n_inner_out, n_inst);
    uint32_t flags = jitc_flags();

    uint32_t n_out = n_inner_out / n_inst, size = 0;
    bool dirty = false;

    JitBackend backend;
    /* Check 'self' */ {
        const Variable *self_v = jitc_var(self);
        size = self_v->size;
        dirty |= self_v->is_dirty();
        backend = (JitBackend) self_v->backend;
        if ((VarType) self_v->type != VarType::UInt32)
            jitc_raise("jit_var_call(): 'self' argument must be an unsigned "
                       "32-bit integer array");
    }

    /* Check 'mask' */ {
        const Variable *mask_v = jitc_var(mask_);
        size = std::max(size, mask_v->size);
        dirty |= (bool) mask_v->is_dirty();
        if ((VarType) mask_v->type != VarType::Bool)
            jitc_raise(
                "jit_var_call(): 'mask' argument must be a boolean array");
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        Variable *v = jitc_var(in[i]);
        if ((VarKind) v->kind == VarKind::CallInput) {
            if (!v->dep[0])
                jitc_raise("jit_var_call(): symbolic variable r%u does not "
                           "reference another input!", in[i]);
            Variable *v2 = jitc_var(v->dep[0]);
            dirty |= v2->is_dirty();
            size = std::max(size, v2->size);
        } else if (!v->is_literal()) {
            jitc_raise("jit_var_call(): input variable r%u must either be a "
                       "literal or symbolic wrapper around another variable!", in[i]);
        } else {
            // Literal field, read temporarily stashed size (see
            // jitc_var_call_input in var.cpp)
            size = std::max(size, v->scratch);
            // Reset 'scratch' to not interfere with visited tracking in jit_eval()
            v->scratch = 0;
        }

        if (v->size != 1)
            jitc_raise(
                "jit_var_call(): size of input variable r%u is %u (must be 1)!",
                in[i], v->size);
    }

    for (uint32_t i = 0; i < n_inner_out; ++i) {
        const Variable *v0 = jitc_var(inner_out[i % n_out]),
                       *v1 = jitc_var(inner_out[i]);

        if (v0->type != v1->type)
            jitc_raise("jit_var_call(): output types don't match between instances.");

        if (v1->size != 1)
            jitc_raise("jit_var_call(): size of output variable r%u is %u "
                       "(must be 1)!", inner_out[i], v1->size);
    }

    ThreadState *ts = thread_state(backend);
    if (ts->side_effects_symbolic.size() !=
        (checkpoints[n_inst] & checkpoint_mask))
        jitc_raise("jitc_var_call(): side effect queue doesn't have the "
                   "expected size!");

    if (dirty) {
        jitc_eval(ts);

        uint32_t dirty_index = 0;
        if (jitc_var(self)->is_dirty())
            dirty_index = self;
        if (jitc_var(mask_)->is_dirty())
            dirty_index = mask_;

        for (uint32_t i = 0; i < n_in; ++i) {
            const Variable *v = jitc_var(in[i]);
            if ((VarKind) v->kind == VarKind::CallInput) {
                if (jitc_var(v->dep[0])->is_dirty()) {
                    dirty_index = v->dep[0];
                    break;
                }
            }
        }

        if (unlikely(dirty_index))
            jitc_raise_dirty_error(dirty_index);
    }

    // =====================================================
    // 3. Apply any masks on the stack, ignore self==NULL
    // =====================================================

    bool optimize = flags & (uint32_t) JitFlag::OptimizeCalls,
         debug = flags & (uint32_t) JitFlag::Debug;

    Ref mask;
    {
        uint32_t zero = 0;
        Ref null_instance = steal(jitc_var_literal(backend, VarType::UInt32, &zero, size, 0)),
            is_non_null   = steal(jitc_var_neq(self, null_instance)),
            mask_2        = steal(jitc_var_and(mask_, is_non_null));

        mask = steal(jitc_var_mask_apply(mask_2, size));

        if (debug)
            mask = steal(jitc_var_check_bounds(BoundsCheckType::Call, self,
                                               mask, max_inst_id + 1));
    }

    // =====================================================
    // 3. Stash information about inputs and outputs
    // =====================================================
    //
    std::unique_ptr<CallData> call(new CallData());
    call->backend = backend;
    call->name = name;
    call->n_in = n_in;
    call->n_out = n_out;
    call->n_inst = n_inst;
    call->optimize = optimize;
    call->inst_id = std::vector<uint32_t>(inst_id, inst_id + n_inst);
    call->inst_hash.resize(n_inst);
    call->outer_in.reserve(n_in);
    call->inner_in.reserve(n_in);
    call->outer_out.reserve(n_out);
    call->inner_out.reserve(n_inner_out);
    call->out_offset.resize(n_out, 0);
    call->in_active.resize(n_in, 0);

    // Move recorded side effects & offsets into 'call'
    call->checkpoints.reserve(n_inst + 1);
    for (uint32_t i = 0; i < n_inst; ++i)
        call->checkpoints.push_back(checkpoints[i] - checkpoints[0]);
    call->checkpoints.push_back(checkpoints[n_inst] - checkpoints[0]);

    call->side_effects = std::vector<uint32_t>(
        ts->side_effects_symbolic.begin() + (checkpoints[0] & checkpoint_mask),
        ts->side_effects_symbolic.begin() + (checkpoints[n_inst] & checkpoint_mask));
    ts->side_effects_symbolic.resize(checkpoints[0] & checkpoint_mask);

    // Collect inputs
    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t index = in[i];
        Variable *v = jitc_var(index);
        jitc_var_inc_ref(index, v);

        call->inner_in.push_back(index);
        if (!v->is_literal())
            index = v->dep[0];
        call->outer_in.push_back(index);
    }

    // =====================================================
    // 4. Collect evaluated data accessed by the instances
    // =====================================================

    call->instance_layout.reserve(n_inst);

    // Collect accesses to evaluated variables/pointers, appending a slot to
    // ``slots`` for each one in traversal order. Byte offsets are assigned per
    // instance by jitc_call_layout_instance(), not by the scan.
    uint32_t data_size = 0, inst_id_max = 0;

    // Scratch reused across instances by the layout pass.
    std::vector<CallData::CaptureSlot> reordered;

    // ``alignment`` is the per-backend block alignment; ``llvm_pkt_cap`` is the
    // LLVM gather width (in elements) that aligns each size bucket, and 1 elsewhere
    // (so those backends pack blocks contiguously).
    uint32_t align = call_data_align(ts, /*uses_optix_=*/ false),
             llvm_pkt_cap = backend == JitBackend::LLVM
                                ? llvm_call_data_packet_cap() : 1u;
    for (uint32_t i = 0; i < n_inst; ++i) {
        if (unlikely(checkpoints[i] > checkpoints[i + 1]))
            jitc_raise("jitc_var_call(): values in 'checkpoints' are not "
                       "monotonically increasing!");

        uint32_t id = inst_id[i];

        // Advance the traversal generation counter
        jitc_visit_new_gen();

        uint32_t slot_start = (uint32_t) call->slots.size();

        for (uint32_t j = 0; j < n_out; ++j)
            jitc_var_call_analyze(call.get(), i, inner_out[j + i * n_out]);

        for (uint32_t j = checkpoints[i]; j != checkpoints[i + 1]; ++j)
            jitc_var_call_analyze(call.get(), i,
                                  call->side_effects[j - checkpoints[0]]);

        jitc_call_layout_instance(call.get(), slot_start,
                                  (uint32_t) call->slots.size(), backend,
                                  align, llvm_pkt_cap, data_size,
                                  reordered);

        inst_id_max = std::max(id, inst_id_max);
    }

    // Pad the data block so each call's slice stays aligned once concatenated.
    call->data_size = align_up(data_size, align);
    call->offset_count = inst_id_max + 1;

    char temp[128];

    // =====================================================
    // 5. Create special variable representing the call op.
    // =====================================================

    Ref call_v = steal(
        jitc_var_new_node_2(backend, VarKind::Call, VarType::Void, size,
                            symbolic, self, jitc_var(self), mask,
                            jitc_var(mask)));

    call->id = call_v;
    {
        Variable *call_var = jitc_var(call_v);
        call_var->optix = call->use_optix;
        call_var->data = call.get();
    }

    // =====================================================
    // 6. Create output variables
    // =====================================================

    // Optimize calling conventions by reordering inputs
    uint32_t n_devirt = 0;

    call->inner_out.resize(n_inner_out, 0);

    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = inner_out[i];
        const Variable *v = jitc_var(index);

        bool uniform = true;
        for (uint32_t j = 0; j < n_inst; ++j) {
            uint32_t index2 = inner_out[j * n_out + i];
            uniform &= index == index2;
            jitc_var_inc_ref(index2);
            call->inner_out[j * n_out + i] = index2;
        }

        // Devirtualize uniform scalar literals when optimizations are enabled
        if (uniform && optimize && v->is_literal()) {
            out[i] = jitc_var_and(index, mask);
            n_devirt++;
            call->outer_out.push_back(WeakRef());
        } else {

            uint32_t out_index =
                jitc_var_new_node_1(backend, VarKind::CallOutput, (VarType) v->type,
                                    size, symbolic, call_v, jitc_var(call_v), i);
            Variable *v2 = jitc_var(out_index);
            v2->literal = i;

            snprintf(temp, sizeof(temp), "Call: %s [out %u]", name, i);
            jitc_var_set_label(out_index, temp);

            call->outer_out.emplace_back(out_index, v2->counter);
            out[i] = out_index;
        }
    }

    uint32_t se_count = checkpoints[n_inst] - checkpoints[0];

    jitc_log(InfoSym,
             "jit_var_call(r%u, self=r%u): call (\"%s\") with %u instance%s, %u "
             "input%s, %u output%s (%u devirtualized), %u side effect%s, %u "
             "byte%s of call data, %u elements%s%s", (uint32_t) call_v, self, name, n_inst,
             n_inst == 1 ? "" : "s", n_in, n_in == 1 ? "" : "s", n_out,
             n_out == 1 ? "" : "s", n_devirt, se_count, se_count == 1 ? "" : "s",
             data_size, data_size == 1 ? "" : "s", size,
             (n_devirt == n_out && se_count == 0) ? " (optimized away)" : "",
             symbolic ? " ([symbolic])" : "");


    // =====================================================
    // 8. Install code generation and deallocation callbacks
    // =====================================================

    snprintf(temp, sizeof(temp), "Call: %s", name);
    jitc_var_set_label(call_v, temp);

    Ref se_v;
    if (se_count) {
        /* The call has side effects. Create a dummy variable to
           ensure that they are evaluated */
        se_v = steal(jitc_var_new_node_1(backend, VarKind::Nop, VarType::Void,
                                         size, symbolic, call_v,
                                         jitc_var(call_v)));

        snprintf(temp, sizeof(temp), "Call: %s [side effects]", name);
        jitc_var_set_label(se_v, temp);
    }

    jitc_var_set_callback(
        call_v,
        [](uint32_t, int free, void *p) {
            if (free)
                delete (CallData *) p;
        },
        call.release(), true);

    jitc_var_mark_side_effect(se_v.release());
}

/// Build a getter as a gather from the shared per-kernel call-data buffer. See
/// the documentation of 'jit_var_call_getter' in the public header for details.
uint32_t jitc_var_call_getter(VarType type, uint32_t count,
                              const uint32_t *values, uint32_t index,
                              uint32_t mask) {
    if (index == 0 || mask == 0)
        jitc_raise("jit_var_call_getter(): 'index'/'mask' are uninitialized!");

    const Variable *index_v = jitc_var(index),
                   *mask_v   = jitc_var(mask);

    JitBackend backend = (JitBackend) index_v->backend;

    if ((VarType) index_v->type != VarType::UInt32)
        jitc_raise("jit_var_call_getter(): 'index' must be an unsigned 32-bit "
                   "integer array!");
    if ((VarType) mask_v->type != VarType::Bool)
        jitc_raise("jit_var_call_getter(): 'mask' must be a boolean array!");
    if ((JitBackend) mask_v->backend != backend)
        jitc_raise("jit_var_call_getter(): 'index' and 'mask' have different "
                   "backends!");

    // Collect the per-callable values. Literals stay as-is; opaque variables are
    // evaluated now so a device pointer exists at upload. Absent -> empty Ref.
    std::unique_ptr<GetterData> gd(new GetterData());
    gd->type = type;
    gd->count = count;
    gd->values.resize(count);

    for (uint32_t j = 0; j < count; ++j) {
        uint32_t vi = values[j];
        if (!vi)
            continue; // absent callable -> empty reference -> reads zero

        const Variable *vv = jitc_var(vi);
        if (vv->size != 1)
            jitc_raise("jit_var_call_getter(): value r%u of callable %u has "
                       "size %u (must be 1)!", vi, j, vv->size);
        if ((VarType) vv->type != type)
            jitc_raise("jit_var_call_getter(): value r%u of callable %u has an "
                       "unexpected type!", vi, j);
        if ((JitBackend) vv->backend != backend)
            jitc_raise("jit_var_call_getter(): value r%u of callable %u has a "
                       "different backend!", vi, j);

        if (vv->is_literal()) {
            gd->values[j] = borrow(vi);
        } else {
            void *ptr = nullptr;
            gd->values[j] = steal(jitc_var_data(vi, /*eval_dirty=*/true, &ptr));
        }
    }

    // Apply the active mask and broadcast the size, as 'jit_var_gather' would.
    uint32_t size = std::max(jitc_var(index)->size, jitc_var(mask)->size);
    Ref mask_2 = steal(jitc_var_mask_apply(mask, size));
    size = std::max(size, jitc_var(mask_2)->size);

    // Build the node by hand with LVN disabled: the value table is outside
    // 'VariableKey', so LVN could wrongly merge getters sharing [index, mask].
    bool symbolic = jitc_var(index)->symbolic || jitc_var(mask_2)->symbolic;

    Variable v;
    v.kind = (uint32_t) VarKind::CallGetter;
    v.type = (uint32_t) type;
    v.backend = (uint32_t) backend;
    v.size = size;
    v.symbolic = symbolic;
    v.dep[0] = index;
    v.dep[1] = (uint32_t) mask_2;

    jitc_var_inc_ref(index);
    jitc_var_inc_ref(mask_2);

    uint32_t result = jitc_var_new(v, /*disable_lvn=*/true);

    // Re-read the node type in case 'jitc_var_new' demoted it (Metal
    // Float64 -> Float32), so the header slice matches the emitted load.
    Variable *rv = jitc_var(result);
    gd->id = result;
    gd->type = (VarType) rv->type;
    rv->data = gd.get();

    jitc_var_set_callback(
        result,
        [](uint32_t, int free, void *p) {
            if (free)
                delete (GetterData *) p;
        },
        gd.release(), true);

    jitc_log(Debug, "jit_var_call_getter(): r%u = getter(index=r%u, mask=r%u) "
             "over %u callable%s", result, index, (uint32_t) mask_2, count,
             count == 1 ? "" : "s");

    return result;
}

/// Reserve a getter's slice of the buffer's header region, register it for
/// upload, and dispatch to the backend-specific masked-load renderer.
void jitc_var_call_getter_assemble(Variable *v, const Variable *index,
                                   const Variable *mask) {
    GetterData *gd = (GetterData *) v->data;
    uint32_t tsize = type_size[(int) gd->type];

    // The header region is physically first, so 'header_offset' is the table's
    // absolute byte offset. Round up to 8 bytes to keep later u64 tables aligned.
    gd->header_offset = call_buffer.fused_offset_size;
    call_buffer.fused_offset_size += align_up((gd->count + 1) * tsize, 8u);

    // Keep the node (hence its value references, hence the source data) alive
    // through the aggregate, mirroring how calls_assembled pins call nodes.
    jitc_var_inc_ref(gd->id);
    call_buffer.getters.push_back(gd);

    JitBackend backend = (JitBackend) v->backend;
    if (jitc_is_llvm(backend))
        jitc_var_call_getter_assemble_llvm(v, index, mask);
#if defined(DRJIT_ENABLE_METAL)
    else if (jitc_is_metal(backend))
        jitc_var_call_getter_assemble_metal(v, index, mask);
#endif
#if defined(DRJIT_ENABLE_CUDA)
    else if (jitc_is_cuda(backend))
        jitc_var_call_getter_assemble_cuda(v, index, mask);
#endif
    else
        jitc_fail("jitc_var_call_getter_assemble(): unsupported backend!");
}

static ProfilerRegion profiler_region_call_assemble("jit_var_call_assemble");

/// Data structure to sort function arguments/return values in order of
/// decreasing size and increasing index
struct PermKey {
    uint32_t size;
    uint32_t index;
    PermKey(uint32_t size, uint32_t index) : size(size), index(index) { }

    bool operator<(const PermKey &k) const {
        if (size > k.size)
            return true;
        else if (size < k.size)
            return false;
        return index < k.index;
    }
};

static std::vector<PermKey> call_perm;

/// Called when Dr.Jit compiles a function call, specifically the 'Call' IR node
void jitc_var_call_assemble(CallData *call, uint32_t call_reg,
                            uint32_t self_reg, uint32_t mask_reg) {

    ProfilerPhase profiler(profiler_region_call_assemble);

    // Reserve this call's slice of the call-data buffer. Nested calls reserve
    // their slices the same way when the recursion below assembles them.
    call->offset_base = call_buffer.fused_offset_size;
    call->data_base   = call_buffer.fused_data_size;
    call_buffer.fused_offset_size += call->offset_count * (uint32_t) sizeof(uint64_t);
    call_buffer.fused_data_size   += call->data_size;

    // Record this call's capture slots
    for (const CallData::CaptureSlot &slot : call->slots) {
        if (!jitc_var(slot.ref))
            continue;
        call_buffer.data_entries.push_back(
            { borrow(slot.ref.index), call->data_base + slot.offset });
    }

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

    /// Restore changes to 'schedule' when the function returns or throw
    struct RestoreGuard {
        std::vector<JitBackupRecord> &backup;

        ~RestoreGuard() {
            for (ScheduledVariable &sv : schedule) {
                Variable *v = jitc_var(sv.index);
                v->reg_index = 0;
                v->output_flag = false;
                jitc_var_dec_ref(sv.index, v);
            }

            schedule.clear();
            for (const JitBackupRecord &b : backup) {
                Variable *v = jitc_var(b.sv.index);
                v->param_type = b.param_type;
                v->output_flag = b.output_flag;
                v->reg_index = b.reg_index;
                v->param_offset = b.param_offset;
                schedule.push_back(b.sv);
            }
        }
    } restore_guard { backup };

    // =========================================================
    // 2. Determine size and alignment of parameter and return
    //    value buffers. Unreferenced return values present an
    //    opportunity for dead code elimination.
    // =========================================================

    bool optimize = call->optimize;

    uint32_t n_inst = call->n_inst,
             n_in = call->n_in,
             n_out = call->n_out,
             n_in_active = 0,
             n_out_active = 0,
             in_size = 0, in_size_all = 0, in_align = 1,
             out_size = 0, out_size_all = 0, out_align = 1;

    call_perm.clear();
    call_perm.reserve(std::max(n_in, n_out));
    for (uint32_t i = 0; i < n_in; ++i)
        call_perm.emplace_back(type_size[jitc_var(call->inner_in[i])->type], i);
    std::sort(call_perm.begin(), call_perm.end());

    for (auto [size, i] : call_perm) {
        Variable *v = jitc_var(call->outer_in[i]);
        Variable *v_in = jitc_var(call->inner_in[i]);
        bool unused =
            !v->reg_index ||
            (optimize && // Optimizations are on
                (v->is_literal() || // Literals are propagated
                 v_in->ref_count == 1)); // CallInput is never used

        call->in_active[i] = !unused;

        if (unused)
            continue;

        in_size_all += size;
        v->param_offset = in_size;
        in_size += size;
        in_align = std::max(size, in_align);
        n_in_active++;
    }

    call_perm.clear();
    for (uint32_t i = 0; i < n_out; ++i)
        call_perm.emplace_back(type_size[jitc_var(call->inner_out[i])->type], i);
    std::sort(call_perm.begin(), call_perm.end());

    if (jitc_is_llvm(call->backend)) {
        // In LLVM mode, the same buffer is used for inputs *and* outputs
        out_size = in_size;
        out_align = in_align;

        // Ensure alignment given that we're appending to the input buffer
        if (!call_perm.empty())
            out_size = align_up(out_size, call_perm[0].size);
    }

    for (PermKey k : call_perm) {
        out_size_all += k.size;

        uint32_t param_offset = (uint32_t) -1;
        Variable *v = jitc_var(call->outer_out[k.index]);
        // Skip outputs that are already evaluated (param_type == Input): their
        // data lives in memory and is read directly as a kernel input, so the
        // call need not compute or return them.
        bool live = v && v->reg_index && v->param_type != ParamType::Input;
        if (live || !optimize) {
            param_offset = out_size;
            out_size += k.size;
            out_align = std::max(k.size, out_align);
            n_out_active++;
        }
        call->out_offset[k.index] = (uint32_t) param_offset;
    }

    // =====================================================
    // 3. Compile code for all instances and collapse
    // =====================================================

    using CallablesSet = std::set<XXH128_hash_t, XXH128Cmp>;
    CallablesSet callables_set;

    int32_t alloca_size_backup = alloca_size;
    int32_t alloca_align_backup = alloca_align;
    alloca_size = alloca_align = -1;

    for (size_t i = 0; i < n_inst; ++i) {
        XXH128_hash_t hash =
            jitc_assemble_func(call, (uint32_t) i, in_size, in_align, out_size, out_align);
        call->inst_hash[i] = hash;
        callables_set.insert(hash);
    }

    alloca_size = alloca_size_backup;
    alloca_align = alloca_align_backup;

    if (jitc_is_llvm(call->backend))
        jitc_var_call_assemble_llvm(call, call_reg, self_reg, mask_reg,
                                    out_size, out_align);

#if defined(DRJIT_ENABLE_METAL)
    else if (jitc_is_metal(call->backend))
        jitc_var_call_assemble_metal(call, call_reg, self_reg, mask_reg,
                                     in_size, in_align,
                                     out_size, out_align);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    else if (jitc_is_cuda(call->backend))
        jitc_var_call_assemble_cuda(call, call_reg, self_reg, mask_reg,
                                    in_size, in_align,
                                    out_size, out_align);
#endif

    else
        jitc_fail("jit_var_call_assemble(): unsupported backend!");

    // =====================================================
    // 4. Restore previously backed-up JIT state
    // =====================================================

    // Main cleanup happens when 'restore_guard' leaves the scope

    // Undo previous change (for more sensible debug out put about buffer sizes)
    if (jitc_is_llvm(call->backend))
        out_size -= in_size;

    jitc_log(InfoSym,
             "jit_var_call_assemble(): call (\"%s\") to %zu/%u functions, "
             "passing %u/%u inputs (%u/%u bytes), %u/%u outputs (%u/%u bytes), "
             "%zu side effects",
             call->name.c_str(), callables_set.size(), call->n_inst,
             n_in_active, n_in, in_size, in_size_all, n_out_active, n_out,
             out_size, out_size_all, call->side_effects.size());

    jitc_var_inc_ref(call->id);
    calls_assembled.push_back(call);
    call->clear_side_effects();
}

/// Collect scalar / pointer variables referenced by a computation
void jitc_var_call_analyze(CallData *call, uint32_t inst_id, uint32_t index) {
    Variable *v = jitc_var(index);

    // Do not visit nodes multiple times. See eval.cpp for details on
    // the visit generation counting scheme.
    uint32_t stamp = visit_gen << 1;
    if (v->scratch == stamp)
        return;
    v->scratch = stamp;

    if (v->optix)
        call->use_optix |= true;

    VarKind kind = (VarKind) v->kind;

    if (kind == VarKind::CallSelf) {
        call->use_self = true;
    } else if (kind == VarKind::Counter) {
        call->use_index = true;
    } else if (kind == VarKind::ThreadIndex) {
        call->use_thread_id = true;
    } else if (kind == VarKind::CallInput) {
        return;
    } else if (kind == VarKind::Call) {
        CallData *call2 = (CallData *) v->data;
        call->use_self  |= call2->use_self;
        call->use_optix |= call2->use_optix;
        call->use_index |= call2->use_index;
        call->use_thread_id |= call2->use_thread_id;

        // The enclosing callable contains a nested call, so it must receive
        // a pointer to the buffer containing captured opaque state.
        call->use_nested = true;
        for (uint32_t index_2: call2->outer_in)
            jitc_var_call_analyze(call, inst_id, index_2);
    } else if (kind == VarKind::CallGetter) {
        // A nested getter reads the shared buffer, so Dr.Jit must forward the
        // base pointer into the callable.
        call->use_nested = true;
    } else if (kind == VarKind::LoopCond) {
        LoopData *loop = (LoopData *) jitc_var(v->dep[0])->data;

        uint32_t loop_state_size = (uint32_t) loop->size;
        for (uint32_t i = 0; i < loop_state_size; ++i) {
            // Loop-invariant variables should not be traversed, as they might
            // never be used in the call. In cases where these variables are
            // used, they will still be traversed when acessing whichever
            // variable/operation used said loop-invariaint variable. Typically,
            // the variables are still used "indirectly", like the source of `gather`.
            if (loop->outer_in[i] == loop->inner_in[i])
                continue;

            jitc_var_call_analyze(call, inst_id, loop->inner_out[i]);
            jitc_var_call_analyze(call, inst_id, loop->outer_in[i]);
        }
    } else if (kind == VarKind::PacketScatter) {
        PacketScatterData *psd = (PacketScatterData *) v->data;
        for (uint32_t i : psd->values)
            jitc_var_call_analyze(call, inst_id, i);
    } else if (kind == VarKind::CoopVecPack) {
        CoopVecPackData *cvid = (CoopVecPackData *) v->data;
        for (uint32_t i : cvid->indices)
            jitc_var_call_analyze(call, inst_id, i);
    } else if (kind == VarKind::TraceRay) {
        TraceData *td = (TraceData *) v->data;
        for (uint32_t index_2: td->indices)
            jitc_var_call_analyze(call, inst_id, index_2);
    } else if (kind == VarKind::TexLookup || kind == VarKind::TexFetchBilerp ||
               kind == VarKind::TexWrite) {
        // Texture writes always reference their coordinates and values. On
        // Metal, reads do too, since the coordinates are passed out-of-band.
        bool refs_coords = kind == VarKind::TexWrite ||
                           (JitBackend) v->backend == JitBackend::Metal;

        if (v->data && refs_coords) {
            TexData *td = (TexData *) v->data;
            for (uint32_t i = 0; i < td->ndim; ++i)
                jitc_var_call_analyze(call, inst_id, td->indices[i]);
            for (uint32_t i = 0; i < td->n_values; ++i)
                jitc_var_call_analyze(call, inst_id, td->values[i]);
        }
    } else if (v->is_evaluated() || (VarType) v->type == VarType::Pointer) {
        // Real data slot. Append it to the current instance's run in ``slots``; the
        // byte offset is assigned later by the layout pass in jitc_var_call().
        call->slots.push_back({ WeakRef(index, v->counter), 0,
                                CallData::SizeBucket(v) });

        if (v->size != 1)
            jitc_raise(
                "jit_var_call(): an indirect call to instance %u accesses an "
                "evaluated variable r%u of type %s and size %u. However, only "
                "*scalar* (size == 1) evaluated variables can be accessed "
                "while recording indirect function calls",
                inst_id, index, type_name[v->type], v->size);
    }

    for (int i = 0; i < 4; ++i) {
        uint32_t index_2 = v->dep[i];
        if (!index_2)
            break;

        jitc_var_call_analyze(call, inst_id, index_2);
    }
}

/// Build the shared call-data buffer (offset tables + data blocks) for every
/// call assembled in the current kernel and bind it to the base pointer
/// parameter. Runs at the end of jitc_{cuda,llvm,metal}_assemble.
void jitc_call_upload(ThreadState *ts) {
    JitBackend backend = (JitBackend) ts->backend;

    if (calls_assembled.empty() && call_buffer.getters.empty()) {
        jitc_assert(call_buffer.base_v == 0,
                    "jitc_call_upload(): base allocated without calls/getters!");
        return;
    }

    jitc_assert(call_buffer.base_v != 0,
                "jitc_call_upload(): kernel performs calls/getters but has no "
                "base pointer!");

    // Align the start of the data region so each call's slice lands at an
    // absolute address suitable for the widest packet load.
    uint32_t align = call_data_align(ts, uses_optix);
    const uint32_t off_size = call_buffer.fused_offset_size,
                   data_start = align_up(off_size, align),
                   total      = data_start + call_buffer.fused_data_size;

    // One aggregation entry per data slot, two per offset-table slot (lo + hi),
    size_t n_entries = call_buffer.data_entries.size();
    for (CallData *call : calls_assembled)
        n_entries += 2 * call->offset_count;

    // ..and 'count + 1' per getter (1 per instance, 1 for the null instance)
    for (GetterData *gd : call_buffer.getters)
        n_entries += gd->count + 1;

    AggregationEntry *agg = (AggregationEntry *) jitc_malloc(
        backend, sizeof(AggregationEntry) * n_entries, /*shared=*/true);
    AggregationEntry *p = agg;

    // Part 1: offset tables. We deliberately store 64-bit offsets via two
    // 32-bit writes so that ``record_ts.cpp`` wont try to to interpret the
    // pointer address as a variable reference.
    for (CallData *call : calls_assembled) {
        offset_entries.assign(call->offset_count, 0);

        for (uint32_t i = 0; i < call->n_inst; ++i) {
            auto it = globals_map.find(GlobalKey(
                call->inst_hash[i], call->n_inst != 1 ? GlobalType::IndirectCallable
                                                      : GlobalType::Callable));
            if (it == globals_map.end())
                jitc_fail("jitc_call_upload(): could not find callable!");
            uint32_t callable_index = it->second.callable_index;

            // entry.hi = absolute byte offset of the instance's data block (from
            // the start of the buffer); entry.lo = callable index.
            uint64_t data_off =
                (uint64_t) data_start + call->data_base +
                call->instance_layout[i].data_offset;
            offset_entries[call->inst_id[i]] = (data_off << 32) | callable_index;
        }

        for (uint32_t i = 0; i < call->offset_count; ++i) {
            uint64_t e = offset_entries[i];
            uint32_t pos = call->offset_base + i * (uint32_t) sizeof(uint64_t);

            p->offset = pos;
            p->size = 4;
            p->resource_kind = 0;
            p->src = (const void *) (uintptr_t) (uint32_t) e;
            p++;

            p->offset = pos + 4;
            p->size = 4;
            p->resource_kind = 0;
            p->src = (const void *) (uintptr_t) (uint32_t) (e >> 32);
            p++;
        }
    }

    // Part 2: capture slots
    for (const CallBufferState::CaptureSlotRef &e : call_buffer.data_entries) {
        const Variable *v = jitc_var((uint32_t) e.src);
        bool is_pointer = (VarType) v->type == VarType::Pointer;
        p->offset        = e.offset + data_start;
        p->size          = is_pointer ? (int16_t) 8 : (int16_t) -(int) type_size[v->type];
        p->resource_kind = is_pointer ? (uint16_t) v->resource_kind() : (uint16_t) 0;
        p->src           = is_pointer ? (const void *) v->literal : v->data;
        p++;
    }

    // Part 3: getter value tables. They live in the header region, so
    // 'header_offset' is absolute (no 'data_start' shift).
    for (GetterData *gd : call_buffer.getters) {
        uint32_t tsize = type_size[(int) gd->type];

        // Null slot (index 0): masked out at read time, zeroed defensively.
        p->offset = gd->header_offset;
        p->size = (int16_t) tsize;
        p->src = 0;
        p->resource_kind = 0;
        p++;

        for (uint32_t j = 0; j < gd->count; ++j) {
            uint32_t off = gd->header_offset + (j + 1) * tsize,
                     vi  = (uint32_t) gd->values[j];
            p->offset = off;
            p->resource_kind = 0;
            if (!vi) { // absent callable -> zero
                p->size = (int16_t) tsize;
                p->src  = 0;
            } else {
                const Variable *vv = jitc_var(vi);
                if (vv->is_literal()) { // literal -> embed by value
                    p->size = (int16_t) tsize;
                    p->src  = (const void *) vv->literal;
                } else { // opaque -> copy by pointer
                    p->size = (int16_t) -(int) tsize;
                    p->src  = vv->data;
                }
            }
            p++;
        }
    }

    uint8_t *buf = (uint8_t *) jitc_malloc(backend, total);
    jitc_aggregate(backend, buf, agg, (uint32_t) (p - agg));

    jitc_free(agg);

    // Hand the freshly built buffer to the base pointer variable. 'base_src'
    // had no backing storage (see jitc_assemble), so there is nothing to free;
    // it now owns 'buf' and releases it after the launch barrier.
    Variable *base_src = jitc_var(call_buffer.base_src);
    base_src->data = buf;
    base_src->size = total;

    jitc_var(call_buffer.base_v)->literal = (uint64_t) (uintptr_t) buf;

    for (CallData *call : calls_assembled)
        jitc_var_dec_ref(call->id);
    calls_assembled.clear();

    // Release getter nodes collected during assembly
    for (GetterData *gd : call_buffer.getters)
        jitc_var_dec_ref(gd->id);
    call_buffer.getters.clear();
}

// Compute a permutation to reorder an array of registered pointers
CallBucket *jitc_var_call_reduce(JitBackend backend, const char *variant,
                                 const char *domain, uint32_t index,
                                 uint32_t *bucket_count_inout) {

    struct CallReduceRecord {
        CallBucket *buckets;
        uint32_t bucket_count;

        ~CallReduceRecord() {
            for (uint32_t i = 0; i < bucket_count; ++i)
                jitc_var_dec_ref(buckets[i].index);
            jitc_free(buckets);
        }
    };

    VariableExtra *extra = jitc_var_extra(jitc_var(index));
    CallReduceRecord *rec = (CallReduceRecord *) extra->callback_data;
    if (rec) {
        *bucket_count_inout = rec->bucket_count;
        return rec->buckets;
    }

    uint32_t bucket_count;
    if (domain)
        bucket_count = jitc_registry_id_bound(variant, domain);
    else
        bucket_count = *bucket_count_inout;

    if (unlikely(bucket_count == 0)) {
        *bucket_count_inout = 0;
        return nullptr;
    }

    bucket_count++;

    // Ensure input index array is fully evaluated
    void *self = nullptr;
    Ref index_ptr = steal(jitc_var_data(index, true, &self));

    uint32_t size = jitc_var(index)->size;

    if (domain)
        jitc_log(InfoSym, "jit_var_call_reduce(r%u, domain=\"%s\")", index, domain);
    else
        jitc_log(InfoSym, "jit_var_call_reduce(r%u)", index);

    if (jitc_flags() & (uint32_t) JitFlag::Debug) {
        Ref max_idx_v = steal(jitc_var_reduce(backend, VarType::UInt32, ReduceOp::Max, index));
        uint32_t max_idx = 0;
        jitc_var_read(max_idx_v, 0, &max_idx);
        if (max_idx >= bucket_count)
            jitc_raise("jit_var_call_reduce(): out-of-bounds callable ID %u "
                       "(must be < %u).", max_idx, bucket_count);
    }

    size_t perm_size    = (size_t) size * (size_t) sizeof(uint32_t),
           offsets_size = (size_t(bucket_count) * 4 + 1) * sizeof(uint32_t);

    if (jitc_is_llvm(backend))
        perm_size += jitc_llvm_vector_width * sizeof(uint32_t);

    uint8_t *offsets = (uint8_t *) jitc_malloc(backend, offsets_size,
                                               /*shared=*/true);
    uint32_t *perm = (uint32_t *) jitc_malloc(backend, perm_size);

    // Compute permutation
    uint32_t unique_count = jitc_block_mkperm(backend, (const uint32_t *) self,
                                               size, size, bucket_count, perm,
                                               (uint32_t *) offsets),
             unique_count_out = unique_count;

    // Register permutation variable with JIT backend and transfer ownership
    uint32_t perm_var = jitc_var_mem_map(backend, VarType::UInt32, perm, size, 1);

    Variable v2;
    v2.kind = (uint32_t) VarKind::Evaluated;
    v2.type = (uint32_t) VarType::UInt32;
    v2.backend = (uint32_t) backend;
    v2.dep[3] = perm_var;
    v2.retain_data = true;
    v2.unaligned = 1;

    struct InputBucket {
        uint32_t id, offset, size, unused;
    };

    InputBucket *input_buckets = (InputBucket *) offsets;

    std::sort(
        input_buckets,
        input_buckets + unique_count,
        [](const InputBucket &b1, const InputBucket &b2) {
            return b1.size > b2.size;
        }
    );

    for (uint32_t i = 0; i < unique_count; ++i) {
        InputBucket bucket = input_buckets[i];

        // Create variable for permutation subrange
        v2.data = perm + bucket.offset;
        v2.size = bucket.size;

        jitc_var_inc_ref(perm_var);

        uint32_t index2 = jitc_var_new(v2);

        CallBucket bucket_out;
        if (domain)
            bucket_out.ptr = jitc_registry_ptr(variant, domain, bucket.id);
        else
            bucket_out.ptr = nullptr;

        bucket_out.index = index2;
        bucket_out.id = bucket.id;

        memcpy(input_buckets + i, &bucket_out, sizeof(CallBucket));

        jitc_trace("jit_var_call_reduce(): registered variable %u: bucket %u "
                   "(" DRJIT_PTR ") of size %u.", index2, bucket_out.id,
                   (uintptr_t) bucket_out.ptr, bucket.size);
    }

    jitc_var_dec_ref(perm_var);

    jitc_var_set_callback(
        index,
        [](uint32_t, int free, void *p) {
            if (free)
                delete (CallReduceRecord *) p;
        },
        new CallReduceRecord{ (CallBucket *) offsets, unique_count_out }, true);

    *bucket_count_inout = unique_count_out;
    return (CallBucket *) offsets;
}
