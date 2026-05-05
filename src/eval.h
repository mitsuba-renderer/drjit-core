/*
    src/eval.h -- Main computation graph evaluation routine

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include "strbuf.h"
#include <map>
#include <vector>
#include <unordered_map>

/// A single variable that is scheduled to execute for a launch with 'size' entries
struct ScheduledVariable {
    uint32_t size;
    uint32_t index;
    uint32_t scope;
    void *data;

    ScheduledVariable(uint32_t size, uint32_t scope, uint32_t index)
        : size(size), index(index), scope(scope), data(nullptr) { }
};

/// Start and end index of a group of variables that will be merged into the same kernel
struct ScheduledGroup {
    uint32_t size;
    uint32_t start;
    uint32_t end;

    ScheduledGroup(uint32_t size, uint32_t start, uint32_t end)
        : size(size), start(start), end(end) { }
};

enum class GlobalType : uint32_t {
    IndirectCallable = 0, // Multi-target vcalls, assembled first
    Callable = 1,         // Single-target vcalls, assembled second
    Global = 2            // Other globals (intrinsics, etc.), assembled last
};

struct GlobalKey {
    XXH128_hash_t hash;
    GlobalType type;

    GlobalKey(XXH128_hash_t hash, GlobalType type)
        : hash(hash), type(type) { }

    /* Order so that callables are defined before other globals, but don't use
       the callable ID itself for ordering (it can be non-deterministic in
       programs that use Dr.Jit with parallelization) */
    bool operator<(const GlobalKey &v) const {
        return std::tie(type, hash.high64, hash.low64) <
               std::tie(v.type, v.hash.high64, v.hash.low64);
    }
};

struct GlobalValue {
    /// Offset and length for the 'globals' buffer
    size_t start, length;

    /// Index within the callable list, if applicable
    uint32_t callable_index;

    GlobalValue(size_t start, size_t length)
        : start(start), length(length), callable_index(0) { }
};

/// Cache data structure for global declarations
using GlobalsMap = std::map<GlobalKey, GlobalValue>;

/// StringBuffer for global definitions (intrinsics, callables, etc.)
extern StringBuffer globals;

/// Mapping that describes the contents of the 'globals' buffer
extern GlobalsMap globals_map;

/// Name of the last generated kernel
extern char kernel_name[52];

/// Are we recording an OptiX kernel?
extern bool uses_optix;

/// Does this Metal kernel use ray tracing?
extern bool uses_metal_rt;

#if defined(DRJIT_ENABLE_METAL)
struct MetalScene;

/// Ordered list of distinct ``MetalScene*`` referenced by ``VarKind::TraceRay``
/// nodes anywhere in the current kernel (top-level schedule + callable
/// bodies + symbolic loops/conds — populated during the assemble pre-walk).
/// The position in this vector becomes the MSL slot index, i.e. scene at
/// index ``i`` is bound to ``[[buffer(1+i)]]`` and referenced as
/// ``accel_<i>`` in the generated MSL.
///
/// Reset at the start of every ``jitc_assemble`` (alongside ``uses_metal_rt``)
/// so each kernel gets a fresh map.
extern std::vector<MetalScene *> metal_kernel_scenes;

/// Look-up table: scene pointer → slot index in ``metal_kernel_scenes``.
/// Maintained in lockstep so per-TraceRay codegen can resolve a slot in O(1).
extern std::unordered_map<MetalScene *, uint32_t> metal_kernel_scene_slot;

/// Per-scene IFT buffer slot. Indexed by the same position as
/// ``metal_kernel_scenes``. Value is the buffer index of that scene's
/// ``ift_<i>`` argument (always >= 1 + N where N is the accel count), or
/// ``-1`` if the scene has no ``intersection_fn_library`` (no IFT param
/// is emitted for it). Populated by ``jitc_metal_finalize_scene_layout()``
/// after the pre-walk and before the kernel signature is emitted.
extern std::vector<int32_t> metal_kernel_ift_slot;

/// Total number of ``[[buffer]]`` slots consumed by accels + IFTs (i.e.
/// 1 + N + (count of scenes with IFT)). Equals the next free buffer slot
/// after the ray-tracing arguments. Mirrors what the kernel signature
/// declared and what callables / call sites need to forward.
extern uint32_t metal_kernel_buffer_count;

/// Register a scene with the kernel, assigning it the next free slot if it
/// hasn't been seen yet. Returns the slot index. ``nullptr`` returns 0 and
/// is not registered (caller is expected to handle that case).
extern uint32_t metal_register_kernel_scene(MetalScene *scene);

/// Compute IFT slot indices for every registered scene. Called once after
/// the recursive pre-walk and before any signature emission so all sites
/// (kernel header, per-TraceRay intersect, callable header, call site,
/// launch binding) agree on which slot belongs to which scene's IFT.
extern void jitc_metal_finalize_scene_layout();
#endif

/// Does this Metal kernel use a simdgroup_matrix-accelerated CoopVecMatVec?
/// When true the kernel emits an [[max_total_threads_per_threadgroup(32)]]
/// attribute (one SIMD-group per threadgroup, so threadgroup memory is
/// per-SG by definition) and reserves ``simdgroup_tgm_floats`` floats of
/// threadgroup memory shared across consecutive matvecs.
extern bool uses_simdgroup_matrix;

/// Maximum (K + M) sum across all simdgroup_matrix-eligible matvecs in the
/// current kernel. The kernel reserves ``simdgroup_tgm_floats * 32`` floats
/// of threadgroup memory at function entry.
extern uint32_t simdgroup_tgm_floats;

/// Size and alignment of auxiliary buffer needed by virtual function calls
extern int32_t alloca_size;
extern int32_t alloca_align;

/// Number of tentative callables that were assembled in the kernel being compiled
extern uint32_t indirect_callable_count;

/// Number of unique callables in the kernel being compiled
extern uint32_t indirect_callable_count_unique;

/// Specifies the nesting level of virtual calls being compiled
extern uint32_t callable_depth;

/// Ordered list of variables that should be computed
extern std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
extern std::vector<ScheduledGroup> schedule_groups;

/// Evaluate all computation that is queued on the current thread
extern void jitc_eval(ThreadState *ts);

/// Used by jitc_eval() to generate PTX source code
extern void jitc_cuda_assemble(ThreadState *ts, ScheduledGroup group,
                               uint32_t n_regs, uint32_t n_params);

/// Used by jitc_eval() to generate LLVM IR source code
extern void jitc_llvm_assemble(ThreadState *ts, ScheduledGroup group);

/// Used by jitc_call() to generate source code for calls
struct CallData;
extern XXH128_hash_t jitc_assemble_func(const CallData *call, uint32_t inst,
                                        uint32_t in_size, uint32_t in_align,
                                        uint32_t out_size, uint32_t out_align);

/// Used by jitc_call() to generate LLVM IR source code for callables
extern void jitc_llvm_assemble_func(const CallData *call, uint32_t inst);

/// Used by jitc_call() to generate PTX source code for callables
extern void jitc_cuda_assemble_func(const CallData *call, uint32_t inst,
                                    uint32_t in_size, uint32_t in_align,
                                    uint32_t out_size, uint32_t out_align,
                                    uint32_t n_regs);

/// Register a global declaration that will be included in the final program
extern void jitc_register_global(const char *str);
