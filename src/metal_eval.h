/*
    src/metal_eval.h -- Metal Shading Language code generation helpers.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_METAL)

#include <vector>
#include <cstdint>

struct ThreadState;
struct ScheduledGroup;
struct MetalScene;
struct Kernel;

/// Ordered list of distinct ``MetalScene*`` referenced by ``VarKind::TraceRay``
/// nodes anywhere in the current kernel (top-level schedule + callable
/// bodies + symbolic loops/conds, populated during the assemble pre-walk).
/// The position in this vector becomes the MSL slot index, i.e. scene at
/// index ``i`` is bound to ``[[buffer(1+i)]]`` and referenced as
/// ``accel_<i>`` in the generated MSL.
extern std::vector<MetalScene *> metal_kernel_scenes;

/// Per-scene IFT buffer slot. Indexed by the same position as
/// ``metal_kernel_scenes``. Value is the buffer index of that scene's
/// ``ift_<i>`` argument (always >= 1 + N where N is the accel count), or
/// ``-1`` if the scene has no ``intersection_fn_library``.
extern std::vector<int32_t> metal_kernel_ift_slot;

/// Register a scene with the kernel, assigning it the next free slot if it
/// hasn't been seen yet. Returns the slot index. ``nullptr`` returns 0 and
/// is not registered (caller is expected to handle that case).
extern uint32_t metal_register_kernel_scene(MetalScene *scene);

/// Compute IFT slot indices for every registered scene. Called once after
/// the recursive pre-walk and before any signature emission.
extern void jitc_metal_finalize_scene_layout();

/// Reset all per-kernel scene state. Called at the start of every
/// ``jitc_assemble``.
extern void jitc_metal_assemble_reset();

/// Persist the per-kernel ``MetalScene*`` list captured by the assemble
/// pre-walk into ``kernel.metal.scenes`` so frozen-function replay
/// (which skips re-assemble) can still bind the right TLASes / IFTs.
extern void jitc_metal_persist_kernel_scenes(Kernel &kernel);

/// Emit a complete MSL kernel for the given scheduled group.
extern void jitc_metal_assemble(ThreadState *ts, ScheduledGroup group,
                                uint32_t n_regs, uint32_t n_params);

/// Emit MSL code for a single callable body within a virtual function call.
struct CallData;
extern void jitc_metal_assemble_func(const CallData *call, uint32_t inst,
                                     uint32_t in_size, uint32_t in_align,
                                     uint32_t out_size, uint32_t out_align,
                                     uint32_t n_regs);

/// Append an MSL helper definition to the kernel's global preamble (deduped
/// by content hash via jitc_register_global). Use for on-demand emission of
/// helpers so that only the ones actually referenced by a kernel land in its
/// source. Mirrors cuda_eval.h / llvm_eval.h.
#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_metal(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__);         \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0)

#endif // defined(DRJIT_ENABLE_METAL)
