/*
    src/metal_eval.h -- Metal Shading Language code generation helpers.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_METAL)

struct ThreadState;
struct ScheduledGroup;

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
/// helpers like the dd_* Float64 primitives — only the helpers actually
/// referenced by a kernel land in its source. Mirrors cuda_eval.h /
/// llvm_eval.h.
#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_metal(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__);         \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0)

#endif // defined(DRJIT_ENABLE_METAL)
