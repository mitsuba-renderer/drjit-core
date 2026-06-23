/*
    src/llvm_packet.h -- Specialized memory operations that read or write
    multiple adjacent values at once.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"

struct CallData;

/// Coalesce an instance's per-call capture loads into packet (vector) loads,
/// setting ``packetized_until[c]`` to the first slot past the packetized prefix
/// of coalesceable size-class bucket ``c`` (0=8B, 1=4B, 2=2B, 3=1B).
extern void jitc_llvm_render_call_data(const CallData *call, uint32_t inst,
                                       uint32_t packetized_until[4]);

extern void jitc_llvm_render_gather_packet(const Variable *v,
                                           const Variable *ptr,
                                           const Variable *index,
                                           const Variable *mask);

extern void jitc_llvm_render_scatter_packet(const Variable *v,
                                            const Variable *ptr,
                                            const Variable *index,
                                            const Variable *mask);
