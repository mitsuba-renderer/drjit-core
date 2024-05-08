/*
    src/llvm_packet.h -- Specialized memory operations that read or write
    multiple adjacent values at once.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"

extern void jitc_llvm_render_gather_packet(const Variable *v,
                                           const Variable *ptr,
                                           const Variable *index,
                                           const Variable *mask);

extern void jitc_llvm_render_scatter_packet(const Variable *v,
                                            const Variable *ptr,
                                            const Variable *index,
                                            const Variable *mask);
