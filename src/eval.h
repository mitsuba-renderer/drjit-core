/*
    src/eval.h -- Main computation graph evaluation routine

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>

/// Evaluate all computation that is queued on the current thread state
extern void jit_eval_ts(ThreadState *ts);

/// Evaluate all computation that is queued on the current thread
extern void jit_eval();

/// Export the intermediate representation of a computation
extern const char *jit_eval_ir(int cuda,
                               const uint32_t *in, uint32_t n_in,
                               const uint32_t *out, uint32_t n_out,
                               uint32_t n_side_effects,
                               uint64_t *hash_out,
                               void ***ptr_out,
                               uint32_t *ptr_count_out);

/// Like jit_eval_ir(), but returns a variable referincing the IR string
extern uint32_t jit_eval_ir_var(int cuda,
                                const uint32_t *in, uint32_t n_in,
                                const uint32_t *out, uint32_t n_out,
                                uint32_t n_side_effects,
                                uint64_t *hash_out,
                                void ***ptr_out,
                                uint32_t *ptr_count_out);

/// Insert an indirect function call into the program
extern void jit_var_vcall(int cuda, uint32_t self, uint32_t n_inst,
                          const uint32_t *inst_ids, const uint64_t *inst_hash,
                          uint32_t n_in, const uint32_t *in, uint32_t n_out,
                          uint32_t *out, uint32_t n_extra, const void **extra,
                          const uint32_t *extra_offset, int side_effects);
