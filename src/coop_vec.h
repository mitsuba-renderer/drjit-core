/*
    src/coop_vec.h -- Backend-independent parts of the Cooperative Vector API

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <vector>

struct CoopVecPackData {
    std::vector<uint32_t> indices;

    ~CoopVecPackData() {
        for (uint32_t index: indices)
            jitc_var_dec_ref(index);
    }
};

struct CoopVecMatVecData {
    MatrixDescr A_descr;
    MatrixDescr b_descr;
    bool transpose;
};

extern uint32_t jitc_coop_vec_pack(uint32_t n, const uint32_t *in);

extern void jitc_coop_vec_unpack(uint32_t index, uint32_t n, uint32_t *out);

extern uint32_t jitc_coop_vec_literal(JitBackend backend, VarType type,
                                      const void *value, size_t size,
                                      uint32_t length);

extern uint32_t jitc_coop_vec_load(uint32_t buffer, uint32_t offset,
                                   uint32_t length);

extern uint32_t jitc_coop_vec_unpack(uint32_t vec, uint32_t index);

extern uint32_t jitc_coop_vec_unary_op(JitOp op, uint32_t a0);

extern uint32_t jitc_coop_vec_binary_op(JitOp op, uint32_t a0, uint32_t a1);

extern uint32_t jitc_coop_vec_ternary_op(JitOp op, uint32_t a0, uint32_t a1, uint32_t a2);

extern void jitc_coop_vec_pack_matrices(uint32_t count, uint32_t in,
                                        const MatrixDescr *in_descr,
                                        uint32_t out,
                                        const MatrixDescr *out_descr);

extern MatrixDescr jitc_coop_vec_compute_layout(uint32_t index,
                                                const MatrixDescr *in,
                                                MatrixLayout layout,
                                                uint32_t offset);

extern uint32_t jitc_coop_vec_matvec(uint32_t A_index,
                                     const MatrixDescr *A_descr,
                                     uint32_t x_index, uint32_t b_index,
                                     const MatrixDescr *b_descr, int transpose);

extern uint32_t jitc_coop_vec_accum(uint32_t target, uint32_t size,
                                    uint32_t offset, uint32_t index);

extern uint32_t jitc_coop_vec_outer_product_accum(uint32_t target, uint32_t size,
                                                  const MatrixDescr *descr,
                                                  uint32_t a, uint32_t b);

extern uint32_t jitc_coop_vec_cast(uint32_t index, VarType vt);
