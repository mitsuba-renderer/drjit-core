/*
    src/op.h -- Conversion of standard operations into PTX and LLVM IR

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

/// Create a variable representing the result of a standard operation
extern uint32_t jitc_var_op(JitOp ot, const uint32_t *dep);

/// Create a variable that reads from another variable
extern uint32_t jitc_var_gather(uint32_t source, uint32_t index,
                                uint32_t mask);

/// Gather a contiguous vector of values
extern void jitc_var_gather_packet(size_t n, uint32_t source, uint32_t index,
                              uint32_t mask, uint32_t *out);

/// Schedule a scatter opartion that writes to an array
extern uint32_t jitc_var_scatter(uint32_t target, uint32_t value,
                                 uint32_t index, uint32_t mask,
                                 ReduceOp op, ReduceMode mode);

/// Scatter or scatter-reduce a contigous vector of values
extern uint32_t jitc_var_scatter_packet(size_t n, uint32_t target,
                                        const uint32_t *values, uint32_t index,
                                        uint32_t mask, ReduceOp op,
                                        ReduceMode mode);

/// Atomic Kahan summation
extern void jitc_var_scatter_add_kahan(uint32_t *target_1, uint32_t *target_2,
                                       uint32_t value, uint32_t index,
                                       uint32_t mask);

/// Atomic scatter-increment
extern uint32_t jitc_var_scatter_inc(uint32_t *target, uint32_t index, uint32_t mask);

/// Perform an ordinary or reinterpreting cast of the variable 'index'
extern uint32_t jitc_var_cast(uint32_t index, VarType target_type,
                              int reinterpret);

/// Check if a backend supports a desired type of scatter-reduction
extern bool jitc_can_scatter_reduce(JitBackend backend, VarType vt, ReduceOp op);

// Common unary operations
extern uint32_t jitc_var_neg(uint32_t a0);
extern uint32_t jitc_var_not(uint32_t a0);
extern uint32_t jitc_var_sqrt(uint32_t a0);
extern uint32_t jitc_var_abs(uint32_t a0);

// Common binary arithmetic operations
extern uint32_t jitc_var_add(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_sub(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_mul(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_div(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_mod(uint32_t a0, uint32_t a1);

// High multiplication
extern uint32_t jitc_var_mulhi(uint32_t a0, uint32_t a1);

// Fused multiply-add
extern uint32_t jitc_var_fma(uint32_t a0, uint32_t a1, uint32_t a2);

// Minimum, maximum
extern uint32_t jitc_var_min(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_max(uint32_t a0, uint32_t a1);

// Rounding operations
extern uint32_t jitc_var_ceil(uint32_t a0);
extern uint32_t jitc_var_floor(uint32_t a0);
extern uint32_t jitc_var_round(uint32_t a0);
extern uint32_t jitc_var_trunc(uint32_t a0);

// Comparisons
extern uint32_t jitc_var_eq(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_neq(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_lt(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_le(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_gt(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_ge(uint32_t a0, uint32_t a1);

// Ternary operator
extern uint32_t jitc_var_select(uint32_t a0, uint32_t a1, uint32_t a2);

// Bit-level counting operations
extern uint32_t jitc_var_popc(uint32_t a0);
extern uint32_t jitc_var_clz(uint32_t a0);
extern uint32_t jitc_var_ctz(uint32_t a0);
extern uint32_t jitc_var_brev(uint32_t a0);

/// Bit-wise operations
extern uint32_t jitc_var_and(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_or(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_xor(uint32_t a0, uint32_t a1);

// Shifts
extern uint32_t jitc_var_shl(uint32_t a0, uint32_t a1);
extern uint32_t jitc_var_shr(uint32_t a0, uint32_t a1);

// Fast approximations
extern uint32_t jitc_var_rcp(uint32_t a0);
extern uint32_t jitc_var_rsqrt(uint32_t a0);

// Multi-function generator (CUDA)
extern uint32_t jitc_var_sin_intrinsic(uint32_t a0);
extern uint32_t jitc_var_cos_intrinsic(uint32_t a0);
extern uint32_t jitc_var_exp2_intrinsic(uint32_t a0);
extern uint32_t jitc_var_log2_intrinsic(uint32_t a0);
extern uint32_t jitc_var_tanh_intrinsic(uint32_t a0);

/// Extra data describing a packet scatter operatoin
struct PacketScatterData {
    ReduceOp op     = ReduceOp::Identity;
    ReduceMode mode = ReduceMode::Auto;

    std::vector<uint32_t> values;
    WeakRef to_reduce;

    ~PacketScatterData() {
        for (uint32_t index: values)
            jitc_var_dec_ref(index);
    }
};
