/*
    src/op.h -- Conversion of standard operations into PTX and LLVM IR

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>

/// Create a variable representing the result of a standard operation
extern uint32_t jitc_var_new_op(JitOp ot, uint32_t n_dep, const uint32_t *dep);

/// Perform an ordinary or reinterpreting cast of the variable 'index'
extern uint32_t jitc_var_new_cast(uint32_t index, VarType target_type,
                                  int reinterpret);

/// Create a variable that reads from another array
extern uint32_t jitc_var_new_gather(uint32_t src, uint32_t index,
                                    uint32_t mask);
