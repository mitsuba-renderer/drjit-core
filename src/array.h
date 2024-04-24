/*
    src/array.h -- Functionality to create, read, and write variable arrays

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit-core/jit.h>

enum class VarKind : uint32_t;
struct Variable;

/// Create a variable array
extern uint32_t jitc_array_create(JitBackend backend, VarType vt, size_t size,
                                  size_t length, uint32_t pred = 0);

/// Initialize an array variable with a literal constant & return a new variable
extern uint32_t jitc_array_init(uint32_t target, uint32_t value);

/// Return the size of a variable array
extern size_t jitc_array_length(uint32_t index);

/// Read from a variable array
extern uint32_t jitc_array_read(uint32_t source, uint32_t offset,
                                uint32_t mask);

/// Write to a variable array
extern uint32_t jitc_array_write(uint32_t target, uint32_t offset,
                                 uint32_t value, uint32_t mask);

/// Return a variable representing the storage region underlying an Array or ArrayWrite variable
extern uint32_t jitc_array_buffer(uint32_t index);
extern uint32_t jitc_array_buffer(const Variable *v);

/// Evolve array state machine during jitc_assemble()
extern void jitc_process_array_op(VarKind kind, Variable *v);
