/*
    src/var.h -- Variable/computation graph-related functions.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>
#include <utility>

struct Variable;

/// Look up a variable by its ID
extern Variable *jit_var(uint32_t index);

/// Append a variable to the instruction trace (no operand)
extern uint32_t jit_var_new_0(int cuda,
                              VarType type,
                              const char *stmt,
                              int stmt_static,
                              uint32_t size);

/// Append a variable to the instruction trace (1 operand)
extern uint32_t jit_var_new_1(int cuda,
                              VarType type,
                              const char *stmt,
                              int stmt_static,
                              uint32_t op1);

/// Append a variable to the instruction trace (2 operands)
extern uint32_t jit_var_new_2(int cuda,
                              VarType type,
                              const char *stmt,
                              int stmt_static,
                              uint32_t op1,
                              uint32_t op2);

/// Append a variable to the instruction trace (3 operands)
extern uint32_t jit_var_new_3(int cuda,
                              VarType type,
                              const char *stmt,
                              int stmt_static,
                              uint32_t op1,
                              uint32_t op2,
                              uint32_t op3);

/// Append a variable to the instruction trace (4 operands)
extern uint32_t jit_var_new_4(int cuda,
                              VarType type,
                              const char *stmt,
                              int stmt_static,
                              uint32_t op1,
                              uint32_t op2,
                              uint32_t op3,
                              uint32_t op4);

/// Append a new variable storing 'size' entries of a literal constant
extern uint32_t jit_var_new_literal(int cuda, VarType type,
                                    uint64_t value, uint32_t size,
                                    int eval);

/// Register an existing variable with the JIT compiler
extern uint32_t jit_var_map_mem(int cuda, VarType type, void *ptr, uint32_t size,
                                int free);

/// Register pointer literal as a special variable within the JIT compiler
extern uint32_t jit_var_copy_ptr(int cuda, const void *ptr, uint32_t index);

/// Copy a memory region onto the device and return its variable index
extern uint32_t jit_var_copy_mem(int cuda, AllocType atype, VarType vtype,
                                 const void *ptr, uint32_t size);

/// Duplicate a variable
extern uint32_t jit_var_copy_var(uint32_t index);

/// Increase the internal reference count of a given variable
extern void jit_var_inc_ref_int(uint32_t index, Variable *v) noexcept(true);

/// Increase the internal reference count of a given variable
extern void jit_var_inc_ref_int(uint32_t index) noexcept(true);

/// Decrease the internal reference count of a given variable
extern void jit_var_dec_ref_int(uint32_t index, Variable *v) noexcept(true);

/// Decrease the internal reference count of a given variable
extern void jit_var_dec_ref_int(uint32_t index) noexcept(true);

/// Increase the external reference count of a given variable
extern void jit_var_inc_ref_ext(uint32_t index, Variable *v) noexcept(true);

/// Increase the external reference count of a given variable
extern void jit_var_inc_ref_ext(uint32_t index) noexcept(true);

/// Decrease the external reference count of a given variable
extern void jit_var_dec_ref_ext(uint32_t index, Variable *v) noexcept(true);

/// Decrease the external reference count of a given variable
extern void jit_var_dec_ref_ext(uint32_t index) noexcept(true);

// Query the pointer variable associated with a given variable
extern void *jit_var_ptr(uint32_t index);

// Query the size of a given variable
extern uint32_t jit_var_size(uint32_t index);

// Query the type of a given variable
extern VarType jit_var_type(uint32_t index);

// Resize a scalar variable
extern uint32_t jit_var_set_size(uint32_t index, uint32_t size);

/// Assign a descriptive label to a given variable
extern void jit_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern const char *jit_var_label(uint32_t index);

/// Assign a callback function that is invoked when the given variable is freed
extern void jit_var_set_free_callback(uint32_t index, void (*callback)(void *),
                                      void *payload);

/// Migrate a variable to a different flavor of memory
extern uint32_t jit_var_migrate(uint32_t index, AllocType type);

/// Mark a variable as a scatter operation that writes to 'target'
extern void jit_var_mark_scatter(uint32_t index, uint32_t target);

/// Is the given variable a literal that equals zero?
extern int jit_var_is_literal_zero(uint32_t index);

/// Is the given variable a literal that equals one?
extern int jit_var_is_literal_one(uint32_t index);

/// Return a human-readable summary of the contents of a variable
const char *jit_var_str(uint32_t index);

/// Read a single element of a variable and write it to 'dst'
extern void jit_var_read(uint32_t index, uint32_t offset, void *dst);

/// Reverse of jit_var_read(). Copy 'src' to a single element of a variable
extern void jit_var_write(uint32_t index, uint32_t offset, const void *src);

/// Schedule a variable \c index for future evaluation via \ref jitc_eval()
extern int jit_var_schedule(uint32_t index);

/// Evaluate the variable \c index right away, if it is unevaluated/dirty.
extern int jit_var_eval(uint32_t index);

/// Return a human-readable summary of registered variables
extern const char *jit_var_whos();

/// Return a GraphViz representation of registered variables
extern const char *jit_var_graphviz();

/// Remove a variable from the cache used for common subexpression elimination
extern void jit_cse_drop(uint32_t index, const Variable *v);

/// Append the given variable to the instruction trace and return its ID
extern std::pair<uint32_t, Variable *> jit_var_new(Variable &v,
                                                   bool disable_cse = false);

/// Query the current (or future, if not yet evaluated) allocation flavor of a variable
extern AllocType jit_var_alloc_type(uint32_t index);

/// Query the device (or future, if not yet evaluated) associated with a variable
extern int jit_var_device(uint32_t index);

/// Descriptive names and byte sizes for the various variable types
extern const char *var_type_name      [(int) VarType::Count];
extern const char *var_type_name_short[(int) VarType::Count];
extern const uint32_t var_type_size   [(int) VarType::Count];

/// Type names and register names for CUDA and LLVM
extern const char *var_type_name_llvm       [(int) VarType::Count];
extern const char *var_type_name_llvm_bin   [(int) VarType::Count];
extern const char *var_type_name_llvm_abbrev[(int) VarType::Count];
extern const char *var_type_name_llvm_big   [(int) VarType::Count];
extern const char *var_type_name_ptx        [(int) VarType::Count];
extern const char *var_type_name_ptx_bin    [(int) VarType::Count];
extern const char *var_type_prefix          [(int) VarType::Count];
extern const char *var_type_size_str        [(int) VarType::Count];
extern const char *var_type_label           [(int) VarType::Count];
