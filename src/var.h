#pragma once

#include <enoki/jit.h>

struct Variable;

/// Look up a variable by its ID
extern Variable *jit_var(uint32_t index);

/// Append a variable to the instruction trace (no operand)
extern uint32_t jit_trace_append_0(VarType type,
                                   const char *stmt,
                                   int stmt_static);

/// Append a variable to the instruction trace (1 operand)
extern uint32_t jit_trace_append_1(VarType type,
                                   const char *stmt,
                                   int stmt_static,
                                   uint32_t arg1);

/// Append a variable to the instruction trace (2 operands)
extern uint32_t jit_trace_append_2(VarType type,
                                   const char *stmt,
                                   int stmt_static,
                                   uint32_t arg1,
                                   uint32_t arg2);

/// Append a variable to the instruction trace (3 operands)
extern uint32_t jit_trace_append_3(VarType type,
                                   const char *stmt,
                                   int stmt_static,
                                   uint32_t arg1,
                                   uint32_t arg2,
                                   uint32_t arg3);

/// Register an existing variable with the JIT compiler
extern uint32_t jit_var_register(VarType type,
                                 void *ptr,
                                 size_t size,
                                 int free);

/// Register pointer literal as a special variable within the JIT compiler
extern uint32_t jit_var_register_ptr(const void *ptr);

/// Copy a memory region onto the device and return its variable index
extern uint32_t jit_var_copy_to_device(VarType type,
                                       const void *ptr,
                                       size_t size);

/// Increase the internal reference count of a given variable
extern void jit_var_int_ref_inc(uint32_t index);

/// Decrease the internal reference count of a given variable
extern void jit_var_int_ref_dec(uint32_t index);

/// Increase the external reference count of a given variable
extern void jit_var_ext_ref_inc(uint32_t index);

/// Decrease the external reference count of a given variable
extern void jit_var_ext_ref_dec(uint32_t index);

// Query the pointer variable associated with a given variable
extern void *jit_var_ptr(uint32_t index);

// Query the size of a given variable
extern size_t jit_var_size(uint32_t index);

/// Set the size of a given variable (if possible, otherwise throw)
extern uint32_t jit_var_set_size(uint32_t index, size_t size, int copy);

/// Assign a descriptive label to a given variable
extern void jit_var_label_set(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern const char *jit_var_label(uint32_t index);

/// Return the size of a given variable type
extern size_t jit_type_size(VarType type);

/// Migrate a variable to a different flavor of memory
extern void jit_var_migrate(uint32_t index, AllocType type);

/// Indicate that evaluation of the given variable causes side effects
extern void jit_var_mark_side_effect(uint32_t index);

/// Mark variable as dirty, e.g. because of pending scatter operations
extern void jit_var_mark_dirty(uint32_t index);

/// Inform the JIT that the next scatter/gather references var. 'index'
extern void jit_set_scatter_gather_operand(uint32_t index, int gather);

/// Return a human-readable summary of the contents of a variable
const char *jit_var_str(uint32_t index);

/// Return a human-readable summary of registered variables
extern const char *jit_var_whos();

/// Remove a variable from the cache used for common subexpression elimination
extern void jit_cse_drop(uint32_t index, const Variable *v);
