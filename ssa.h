#pragma once

#include "api.h"

/// Register an existing variable with the JIT compiler
uint32_t jit_var_register(uint32_t type,
                          void *ptr,
                          size_t size,
                          bool free);

/// Append a variable to the instruction trace (no operand)
uint32_t jit_trace_append(uint32_t type,
                          const char *cmd);

/// Append a variable to the instruction trace (1 operand)
uint32_t jit_trace_append(uint32_t type,
                          const char *cmd,
                          uint32_t arg1);

/// Append a variable to the instruction trace (2 operands)
uint32_t jit_trace_append(uint32_t type,
                          const char *cmd,
                          uint32_t arg1,
                          uint32_t arg2);

/// Append a variable to the instruction trace (3 operands)
uint32_t jit_trace_append(uint32_t type,
                          const char *cmd,
                          uint32_t arg1,
                          uint32_t arg2,
                          uint32_t arg3);

/// Increase the internal reference count of a given variable
void jit_inc_ref_int(uint32_t index);

/// Decrease the internal reference count of a given variable
void jit_dec_ref_int(uint32_t index);

/// Increase the external reference count of a given variable
void jit_inc_ref_ext(uint32_t index);

/// Decrease the external reference count of a given variable
void jit_dec_ref_ext(uint32_t index);

// Query the pointer variable associated with a given variable
void *jit_var_ptr(uint32_t index);

// Query the size of a given variable
size_t jit_var_size(uint32_t index);

/// Set the size of a given variable (if possible, otherwise throw)
uint32_t jit_var_set_size(uint32_t index, size_t size, bool copy);

/// Assign a descriptive label to a given variable
void jit_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
const char *jit_var_label(uint32_t index);
