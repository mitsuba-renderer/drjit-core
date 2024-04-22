/*
    src/var.h -- Variable/computation graph-related functions.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <utility>

enum VarKind : uint32_t;

struct Variable;
struct VariableExtra;
struct WeakRef;

/// Access a variable by ID, terminate with an error if it doesn't exist
extern Variable *jitc_var(uint32_t index);

/// Access a variable through a weak reference. May return ``nullptr``
extern Variable *jitc_var(WeakRef ref);

/// Create a value constant variable of the given size
extern uint32_t jitc_var_literal(JitBackend backend, VarType type,
                                 const void *value, size_t size,
                                 int eval);

/// Create a variable counting from 0 ... size - 1
extern uint32_t jitc_var_counter(JitBackend backend, size_t size,
                                 bool simplify_scalar);

/// Create a new IR node. Just a wrapper around jitc_var_new without any error checking
extern uint32_t jitc_var_new_node_0(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool symbolic,
                                    uint64_t payload = 0);

extern uint32_t jitc_var_new_node_1(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool symbolic,
                                    uint32_t a0, Variable *v0,
                                    uint64_t payload = 0);

extern uint32_t jitc_var_new_node_2(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool symbolic,
                                    uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                                    uint64_t payload = 0);

extern uint32_t jitc_var_new_node_3(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool symbolic,
                                    uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                                    uint32_t a2, Variable *v2, uint64_t payload = 0);

extern uint32_t jitc_var_new_node_4(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool symbolic,
                                    uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                                    uint32_t a2, Variable *v2, uint32_t a3, Variable *v4,
                                    uint64_t payload = 0);

/// Create a variable representing uninitialized memory
extern uint32_t jitc_var_undefined(JitBackend backend, VarType type, size_t size);

/// Create a variable that refers to a memory region
extern uint32_t jitc_var_pointer(JitBackend backend, const void *value,
                                 uint32_t dep, int write);

/// Wrap an input variable of a virtual function call before recording computation
extern uint32_t jitc_var_call_input(uint32_t index);

/// Register an existing variable with the JIT compiler
extern uint32_t jitc_var_mem_map(JitBackend backend, VarType type, void *ptr,
                                 size_t size, int free);

/// Copy a memory region onto the device and return its variable index
extern uint32_t jitc_var_mem_copy(JitBackend backend, AllocType atype,
                                  VarType vtype, const void *ptr,
                                  size_t size);

/// Duplicate a variable
extern uint32_t jitc_var_copy(uint32_t index);

/// Create a resized copy of a variable
extern uint32_t jitc_var_resize(uint32_t index, size_t size);

/// Increase the external reference count of a given variable
extern void jitc_var_inc_ref(uint32_t index, Variable *v) noexcept;

/// Increase the external reference count of a given variable
extern void jitc_var_inc_ref(uint32_t index) noexcept;

/// Increase the external reference count of a given variable
inline uint32_t jitc_var_new_ref(uint32_t index) noexcept {
    jitc_var_inc_ref(index);
    return index;
}

/// Decrease the external reference count of a given variable
extern void jitc_var_dec_ref(uint32_t index, Variable *v) noexcept;

/// Decrease the external reference count of a given variable
extern void jitc_var_dec_ref(uint32_t index) noexcept;

/// Increase the side effect reference count of a given variable
extern void jitc_var_inc_ref_se(uint32_t index, Variable *v) noexcept;

/// Increase the side effect reference count of a given variable
extern void jitc_var_inc_ref_se(uint32_t index) noexcept;

/// Decrease the side effect reference count of a given variable
extern void jitc_var_dec_ref_se(uint32_t index, Variable *v) noexcept;

/// Decrease the side effect reference count of a given variable
extern void jitc_var_dec_ref_se(uint32_t index) noexcept;

// Query the type of a given variable
extern VarType jitc_var_type(uint32_t index);

/// Assign a descriptive label to a variable with only 1 reference
extern void jitc_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern const char *jitc_var_label(uint32_t index);

/// Assign a callback function that is invoked when the given variable is freed
extern void jitc_var_set_callback(uint32_t index,
                                  void (*callback)(uint32_t, int, void *),
                                  void *data,
                                  bool is_internal);

/// Migrate a variable to a different flavor of memory
extern uint32_t jitc_var_migrate(uint32_t index, AllocType type);

/// Indicate to the JIT compiler that a variable has side effects
extern void jitc_var_mark_side_effect(uint32_t index);

/// Return a human-readable summary of the contents of a variable
const char *jitc_var_str(uint32_t index);

/// Read a single element of a variable and write it to 'dst'
extern void jitc_var_read(uint32_t index, size_t offset, void *dst);

/// Reverse of jitc_var_read(). Copy 'src' to a single element of a variable
extern uint32_t jitc_var_write(uint32_t index, size_t offset, const void *src);

/// Schedule a variable \c index for future evaluation via \ref jit_eval()
extern int jitc_var_schedule(uint32_t index);

/// More aggressive version of the above function
extern uint32_t jitc_var_schedule_force(uint32_t index, int *rv);

/// Evaluate a value constant variable
extern void jitc_var_eval_literal(uint32_t index, Variable *v);

/// Evaluate an uninitialized variable
extern void jitc_var_eval_undefined(uint32_t index, Variable *v);

/// Evaluate the variable \c index right away, if it is unevaluated/dirty.
extern int jitc_var_eval(uint32_t index);

/// Return the pointer location of the variable, evaluate if needed
extern uint32_t jitc_var_data(uint32_t index, bool eval_dirty, void **ptr_out);

/// Return a human-readable summary of registered variables
extern const char *jitc_var_whos();

/// Return a GraphViz representation of registered variables
extern const char *jitc_var_graphviz();

/// Remove a variable from the cache used for common subexpression elimination
extern void jitc_lvn_drop(uint32_t index, const Variable *v);

/// Register a variable with cache used for common subexpression elimination
extern void jitc_lvn_put(uint32_t index, const Variable *v);

/// Append the given variable to the instruction trace and return its ID
extern uint32_t jitc_var_new(Variable &v, bool disable_lvn = false);

/// Query the current (or future, if not yet evaluated) allocation flavor of a variable
extern AllocType jitc_var_alloc_type(uint32_t index);

/// Query the device (or future, if not yet evaluated) associated with a variable
extern int jitc_var_device(uint32_t index);

/// Return a mask of currently active lanes
extern uint32_t jitc_var_mask_peek(JitBackend backend);

/// Push an active mask
extern void jitc_var_mask_push(JitBackend backend, uint32_t index);

/// Pop an active mask
extern void jitc_var_mask_pop(JitBackend backend);

/// Combine the given mask 'index' with the mask stack. 'size' indicates the wavefront size
extern uint32_t jitc_var_mask_apply(uint32_t index, uint32_t size);

/// Return the default mask
extern uint32_t jitc_var_mask_default(JitBackend backend, size_t size);

/// Start a new scope of the program being recorded
extern uint32_t jitc_new_scope(JitBackend backend);

/// Reduce (And) a boolean array to a single value, synchronizes.
extern bool jitc_var_all(uint32_t index);

/// Reduce (Or) a boolean array to a single value, synchronizes.
extern bool jitc_var_any(uint32_t index);

/// Reduce a variable to a single value
extern uint32_t jitc_var_reduce(JitBackend backend, VarType vt,
                                ReduceOp reduce_op, uint32_t index);

/// Dot product reduction of two variables
extern uint32_t jitc_var_reduce_dot(uint32_t index_1,
                                    uint32_t index_2);

/// Reduce a variable over blocks
uint32_t jitc_var_block_reduce(ReduceOp op, uint32_t index, uint32_t block_size, int symbolic);

/// Replicate entries of a a variable into blocks
uint32_t jitc_var_tile(uint32_t index, uint32_t count);

/// Compute an inclusive or exclusive prefix sum of a given variable
extern uint32_t jitc_var_prefix_sum(uint32_t index, bool exclusive);

/// Create a variable containing the buffer storing a specific attribute
extern uint32_t jitc_var_registry_attr(JitBackend backend, VarType type,
                                       const char *domain, const char *name);

/// Return an implicit mask for operations within a virtual function call
extern uint32_t jitc_var_call_mask(JitBackend);

/// Register the current Python source code location with Dr.Jit
extern void jitc_set_source_location(const char *fname, size_t lineno) noexcept;

/// Set the 'self' variable, which plays a special role when tracing method calls
extern void jitc_var_set_self(JitBackend backend, uint32_t value, uint32_t index);

/// Return the 'self' variable, which plays a special role when tracing method calls
extern void jitc_var_self(JitBackend backend, uint32_t *value, uint32_t *index);

/// Return the 'VariableExtra' record associated with a variable (or create it)
extern VariableExtra *jitc_var_extra(Variable *v);

/// Create a view of an existing variable that has a smaller size
extern uint32_t jitc_var_shrink(uint32_t index, size_t size);

/// Compress a sparse boolean array into an index array of the active indices
extern uint32_t jitc_var_compress(uint32_t index);

/// Temporarily stash the reference count of a variable used to make
/// copy-on-write (COW) decisions in jit_var_scatter. Returns a handle for
/// \ref jitc_var_unstash_ref().
extern uint64_t jitc_var_stash_ref(uint32_t index);

/// Undo the change performed by \ref jitc_var_stash_ref()
extern void jitc_var_unstash_ref(uint64_t handle);

/// Return the identity element for different horizontal reductions
extern uint64_t jitc_reduce_identity(ReduceOp reduce_op, VarType vt);

/// LLVM: expand a variable to a larger storage area to avoid atomic scatters
extern std::pair<uint32_t, uint32_t> jitc_var_expand(uint32_t index, ReduceOp reduce_op);

/// Undo the above
extern void jitc_var_reduce_expanded(uint32_t index);

/// Generate an informative error message about variables that remain dirty following evaluation
extern void jitc_raise_dirty_error(uint32_t index);

/// Identify different types of bounds checks (used to choose a suitable error message)
enum class BoundsCheckType {
    Scatter,
    ScatterReduce,
    ScatterAddKahan,
    ScatterInc,
    Gather,
    Call
};

/// In debug mode: insert a bounds check, e.g. before gathering/scattering
extern uint32_t jitc_var_check_bounds(BoundsCheckType bct, uint32_t index,
                                      uint32_t mask, uint32_t size);

/// Descriptive names and byte sizes for the various variable types
extern const char *type_name      [(int) VarType::Count];
extern const char *type_name_short[(int) VarType::Count];
extern const uint32_t type_size   [(int) VarType::Count];

/// Type names and register names for CUDA and LLVM
extern const char *type_name_llvm       [(int) VarType::Count];
extern const char *type_name_llvm_bin   [(int) VarType::Count];
extern const char *type_name_llvm_abbrev[(int) VarType::Count];
extern const char *type_name_llvm_big   [(int) VarType::Count];
extern const char *type_name_ptx        [(int) VarType::Count];
extern const char *type_name_ptx_bin    [(int) VarType::Count];
extern const char *type_prefix          [(int) VarType::Count];
extern const char *type_size_str        [(int) VarType::Count];
