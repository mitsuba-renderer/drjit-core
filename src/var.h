/*
    src/var.h -- Variable data structure, operations to create variables

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "core.h"
#include "hash.h"
#include "alloc.h"

enum VarKind : uint32_t {
    // Invalid node (default)
    Invalid,

    // An evaluated node representing data
    Data,

    // Legacy string-based IR statement
    Stmt,

    // A literal constant
    // (note: this must be the last enumeration entry before the regular nodes start)
    Literal,

    /// A no-op (generates no code)
    Nop,

    // Common unary operations
    Neg, Not, Sqrt, Abs,

    // Common binary arithmetic operations
    Add, Sub, Mul, Div, Mod,

    // High multiplication
    Mulhi,

    // Fused multiply-add
    Fma,

    // Minimum, maximum
    Min, Max,

    // Rounding operations
    Ceil, Floor, Round, Trunc,

    // Comparisons
    Eq, Neq, Lt, Le, Gt, Ge,

    // Ternary operator
    Select,

    // Bit-level counting operations
    Popc, Clz, Ctz,

    /// Bit-wise operations
    And, Or, Xor,

    // Shifts
    Shl, Shr,

    // Fast approximations
    Rcp, Rsqrt,

    // Multi-function generator (CUDA)
    Sin, Cos, Exp2, Log2,

    // Casts
    Cast, Bitcast,

    // Memory-related operations
    Gather, Scatter,

    // Specialized nodes for vcalls
    VCallMask, VCallSelf,

    // Counter node to determine the current lane ID
    Counter,

    // Recorded 'printf' instruction for debugging purposes
    Printf,

    // A polymorphic function call
    Dispatch,

    // Perform a standard texture lookup (CUDA)
    TexLookup,

    // Load all texels used for bilinear interpolation (CUDA)
    TexFetchBilerp,

    // Perform a ray tracing call
    TraceRay,

    // Extract a component from an operation that produced multiple results
    Extract,

    // Denotes the number of different node types
    Count
};

#pragma pack(push, 1)

/// Central variable data structure, which represents an assignment in SSA form
struct Variable {
    #if defined(__GNUC__)
    #  pragma GCC diagnostic push
    #  if defined(__has_warning)
    #    if __has_warning("-Wclass-memaccess")
    #      pragma GCC diagnostic ignored "-Wclass-memaccess"
    #    endif
    #  else
    #    pragma GCC diagnostic ignored "-Wclass-memaccess"
    #  endif
    #elif defined(_MSC_VER)
    #  pragma warning (disable:4201) // nonstandard extension used: nameless struct/union
    #endif

    /// Zero-initialize by default
    Variable() { memset(this, 0, sizeof(Variable)); }

    #if defined(__GNUC__)
    #  pragma GCC diagnostic pop
    #endif

    // =================  Reference count, dependencies, scope ================

    /// Number of times that this variable is referenced elsewhere
    uint32_t ref_count;

    /// Identifier of the basic block containing this variable
    uint32_t scope;

    /**
     * \brief Up to 4 dependencies of this instruction
     *
     * Certain complex operations (e.g. a ray tracing call) may reference
     * further operands via an entry in the 'State::extra' map. They must set
     * the 'Variable::extra' bit to 1 to indicate the presence of such
     * supplemental information.
     */
    uint32_t dep[4];

    // ======  Size & encoded instruction (IR statement, literal, data) =======

    /// The 'kind' field determines which entry of the following union is used
    union {
        // Floating point/integer value, reinterpreted as u64
        uint64_t literal;

        /// Pointer to device memory. Used when kind == VarKind::Data
        void *data;

        // Legacy string-based IR (to be removed)
        char *stmt;
    };

    /// Number of entries
    uint32_t size;

    /// Unused, to be eventually used to upgrade to 64 bit array sizes
    uint32_t unused;

    // ================  Essential flags used in the LVN key  =================

    // Variable kind (IR statement / literal constant / data)
    uint32_t kind : 8;

    /// Backend associated with this variable
    uint32_t backend : 2;

    /// Variable type (Bool/Int/Float/....)
    uint32_t type : 4;

    /// Is this a pointer variable that is used to write to some array?
    uint32_t write_ptr : 1;

    /// Free the 'stmt' variables at destruction time?
    uint32_t free_stmt : 1;

    // =======================  Miscellaneous fields =========================

    /// If set, 'data' will not be deallocated when the variable is destructed
    uint32_t retain_data : 1;

    /// Is this a placeholder variable used to record arithmetic symbolically?
    uint32_t placeholder : 1;

    /// Is this a placeholder variable used to record arithmetic symbolically?
    uint32_t vcall_iface : 1;

    /// Must be set if the variable is associated with an 'Extra' instance
    uint32_t extra : 1;

    /// Must be set if 'data' is not properly aligned in memory
    uint32_t unaligned : 1;

    /// Does this variable perform an OptiX operation?
    uint32_t optix : 1;

    /// If set, evaluation will have side effects on other variables
    uint32_t side_effect : 1;

    // =========== Entries that are temporarily used in jitc_eval() ============

    /// Argument type
    uint32_t param_type : 2;

    /// Is this variable marked as an output?
    uint32_t output_flag : 1;

    /// Unused for now
    uint32_t unused_2 : 6;

    /// Offset of the argument in the list of kernel parameters
    uint32_t param_offset;

    /// Register index
    uint32_t reg_index;

    // ========================  Side effect tracking  =========================

    /// Number of queued side effects
    uint32_t ref_count_se;

    // =========================   Helper functions   ==========================

    bool is_data()    const { return kind == (uint32_t) VarKind::Data;    }
    bool is_literal() const { return kind == (uint32_t) VarKind::Literal; }
    bool is_stmt()    const { return kind == (uint32_t) VarKind::Stmt;    }
    bool is_node()    const { return (uint32_t) kind > VarKind::Literal; }
    bool is_dirty()   const { return ref_count_se > 0; }
};

/// Abbreviated version of the Variable data structure
struct VariableKey {
    uint32_t size;
    uint32_t scope;
    uint32_t dep[4];
    uint32_t kind      : 8;
    uint32_t backend   : 2;
    uint32_t type      : 4;
    uint32_t write_ptr : 1;
    uint32_t free_stmt : 1;
    uint32_t unused    : 16;

    union {
        uint64_t literal;
        char *stmt;
    };

    VariableKey(const Variable &v) {
        size = v.size;
        scope = v.scope;
        for (int i = 0; i < 4; ++i)
            dep[i] = v.dep[i];
        if (v.is_stmt())
            stmt = v.stmt;
        else
            literal = v.literal;

        kind = v.kind;
        backend = v.backend;
        type = v.type;
        write_ptr = v.write_ptr;
        free_stmt = v.free_stmt;
        unused = 0;
    }

    bool operator==(const VariableKey &v) const {
        if (memcmp(this, &v, 7 * sizeof(uint32_t)) != 0)
            return false;
        if ((VarKind) kind != VarKind::Stmt)
            return literal == v.literal;
        else if (!free_stmt)
            return stmt == v.stmt;
        else
            return stmt == v.stmt || strcmp(stmt, v.stmt) == 0;
    }
};

#pragma pack(pop)

enum ParamType { Register, Input, Output };

/// Helper class to hash VariableKey instances
struct VariableKeyHasher {
    size_t operator()(const VariableKey &k) const {
        uint64_t hash_1;
        if ((VarKind) k.kind != VarKind::Stmt)
            hash_1 = k.literal;
        else if (!k.free_stmt)
            hash_1 = (uintptr_t) k.stmt;
        else
            hash_1 = hash_str(k.stmt);

        uint32_t buf[7];
        size_t size = 7 * sizeof(uint32_t);
        memcpy(buf, &k, size);
        return hash(buf, size, hash_1);
    }
};

/// Cache data structure for local value numbering
using LVNMap =
    tsl::robin_map<VariableKey, uint32_t, VariableKeyHasher,
                   std::equal_to<VariableKey>,
                   std::allocator<std::pair<VariableKey, uint32_t>>,
                   /* StoreHash = */ true>;


/// Maps from variable ID to a Variable instance
using VariableMap =
    tsl::robin_map<uint32_t, Variable, UInt32Hasher,
                   std::equal_to<uint32_t>,
                   aligned_allocator<std::pair<uint32_t, Variable>, 64>,
                   /* StoreHash = */ false>;

struct Extra {
    /// Optional descriptive label
    char *label = nullptr;

    /// Additional references
    uint32_t *dep = nullptr;
    uint32_t n_dep = 0;

    /// Callback to be invoked when the variable is evaluated/deallocated
    void (*callback)(uint32_t, int, void *) = nullptr;
    void *callback_data = nullptr;
    bool callback_internal = false;

    /// Bucket decomposition for virtual function calls
    uint32_t vcall_bucket_count = 0;
    VCallBucket *vcall_buckets = nullptr;

    /// Code generation callback
    void (*assemble)(const Variable *v, const Extra &extra) = nullptr;
};

using ExtraMap = tsl::robin_map<uint32_t, Extra, UInt32Hasher>;

// ===========================================================================
// Helper functions to classify different variable types
// ===========================================================================

inline bool jitc_is_arithmetic(VarType type) {
    return type != VarType::Void && type != VarType::Bool;
}

inline bool jitc_is_float(VarType type) {
    return type == VarType::Float16 ||
           type == VarType::Float32 ||
           type == VarType::Float64;
}

inline bool jitc_is_single(VarType type) { return type == VarType::Float32; }
inline bool jitc_is_double(VarType type) { return type == VarType::Float64; }
inline bool jitc_is_bool(VarType type) { return type == VarType::Bool; }

inline bool jitc_is_sint(VarType type) {
    return type == VarType::Int8 ||
           type == VarType::Int16 ||
           type == VarType::Int32 ||
           type == VarType::Int64;
}

inline bool jitc_is_uint(VarType type) {
    return type == VarType::UInt8 ||
           type == VarType::UInt16 ||
           type == VarType::UInt32 ||
           type == VarType::UInt64;
}

inline bool jitc_is_int(VarType type) {
    return jitc_is_sint(type) || jitc_is_uint(type);
}

inline bool jitc_is_void(VarType type) {
    return type == VarType::Void;
}

inline bool jitc_is_arithmetic(const Variable *v) { return jitc_is_arithmetic((VarType) v->type); }
inline bool jitc_is_float(const Variable *v) { return jitc_is_float((VarType) v->type); }
inline bool jitc_is_single(const Variable *v) { return jitc_is_single((VarType) v->type); }
inline bool jitc_is_double(const Variable *v) { return jitc_is_double((VarType) v->type); }
inline bool jitc_is_sint(const Variable *v) { return jitc_is_sint((VarType) v->type); }
inline bool jitc_is_uint(const Variable *v) { return jitc_is_uint((VarType) v->type); }
inline bool jitc_is_int(const Variable *v) { return jitc_is_int((VarType) v->type); }
inline bool jitc_is_void(const Variable *v) { return jitc_is_void((VarType) v->type); }
inline bool jitc_is_bool(const Variable *v) { return jitc_is_bool((VarType) v->type); }
inline bool jitc_is_zero(Variable *v) { return v->is_literal() && v->literal == 0; }

inline bool jitc_is_one(Variable *v) {
    if (!v->is_literal())
        return false;

    uint64_t one;
    switch ((VarType) v->type) {
        case VarType::Float16: one = 0x3c00ull; break;
        case VarType::Float32: one = 0x3f800000ull; break;
        case VarType::Float64: one = 0x3ff0000000000000ull; break;
        default: one = 1; break;
    }

    return v->literal == one;
}

/// Look up a variable by its ID
extern Variable *jitc_var(uint32_t index);

/// Create a value constant variable of the given size
extern uint32_t jitc_var_literal(JitBackend backend, VarType type,
                                 const void *value, size_t size,
                                 int eval, int is_class = 0);

/// Create a variable counting from 0 ... size - 1
extern uint32_t jitc_var_counter(JitBackend backend, size_t size,
                                 bool simplify_scalar);

/// Create a variable representing the result of a custom IR statement
extern uint32_t jitc_var_stmt(JitBackend backend,
                              VarType type,
                              const char *stmt,
                              int stmt_static,
                              uint32_t n_dep,
                              const uint32_t *dep);

/// Create a new IR node. Just a wrapper around jitc_var_new without any error checking
extern uint32_t jitc_var_new_node_0(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool placeholder,
                                    uint64_t payload = 0);

extern uint32_t jitc_var_new_node_1(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool placeholder,
                                    uint32_t a0, Variable *v0,
                                    uint64_t payload = 0);

extern uint32_t jitc_var_new_node_2(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool placeholder,
                                    uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                                    uint64_t payload = 0);

extern uint32_t jitc_var_new_node_3(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool placeholder,
                                    uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                                    uint32_t a2, Variable *v2, uint64_t payload = 0);

extern uint32_t jitc_var_new_node_4(JitBackend backend, VarKind kind,
                                    VarType vt, uint32_t size, bool placeholder,
                                    uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                                    uint32_t a2, Variable *v2, uint32_t a3, Variable *v4,
                                    uint64_t payload = 0);

/// Create a variable that refers to a memory region
extern uint32_t jitc_var_pointer(JitBackend backend, const void *value,
                                 uint32_t dep, int write);

/// Wrap an input variable of a virtual function call before recording computation
extern uint32_t jitc_var_wrap_vcall(uint32_t index);

/// Register an existing variable with the JIT compiler
extern uint32_t jitc_var_mem_map(JitBackend backend, VarType type, void *ptr,
                                 size_t size, int free);

/// Copy a memory region onto the device and return its variable index
extern uint32_t jitc_var_mem_copy(JitBackend dst, JitBackend src,
                                  VarType vtype, const void *ptr,
                                  size_t size);

/// Duplicate a variable
extern uint32_t jitc_var_copy(uint32_t index);

/// Create a resized copy of a variable
extern uint32_t jitc_var_resize(uint32_t index, size_t size);

/// Increase the external reference count of a given variable
extern void jitc_var_inc_ref(uint32_t index, Variable *v) noexcept(true);

/// Increase the external reference count of a given variable
extern void jitc_var_inc_ref(uint32_t index) noexcept(true);

/// Increase the external reference count of a given variable
inline uint32_t jitc_var_new_ref(uint32_t index) noexcept(true) {
    jitc_var_inc_ref(index);
    return index;
}

/// Decrease the external reference count of a given variable
extern void jitc_var_dec_ref(uint32_t index, Variable *v) noexcept(true);

/// Decrease the external reference count of a given variable
extern void jitc_var_dec_ref(uint32_t index) noexcept(true);

/// Increase the side effect reference count of a given variable
extern void jitc_var_inc_ref_se(uint32_t index, Variable *v) noexcept(true);

/// Increase the side effect reference count of a given variable
extern void jitc_var_inc_ref_se(uint32_t index) noexcept(true);

/// Decrease the side effect reference count of a given variable
extern void jitc_var_dec_ref_se(uint32_t index, Variable *v) noexcept(true);

/// Decrease the side effect reference count of a given variable
extern void jitc_var_dec_ref_se(uint32_t index) noexcept(true);

// Query the type of a given variable
extern VarType jitc_var_type(uint32_t index);

/// Assign a descriptive label to a variable with only 1 reference
extern void jitc_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern const char *jitc_var_label(uint32_t index);

/// Assign a callback function that is invoked when the given variable is freed
extern void jitc_var_set_callback(uint32_t index,
                                  void (*callback)(uint32_t, int, void *),
                                  void *payload);

/// Migrate a variable to a different flavor of memory
// extern uint32_t jitc_var_migrate(uint32_t index, AllocType type);

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

/// Evaluate a value constant variable
extern void jitc_var_eval_literal(uint32_t index, Variable *v);

/// Evaluate the variable \c index right away, if it is unevaluated/dirty.
extern int jitc_var_eval(uint32_t index);

/// Return the pointer location of the variable, evaluate if needed
extern void *jitc_var_ptr(uint32_t index);

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
extern uint32_t jitc_var_mask_default(JitBackend backend, uint32_t size);

/// Start a new scope of the program being recorded
extern void jitc_new_scope(JitBackend backend);

/// Reduce (And) a boolean array to a single value, synchronizes.
extern bool jitc_var_all(uint32_t index);

/// Reduce (Or) a boolean array to a single value, synchronizes.
extern bool jitc_var_any(uint32_t index);

/// Reduce a variable to a single value
extern uint32_t jitc_var_reduce(uint32_t index, ReduceOp reduce_op);

/// Create a variable containing the buffer storing a specific attribute
extern uint32_t jitc_var_registry_attr(JitBackend backend, VarType type,
                                       const char *domain, const char *name);

/// Return an implicit mask for operations within a virtual function call
extern uint32_t jitc_var_vcall_mask(JitBackend);

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
extern const char *var_kind_name        [(int) VarKind::Count];

