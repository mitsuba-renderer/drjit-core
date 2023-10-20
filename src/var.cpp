/*
    src/var.cpp -- Operations for creating and querying variables

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "var.h"
#include "internal.h"
#include "log.h"
#include "eval.h"
#include "util.h"
#include "op.h"
#include "registry.h"

// When debugging via valgrind, this will make iterator invalidation more obvious
// #define DRJIT_VALGRIND 1

/// Descriptive names for the various variable types
const char *type_name[(int) VarType::Count] {
    "void",   "bool",  "int8",   "uint8",   "int16",   "uint16",  "int32",
    "uint32", "int64", "uint64", "pointer", "float16", "float32", "float64"
};

/// Descriptive names for the various variable types (extra-short version)
const char *type_name_short[(int) VarType::Count] {
    "void ", "msk", "i8",  "u8",  "i16", "u16", "i32",
    "u32", "i64", "u64", "ptr", "f16", "f32", "f64"
};

/// CUDA PTX type names
const char *type_name_ptx[(int) VarType::Count] {
    "???", "pred", "s8",  "u8",  "s16", "u16", "s32",
    "u32", "s64",  "u64", "u64", "f16", "f32", "f64"
};

/// CUDA PTX type names (binary view)
const char *type_name_ptx_bin[(int) VarType::Count] {
    "???", "pred", "b8",  "b8",  "b16", "b16", "b32",
    "b32", "b64",  "b64", "b64", "b16", "b32", "b64"
};

/// LLVM IR type names (does not distinguish signed vs unsigned)
const char *type_name_llvm[(int) VarType::Count] {
    "???", "i1",  "i8",  "i8",   "i16",   "i16",   "i32",
    "i32", "i64", "i64", "i64", "half", "float", "double"
};

/// Double size integer arrays for mulhi()
const char *type_name_llvm_big[(int) VarType::Count] {
    "???", "???",  "i16",  "i16", "i32", "i32", "i64",
    "i64", "i128", "i128", "???", "???", "???", "???"
};

/// Abbreviated LLVM IR type names
const char *type_name_llvm_abbrev[(int) VarType::Count] {
    "???", "i1",  "i8",  "i8",  "i16", "i16", "i32",
    "i32", "i64", "i64", "i64", "f16", "f32", "f64"
};

/// LLVM IR type names (binary view)
const char *type_name_llvm_bin[(int) VarType::Count] {
    "???", "i1",  "i8",  "i8",  "i16", "i16", "i32",
    "i32", "i64", "i64", "i64", "i16", "i32", "i64"
};

/// LLVM/CUDA register name prefixes
const char *type_prefix[(int) VarType::Count] {
    "%u", "%p", "%b", "%b", "%w", "%w", "%r",
    "%r", "%rd", "%rd", "%rd", "%h", "%f", "%d"
};

/// Maps types to byte sizes
const uint32_t type_size[(int) VarType::Count] {
    0, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 2, 4, 8
};

/// String version of the above
const char *type_size_str[(int) VarType::Count] {
    "0", "1", "1", "1", "2", "2", "4",
    "4", "8", "8", "8", "2", "4", "8"
};

///
const char *var_kind_name[(int) VarKind::Count] {
    "invalid",

    // An evaluated node representing data
    "data",

    // Legacy string-based IR statement
    "stmt",

    // A literal constant
    "literal",

    // A no-op (generates no code)
    "nop",

    // Common unary operations
    "neg", "not", "sqrt", "abs",

    // Common binary arithmetic operations
    "add", "sub", "mul", "div", "mod",

    // High multiplication
    "mulhi",

    // Fused multiply-add
    "fma",

    // Minimum, maximum
    "min", "max",

    // Rounding operations
    "ceil", "floor", "round", "trunc",

    // Comparisons
    "eq", "neq", "lt", "le", "gt", "ge",

    // Ternary operator
    "select",

    // Bit-level counting operations
    "popc", "clz", "ctz",

    /// Bit-wise operations
    "and", "or", "xor",

    // Shifts
    "shl", "shr",

    // Fast approximations
    "rcp", "rsqrt",

    // Multi-function generator (CUDA)
    "sin", "cos", "exp2", "log2",

    // Casts
    "cast", "bitcast",

    // Memory-related operations
    "gather", "scatter", "scatter_inc", "scatter_kahan",

    // Specialized nodes for vcalls
    "vcall_mask", "self",

    // Counter node to determine the current lane ID
    "counter",

    // Default mask used to ignore out-of-range SIMD lanes (LLVM)
    "default_mask",

    // Recorded 'printf' instruction for debugging purposes
    "printf",

    // A polymorphic function call
    "dispatch",

    // Perform a standard texture lookup (CUDA)
    "tex_lookup",

    // Load all texels used for bilinear interpolation (CUDA)
    "tex_fetch_bilerp",

    // Perform a ray tracing call
    "trace_ray",

    // Extract a component from an operation that produced multiple results
    "extract"
};


/// Temporary string buffer for miscellaneous variable-related tasks
StringBuffer var_buffer(0);

#define jitc_check_size(name, size)                                            \
    if (unlikely(size > 0xFFFFFFFF))                                           \
        jitc_raise("%s(): tried to create an array with %zu entries, which "   \
                   "exceeds the limit of 2^32 == 4294967296 entries.",         \
                   name, size);

/// Cleanup handler, called when the internal/external reference count reaches zero
void jitc_var_free(uint32_t index, Variable *v) {
    jitc_trace("jit_var_free(r%u)", index);

    if (v->is_data()) {
        // Release memory referenced by this variable
        if (!v->retain_data)
            jitc_free(v->data);
    } else {
        // Unevaluated variable, drop from CSE cache
        jitc_lvn_drop(index, v);
    }

    // Free IR string if needed
    if (unlikely(v->free_stmt))
        free(v->stmt);

    uint32_t dep[4];
    bool write_ptr = v->write_ptr;
    memcpy(dep, v->dep, sizeof(uint32_t) * 4);

    if (unlikely(v->extra)) {
        auto it = state.extra.find(index);
        if (it == state.extra.end())
            jitc_fail("jit_var_free(): entry in 'extra' hash table not found!");
        Extra extra = it.value();
        state.extra.erase(it);

        /* Notify callback that the variable was freed.
           Do this first, before freeing any dependencies */
        if (extra.callback) {
            if (extra.callback_internal) {
                extra.callback(index, 1, extra.callback_data);
            } else {
                unlock_guard guard(state.lock);
                extra.callback(index, 1, extra.callback_data);
            }
        }

        // Decrease reference counts of extra references if needed
        if (extra.dep) {
            for (uint32_t i = 0; i < extra.n_dep; ++i)
                jitc_var_dec_ref(extra.dep[i]);
            free(extra.dep);
        }

        // If jitc_vcall() was invoked on this variable, free bucket list
        if (extra.vcall_bucket_count) {
            for (uint32_t i = 0; i < extra.vcall_bucket_count; ++i)
                jitc_var_dec_ref(extra.vcall_buckets[i].index);
            jitc_free(extra.vcall_buckets);
        }

        // Free descriptive label
        free(extra.label);
    }

    // Remove from hash table
    state.variables.erase(index);

#if defined(DRJIT_VALGRIND)
    VariableMap var_new(state.variables);
    state.variables.swap(var_new);
#endif

    if (likely(!write_ptr)) {
        // Decrease reference count of dependencies
        for (int i = 0; i < 4; ++i)
            jitc_var_dec_ref(dep[i]);
    } else {
        jitc_var_dec_ref_se(dep[3]);
    }
}

/// Access a variable by ID, terminate with an error if it doesn't exist
Variable *jitc_var(uint32_t index) {
    auto it = state.variables.find(index);
    if (unlikely(it == state.variables.end()))
        jitc_fail("jit_var(r%u): unknown variable!", index);
    return &it.value();
}

/// Increase the external reference count of a given variable
void jitc_var_inc_ref(uint32_t index, Variable *v) noexcept(true) {
    (void) index; // jitc_trace may be disabled
    v->ref_count++;
    jitc_trace("jit_var_inc_ref(r%u): %u", index, (uint32_t) v->ref_count);
}

/// Increase the external reference count of a given variable
void jitc_var_inc_ref(uint32_t index) noexcept(true) {
    if (index)
        jitc_var_inc_ref(index, jitc_var(index));
}

/// Increase the side effect reference count of a given variable
void jitc_var_inc_ref_se(uint32_t index, Variable *v) noexcept(true) {
    (void) index; // jitc_trace may be disabled
    v->ref_count_se++;
    jitc_trace("jit_var_inc_ref_se(r%u): %u", index, (uint32_t) v->ref_count_se);
}

/// Increase the side effect reference count of a given variable
void jitc_var_inc_ref_se(uint32_t index) noexcept(true) {
    if (index)
        jitc_var_inc_ref_se(index, jitc_var(index));
}

/// Decrease the external reference count of a given variable
void jitc_var_dec_ref(uint32_t index, Variable *v) noexcept(true) {
    if (unlikely(v->ref_count == 0))
        jitc_fail("jit_var_dec_ref(): variable r%u has no external references!", index);

    jitc_trace("jit_var_dec_ref(r%u): %u", index, (uint32_t) v->ref_count - 1);
    v->ref_count--;

    if (v->ref_count == 0 && v->ref_count_se == 0)
        jitc_var_free(index, v);
}

/// Decrease the external reference count of a given variable
void jitc_var_dec_ref(uint32_t index) noexcept(true) {
    if (index != 0)
        jitc_var_dec_ref(index, jitc_var(index));
}

/// Decrease the side effect reference count of a given variable
void jitc_var_dec_ref_se(uint32_t index, Variable *v) noexcept(true) {
    if (unlikely(v->ref_count_se == 0))
        jitc_fail("jit_var_dec_ref_se(): variable r%u has no side effect references!", index);

    jitc_trace("jit_var_dec_ref_se(r%u): %u", index, (uint32_t) v->ref_count_se - 1);
    v->ref_count_se--;

    if (v->ref_count == 0 && v->ref_count_se == 0)
        jitc_var_free(index, v);
}

/// Decrease the side effect reference count of a given variable
void jitc_var_dec_ref_se(uint32_t index) noexcept(true) {
    if (index != 0)
        jitc_var_dec_ref_se(index, jitc_var(index));
}

/// Remove a variable from the cache used for common subexpression elimination
void jitc_lvn_drop(uint32_t index, const Variable *v) {
    VariableKey key(*v);
    LVNMap &cache = state.lvn_map;
    auto it = cache.find(key);
    if (it != cache.end() && it.value() == index)
        cache.erase(it);
}

/// Register a variable with cache used for common subexpression elimination
void jitc_lvn_put(uint32_t index, const Variable *v) {
    state.lvn_map.try_emplace(VariableKey(*v), index);
}

/// Query the type of a given variable
VarType jitc_var_type(uint32_t index) {
    return (VarType) jitc_var(index)->type;
}

/// Query the descriptive label associated with a given variable
const char *jitc_var_label(uint32_t index) {
    ExtraMap::iterator it = state.extra.find(index);
    if (it == state.extra.end()) {
        return nullptr;
    } else {
        const char *label = it.value().label;
        if (label) {
            const char *delim = strrchr(label, '/');
            return delim ? (delim + 1) : label;
        } else {
            return nullptr;
        }
    }
}

void jitc_var_set_label(uint32_t index, const char *label) {
    if (unlikely(index == 0))
        return;

    Variable *v = jitc_var(index);
    size_t len = label ? strlen(label) : 0;

    for (size_t i = 0; i < len; ++i) {
        if (label[i] == '\n' || label[i] == '/')
            jitc_raise("jit_var_set_label(): invalid string (may not "
                       "contain newline or '/' characters)");
    }

    v->extra = true;
    Extra &extra = state.extra[index];
    free(extra.label);

    ThreadState *ts = thread_state(v->backend);
    if (!ts->prefix) {
        if (!label) {
            extra.label = nullptr;
        } else {
            extra.label = (char *) malloc_check(len + 1);
            memcpy(extra.label, label, len + 1);
        }
    } else {
        size_t prefix_len = strlen(ts->prefix);
        char *combined = (char *) malloc(prefix_len + len + 1);
        memcpy(combined, ts->prefix, prefix_len);
        if (len)
            memcpy(combined + prefix_len, label, len);
        combined[prefix_len + len] = '\0';
        extra.label = combined;
    }

    jitc_log(Debug, "jit_var_set_label(r%u): \"%s\"", index,
             label ? label : "(null)");
}

// Print a value variable to 'var_buffer' (for debug/GraphViz output)
void jitc_value_print(const Variable *v, bool graphviz = false) {
    #define JIT_LITERAL_PRINT(type, ptype, fmtstr)  {  \
            type value;                                \
            memcpy(&value, &v->literal, sizeof(type)); \
            var_buffer.fmt(fmtstr, (ptype) value);     \
        }                                              \
        break;

    switch ((VarType) v->type) {
        case VarType::Float32: JIT_LITERAL_PRINT(float, float, "%g");
        case VarType::Float64: JIT_LITERAL_PRINT(double, double, "%g");
        case VarType::Bool:    JIT_LITERAL_PRINT(bool, int, "%i");
        case VarType::Int8:    JIT_LITERAL_PRINT(int8_t, int, "%i");
        case VarType::Int16:   JIT_LITERAL_PRINT(int16_t, int, "%i");
        case VarType::Int32:   JIT_LITERAL_PRINT(int32_t, int, "%i");
        case VarType::Int64:   JIT_LITERAL_PRINT(int64_t, long long int, "%lli");
        case VarType::UInt8:   JIT_LITERAL_PRINT(uint8_t, unsigned, "%u");
        case VarType::UInt16:  JIT_LITERAL_PRINT(uint16_t, unsigned, "%u");
        case VarType::UInt32:  JIT_LITERAL_PRINT(uint32_t, unsigned, "%u");
        case VarType::UInt64:  JIT_LITERAL_PRINT(uint64_t, long long unsigned int, "%llu");
        case VarType::Pointer: JIT_LITERAL_PRINT(uintptr_t, uintptr_t, (graphviz ? ("0x%" PRIxPTR) : (DRJIT_PTR)));
        default:
            jitc_fail("jit_value_print(): unsupported type!");
    }

    #undef JIT_LITERAL_PRINT
}

/// Append the given variable to the instruction trace and return its ID
uint32_t jitc_var_new(Variable &v, bool disable_lvn) {
    ThreadState *ts = thread_state(v.backend);

    bool lvn = !disable_lvn && (VarType) v.type != VarType::Void &&
               !v.is_data() &&
               jit_flag(JitFlag::ValueNumbering);

    v.scope = ts->scope;

    // Check if this exact statement already exists ..
    LVNMap::iterator key_it;
    bool lvn_key_inserted = false;
    if (lvn)
        std::tie(key_it, lvn_key_inserted) =
            state.lvn_map.try_emplace(VariableKey(v), 0);

    uint32_t index;
    Variable *vo;

    if (likely(!lvn || lvn_key_inserted)) {
        #if defined(DRJIT_VALGRIND)
            VariableMap var_new(state.variables);
            state.variables.swap(var_new);
        #endif

        // .. nope, it is new.
        VariableMap::iterator var_it;
        bool var_inserted = false;
        do {
            index = state.variable_index++;

            if (unlikely(index == 0)) // overflow
                continue;

            std::tie(var_it, var_inserted) =
                state.variables.try_emplace(index, v);
        } while (!var_inserted);

        state.variable_watermark = std::max(state.variable_watermark,
                                            (uint32_t) state.variables.size());

        if (lvn_key_inserted)
            key_it.value() = index;

        vo = &var_it.value();

        if (unlikely(ts->prefix)) {
            vo->extra = true;
            state.extra[index].label = strdup(ts->prefix);
        }
    } else {
        // .. found a match! Deallocate 'v'.
        if (v.free_stmt)
            free(v.stmt);

        if (likely(!v.write_ptr)) {
            for (int i = 0; i < 4; ++i)
                jitc_var_dec_ref(v.dep[i]);
        } else {
            jitc_var_dec_ref_se(v.dep[3]);
        }

        index = key_it.value();
        vo = jitc_var(index);
    }

    if (unlikely(std::max(state.log_level_stderr, state.log_level_callback) >=
                 LogLevel::Debug)) {
        var_buffer.clear();
        var_buffer.fmt("jit_var_new(%s r%u", type_name[v.type], index);
        if (v.size > 1)
            var_buffer.fmt("[%u]", v.size);

        uint32_t n_dep = 0;
        for (int i = 0; i < 4; ++i) {
            if (v.dep[i])
                n_dep = i + 1;
        }
        if (n_dep)
            var_buffer.put(" <- ");
        for (uint32_t i = 0; i < n_dep; ++i)
            var_buffer.fmt("r%u%s", v.dep[i], i + 1 < n_dep ? ", " : "");
        var_buffer.put("): ");

        if (v.is_literal()) {
            var_buffer.put("literal = ");
            jitc_value_print(&v);
        } else if (v.is_data()) {
            var_buffer.fmt(DRJIT_PTR, (uintptr_t) v.data);
        } else if (v.is_stmt()) {
            var_buffer.put(v.stmt, strlen(v.stmt));
        } else if (v.is_node()) {
            var_buffer.fmt("\"%s\" operation", var_kind_name[v.kind]);
        }

        bool lvn_hit = lvn && !lvn_key_inserted;
        if (v.placeholder || lvn_hit) {
            var_buffer.put(" (");
            if (v.placeholder)
                var_buffer.put("placeholder");
            if (v.placeholder && lvn_hit)
                var_buffer.put(", ");
            if (lvn_hit)
                var_buffer.put("lvn hit");
            var_buffer.put(")");
        }

        jitc_log(Debug, "%s", var_buffer.get());
    }

    jitc_var_inc_ref(index, vo);

    return index;
}

uint32_t jitc_var_literal(JitBackend backend, VarType type, const void *value,
                          size_t size, int eval, int is_class) {
    if (unlikely(size == 0))
        return 0;

    jitc_check_size("jit_var_literal", size);

    /* When initializing a value pointer array while recording a virtual
       function, we can leverage the already available `self` variable
       instead of creating a new one. */
    if (is_class) {
        ThreadState *ts = thread_state(backend);
        if (ts->vcall_self_value &&
            *((uint32_t *) value) == ts->vcall_self_value) {
            jitc_var_inc_ref(ts->vcall_self_index);
            return ts->vcall_self_index;
        }
    }

    if (!eval) {
        Variable v;
        memcpy(&v.literal, value, type_size[(uint32_t) type]);
        v.kind = (uint32_t) VarKind::Literal;
        v.type = (uint32_t) type;
        v.size = (uint32_t) size;
        v.backend = (uint32_t) backend;

        return jitc_var_new(v);
    } else {
        uint32_t isize = type_size[(int) type];
        void *data =
            jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                    : AllocType::HostAsync,
                        size * (size_t) isize);
        jitc_memset_async(backend, data, (uint32_t) size, isize, value);
        return jitc_var_mem_map(backend, type, data, size, 1);
    }
}

uint32_t jitc_var_pointer(JitBackend backend, const void *value,
                              uint32_t dep, int write) {
    Variable v;
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::Pointer;
    v.size = 1;
    v.literal = (uint64_t) (uintptr_t) value;
    v.backend = (uint32_t) backend;

    /* A value variable (especially a pointer to some memory region) can
       specify an optional dependency to keep that memory region alive. The
       last entry (v.dep[3]) is strategically chosen as jitc_var_traverse()
       will ignore it given that preceding entries (v.dep[0-2]) are all
       zero, keeping the referenced variable from being merged into
       programs that make use of this value. */
    v.dep[3] = dep;

    // Write pointers create a special type of reference to indicate pending writes
    v.write_ptr = write != 0;
    if (write)
        jitc_var_inc_ref_se(dep);
    else
        jitc_var_inc_ref(dep);

    return jitc_var_new(v);
}

uint32_t jitc_var_counter(JitBackend backend, size_t size,
                          bool simplify_scalar) {
    if (size == 1 && simplify_scalar) {
        uint32_t zero = 0;
        return jitc_var_literal(backend, VarType::UInt32, &zero, 1, 0);
    }

    jitc_check_size("jit_var_counter", size);
    Variable v;
    v.kind = VarKind::Counter;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) VarType::UInt32;
    v.size = (uint32_t) size;
    return jitc_var_new(v);
}

uint32_t jitc_var_wrap_vcall(uint32_t index) {
    if (index == 0)
        jitc_raise("jit_var_wrap_vcall(): invoked with an "
                   "uninitialized variable!");

    const Variable *v = jitc_var(index);
    if (v->is_literal() && (jitc_flags() & (uint32_t) JitFlag::VCallOptimize)) {
        Variable v2;
        v2.backend = v->backend;
        v2.kind = (uint32_t) VarKind::Literal;
        v2.literal = v->literal;
        v2.type = v->type;
        v2.size = 1;
        return jitc_var_new(v2);
    }

    Variable v2;
    v2.stmt = (char *) (((JitBackend) v->backend == JitBackend::CUDA)
                            ? "mov.$t0 $r0, $r1"
                            : "$r0 = bitcast <$w x $t0> $r1 to <$w x $t0>");
    v2.backend = v->backend;
    v2.kind = (uint32_t) VarKind::Stmt;
    v2.type = v->type;
    v2.size = 1;
    v2.placeholder = v2.vcall_iface = 1;
    v2.dep[0] = index;
    jitc_var_inc_ref(index);

    uint32_t result = jitc_var_new(v2);
    jitc_log(Debug, "jit_var_wrap_vcall(%s r%u <- r%u)", type_name[v2.type],
             result, index);
    return result;
}

void jitc_new_scope(JitBackend backend) {
    uint32_t scope_index = ++state.scope_ctr;
    if (unlikely(scope_index == 0))
        jitc_raise("jit_new_scope(): overflow (more than 2^32=4294967296 scopes created!");
    jitc_trace("jit_new_scope(%u)", scope_index);
    thread_state(backend)->scope = scope_index;
}

uint32_t jitc_var_stmt(JitBackend backend, VarType vt, const char *stmt,
                       int stmt_static, uint32_t n_dep,
                       const uint32_t *dep) {
    uint32_t size = n_dep == 0 ? 1 : 0;
    bool dirty = false, uninitialized = false, placeholder = false;
    Variable *v[4] { };

    if (unlikely(n_dep > 4))
        jitc_fail("jit_var_stmt(): 0-4 dependent variables supported!");

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (likely(dep[i])) {
            Variable *vi = jitc_var(dep[i]);
            size = std::max(size, vi->size);
            dirty |= vi->is_dirty();
            placeholder |= (bool) vi->placeholder;
            v[i] = vi;
        } else {
            uninitialized = true;
        }
    }

    if (unlikely(size == 0)) {
        if (!stmt_static)
            free((char *) stmt);
        return 0;
    } else if (unlikely(uninitialized)) {
        jitc_raise("jit_var_stmt(): arithmetic involving an "
                   "uninitialized variable!");
    }

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (v[i]->size != size && v[i]->size != 1)
            jitc_raise("jit_var_stmt(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(backend));
        dirty = false;
        for (uint32_t i = 0; i < n_dep; ++i) {
            v[i] = jitc_var(dep[i]);
            dirty |= v[i]->is_dirty();
        }
        if (dirty)
            jitc_raise("jit_var_stmt(): variable remains dirty following evaluation!");
    }

    Variable v2;
    for (uint32_t i = 0; i < n_dep; ++i) {
        v2.dep[i] = dep[i];
        jitc_var_inc_ref(dep[i], v[i]);
    }

    v2.kind = (uint32_t) VarKind::Stmt;
    v2.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v2.size = size;
    v2.type = (uint32_t) vt;
    v2.backend = (uint32_t) backend;
    v2.free_stmt = stmt_static == 0;
    v2.placeholder = placeholder;

    return jitc_var_new(v2);
}

/**
 * \brie Create a new IR node
 *
 * The following functions build on `jitc_var_new()`. They additionally
 *
 * - increase the reference count of operands.
 *
 * - ensure that no operands have pending side effects. Otherwise, they are
 *   evaluated, and the function checks that this worked as expected.
 */
uint32_t jitc_var_new_node_0(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool placeholder, uint64_t payload) {

    Variable v;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.placeholder = placeholder;

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_1(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool placeholder,
                             uint32_t a0, Variable *v0, uint64_t payload) {

    if (unlikely(v0->is_dirty())) {
        jitc_eval(thread_state(backend));
        v0 = jitc_var(a0);
        if (v0->is_dirty())
            jitc_fail("jit_var_new_node(): variable remains dirty following "
                      "evaluation!");
    }

    Variable v;
    v.dep[0] = a0;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.placeholder = placeholder;

    jitc_var_inc_ref(a0, v0);

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_2(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool placeholder,
                             uint32_t a0, Variable *v0,
                             uint32_t a1, Variable *v1, uint64_t payload) {

    if (unlikely(v0->is_dirty() || v1->is_dirty())) {
        jitc_eval(thread_state(backend));
        v0 = jitc_var(a0); v1 = jitc_var(a1);
        if (v0->is_dirty() || v1->is_dirty())
            jitc_fail("jit_var_new_node(): variable remains dirty!");
    }

    Variable v;
    v.dep[0] = a0;
    v.dep[1] = a1;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.placeholder = placeholder;

    jitc_var_inc_ref(a0, v0);
    jitc_var_inc_ref(a1, v1);

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_3(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool placeholder,
                             uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                             uint32_t a2, Variable *v2, uint64_t payload) {
    if (unlikely(v0->is_dirty() || v1->is_dirty() || v2->is_dirty())) {
        jitc_eval(thread_state(backend));
        v0 = jitc_var(a0); v1 = jitc_var(a1); v2 = jitc_var(a2);
        if (v0->is_dirty() || v1->is_dirty() || v2->is_dirty())
            jitc_fail("jit_var_new_node(): variable remains dirty!");
    }

    Variable v;
    v.dep[0] = a0;
    v.dep[1] = a1;
    v.dep[2] = a2;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.placeholder = placeholder;

    jitc_var_inc_ref(a0, v0);
    jitc_var_inc_ref(a1, v1);
    jitc_var_inc_ref(a2, v2);

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_4(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool placeholder,
                             uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                             uint32_t a2, Variable *v2, uint32_t a3, Variable *v3,
                             uint64_t payload) {
    if (unlikely(v0->is_dirty() || v1->is_dirty() || v2->is_dirty() || v3->is_dirty())) {
        jitc_eval(thread_state(backend));
        v0 = jitc_var(a0); v1 = jitc_var(a1); v2 = jitc_var(a2); v3 = jitc_var(a3);
        if (v0->is_dirty() || v1->is_dirty() || v2->is_dirty() || v3->is_dirty())
            jitc_fail("jit_var_new_node(): variable remains dirty!");
    }

    Variable v;
    v.dep[0] = a0;
    v.dep[1] = a1;
    v.dep[2] = a2;
    v.dep[3] = a3;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.placeholder = placeholder;

    jitc_var_inc_ref(a0, v0);
    jitc_var_inc_ref(a1, v1);
    jitc_var_inc_ref(a2, v2);
    jitc_var_inc_ref(a3, v3);

    return jitc_var_new(v);
}

void jitc_var_set_callback(uint32_t index,
                           void (*callback)(uint32_t, int, void *),
                           void *callback_data) {
    Variable *v = jitc_var(index);

    jitc_log(Debug, "jit_var_set_callback(r%u): " DRJIT_PTR " (" DRJIT_PTR ")",
            index, (uintptr_t) callback, (uintptr_t) callback_data);

    Extra &extra = state.extra[index];
    if (callback && unlikely(extra.callback))
        jitc_fail("jit_var_set_callback(): a callback was already set!");
    extra.callback = callback;
    extra.callback_data = callback_data;
    extra.callback_internal = false;
    if (callback)
        v->extra = true;
}

/// Query the current (or future, if not yet evaluated) allocation flavor of a variable
AllocType jitc_var_alloc_type(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (v->is_data())
        return jitc_malloc_type(v->data);

    return (JitBackend) v->backend == JitBackend::CUDA ? AllocType::Device
                                                       : AllocType::HostAsync;
}

/// Query the device associated with a variable
int jitc_var_device(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (v->is_data())
        return jitc_malloc_device(v->data);

    return thread_state(v->backend)->device;
}

/// Mark a variable as a scatter operation that writes to 'target'
void jitc_var_mark_side_effect(uint32_t index) {
    if (index == 0)
        return;

    Variable *v = jitc_var(index);
    v->side_effect = true;

    /* Include all side effects in recorded program, even if
       they don't depend on other placeholder variables */
    v->placeholder |= jit_flag(JitFlag::Recording);

    jitc_log(Debug, "jit_var_mark_side_effect(r%u)%s", index,
             v->placeholder ? " (part of a recorded computation)" : "");

    ThreadState *ts = thread_state(v->backend);
    std::vector<uint32_t> &output =
        v->placeholder ? ts->side_effects_recorded : ts->side_effects;
    output.push_back(index);
}

/// Return a human-readable summary of the contents of a variable
const char *jitc_var_str(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (!v->is_literal() && (!v->is_data() || v->is_dirty())) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }

    size_t size            = v->size,
           isize           = type_size[v->type],
           limit_remainder = std::min(5u, (state.print_limit + 3) / 4) * 2;

    uint8_t dst[8] { };

    if (v->is_literal())
        memcpy(dst, &v->literal, isize);

    var_buffer.clear();
    var_buffer.put('[');
    for (uint32_t i = 0; i < size; ++i) {
        if (size > state.print_limit && i == limit_remainder / 2) {
            var_buffer.fmt(".. %zu skipped .., ", size - limit_remainder);
            i = (uint32_t) (size - limit_remainder / 2 - 1);
            continue;
        }

        if (v->is_data()) {
            const uint8_t *src_offset = (const uint8_t *) v->data + i * isize;
            jitc_memcpy((JitBackend) v->backend, dst, src_offset, isize);
        }

        const char *comma = i + 1 < (uint32_t) size ? ", " : "";
        switch ((VarType) v->type) {
            case VarType::Bool:    var_buffer.fmt("%"   PRIu8  "%s", *(( uint8_t *) dst), comma); break;
            case VarType::Int8:    var_buffer.fmt("%"   PRId8  "%s", *((  int8_t *) dst), comma); break;
            case VarType::UInt8:   var_buffer.fmt("%"   PRIu8  "%s", *(( uint8_t *) dst), comma); break;
            case VarType::Int16:   var_buffer.fmt("%"   PRId16 "%s", *(( int16_t *) dst), comma); break;
            case VarType::UInt16:  var_buffer.fmt("%"   PRIu16 "%s", *((uint16_t *) dst), comma); break;
            case VarType::Int32:   var_buffer.fmt("%"   PRId32 "%s", *(( int32_t *) dst), comma); break;
            case VarType::UInt32:  var_buffer.fmt("%"   PRIu32 "%s", *((uint32_t *) dst), comma); break;
            case VarType::Int64:   var_buffer.fmt("%"   PRId64 "%s", *(( int64_t *) dst), comma); break;
            case VarType::UInt64:  var_buffer.fmt("%"   PRIu64 "%s", *((uint64_t *) dst), comma); break;
            case VarType::Float32: var_buffer.fmt("%g%s", *((float *) dst), comma); break;
            case VarType::Float64: var_buffer.fmt("%g%s", *((double *) dst), comma); break;
            default: jitc_fail("jit_var_str(): unsupported type!");
        }
    }
    var_buffer.put(']');
    return var_buffer.get();
}

static void jitc_raise_placeholder_error(const char *func, uint32_t index) {
    jitc_raise(
        "%s(r%u): placeholder variables are used to record computation symbolically\n"
        "and cannot be scheduled for evaluation! This error message could appear for\n"
        "the following reasons:\n"
        "\n"
        "1. You are using DrJit's loop or virtual function call recording feature\n"
        "   and tried to perform an operation that is not permitted in this restricted\n"
        "   execution mode. Please see the documentation of recorded loops/virtual\n"
        "   function calls to learn about these restrictions.\n"
        "\n"
        "2. You are accessing a variable that was modified as part of a recorded\n"
        "   loop and forgot to specify it as a loop variable. Please see the\n"
        "   drjit::Loop documentation for details.", func, index
    );
}

static void jitc_raise_consumed_error(const char *func, uint32_t index) {
    jitc_raise("%s(r%u): the provided variable of kind \"%s\" can only be "
               "evaluated once and was consumed by a prior operation",
               func, index, var_kind_name[jitc_var(index)->kind]);
}

/// Schedule a variable \c index for future evaluation via \ref jit_eval()
int jitc_var_schedule(uint32_t index) {
    auto it = state.variables.find(index);
    if (unlikely(it == state.variables.end()))
        jitc_raise("jit_var_schedule(r%u): unknown variable!", index);
    Variable *v = &it.value();

    if (unlikely(v->placeholder))
        jitc_raise_placeholder_error("jitc_var_schedule", index);
    if (unlikely(v->consumed))
        jitc_raise_placeholder_error("jitc_var_schedule", index);

    if (v->is_stmt() || v->is_node()) {
        thread_state(v->backend)->scheduled.push_back(index);
        jitc_log(Debug, "jit_var_schedule(r%u)", index);
        return 1;
    } else if (v->is_dirty()) {
        return 1;
    }

    return 0;
}

void *jitc_var_ptr(uint32_t index) {
    if (index == 0)
        return nullptr;
    Variable *v = jitc_var(index);

    /* If 'v' is a constant, initialize it directly instead of
       generating code to do so.. */
    if (v->is_literal())
        jitc_var_eval_literal(index, v);
    else if (v->is_stmt() || v->is_node())
        jitc_var_eval(index);

    return jitc_var(index)->data;
}

/// Evaluate a literal constant variable
void jitc_var_eval_literal(uint32_t index, Variable *v) {
    jitc_log(Debug,
            "jit_var_eval_literal(r%u): writing %s literal of size %u",
            index, type_name[v->type], v->size);

    jitc_lvn_drop(index, v);

    JitBackend backend = (JitBackend) v->backend;
    uint32_t isize = type_size[v->type];
    void* data = jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                         : AllocType::HostAsync,
                             (size_t) v->size * (size_t) isize);
    v = jitc_var(index);
    jitc_memset_async(backend, data, v->size, isize, &v->literal);

    v->kind = (uint32_t) VarKind::Data;
    v->data = data;
}

/// Evaluate the variable \c index right away if it is unevaluated/dirty.
int jitc_var_eval(uint32_t index) {
    Variable *v = jitc_var(index);

    if (unlikely(v->placeholder))
        jitc_raise_placeholder_error("jitc_var_eval", index);
    if (unlikely(v->consumed))
        jitc_raise_consumed_error("jitc_var_eval", index);

    if (v->is_stmt() || v->is_node() || (v->is_data() && v->is_dirty())) {
        ThreadState *ts = thread_state(v->backend);

        if (!v->is_data())
            ts->scheduled.push_back(index);

        jitc_eval(ts);
        v = jitc_var(index);

        if (unlikely(v->is_dirty()))
            jitc_raise("jit_var_eval(): variable r%u remains dirty after evaluation!", index);
        else if (unlikely(!v->is_data() || !v->data))
            jitc_raise("jit_var_eval(): invalid/uninitialized variable r%u!", index);

        return 1;
    }

    return 0;
}

/// Read a single element of a variable and write it to 'dst'
void jitc_var_read(uint32_t index, size_t offset, void *dst) {
    const Variable *v = jitc_var(index);

    if (v->is_stmt() || v->is_node() || (v->is_data() && v->is_dirty())) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }

    if (v->size == 1)
        offset = 0;
    else if (unlikely(offset >= (size_t) v->size))
        jitc_raise("jit_var_read(): attempted to access entry %zu in an array of "
                   "size %u!", offset, v->size);

    uint32_t isize = type_size[v->type];
    if (v->is_literal())
        memcpy(dst, &v->literal, isize);
    else if (v->is_data())
        jitc_memcpy((JitBackend) v->backend, dst,
                    (const uint8_t *) v->data + offset * isize, isize);
    else
        jitc_fail("jit_var_read(): internal error!");
}

/// Reverse of jitc_var_read(). Copy 'dst' to a single element of a variable
uint32_t jitc_var_write(uint32_t index, size_t offset, const void *src) {
    Variable *v = jitc_var(index);
    if (v->is_dirty() || v->ref_count > 1) {
        // Not safe to directly write to 'v'
        index = jitc_var_copy(index);
    } else {
        jitc_var_inc_ref(index);
    }

    jitc_var_ptr(index); // ensure variable is evaluated, even if it is a value

    v = jitc_var(index);
    if (unlikely(offset >= (size_t) v->size))
        jitc_raise("jit_var_write(): attempted to access entry %zu in an array of "
                   "size %u!", offset, v->size);

    uint32_t isize = type_size[v->type];
    uint8_t *dst = (uint8_t *) v->data + offset * isize;
    jitc_poke((JitBackend) v->backend, dst, src, isize);

    return index;
}

/// Register an existing variable with the JIT compiler
uint32_t jitc_var_mem_map(JitBackend backend, VarType type, void *ptr,
                          size_t size, int free) {
    if (unlikely(size == 0))
        return 0;

    jitc_check_size("jit_var_mem_map", size);

    Variable v;
    v.kind = (uint32_t) VarKind::Data;
    v.type = (uint32_t) type;
    v.backend = (uint32_t) backend;
    v.data = ptr;
    v.size = (uint32_t) size;
    v.retain_data = free == 0;

    if (backend == JitBackend::LLVM) {
        uintptr_t align =
            std::min(64u, jitc_llvm_vector_width * type_size[(int) type]);
        v.unaligned = uintptr_t(ptr) % align != 0;
    }

    return jitc_var_new(v, true);
}

/// Copy a memory region onto the device and return its variable index
uint32_t jitc_var_mem_copy(JitBackend backend, AllocType atype, VarType vtype,
                           const void *ptr, size_t size) {
    if (unlikely(size == 0))
        return 0;

    jitc_check_size("jit_var_mem_copy", size);

    size_t total_size = (size_t) size * (size_t) type_size[(int) vtype];
    void *target_ptr;

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        target_ptr = jitc_malloc(AllocType::Device, total_size);

        scoped_set_context guard(ts->context);
        if (atype == AllocType::HostAsync) {
            jitc_fail("jit_var_mem_copy(): copy from HostAsync to GPU memory not supported!");
        } else if (atype == AllocType::Host) {
            void *host_ptr = jitc_malloc(AllocType::HostPinned, total_size);
            CUresult rv;
            {
                unlock_guard guard2(state.lock);
                memcpy(host_ptr, ptr, total_size);
                rv = cuMemcpyAsync((CUdeviceptr) target_ptr,
                                   (CUdeviceptr) host_ptr, total_size,
                                   ts->stream);
            }
            cuda_check(rv);
            jitc_free(host_ptr);
        } else {
            cuda_check(cuMemcpyAsync((CUdeviceptr) target_ptr,
                                     (CUdeviceptr) ptr, total_size,
                                     ts->stream));
        }
    } else {
        if (atype == AllocType::HostAsync) {
            target_ptr = jitc_malloc(AllocType::HostAsync, total_size);
            jitc_memcpy_async(backend, target_ptr, ptr, total_size);
        } else if (atype == AllocType::Host) {
            target_ptr = jitc_malloc(AllocType::Host, total_size);
            {
                unlock_guard guard(state.lock);
                memcpy(target_ptr, ptr, total_size);
            }
            target_ptr = jitc_malloc_migrate(target_ptr, AllocType::HostAsync, 1);
        } else {
            target_ptr = jitc_malloc(AllocType::HostPinned, total_size);
            cuda_check(cuMemcpyAsync((CUdeviceptr) target_ptr,
                                     (CUdeviceptr) ptr, total_size,
                                     ts->stream));
        }
    }

    uint32_t index = jitc_var_mem_map(backend, vtype, target_ptr, size, true);
    jitc_log(Debug, "jit_var_mem_copy(%s r%u[%zu] <- " DRJIT_PTR ")",
             type_name[(int) vtype], index, size, (uintptr_t) ptr);
    return index;
}

uint32_t jitc_var_copy(uint32_t index) {
    if (index == 0)
        return 0;

    Variable *v = jitc_var(index);
    if (v->is_dirty()) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }
    if (unlikely(v->consumed))
        jitc_raise_consumed_error("jitc_var_copy", index);

    uint32_t result;
    if (v->is_data()) {
        JitBackend backend = (JitBackend) v->backend;
        AllocType atype = backend == JitBackend::CUDA ? AllocType::Device
                                                      : AllocType::HostAsync;
        result = jitc_var_mem_copy(backend, atype, (VarType) v->type, v->data,
                                   v->size);
    } else {
        Variable v2;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.placeholder = v->placeholder;
        v2.size = v->size;

        if (v->is_literal()) {
            v2.kind = (uint32_t) VarKind::Literal;
            v2.literal = v->literal;
        } else {
            v2.kind = (uint32_t) VarKind::Stmt;
            v2.stmt = (char *) (((JitBackend) v->backend == JitBackend::CUDA)
                                ? "mov.$t0 $r0, $r1"
                                : "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>");
            v2.dep[0] = index;
            jitc_var_inc_ref(index, v);
        }

        result = jitc_var_new(v2, true);
    }

    jitc_log(Debug, "jit_var_copy(r%u <- r%u)", result, index);
    return result;
}

uint32_t jitc_var_resize(uint32_t index, size_t size) {
    if (index == 0 && size == 0)
        return 0;

    jitc_check_size("jit_var_resize", size);

    Variable *v = jitc_var(index);
    if (unlikely(v->consumed))
        jitc_raise_consumed_error("jitc_var_resize", index);

    if (v->size == size) {
        jitc_var_inc_ref(index, v);
        return index; // Nothing to do
    } else if (v->size != 1 && !v->is_literal()) {
        jitc_raise("jit_var_resize(): variable %u must be scalar or value!", index);
    }

    if (v->is_dirty()) {
        jitc_eval(thread_state(v->backend));
        v = jitc_var(index);
        if (v->is_dirty())
            jitc_raise("jit_var_resize(): variable remains dirty following evaluation!");
    }

    uint32_t result;
    if (!v->is_data() && v->ref_count == 1) {
        // Nobody else holds a reference -- we can directly resize this variable
        jitc_var_inc_ref(index, v);
        jitc_lvn_drop(index, v);
        v->size = (uint32_t) size;
        jitc_lvn_put(index, v);
        result = index;
    } else if (v->is_literal()) {
        result = jitc_var_literal((JitBackend) v->backend, (VarType) v->type,
                                  &v->literal, size, 0);
    } else {
        Variable v2;
        v2.kind = (uint32_t) VarKind::Stmt;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.placeholder = v->placeholder;
        v2.size = (uint32_t) size;
        v2.dep[0] = index;
        v2.stmt = (char *) (((JitBackend) v->backend == JitBackend::CUDA)
                            ? "mov.$t0 $r0, $r1"
                            : "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>");
        jitc_var_inc_ref(index, v);
        result = jitc_var_new(v2, true);
    }

    jitc_log(Debug, "jit_var_resize(r%u <- r%u, size=%zu)", result, index, size);

    return result;
}

/// Migrate a variable to a different flavor of memory
uint32_t jitc_var_migrate(uint32_t src_index, AllocType dst_type) {
    if (src_index == 0)
        return 0;

    Variable *v = jitc_var(src_index);
    JitBackend backend = (JitBackend) v->backend;

    if (v->is_literal()) {
        size_t size = v->size;
        void *ptr = jitc_malloc(dst_type, type_size[v->type] * size);
        if (dst_type == AllocType::Host) {
            switch (type_size[v->type]) {
                case 1: {
                    uint8_t *p = (uint8_t *) ptr, q = (uint8_t) v->literal;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = q;
                    break;
                }
                case 2: {
                    uint16_t *p = (uint16_t *) ptr, q = (uint16_t) v->literal;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = q;
                    break;
                }
                case 4: {
                    uint32_t *p = (uint32_t *) ptr, q = (uint32_t) v->literal;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = q;
                    break;
                }
                case 8: {
                    uint64_t *p = (uint64_t *) ptr, q = (uint64_t) v->literal;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = q;
                    break;
                }
                default:
                    jitc_fail("jit_var_migrate(): invalid element size!");
            }
        } else {
            jitc_memset_async(dst_type == AllocType::HostAsync
                                  ? JitBackend::LLVM
                                  : JitBackend::CUDA,
                              ptr, (uint32_t) size, type_size[v->type], &v->literal);
        }

        return jitc_var_mem_map(backend, (VarType) v->type, ptr, v->size, 1);
    }

    if (!v->is_data() || v->is_dirty()) {
        jitc_var_eval(src_index);
        v = jitc_var(src_index);
    }

    AllocType src_type;
    void *src_ptr = v->data,
         *dst_ptr;

    auto it = state.alloc_used.find((uintptr_t) v->data);
    if (unlikely(it == state.alloc_used.end())) {
        /* Cannot resolve pointer to allocation, it was likely
           likely created by another framework */
        if ((JitBackend) v->backend == JitBackend::CUDA) {
            int type;
            ThreadState *ts = thread_state(v->backend);
            scoped_set_context guard(ts->context);
            cuda_check(cuPointerGetAttribute(
                &type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr) v->data));
            if (type == CU_MEMORYTYPE_HOST)
                src_type = AllocType::Host;
            else
                src_type = AllocType::Device;
        } else {
            src_type = AllocType::Host;
        }

        size_t size = type_size[v->type] * v->size;
        dst_ptr = jitc_malloc(dst_type, size);
        jitc_memcpy_async(backend, dst_ptr, src_ptr, size);
    } else {
        auto [size, type, device] = alloc_info_decode(it->second);
        (void) size; (void) device;
        src_type = type;
        dst_ptr = jitc_malloc_migrate(src_ptr, dst_type, 0);
    }

    uint32_t dst_index = src_index;

    v = jitc_var(src_index);
    if (src_ptr != dst_ptr) {
        Variable v2 = *v;
        v2.kind = (uint32_t) VarKind::Data;
        v2.data = dst_ptr;
        v2.retain_data = false;
        v2.ref_count = 0;
        v2.ref_count_se = 0;
        v2.extra = 0;
        dst_index = jitc_var_new(v2);
    } else {
        jitc_var_inc_ref(dst_index, v);
    }

    jitc_log(Debug,
             "jit_var_migrate(r%u <- r%u, " DRJIT_PTR " <- " DRJIT_PTR
             ", %s <- %s)",
             dst_index, src_index, (uintptr_t) dst_ptr, (uintptr_t) src_ptr,
             alloc_type_name[(int) dst_type], alloc_type_name[(int) src_type]);

    return dst_index;
}

uint32_t jitc_var_mask_default(JitBackend backend, uint32_t size) {
    if (backend == JitBackend::CUDA) {
        bool value = true;
        return jitc_var_literal(backend, VarType::Bool, &value, size, 0);
    } else {
        // Ignore SIMD lanes that lie beyond the end of the range
        Ref counter = steal(jitc_var_counter(backend, size, false));
        Variable *v_counter = jitc_var(counter);
        return jitc_var_new_node_1(backend, VarKind::DefaultMask, VarType::Bool,
                                   v_counter->size, v_counter->placeholder,
                                   counter, v_counter);
    }
}

uint32_t jitc_var_mask_peek(JitBackend backend) {
    auto &stack = thread_state(backend)->mask_stack;

    if (stack.empty()) {
        return 0;
    } else {
        uint32_t index = stack.back();
        jitc_var_inc_ref(index);
        return index;
    }
}

void jitc_var_mask_push(JitBackend backend, uint32_t index) {
    jitc_log(Debug, "jit_var_mask_push(index=r%u)", index);
    jitc_var_inc_ref(index);
    thread_state(backend)->mask_stack.push_back(index);
}

uint32_t jitc_var_mask_apply(uint32_t index, uint32_t size) {
    const Variable *v = jitc_var(index);
    JitBackend backend = (JitBackend) v->backend;

    if ((VarType) v->type != VarType::Bool)
        jitc_raise("jit_var_mask_apply(): the mask parameter was not a boolean array!");

    auto &stack = thread_state(backend)->mask_stack;
    Ref mask;
    if (!stack.empty()) {
        uint32_t index_2 = stack.back(),
                 size_2  = jitc_var(index_2)->size;

        // Use mask from the mastk stack if its size is compatible
        if (size == 1 || size_2 == 1 || size_2 == size)
            mask = borrow(index_2);
    }

    if (!mask && backend == JitBackend::LLVM)
        mask = steal(jitc_var_mask_default(backend, size));

    uint32_t result;
    if (mask) {
        // Combine given mask with mask stack
        result = jitc_var_and(mask, index);
    } else {
        result = jitc_var_resize(index, size);
    }

    jitc_log(Debug, "jit_var_apply_mask(r%u <- r%u, size=%u)", result, index, size);
    return result;
}

void jitc_var_mask_pop(JitBackend backend) {
    auto &stack = thread_state(backend)->mask_stack;
    if (unlikely(stack.empty()))
        jitc_raise("jit_var_mask_pop(): stack underflow!");

    jitc_log(Debug, "jit_var_mask_pop()");

    uint32_t index = stack.back();
    stack.pop_back();
    jitc_var_dec_ref(index);
}

/// Return an implicit mask for operations within a virtual function call
uint32_t jitc_var_vcall_mask(JitBackend backend) {
    return jitc_var_new_node_0(backend, VarKind::VCallMask, VarType::Bool, 1, 1);
}

bool jitc_var_any(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (unlikely((VarType) v->type != VarType::Bool))
        jitc_raise("jit_var_any(r%u): requires a boolean array as input!", index);

    if (v->is_literal())
        return (bool) v->literal;

    if (jitc_var_eval(index))
        v = jitc_var(index);

    return jitc_any((JitBackend) v->backend, (uint8_t *) v->data, v->size);
}

bool jitc_var_all(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (unlikely((VarType) v->type != VarType::Bool))
        jitc_raise("jit_var_all(r%u): requires a boolean array as input!", index);

    if (v->is_literal())
        return (bool) v->literal;

    if (jitc_var_eval(index))
        v = jitc_var(index);

    return jitc_all((JitBackend) v->backend, (uint8_t *) v->data, v->size);
}

template <typename T> static void jitc_var_reduce_scalar(uint32_t size, void *ptr) {
    T value;
    memcpy(&value, ptr, sizeof(T));
    value = T(value * T(size));
    memcpy(ptr, &value, sizeof(T));
}

uint32_t jitc_var_reduce(uint32_t index, ReduceOp reduce_op) {
    if (unlikely(reduce_op == ReduceOp::And || reduce_op == ReduceOp::Or))
        jitc_raise("jitc_var_reduce: doesn't support And/Or operation!");
    else if (index == 0)
        return 0;

    const Variable *v = jitc_var(index);

    JitBackend backend = (JitBackend) v->backend;
    VarType type = (VarType) v->type;

    if (v->is_literal()) {
        uint64_t value = v->literal;
        uint32_t size = v->size;

        // Tricky cases
        if (size != 1 && (reduce_op == ReduceOp::Add)) {
            switch ((VarType) v->type) {
                case VarType::Int8:    jitc_var_reduce_scalar<int8_t>  (size, &value); break;
                case VarType::UInt8:   jitc_var_reduce_scalar<uint8_t> (size, &value); break;
                case VarType::Int16:   jitc_var_reduce_scalar<int16_t> (size, &value); break;
                case VarType::UInt16:  jitc_var_reduce_scalar<uint16_t>(size, &value); break;
                case VarType::Int32:   jitc_var_reduce_scalar<int32_t> (size, &value); break;
                case VarType::UInt32:  jitc_var_reduce_scalar<uint32_t>(size, &value); break;
                case VarType::Int64:   jitc_var_reduce_scalar<int64_t> (size, &value); break;
                case VarType::UInt64:  jitc_var_reduce_scalar<uint64_t>(size, &value); break;
                case VarType::Float32: jitc_var_reduce_scalar<float>   (size, &value); break;
                case VarType::Float64: jitc_var_reduce_scalar<double>  (size, &value); break;
                default: jitc_raise("jit_var_reduce(): unsupported operand type!");
            }
        } else if (size != 1 && (reduce_op == ReduceOp::Mul)) {
            jitc_raise("jit_var_reduce(): ReduceOp::Mul is not supported for vector values!");
        }

        return jitc_var_literal(backend, type, &value, 1, 0);
    }

    jitc_log(Debug, "jit_var_reduce(index=%u, reduce_op=%s)", index, reduction_name[(int) reduce_op]);

    if (jitc_var_eval(index))
        v = jitc_var(index);

    uint8_t *values = (uint8_t *) v->data;
    uint32_t size = v->size;

    void *data =
        jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                : AllocType::HostAsync,
                    (size_t) type_size[(int) type]);
    jitc_reduce(backend, type, reduce_op, values, size, data);
    return jitc_var_mem_map(backend, type, data, 1, 1);
}

uint32_t jitc_var_registry_attr(JitBackend backend, VarType type,
                                const char *domain, const char *name) {
    uint32_t index = 0;
    Registry* registry = state.registry(backend);
    auto it = registry->attributes.find(AttributeKey(domain, name));
    if (unlikely(it == registry->attributes.end())) {
        if (jitc_registry_get_max(backend, domain) > 0) {
            jitc_log(Warn,
                     "jit_var_registry_attr(): entry with domain=\"%s\", "
                     "name=\"%s\" not found!",
                     domain, name);
        }
    } else {
        AttributeValue &val = it.value();
        index = jitc_var_mem_map(backend, type, val.ptr, val.count, false);
    }
    jitc_log(Debug, "jit_var_registry_attr(\"%s\", \"%s\"): r%u", domain, name, index);
    return index;
}

/// Return a human-readable summary of registered variables
const char *jitc_var_whos() {
    var_buffer.clear();
    var_buffer.put("\n  ID        Type       Status       Refs    Entries   Storage     Label");
    var_buffer.put("\n  =======================================================================\n");

    std::vector<uint32_t> indices;
    indices.reserve(state.variables.size());
    for (const auto& it : state.variables)
        indices.push_back(it.first);
    std::sort(indices.begin(), indices.end());

    size_t mem_size_evaluated = 0,
           mem_size_unevaluated = 0;

    for (uint32_t index: indices) {
        const Variable *v = jitc_var(index);
        size_t mem_size = (size_t) v->size * (size_t) type_size[v->type];

        var_buffer.fmt("  %-9u %s %-5s ", index,
                       (JitBackend) v->backend == JitBackend::CUDA ? "cuda"
                                                                   : "llvm",
                       type_name_short[v->type]);

        if (v->is_literal()) {
            var_buffer.put("const.     ");
        } else if (v->is_data()) {
            auto it = state.alloc_used.find((uintptr_t) v->data);
            if (unlikely(it == state.alloc_used.end())) {
                if (!v->retain_data)
                    jitc_raise("jit_var_whos(): Cannot resolve pointer to actual allocation!");
                else
                    var_buffer.put("mapped mem.");
            } else {
                auto [size, type, device] = alloc_info_decode(it->second);
                (void) size;

                if ((AllocType) type == AllocType::Device) {
                    var_buffer.fmt("device %-4i", (int) device);
                } else {
                    const char *tname = alloc_type_name_short[(int) type];
                    var_buffer.put(tname, strlen(tname));
                }
            }
        } else {
            var_buffer.put("           ");
        }

        size_t sz = var_buffer.fmt("  %u", (uint32_t) v->ref_count);
        const char *label = jitc_var_label(index);

        var_buffer.fmt("%*s%-10u%-8s   %s\n", 10 - (int) sz, "", v->size,
                   jitc_mem_string(mem_size), label ? label : "");

        if (v->is_data())
            mem_size_evaluated += mem_size;
        else
            mem_size_unevaluated += mem_size;
    }
    if (indices.empty())
        var_buffer.put("                       -- No variables registered --\n");

    constexpr size_t BucketSize1 = sizeof(tsl::detail_robin_hash::bucket_entry<VariableMap::value_type, false>);
    constexpr size_t BucketSize2 = sizeof(tsl::detail_robin_hash::bucket_entry<LVNMap::value_type, false>);

    var_buffer.put("  =======================================================================\n\n");
    var_buffer.put("  JIT compiler\n");
    var_buffer.put("  ============\n");
    var_buffer.fmt("   - Storage           : %s on device, ",
               jitc_mem_string(mem_size_evaluated));
    var_buffer.fmt("%s unevaluated.\n",
               jitc_mem_string(mem_size_unevaluated));
    var_buffer.fmt("   - Variables created : %u (peak: %u, table size: %s).\n",
               state.variable_index, state.variable_watermark,
               jitc_mem_string(
                   state.variables.bucket_count() * BucketSize1 +
                   state.lvn_map.bucket_count() * BucketSize2));
    var_buffer.fmt("   - Kernel launches   : %zu (%zu cache hits, "
               "%zu soft, %zu hard misses).\n\n",
               state.kernel_launches, state.kernel_hits,
               state.kernel_soft_misses, state.kernel_hard_misses);

    var_buffer.put("  Memory allocator\n");
    var_buffer.put("  ================\n");
    for (int i = 0; i < (int) AllocType::Count; ++i)
        var_buffer.fmt("   - %-18s: %s/%s used (peak: %s).\n",
                   alloc_type_name[i],
                   std::string(jitc_mem_string(state.alloc_usage[i])).c_str(),
                   std::string(jitc_mem_string(state.alloc_allocated[i])).c_str(),
                   std::string(jitc_mem_string(state.alloc_watermark[i])).c_str());

    return var_buffer.get();
}

/// Return a GraphViz representation of registered variables
const char *jitc_var_graphviz() {
    std::vector<uint32_t> indices;
    indices.reserve(state.variables.size());
    for (const auto& it : state.variables)
        indices.push_back(it.first);

    std::sort(indices.begin(), indices.end());
    var_buffer.clear();
    var_buffer.put("digraph {\n"
                   "    rankdir=TB;\n"
                   "    graph [dpi=50 fontname=Consolas];\n"
                   "    node [shape=record fontname=Consolas];\n"
                   "    edge [fontname=Consolas];\n");

    size_t current_hash = 0, current_depth = 1;

    for (int32_t index : indices) {
        ExtraMap::iterator it = state.extra.find(index);

        const char *label = nullptr;
        if (it != state.extra.end())
            label = it.value().label;

        const char *label_without_prefix = label;

        size_t prefix_hash = 0;
        if (label) {
            const char *sep = strrchr(label, '/');
            if (sep) {
                prefix_hash = hash(label, sep - label);
                label_without_prefix = sep + 1;
            }
        }

        if (prefix_hash != current_hash) {
            for (size_t i = current_depth - 1; i > 0; --i) {
                var_buffer.put(' ', 4 * i);
                var_buffer.put("}\n");
            }

            current_hash = prefix_hash;
            current_depth = 1;

            const char *p = label;
            while (true) {
                const char *pn = p ? strchr(p, '/') : nullptr;
                if (!pn)
                    break;

                var_buffer.put(' ', 4 * current_depth);
                var_buffer.fmt("subgraph cluster_%08llx {\n",
                               (unsigned long long) hash(label, pn - label));
                current_depth++;
                var_buffer.put(' ', 4 * current_depth);
                var_buffer.put("label=\"");
                var_buffer.put(p, pn - p);
                var_buffer.put("\";\n");
                var_buffer.put(' ', 4 * current_depth);
                var_buffer.put("color=gray95;\n");
                var_buffer.put(' ', 4 * current_depth);
                var_buffer.put("style=filled;\n");

                p = pn + 1;
            }
        }

        const Variable *v = jitc_var(index);
        var_buffer.put(' ', 4 * current_depth);
        var_buffer.put_u32(index);
        var_buffer.put(" [label=\"{");

        auto print_escape = [](const char *s) {
            char c;
            while (c = *s++, c != '\0') {
                bool escape = false;
                switch (c) {
                    case '$':
                        if (s[0] == 'n') {
                            s++;
                            var_buffer.put("\\l");
                            continue;
                        }
                        break;

                    case '\n':
                        var_buffer.put("\\l");
                        continue;

                    case '"':
                    case '|':
                    case '{':
                    case '}':
                    case '<':
                    case '>':
                        escape = true;
                        break;
                    default:
                        break;
                }
                if (escape)
                    var_buffer.put('\\');
                var_buffer.put(c);
            }
        };

        const char *color = nullptr;
        bool labeled = false;
        if (label_without_prefix && strlen(label_without_prefix) != 0) {
            var_buffer.put("Label: \\\"");
            print_escape(label_without_prefix);
            var_buffer.put("\\\"|");
            labeled = true;
        }

        if (v->is_literal()) {
            var_buffer.put("Literal: ");
            jitc_value_print(v, true);
            color = "gray90";
        } else if (v->is_data()) {
            if (v->is_dirty()) {
                var_buffer.put("Evaluated (dirty)");
                color = "salmon";
            } else {
                var_buffer.put("Evaluated");
                color = "lightblue2";
            }
        } else if (v->is_stmt()) {
            if ((VarType) v->type == VarType::Void)
                color = "yellowgreen";

            if (*v->stmt != '\0') {
                print_escape(v->stmt);
                var_buffer.put("\\l");
            } else if (labeled) {
                var_buffer.rewind_to(var_buffer.size() - 1);
            }
        } else {
            const char *name = var_kind_name[v->kind];
            var_buffer.put(name, strlen(name));
        }

        if (v->placeholder && !color)
            color = "yellow";
        if (labeled && !color)
            color = "wheat";

        var_buffer.fmt("|{Type: %s %s|Size: %u}|{r%u|Refs: %u}}",
            (JitBackend) v->backend == JitBackend::CUDA ? "cuda" : "llvm",
            type_name_short[v->type], v->size, index,
            (uint32_t) v->ref_count);

        var_buffer.put("}\"");
        if (color)
            var_buffer.fmt(" fillcolor=%s style=filled", color);
        var_buffer.put("];\n");
    }

    for (size_t i = current_depth - 1; i > 0; --i) {
        var_buffer.put(' ', 4 * i);
        var_buffer.put("}\n");
    }

    for (int32_t index : indices) {
        const Variable *v = jitc_var(index);

        int n_dep = 0;
        for (uint32_t i = 0; i < 4; ++i)
            n_dep += v->dep[i] ? 1 : 0;

        const Extra *extra = nullptr;
        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_var_graphviz(): could not find matching 'extra' "
                          "record!");
            extra = &it->second;
            n_dep += extra->n_dep;
        }

        uint32_t edge_index = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            if (!v->dep[i])
                continue;

            var_buffer.fmt("    %u -> %u", v->dep[i], index);
            bool special = i == 3 && v->dep[2] == 0;
            if (n_dep > 1 || special) {
                var_buffer.put(" [");
                if (n_dep > 1)
                    var_buffer.fmt("label=\" %u\"", ++edge_index);
                if (special)
                    var_buffer.fmt("%sstyle=dashed", n_dep > 1 ? " " : "");
                var_buffer.put("]");
            }
            var_buffer.put(";\n");
        }

        for (uint32_t i = 0; i < (extra ? extra->n_dep : 0); ++i) {
            if (!extra->dep[i])
                continue;
            var_buffer.fmt("    %u -> %u", extra->dep[i], index);
            if (n_dep > 1)
                var_buffer.fmt(" [label=\" %u\"]", ++edge_index);
            var_buffer.put(";\n");
        }
    }

    var_buffer.put(
        "    subgraph cluster_legend {\n"
        "        label=\"Legend\";\n"
        "        l5 [style=filled fillcolor=yellow label=\"Placeholder\"];\n"
        "        l4 [style=filled fillcolor=yellowgreen label=\"Special\"];\n"
        "        l3 [style=filled fillcolor=salmon label=\"Dirty\"];\n"
        "        l2 [style=filled fillcolor=lightblue2 label=\"Evaluated\"];\n"
        "        l1 [style=filled fillcolor=wheat label=\"Labeled\"];\n"
        "        l0 [style=filled fillcolor=gray90 label=\"Constant\"];\n"
        "    }\n"
        "}\n");

    return var_buffer.get();
}
