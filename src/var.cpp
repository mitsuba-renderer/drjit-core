/*
    src/var.cpp -- Operations for creating and querying variables

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "var.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "util.h"
#include "op.h"
#include "registry.h"

/// Descriptive names for the various variable types
const char *type_name[(int) VarType::Count] {
    "void",   "bool",  "int8",   "uint8",   "int16",   "uint16",  "int32",
    "uint32", "int64", "uint64", "pointer", "float16", "float32", "float64"
};

/// Descriptive names for the various variable types (extra-short version)
const char *type_name_short[(int) VarType::Count] {
    "void ", "bool", "i8", "u8",  "i16", "u16", "i32",
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

// Representation an all-1 bit vector
const uint64_t type_all_ones[(int) VarType::Count] {
    0, 1, 0xff, 0xff, 0xffff, 0xffff, 0xffffffffu,
    0xffffffffu, 0xffffffffffffffffull,
    0xffffffffffffffffull, 0xffffffffffffffffull,
    0xffff, 0xffffffffu, 0xffffffffffffffffull
};

// Representation of the value 1
const uint64_t type_one[(int) VarType::Count] {
    0, 1, 1, 1, 1,      1,          1,
    1, 1, 1, 0, 0x3c00, 0x3f800000, 0x3ff0000000000000ull
};

/// Smallest representable value (neg. infinity for FP values)
const uint64_t type_min[(int) VarType::Count] {
    0, 0, 0x80, 0, 0x8000, 0, 0x80000000, 0,
    0x8000000000000000ull, 0, 0,
    0xfc00, 0xff800000, 0xfff0000000000000
};

/// Largest representable value (infinity for FP values)
const uint64_t type_max[(int) VarType::Count] {
    0, 1, 0x7f, 0xff, 0x7fff, 0xffff, 0x7fffffff, 0xffffffff,
    0x7fffffffffffffffull, 0xffffffffffffffffull, 0xffffffffffffffffull,
    0x7c00, 0x7f800000, 0x7ff0000000000000ull
};

bool jitc_is_max(Variable *v) {
    return v->is_literal() && type_max[v->type] == v->literal;
}

bool jitc_is_min(Variable *v) {
    return v->is_literal() && type_min[v->type] == v->literal;
}

/// Label for each VarKind entry (used for trace messages in jitc_var_new)
const char *var_kind_name[(int) VarKind::Count] {
    "invalid",

    // An evaluated node representing data
    "evaluated",

    // Undefined memory
    "undefined",

    // A literal constant
    "literal",

    // A no-op (generates no code)
    "nop",

    // Common unary operations
    "neg", "not", "sqrt", "sqrt.approx", "abs",

    // Common binary arithmetic operations
    "add", "sub", "mul", "div", "div.approx", "mod",

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
    "popc", "clz", "ctz", "brev",

    // Bit-wise operations
    "and", "or", "xor",

    // Shifts
    "shl", "shr",

    // Fast approximations
    "rcp", "rcp.approx", "rsqrt.approx",

    // Multi-function generator (CUDA)
    "sin", "cos", "exp2", "log2",

    // Casts
    "cast", "bitcast",

    // Ensure that an index is within the array bounds
    "bounds_check",

    // Memory-related operations
    "gather", "scatter", "scatter_inc", "scatter_kahan",

    // Counter node to determine the current lane ID
    "counter",

    // Default mask used to ignore out-of-range SIMD lanes (LLVM)
    "default_mask",

    // A polymorphic function call
    "call",

    // Specialized nodes for calls
    "call_mask", "call_self",

    // Input argument to a function call
    "call_input",

    // Output of a function call
    "call_output",

    // Perform a standard texture lookup (CUDA)
    "tex_lookup",

    // Load all texels used for bilinear interpolation (CUDA)
    "tex_fetch_bilerp",

    // Perform a ray tracing call
    "trace_ray",

    // Extract a component from an operation that produced multiple results
    "extract",

    /// Retrieve the index of the current thread (LLVM mode)
    "thread_index",

    // Variable marking the start of a loop
    "loop_start",

    // Variable marking the loop condition
    "loop_cond",

    // Variable marking the end of a loop
    "loop_end",

    // SSA Phi variable at start of loop
    "loop_phi",

    // SSA Phi variable at end of loop
    "loop_output",

    // Variable marking the start of a conditional statement
    "cond_start",

    // Variable marking the 'false' branch of a conditional statement
    "cond_mid",

    // Variable marking the end of a conditional statement
    "cond_end",

    // SSA Phi variable marking an output of a conditional statement
    "cond_output"
};

/// Temporary string buffer for miscellaneous variable-related tasks
StringBuffer var_buffer(0);

#define jitc_check_size(name, size)                                            \
    if (unlikely(size > 0xFFFFFFFF))                                           \
    jitc_raise(name "(): tried to create an array with %zu entries, "          \
                    "which exceeds the limit of 2^32 == 4294967296 entries.",  \
               size)

/// Cleanup handler, called when the internal/external reference count reaches zero
void jitc_var_free(uint32_t index, Variable *v) {
    jitc_trace("jit_var_free(r%u)", index);

    if (v->is_evaluated()) {
        // Release memory referenced by this variable
        if (!v->retain_data)
            jitc_free(v->data);
    } else {
        // Unevaluated variable, drop from CSE cache
        jitc_lvn_drop(index, v);
    }

    uint32_t dep[4];
    bool write_ptr = v->write_ptr;
    memcpy(dep, v->dep, sizeof(uint32_t) * 4);
    v->counter++;

    if (unlikely(v->extra)) {
        uint32_t index2 = v->extra;
        VariableExtra &extra = state.extra[index2];
        char *label = extra.label;

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

        free(label);
        state.unused_extra.push(index2);
    }

    // Remove from unused variable list
    state.unused_variables.push(index);

    if (likely(!write_ptr)) {
        // Decrease reference count of dependencies
        for (int i = 0; i < 4; ++i)
            jitc_var_dec_ref(dep[i]);
    } else {
        jitc_var_dec_ref_se(dep[3]);
    }

    // Optional: intense internal sanitation instrumentation
#if defined(DRJIT_SANITIZE_INTENSE)
    jitc_sanitation_checkpoint();
#endif
}

/// Access a variable by ID, terminate with an error if it doesn't exist
Variable *jitc_var(uint32_t index) {
    std::vector<Variable> &variables = state.variables;
    Variable *v = variables.data() + index;

    if (unlikely(index == 0 || index >= variables.size() ||
                 (v->ref_count == 0 && v->ref_count_se == 0)))
        jitc_fail("jit_var(r%u): unknown variable!", index);

    return v;
}

/// Access a variable through a weak reference. May return ``nullptr``
Variable *jitc_var(WeakRef ref) {
    std::vector<Variable> &variables = state.variables;
    if (unlikely(ref.index >= variables.size()))
        jitc_fail("jit_var(r%u): unknown variable!", ref.index);

    Variable *v = variables.data() + ref.index;
    if (ref.index == 0 || v->counter != ref.counter)
        return nullptr;

    return v;
}

/// Increase the external reference count of a given variable
void jitc_var_inc_ref(uint32_t index, Variable *v) noexcept {
    (void) index; // jitc_trace may be disabled
    v->ref_count++;
    jitc_trace("jit_var_inc_ref(r%u): %u", index, (uint32_t) v->ref_count);
}

/// Temporarily stash the reference count of a variable used to make
/// copy-on-write (COW) decisions in jit_var_scatter. Returns a handle for
/// \ref jitc_var_unstash_ref().
uint64_t jitc_var_stash_ref(uint32_t index) {
    if (!index)
        return 0;

    Variable *v = jitc_var(index);
    if (v->ref_count_stashed)
        return 0;

    v->ref_count_stashed = v->ref_count;
    jitc_trace("jit_var_stash_ref(r%u): %u", index, v->ref_count);
    return (((uint64_t) v->counter) << 32) | index;
}

/// Undo the change performed by jitc_var_stash_ref
void jitc_var_unstash_ref(uint64_t handle) {
    if (!handle)
        return;

    uint32_t index = (uint32_t) handle;

    Variable *v = jitc_var(WeakRef(index, (uint32_t) (handle >> 32)));
    if (v) {
        v->ref_count_stashed = 0;
        jitc_trace("jit_var_unstash_ref(r%u)", index);
    } else {
        jitc_trace("jit_var_unstash_ref(r%u): expired.", index);
    }
}

/// Increase the external reference count of a given variable
void jitc_var_inc_ref(uint32_t index) noexcept {
    if (index)
        jitc_var_inc_ref(index, jitc_var(index));
}

/// Increase the side effect reference count of a given variable
void jitc_var_inc_ref_se(uint32_t index, Variable *v) noexcept {
    (void) index; // jitc_trace may be disabled
    v->ref_count_se++;
    jitc_trace("jit_var_inc_ref_se(r%u): %u", index, (uint32_t) v->ref_count_se);
}

/// Increase the side effect reference count of a given variable
void jitc_var_inc_ref_se(uint32_t index) noexcept {
    if (index)
        jitc_var_inc_ref_se(index, jitc_var(index));
}

/// Decrease the external reference count of a given variable
void jitc_var_dec_ref(uint32_t index, Variable *v) noexcept {
    jitc_assert(v->ref_count != 0,
                "jit_var_dec_ref(): reference count underflow in variable r%u!",
                index);

    jitc_trace("jit_var_dec_ref(r%u): %u", index, (uint32_t) v->ref_count - 1);
    v->ref_count--;

    if (v->ref_count == 0 && v->ref_count_se == 0)
        jitc_var_free(index, v);
}

/// Decrease the external reference count of a given variable
void jitc_var_dec_ref(uint32_t index) noexcept {
    if (index != 0)
        jitc_var_dec_ref(index, jitc_var(index));
}

/// Decrease the side effect reference count of a given variable
void jitc_var_dec_ref_se(uint32_t index, Variable *v) noexcept {
    if (unlikely(v->ref_count_se == 0))
        jitc_fail("jit_var_dec_ref_se(): variable r%u has no side effect references!", index);

    jitc_trace("jit_var_dec_ref_se(r%u): %u", index, (uint32_t) v->ref_count_se - 1);
    v->ref_count_se--;

    if (v->ref_count == 0 && v->ref_count_se == 0)
        jitc_var_free(index, v);
}

/// Decrease the side effect reference count of a given variable
void jitc_var_dec_ref_se(uint32_t index) noexcept {
    if (index != 0)
        jitc_var_dec_ref_se(index, jitc_var(index));
}

/// Remove a variable from the cache used for common subexpression elimination
void jitc_lvn_drop(uint32_t index, const Variable *v) {
    VariableKey key(*v);
    size_t hash = VariableKeyHasher()(key);
    LVNMap &cache = state.lvn_map;
    LVNMap::iterator it = cache.find(key, hash);
    if (it != cache.end() && it.value() == index)
        cache.erase_fast(it);
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
    if (index) {
        const Variable *v = jitc_var(index);
        if (v->extra) {
            const char *label = state.extra[v->extra].label;
            if (label) {
                const char *delim = strrchr(label, '/');
                return delim ? (delim + 1) : label;
            }
        }
    }
    return nullptr;
}

void jitc_var_set_label(uint32_t index, const char *label) {
    if (unlikely(index == 0))
        return;

    size_t len = label ? strlen(label) : 0;

    for (size_t i = 0; i < len; ++i) {
        if (label[i] == '\n' || label[i] == '/')
            jitc_raise("jit_var_set_label(): invalid string (may not "
                       "contain newline or '/' characters)");
    }

    Variable *v = jitc_var(index);
    ThreadState *ts = thread_state(v->backend);
    VariableExtra *e = jitc_var_extra(v);
    free(e->label);

    if (!ts->prefix) {
        if (!label) {
            e->label = nullptr;
        } else {
            e->label = (char *) malloc_check(len + 1);
            memcpy(e->label, label, len + 1);
        }
    } else {
        size_t prefix_len = strlen(ts->prefix);
        char *combined = (char *) malloc_check(prefix_len + len + 1);
        memcpy(combined, ts->prefix, prefix_len);
        if (len)
            memcpy(combined + prefix_len, label, len);
        combined[prefix_len + len] = '\0';
        e->label = combined;
    }

    jitc_log(Debug, "jit_var_set_label(): r%u.label = \"%s\"", index,
             label ? label : "(null)");
}

// Print a value variable to 'var_buffer' (for debug/GraphViz output)
void jitc_value_print(const Variable *v, bool graphviz = false) {
    #define JIT_LITERAL_PRINT(type, ptype, fmtstr)  {  \
            type value;                                \
            memcpy((void*)&value, &v->literal, sizeof(type)); \
            var_buffer.fmt(fmtstr, (ptype) value);     \
        }                                              \
        break;

    switch ((VarType) v->type) {
        case VarType::Float16: JIT_LITERAL_PRINT(drjit::half, float, "%g");
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

static char source_location_buf[256] { 0 };

void jitc_set_source_location(const char *fname, size_t lineno) noexcept {
    snprintf(source_location_buf, sizeof(source_location_buf), "%s:%zu", fname, lineno);
}

/// Append the given variable to the instruction trace and return its ID
uint32_t jitc_var_new(Variable &v, bool disable_lvn) {
    State &st = ::state;
    if (unlikely(v.backend == (uint32_t) JitBackend::None))
        v.backend = (uint32_t) default_backend;

    ThreadState *ts = thread_state(v.backend);
    uint32_t flags = jitc_flags();

    bool lvn = !disable_lvn && (VarType) v.type != VarType::Void &&
               !v.is_evaluated() && (flags & (uint32_t) JitFlag::ValueNumbering);

    v.scope = ts->scope;

    // Check if this exact statement already exists ..
    LVNMap::iterator key_it;
    bool lvn_key_inserted = false;
    if (lvn)
        std::tie(key_it, lvn_key_inserted) =
            st.lvn_map.try_emplace(VariableKey(v), 0);

    uint32_t index;
    Variable *vo;

    if (likely(!lvn || lvn_key_inserted)) {
        bool reuse_indices = flags & (uint32_t) JitFlag::ReuseIndices;
        UnusedPQ &unused = st.unused_variables;

        if (unlikely(unused.empty() || !reuse_indices)) {
            index = (uint32_t) st.variables.size();
            st.variables.emplace_back();
        } else {
            index = unused.top();
            unused.pop();
        }

        if (lvn_key_inserted)
            key_it.value() = index;

        vo = &st.variables[index];
        jitc_assert(vo->ref_count == 0 && vo->ref_count_se == 0,
                    "jit_var_new(): selected entry of variable r%u "
                    "is already used.", index);

        v.counter = vo->counter;
        *vo = v;

        bool has_prefix = ts->prefix != nullptr,
             has_loc = (flags & (uint32_t) JitFlag::Debug) && (source_location_buf[0] != '\0');

        if (unlikely(has_prefix || has_loc)) {
            size_t size_prefix = has_prefix ? strlen(ts->prefix) : 0,
                   size_loc    = has_loc ? strlen(source_location_buf) : 0;
            char *s = (char *) malloc_check(size_prefix + size_loc + 1), *p = s;
            if (has_prefix) {
                memcpy(p, ts->prefix, size_prefix);
                p += size_prefix;
            }

            if (size_loc) {
                memcpy(p, source_location_buf, size_loc);
                p += size_loc;
            }

            *p++ = '\0';
            jitc_var_extra(vo)->label = s;
        }

        st.variable_counter++;
    } else {
        if (likely(!v.write_ptr)) {
            for (int i = 0; i < 4; ++i)
                jitc_var_dec_ref(v.dep[i]);
        } else {
            jitc_var_dec_ref_se(v.dep[3]);
        }

        index = key_it.value();
        vo = &st.variables[index];
        jitc_assert(VariableKey(*vo) == VariableKey(v),
                    "jit_var_new(): LVN data structure is out of sync! (1)");
        jitc_assert(vo->ref_count != 0 || vo->ref_count_se != 0,
                    "jit_var_new(): LVN data structure is out of sync! (2)");
    }

    if (unlikely(std::max(st.log_level_stderr, st.log_level_callback) >=
                 LogLevel::Debug)) {
        var_buffer.clear();
        var_buffer.fmt("jit_var_new(): %s r%u", type_name[v.type], index);
        if (v.size > 1)
            var_buffer.fmt("[%u]", v.size);

        var_buffer.put(" = ");

        if (v.is_node() || v.is_undefined()) {
            var_buffer.fmt("%s(", var_kind_name[v.kind]);
            for (int i = 0; i < 4; ++i) {
                if (!v.dep[i])
                    break;
                if (i > 0)
                    var_buffer.put(", ");
                var_buffer.fmt("r%u", v.dep[i]);
            }
            var_buffer.put(")");
        } else if (v.is_literal()) {
            jitc_value_print(&v);
        } else if (v.is_evaluated()) {
            var_buffer.put("data(");
            var_buffer.fmt(DRJIT_PTR, (uintptr_t) v.data);
            var_buffer.put(")");
        }

        bool lvn_hit = lvn && !lvn_key_inserted,
             show_lit = v.is_node() && !v.is_undefined() &&
                        (VarKind) v.kind != VarKind::BoundsCheck &&
                        (VarType) v.type != VarType::Void && v.literal;
        if (v.symbolic || lvn_hit || show_lit) {
            var_buffer.put(" [");
            bool prev = false;
            if (v.symbolic) {
                var_buffer.put("symbolic");
                prev = true;
            }
            if (show_lit) {
                if (prev)
                    var_buffer.put(", ");
                prev = true;
                var_buffer.fmt("#%llu", (unsigned long long) v.literal);
            }
            if (lvn_hit) {
                if (prev)
                    var_buffer.put(", ");
                var_buffer.put("lvn hit");
            }
            var_buffer.put("]");
        }

        jitc_log(Debug, "%s", var_buffer.get());
    }

    jitc_var_inc_ref(index, vo);

    // Optional: intense internal sanitation instrumentation
#if defined(DRJIT_SANITIZE_INTENSE)
    jitc_sanitation_checkpoint();
#endif

    return index;
}

uint32_t jitc_var_literal(JitBackend backend, VarType type, const void *value,
                          size_t size, int eval) {
    if (unlikely(size == 0))
        return 0;

    jitc_check_size("jit_var_literal", size);

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

uint32_t jitc_var_undefined(JitBackend backend, VarType type, size_t size) {
    if (size == 0)
        return 0;

    jitc_check_size("jit_var_undefined", size);
    Variable v;
    v.kind = VarKind::Undefined;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) type;
    v.size = (uint32_t) size;
    v.literal = (uint64_t) -1;
    return jitc_var_new(v);
}

uint32_t jitc_var_counter(JitBackend backend, size_t size,
                          bool simplify_scalar) {
    if (size == 1 && simplify_scalar && !jit_flag(JitFlag::SymbolicScope)) {
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

uint32_t jitc_var_call_input(uint32_t index) {
    if (index == 0)
        jitc_raise("jit_var_call_input(): invoked with an "
                   "uninitialized variable!");

    const Variable *v = jitc_var(index);

    Variable v2;
    v2.backend = v->backend;
    v2.type = v->type;
    v2.size = 1;

    bool optimize = jitc_flags() & (uint32_t) JitFlag::OptimizeCalls;

    if (v->is_literal() && optimize) {
        v2.kind = (uint32_t) VarKind::Literal;
        v2.literal = v->literal;
        return jitc_var_new(v2);
    } else {
        v2.kind = (uint32_t) VarKind::CallInput;
        v2.symbolic = 1;
        v2.dep[0] = index;
        jitc_var_inc_ref(index);
    }

    return jitc_var_new(v2, !optimize);
}

uint32_t jitc_new_scope(JitBackend backend) {
    uint32_t scope_index = ++state.scope_ctr;
    if (unlikely(scope_index == 0))
        jitc_raise("jit_new_scope(): overflow (more than 2^32=4294967296 scopes created!");
    jitc_trace("jit_new_scope(%u)", scope_index);
    thread_state(backend)->scope = scope_index;
    return scope_index;
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
                             uint32_t size, bool symbolic, uint64_t payload) {

    Variable v;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.symbolic = symbolic;

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_1(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool symbolic,
                             uint32_t a0, Variable *v0, uint64_t payload) {

    if (unlikely(v0->is_dirty())) {
        jitc_eval(thread_state(backend));

        v0 = jitc_var(a0);
        if (v0->is_dirty())
            jitc_raise_dirty_error(a0);
    }

    Variable v;
    v.dep[0] = a0;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.symbolic = symbolic;

    jitc_var_inc_ref(a0, v0);

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_2(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool symbolic,
                             uint32_t a0, Variable *v0,
                             uint32_t a1, Variable *v1, uint64_t payload) {

    if (unlikely(v0->is_dirty() || v1->is_dirty())) {
        jitc_eval(thread_state(backend));

        v0 = jitc_var(a0);
        if (v0->is_dirty())
            jitc_raise_dirty_error(a0);

        v1 = jitc_var(a1);
        if (v1->is_dirty())
            jitc_raise_dirty_error(a1);
    }

    Variable v;
    v.dep[0] = a0;
    v.dep[1] = a1;
    v.literal = payload;
    v.size = size;
    v.kind = kind;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.symbolic = symbolic;

    jitc_var_inc_ref(a0, v0);
    jitc_var_inc_ref(a1, v1);

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_3(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool symbolic,
                             uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                             uint32_t a2, Variable *v2, uint64_t payload) {
    if (unlikely(v0->is_dirty() || v1->is_dirty() || v2->is_dirty())) {
        jitc_eval(thread_state(backend));

        v0 = jitc_var(a0);
        if (v0->is_dirty())
            jitc_raise_dirty_error(a0);

        v1 = jitc_var(a1);
        if (v1->is_dirty())
            jitc_raise_dirty_error(a1);

        v2 = jitc_var(a2);
        if (v1->is_dirty())
            jitc_raise_dirty_error(a2);
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
    v.symbolic = symbolic;

    jitc_var_inc_ref(a0, v0);
    jitc_var_inc_ref(a1, v1);
    jitc_var_inc_ref(a2, v2);

    return jitc_var_new(v);
}

uint32_t jitc_var_new_node_4(JitBackend backend, VarKind kind, VarType vt,
                             uint32_t size, bool symbolic,
                             uint32_t a0, Variable *v0, uint32_t a1, Variable *v1,
                             uint32_t a2, Variable *v2, uint32_t a3, Variable *v3,
                             uint64_t payload) {
    if (unlikely(v0->is_dirty() || v1->is_dirty() || v2->is_dirty() || v3->is_dirty())) {
        jitc_eval(thread_state(backend));

        v0 = jitc_var(a0);
        if (v0->is_dirty())
            jitc_raise_dirty_error(a0);

        v1 = jitc_var(a1);
        if (v1->is_dirty())
            jitc_raise_dirty_error(a1);

        v2 = jitc_var(a2);
        if (v1->is_dirty())
            jitc_raise_dirty_error(a2);

        v3 = jitc_var(a3);
        if (v1->is_dirty())
            jitc_raise_dirty_error(a3);
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
    v.symbolic = symbolic;

    jitc_var_inc_ref(a0, v0);
    jitc_var_inc_ref(a1, v1);
    jitc_var_inc_ref(a2, v2);
    jitc_var_inc_ref(a3, v3);

    return jitc_var_new(v);
}

void jitc_var_set_callback(uint32_t index,
                           void (*callback)(uint32_t, int, void *),
                           void *data,
                           bool is_internal) {
    Variable *v = jitc_var(index);

    jitc_log(Debug, "jit_var_set_callback(r%u): " DRJIT_PTR " (" DRJIT_PTR ")",
            index, (uintptr_t) callback, (uintptr_t) data);

    VariableExtra *extra = jitc_var_extra(v);
    if (unlikely(callback && extra->callback))
        jitc_fail("jit_var_set_callback(): a callback was already set!");

    extra->callback = callback;
    extra->callback_data = data;
    extra->callback_internal = is_internal;
}

/// Query the current (or future, if not yet evaluated) allocation flavor of a variable
AllocType jitc_var_alloc_type(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (v->is_evaluated())
        return jitc_malloc_type(v->data);

    return (JitBackend) v->backend == JitBackend::CUDA ? AllocType::Device
                                                       : AllocType::HostAsync;
}

/// Query the device associated with a variable
int jitc_var_device(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (v->is_evaluated())
        return jitc_malloc_device(v->data);

    return thread_state(v->backend)->device;
}

/// Mark a variable as a scatter operation that writes to 'target'
void jitc_var_mark_side_effect(uint32_t index) {
    if (index == 0)
        return;

    Variable *v = jitc_var(index);
    v->side_effect = true;

    jitc_log(Debug, "jit_var_mark_side_effect(r%u)%s", index,
             v->symbolic ? " [symbolic]" : "");

    ThreadState *ts = thread_state(v->backend);
    std::vector<uint32_t> &output =
        v->symbolic ? ts->side_effects_symbolic : ts->side_effects;
    output.push_back(index);
}

/// Return a human-readable summary of the contents of a variable
const char *jitc_var_str(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (!v->is_literal() && (!v->is_evaluated() || v->is_dirty())) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }

    size_t size            = v->size,
           isize           = type_size[v->type],
           limit_remainder = std::min(5u, (state.print_limit + 3) / 4) * 2;
    VarType vt = (VarType) v->type;
    JitBackend backend = (JitBackend) v->backend;
    bool is_evaluated = v->is_evaluated();
    const void *v_data = v->data;

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

        if (is_evaluated) {
            const uint8_t *src_offset = (const uint8_t *) v_data + i * isize;
            jitc_memcpy((JitBackend) backend, dst, src_offset, isize);
        }

        const char *comma = i + 1 < (uint32_t) size ? ", " : "";
        switch (vt) {
            case VarType::Bool:    var_buffer.fmt("%"   PRIu8  "%s", *(( uint8_t *) dst), comma); break;
            case VarType::Int8:    var_buffer.fmt("%"   PRId8  "%s", *((  int8_t *) dst), comma); break;
            case VarType::UInt8:   var_buffer.fmt("%"   PRIu8  "%s", *(( uint8_t *) dst), comma); break;
            case VarType::Int16:   var_buffer.fmt("%"   PRId16 "%s", *(( int16_t *) dst), comma); break;
            case VarType::UInt16:  var_buffer.fmt("%"   PRIu16 "%s", *((uint16_t *) dst), comma); break;
            case VarType::Int32:   var_buffer.fmt("%"   PRId32 "%s", *(( int32_t *) dst), comma); break;
            case VarType::UInt32:  var_buffer.fmt("%"   PRIu32 "%s", *((uint32_t *) dst), comma); break;
            case VarType::Int64:   var_buffer.fmt("%"   PRId64 "%s", *(( int64_t *) dst), comma); break;
            case VarType::UInt64:  var_buffer.fmt("%"   PRIu64 "%s", *((uint64_t *) dst), comma); break;
            case VarType::Float16: var_buffer.fmt("%g%s", float(*((drjit::half *) dst)), comma); break;
            case VarType::Float32: var_buffer.fmt("%g%s", *((float *) dst), comma); break;
            case VarType::Float64: var_buffer.fmt("%g%s", *((double *) dst), comma); break;
            default: jitc_fail("jit_var_str(): unsupported type!");
        }
    }
    var_buffer.put(']');
    return var_buffer.get();
}

void jitc_raise_dirty_error(uint32_t index) {
    jitc_raise(
        "Variable r%u remains dirty despite an attempt to evaluate it.\n"
        "This normally indicates the following situation:\n"
        "\n"
        "1. You executed an operation causing a *side effect*, e.g., a\n"
        "\n"
        "    - Scatter: dr.scatter(), dr.scatter_reduce(), etc., or\n"
        "\n"
        "    - Reduction (via the 'symbolic' strategy): dr.reduce(), \n"
        "      dr.block_reduce(), dr.all(), dr.sum(), dr.min(), etc.\n"
        "\n"
        "    - Indirect assignment: a[index_vector] = ...\n"
        "\n"
        "2. You did so within a symbolic operation such as dr.switch(),\n"
        "   dr.if_stmt(), dr.while_loop(), etc. Or perhaps you are using\n"
        "   @dr.syntax-decorated functions that automatically insert\n"
        "   calls to such symbolic operations. You can read more about\n"
        "   symbolic execution here:\n"
        "   https://drjit.readthedocs.io/en/latest/cflow.html#symbolic-versus-evaluated-modes\n"
        "\n"
        "3. You then tried to evaluate the modified array, while still\n"
        "   inside the symbolic operation.\n"
        "\n"
        "This is not permitted. Side effects inside symbolic regions are\n"
        "tracked by Dr.Jit, but they cannot be materialized until the\n"
        "program has exited the outermost symbolic operation.\n"
        "\n"
        "Here is an example of a histogram routine with this flaw:\n"
        "\n"
        "from drjit.auto import UInt32, Float\n"
        "\n"
        "@dr.syntax\n"
        "def histogram(index: UInt32, bin_count: int = 10) -> Float:\n"
        "    hist = dr.zeros(Float, bin_count)\n"
        "    if index < bin_count:\n"
        "        dr.scatter_add(hist, 1, index)\n"
        "        hist /= len(index) # <-- oops\n"
        "    return hist\n"
        "\n"
        "This example can be fixed dedenting the commented line so that\n"
        "it is no longer contained within the symbolic 'if' statement.\n",
        index
    );
}


static void jitc_raise_symbolic_error(const char *func, uint32_t index) {
    jitc_raise(
        "%s(r%u): not permitted.\n\n"
        "You performed an operation that tried to evalute a *symbolic*\n"
        "variable which is not permitted.\n\n"
        "Tracing operations like dr.while_loop(), dr.if_stmt(), dr.switch(),\n"
        "dr.dispatch(), etc., employ such symbolic variables to call code with\n"
        "abstract inputs and record the resulting computation. It is also\n"
        "possible that you used ordinary Python loops/if statements together\n"
        "with the @dr.syntax decorator, which automatically rewrites code to\n"
        "use such tracing operations. The following operations cannot be \n"
        "performed on symbolic variables:\n"
        "\n"
        " - You cannot use dr.eval() or dr.schedule() to evaluate them.\n"
        "\n"
        " - You cannot print() their contents. (But you may use dr.print() to\n"
        "   print them *asynchronously*)\n"
        "\n"
        " - You cannot perform reductions that would require evaluation of the\n"
        "   entire input array (e.g. dr.all(x > 0, axis=None) to check if the\n"
        "   elements of an array are positive).\n"
        "\n"
        " - you cannot convert them to NumPy/PyTorch/TensorFlow/JAX arrays.\n"
        "\n"
        " - You cannot access specific elements of 1D arrays using indexing\n"
        "   operations (this would require the contents to be known.)\n"
        "\n"
        "The common pattern of these limitation is that the contents of symbolic\n"
        "of variables are *simply not known*. Any attempt to access or otherwise\n"
        "reveal their contents is therefore doomed to fail.\n"
        "\n"
        "Symbolic variables can be inconvenient for debugging, which often\n"
        "requires adding print() statements or stepping through a program while\n"
        "investigating intermediate results. If you wish to do this, then you\n"
        "should switch Dr.Jit from *symbolic* into *evaluated* mode.\n"
        "\n"
        "This mode tends to be more expensive in terms of memory storage and\n"
        "bandwidth, which is why it is not enabled by default. Please see the\n"
        "Dr.Jit documentation for more information on symbolic and evaluated\n"
        "evaluation modes:\n\n"
        "https://drjit.readthedocs.io/en/latest/cflow.html#symbolic-versus-evaluated-modes",
        func, index);
}

static void jitc_raise_consumed_error(const char *func, uint32_t index) {
    jitc_raise("%s(r%u): the provided variable of kind \"%s\" can only be "
               "evaluated once and was consumed by a prior operation",
               func, index, var_kind_name[jitc_var(index)->kind]);
}

/// Force-evaluate a variable of type 'literal' or 'undefined'
uint32_t jitc_var_eval_force(uint32_t index, Variable v, void **ptr_out) {
    uint32_t isize = type_size[v.type];

    void *ptr = jitc_malloc((JitBackend) v.backend == JitBackend::CUDA
                                ? AllocType::Device
                                : AllocType::HostAsync,
                            v.size * (size_t) isize);

    if (v.is_literal()) {
        jitc_memset_async((JitBackend) v.backend, ptr, v.size, isize,
                          &v.literal);
    }

    uint32_t result = jitc_var_mem_map((JitBackend) v.backend, (VarType) v.type,
                                       ptr, v.size, 1);

    jitc_log(Debug,
             "jit_var_eval(): %s r%u[%u] = data(" DRJIT_PTR ") [copy of r%u]",
             type_name[v.type], result, v.size, (uintptr_t) ptr, index);

    *ptr_out = ptr;

    return result;
}

uint32_t jitc_var_data(uint32_t index, bool eval_dirty, void **ptr_out) {
    if (index == 0) {
        *ptr_out = nullptr;
        return 0;
    }

    Variable *v = jitc_var(index);
    if (v->is_literal() || v->is_undefined()) {
        return jitc_var_eval_force(index, *v, ptr_out);
    } else if (v->is_evaluated()) {
        if (v->is_dirty() && eval_dirty) {
            jitc_eval(thread_state(v->backend));
            v = jitc_var(index);

            if (unlikely(v->is_dirty()))
                jitc_raise_dirty_error(index);
        }
    } else if (v->is_node()) {
        jitc_var_eval(index);
        v = jitc_var(index);
        if (unlikely(!v->is_evaluated()))
            jitc_fail("jitc_var_data(): evaluation of variable r%u failed!", index);
    } else {
        jitc_fail("jitc_var_data(): unhandled variable r%u!", index);
    }

    *ptr_out = v->data;
    jitc_var_inc_ref(index, v);

    return index;
}

/// Schedule a variable \c index for future evaluation via \ref jit_eval()
int jitc_var_schedule(uint32_t index) {
    if (index == 0)
        return 0;

    Variable *v = jitc_var(index);
    if (unlikely(v->symbolic))
        jitc_raise_symbolic_error("jit_var_schedule", index);

    if (unlikely(v->consumed))
        jitc_raise_consumed_error("jit_var_schedule", index);

    if (v->is_node()) {
        thread_state(v->backend)->scheduled.emplace_back(index, v->counter);
        jitc_log(Debug, "jit_var_schedule(r%u)", index);
        return 1;
    } else if (v->is_dirty()) {
        return 1;
    }

    return 0;
}

/// More aggressive version of the above
uint32_t jitc_var_schedule_force(uint32_t index, int *rv) {
    if (index == 0) {
        *rv = 0;
        return 0;
    }

    Variable *v = jitc_var(index);

    if (unlikely(v->symbolic))
        jitc_raise_symbolic_error("jit_var_schedule_force", index);

    if (unlikely(v->consumed))
        jitc_raise_consumed_error("jit_var_schedule_force", index);

    if (v->is_evaluated()) {
        *rv = v->is_dirty();
    } else {
        jitc_log(Debug, "jit_var_schedule_force(r%u)", index);

        if (v->is_literal() || v->is_undefined()) {
            *rv = 0;
            void *unused = nullptr;
            return jitc_var_eval_force(index, *v, &unused);
        } else if (v->is_node()) {
            thread_state(v->backend)->scheduled.emplace_back(index, v->counter);
            *rv = 1;
        } else {
            *rv = 0;
        }
    }

    jitc_var_inc_ref(index, v);

    return index;
}


/// Evaluate the variable \c index right away if it is unevaluated/dirty.
int jitc_var_eval(uint32_t index) {
    if (!jitc_var_schedule(index))
        return 0;

    Variable *v = jitc_var(index);
    jitc_eval(thread_state(v->backend));

    v = jitc_var(index);
    if (unlikely(v->is_dirty()))
        jitc_raise_dirty_error(index);
    else if (unlikely(!v->is_evaluated() || !v->data))
        jitc_raise("jit_var_eval(): invalid/uninitialized variable r%u!", index);

    return 1;
}

/// Read a single element of a variable and write it to 'dst'
void jitc_var_read(uint32_t index, size_t offset, void *dst) {
    jitc_var_eval(index);

    const Variable *v = jitc_var(index);
    if (v->size == 1)
        offset = 0;
    else if (unlikely(offset >= (size_t) v->size))
        jitc_raise("jit_var_read(): attempted to access entry %zu in an array of "
                   "size %u!", offset, v->size);

    uint32_t isize = type_size[v->type];
    if (v->is_literal() || v->is_undefined()) {
        memcpy(dst, &v->literal, isize);
    } else if (v->is_evaluated()) {
        jitc_memcpy((JitBackend) v->backend, dst,
                    (const uint8_t *) v->data + offset * isize, isize);
    } else {
        jitc_fail("jit_var_read(): unhandled variable type!");
    }
}

/// Reverse of jitc_var_read(). Copy 'dst' to a single element of a variable
uint32_t jitc_var_write(uint32_t index_, size_t offset, const void *src) {
    void *ptr = nullptr;
    Ref index = steal(jitc_var_data(index_, true, &ptr));
    Variable *v = jitc_var(index);

    // Check if it is safe to write directly
    if (v->ref_count > 2) { // 1 from original array, 1 from jitc_var_data() above
        index = steal(jitc_var_copy(index));

        // The above operation may have invalidated 'v'
        v = jitc_var(index);
    }

    if (unlikely(offset >= (size_t) v->size))
        jitc_raise("jit_var_write(): attempted to access entry %zu in an array of "
                   "size %u!", offset, v->size);

    uint32_t isize = type_size[v->type];
    uint8_t *dst = (uint8_t *) v->data + offset * isize;
    jitc_poke((JitBackend) v->backend, dst, src, isize);

    return index.release();
}

/// Register an existing variable with the JIT compiler
uint32_t jitc_var_mem_map(JitBackend backend, VarType type, void *ptr,
                          size_t size, int free) {
    if (unlikely(size == 0))
        return 0;

    jitc_check_size("jit_var_mem_map", size);

    Variable v;
    v.kind = (uint32_t) VarKind::Evaluated;
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
    jitc_log(Debug,
             "jit_var_mem_copy(): %s r%u[%zu] = copy_from(%s, " DRJIT_PTR ")",
             type_name[(int) vtype], index, size, alloc_type_name[(int) atype],
             (uintptr_t) ptr);
    return index;
}

uint32_t jitc_var_copy(uint32_t index) {
    if (index == 0)
        return 0;

    Variable *v = jitc_var(index);
    if (v->is_dirty()) {
        jitc_var_eval(index);
        v = jitc_var(index);
        if (unlikely(v->is_dirty()))
            jitc_raise_dirty_error(index);
    }

    if (unlikely(v->consumed))
        jitc_raise_consumed_error("jitc_var_copy", index);

    uint32_t result;
    VarType vt = (VarType) v->type;
    uint32_t size = v->size;

    if (v->is_evaluated()) {
        JitBackend backend = (JitBackend) v->backend;
        AllocType atype = backend == JitBackend::CUDA ? AllocType::Device
                                                      : AllocType::HostAsync;
        result = jitc_var_mem_copy(backend, atype, vt, v->data, size);
    } else {
        Variable v2;
        v2.type = (uint32_t) vt;
        v2.backend = v->backend;
        v2.symbolic = v->symbolic;
        v2.size = size;

        if (v->is_literal() || v->is_undefined()) {
            v2.kind = v->kind;
            v2.literal = v->literal;
        } else {
            v2.kind = (uint32_t) VarKind::Bitcast;
            v2.dep[0] = index;
            jitc_var_inc_ref(index, v);
        }

        result = jitc_var_new(v2, true);
    }

    jitc_log(Debug, "jit_var_copy(): %s r%u[%u] = r%u", type_name[(int) vt],
             result, size, index);

    return result;
}

uint32_t jitc_var_shrink(uint32_t index, size_t size) {
    if (index == 0 || size == 0)
        return 0;
    Variable *v = jitc_var(index);
    if ((size_t) v->size == size) {
        jitc_var_inc_ref(index, v);
        return index;
    }
    if ((size_t) v->size < size)
        jitc_raise("jit_var_shrink(r%u): requested size (%zu) exceeds current "
                   "size (%u)!", index, size, v->size);

    VarType vt = (VarType) v->type;
    JitBackend backend = (JitBackend) v->backend;

    uint32_t result;
    if (v->is_literal()) {
        result = jitc_var_literal(backend, vt, &v->literal, size, 0);
    } else {
        void *dst_addr = nullptr;
        Ref dst = steal(jitc_var_data(index, false, &dst_addr));

        result = jitc_var_mem_map(backend, vt, dst_addr, size, false);
        jitc_var(result)->dep[3] = index;
        jitc_var_inc_ref(index);
    }

    jitc_log(Debug, "jit_var_shrink(): %s r%u[%zu] = shrink(r%u)",
             type_name[(int) vt], result, size, index);

    return result;
}

uint32_t jitc_var_resize(uint32_t index, size_t size) {
    if (index == 0 && size == 0)
        return 0;

    jitc_check_size("jit_var_resize", size);

    Variable *v = jitc_var(index);
    VarType vt = (VarType) v->type;
    if (unlikely(v->consumed))
        jitc_raise_consumed_error("jitc_var_resize", index);

    if (v->size == size) {
        jitc_var_inc_ref(index, v);
        return index; // Nothing to do
    } else if (v->size != 1 && !v->is_literal()) {
        jitc_raise("jit_var_resize(): variable %u must be scalar or a literal constant!", index);
    }

    if (v->is_dirty()) {
        jitc_eval(thread_state(v->backend));
        v = jitc_var(index);
        if (v->is_dirty())
            jitc_raise_dirty_error(index);
    }

    uint32_t result;
    if (!v->is_evaluated() && v->ref_count == 1) {
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
        v2.kind = (uint32_t) VarKind::Bitcast;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.symbolic = v->symbolic;
        v2.size = (uint32_t) size;
        v2.dep[0] = index;
        jitc_var_inc_ref(index, v);
        result = jitc_var_new(v2, true);
    }

    jitc_log(Debug, "jit_var_resize(): %s r%u[%zu] = resize(r%u)",
             type_name[(int) vt], result, size, index);

    return result;
}

/// Migrate a variable to a different flavor of memory
uint32_t jitc_var_migrate(uint32_t src_index, AllocType dst_type) {
    if (src_index == 0)
        return 0;

    Variable *v = jitc_var(src_index);
    JitBackend backend = (JitBackend) v->backend;

    if (v->is_literal() || v->is_undefined()) {
        size_t size = v->size;
        void *ptr = jitc_malloc(dst_type, type_size[v->type] * size);
        v = jitc_var(src_index);

        if (v->is_literal()) {
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
        }

        return jitc_var_mem_map(backend, (VarType) v->type, ptr, v->size, 1);
    }

    if (!v->is_evaluated() || v->is_dirty()) {
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
        v2.kind = (uint32_t) VarKind::Evaluated;
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

uint32_t jitc_var_mask_default(JitBackend backend, size_t size) {
    jitc_check_size("jit_var_mask_default", size);

    if (backend == JitBackend::CUDA) {
        bool value = true;
        return jitc_var_literal(backend, VarType::Bool, &value, size, 0);
    } else {
        // Ignore SIMD lanes that lie beyond the end of the range
        Ref counter = steal(jitc_var_counter(backend, size, false));
        Variable *v_counter = jitc_var(counter);
        return jitc_var_new_node_1(backend, VarKind::DefaultMask, VarType::Bool,
                                   v_counter->size, v_counter->symbolic,
                                   counter, v_counter);
    }
}

/// Return the 'VariableExtra' record associated with a variable (or create it)
VariableExtra *jitc_var_extra(Variable *v) {
    State &st = ::state;

    uint32_t index = v->extra;
    if (!index) {
        UnusedPQ &unused = st.unused_extra;
        if (unused.empty()) {
            index = (uint32_t) st.extra.size();
            st.extra.emplace_back();
        } else {
            index = unused.top();
            unused.pop();
        }
        v->extra = index;
        st.extra[index] = VariableExtra();
    }
    return &st.extra[index];
}

uint32_t jitc_var_mask_peek(JitBackend backend) {
    std::vector<uint32_t> &stack = thread_state(backend)->mask_stack;

    if (stack.empty()) {
        return 0;
    } else {
        uint32_t index = stack.back();
        jitc_var_inc_ref(index);
        return index;
    }
}

void jitc_var_mask_push(JitBackend backend, uint32_t index) {
    jitc_log(Debug, "jit_var_mask_push(r%u)", index);
    jitc_var_inc_ref(index);
    thread_state(backend)->mask_stack.push_back(index);
}

uint32_t jitc_var_mask_apply(uint32_t index, uint32_t size) {
    const Variable *v = jitc_var(index);
    JitBackend backend = (JitBackend) v->backend;

    if ((VarType) v->type != VarType::Bool)
        jitc_raise("jit_var_mask_apply(): the mask parameter was not a boolean array!");

    std::vector<uint32_t> &stack = thread_state(backend)->mask_stack;
    Ref mask;
    if (!stack.empty()) {
        uint32_t index_2 = stack.back(),
                 size_2  = jitc_var(index_2)->size;

        // Use mask from the mask stack if its size is compatible
        if (size == 1 || size_2 == 1 || size_2 == size)
            mask = borrow(index_2);
    }

    if (!mask && backend == JitBackend::LLVM)
        mask = steal(jitc_var_mask_default(backend, size));

    uint32_t result;
    // Combine given mask with mask stack
    if (mask) {
        result = jitc_var_and(mask, index);
    } else {
        // Hold a temporary reference so that the following operation does not mutate 'index'
        Ref temp = borrow(index);
        result = jitc_var_resize(index, size);
    }

    return result;
}

void jitc_var_mask_pop(JitBackend backend) {
    std::vector<uint32_t> &stack = thread_state(backend)->mask_stack;
    if (unlikely(stack.empty()))
        jitc_raise("jit_var_mask_pop(): stack underflow!");

    jitc_log(Debug, "jit_var_mask_pop()");

    uint32_t index = stack.back();
    stack.pop_back();
    jitc_var_dec_ref(index);
}

/// Return an implicit mask for operations within a virtual function call
uint32_t jitc_var_call_mask(JitBackend backend) {
    if (backend == JitBackend::LLVM) {
        return jitc_var_new_node_0(backend, VarKind::CallMask, VarType::Bool, 1, 1);
    } else {
        bool value = true;
        return jitc_var_literal(backend, VarType::Bool, &value, 1, 0);
    }
}

bool jitc_var_any(uint32_t index) {
    if (!index)
        return false;

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
    if (!index)
        return true;

    const Variable *v = jitc_var(index);

    if (unlikely((VarType) v->type != VarType::Bool))
        jitc_raise("jit_var_all(r%u): requires a boolean array as input!", index);

    if (v->is_literal())
        return (bool) v->literal;

    if (jitc_var_eval(index))
        v = jitc_var(index);

    return jitc_all((JitBackend) v->backend, (uint8_t *) v->data, v->size);
}

uint32_t jitc_var_compress(uint32_t index) {
    if (!index)
        return 0;

    const Variable *v = jitc_var(index);

    if (unlikely((VarType) v->type != VarType::Bool))
        jitc_raise("jit_var_compress(r%u): requires a boolean array as input!", index);

    if (jitc_var_eval(index))
        v = jitc_var(index);

    JitBackend backend = (JitBackend) v->backend;
    const uint32_t size_in = v->size;
    const uint8_t *ptr = (const uint8_t *) v->data;

    uint32_t *indices_out = (uint32_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync,
        size_in * sizeof(uint32_t));

    uint32_t size_out = jitc_compress(backend, ptr, size_in, indices_out);
    if (size_out > 0) {
        return jitc_var_mem_map(backend, VarType::UInt32, indices_out, size_out, 1);
    } else {
        jitc_free(indices_out);
        return 0;
    }
}

uint32_t jitc_var_block_reduce(ReduceOp op, uint32_t index, uint32_t block_size, int symbolic) {
    if (index == 0) {
        return 0;
    } else if (block_size == 1) {
        jitc_var_inc_ref(index);
        return index;
    }

    const Variable *v = jitc_var(index);

    JitBackend backend = (JitBackend) v->backend;
    VarType vt = (VarType) v->type;
    uint32_t size = v->size,
             reduced = size / block_size;

    if (reduced * block_size != size)
        jitc_raise("jit_var_block_reduce(r%u): variable size (%u) must be an integer "
                   "multiple of 'block_size' (%u)", index, size, block_size);

    if (symbolic == -1) {
        bool can_scatter_reduce = jitc_can_scatter_reduce(backend, vt, op),
             is_evaluated = v->is_evaluated(),
             can_evaluate =
                 !v->symbolic && !(backend == JitBackend::CUDA &&
                                   (block_size & (block_size - 1)) != 0);

        if (can_scatter_reduce != can_evaluate)
            symbolic = can_scatter_reduce; // one strategy admissible
        else if (!can_scatter_reduce) // no strategy
            jitc_raise("jit_var_block_reduce(): neither evaluated nor symbolic "
                       "strategies are available. Please see "
                       "https://drjit.readthedocs.io/en/latest/"
                       "reference.html#drjit.block_reduce for details");
        else if (is_evaluated)
            symbolic = false;
        else // choose based on size
            symbolic = size * type_size[(int) vt] >= 1024u * 1024u * 1024u; // 1 GiB
    }

    if (symbolic == 1) {
        uint64_t identity = jitc_reduce_identity(op, vt);

        uint64_t bsize_u64 = block_size, one_u64 = 1;

        Ref counter = steal(jitc_var_counter(backend, size, true)),
            bsize   = steal(jitc_var_literal(backend, VarType::UInt32, &bsize_u64, 1, 0)),
            offset  = steal(jitc_var_div(counter, bsize)),
            t_mask  = steal(jitc_var_literal(backend, VarType::Bool, &one_u64, 1, 0)),
            target  = steal(jitc_var_literal(backend, vt, &identity, reduced, 0));

        return jitc_var_scatter(target, index, offset, t_mask, op, ReduceMode::Auto);
    } else if (symbolic == 0) {
        jitc_var_eval(index);

        jitc_log(Debug, "jit_var_block_reduce(r%u, block_size=%u)", index, block_size);

        void *out =
            jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                    : AllocType::HostAsync,
                        reduced * (size_t) type_size[(int) vt]);
        Ref out_v = steal(jitc_var_mem_map(backend, vt, out, reduced, 1));
        jitc_block_reduce(backend, vt, op, jitc_var(index)->data, size, block_size, out);

        return out_v.release();
    } else {
        jitc_raise("jit_var_block_reduce(): 'symbolic' must equal -1, 0, or -1");
    }
}

uint32_t jitc_var_tile(uint32_t index, uint32_t block_size) {
    const Variable *v = jitc_var(index);

    JitBackend backend = (JitBackend) v->backend;
    size_t size = v->size,
           out_size = size * block_size;

    jitc_check_size("jit_var_tile", out_size);

    uint64_t bsize_u64 = block_size, one_u64 = 1;
    Ref counter = steal(jitc_var_counter(backend, out_size, true)),
        bsize   = steal(jitc_var_literal(backend, VarType::UInt32, &bsize_u64, 1, 0)),
        offset  = steal(jitc_var_div(counter, bsize)),
        t_mask  = steal(jitc_var_literal(backend, VarType::Bool, &one_u64, 1, 0));

    return jitc_var_gather(index, offset, t_mask);
}

template <typename T> static void jitc_var_reduce_scalar(uint32_t size, void *ptr) {
    T value;
    memcpy(&value, ptr, sizeof(T));
    value = T(value * T(size));
    memcpy(ptr, &value, sizeof(T));
}

void jitc_var_set_self(JitBackend backend, uint32_t value, uint32_t index) {
    ThreadState *ts = thread_state(backend);

    if (ts->call_self_index) {
        jitc_var_dec_ref(ts->call_self_index);
        ts->call_self_index = 0;
    }

    ts->call_self_value = value;

    if (value) {
        if (index) {
            jitc_var_inc_ref(index);
            ts->call_self_index = index;
        } else {
            Variable v;
            v.kind = VarKind::CallSelf;
            v.backend = (uint32_t) backend;
            v.size = 1u;
            v.type = (uint32_t) VarType::UInt32;
            v.symbolic = true;
            ts->call_self_index = jitc_var_new(v, true);
        }
    }
}

void jitc_var_self(JitBackend backend, uint32_t *value, uint32_t *index) {
    ThreadState *ts = thread_state(backend);
    *value = ts->call_self_value;
    *index = ts->call_self_index;
}

/// Return the identity element for different horizontal reductions
uint64_t jitc_reduce_identity(ReduceOp reduce_op, VarType vt) {
    switch (reduce_op) {
        case ReduceOp::Or:
        case ReduceOp::Add: return 0; break;
        case ReduceOp::And: return type_all_ones[(int) vt]; break;
        case ReduceOp::Mul: return type_one[(int) vt]; break;
        case ReduceOp::Min: return type_max[(int) vt]; break;
        case ReduceOp::Max: return type_min[(int) vt]; break;
        default: jitc_fail("jitc_reduce_identity(): unsupported reduction type!");
                 return 0;
    }
}

uint32_t jitc_var_reduce(JitBackend backend, VarType vt, ReduceOp reduce_op,
                         uint32_t index) {
    if (unlikely(reduce_op == ReduceOp::And || reduce_op == ReduceOp::Or))
        jitc_raise("jit_var_reduce(): does not support And/Or operation!");

    if (unlikely(index == 0)) {
        if (backend == JitBackend::None || vt == VarType::Void)
            jitc_raise("jit_var_reduce(): missing backend/type information!");

        uint64_t identity = jitc_reduce_identity(reduce_op, vt);
        return jitc_var_literal(backend, vt, &identity, 1, 0);
    }

    const Variable *v = jitc_var(index);

    if (vt == VarType::Void)
        vt = (VarType) v->type;
    if (backend == JitBackend::None)
        backend = (JitBackend) v->backend;

    if ((VarType) v->type != vt || (JitBackend) v->backend != backend)
        jitc_raise("jit_var_reduce(): variable mismatch!");

    if (v->is_literal()) {
        uint64_t value = v->literal;
        uint32_t size = v->size;

        // Tricky cases
        if (size != 1 && reduce_op == ReduceOp::Add) {
            using half = drjit::half;
            switch ((VarType) v->type) {
                case VarType::Int8:    jitc_var_reduce_scalar<int8_t>  (size, &value); break;
                case VarType::UInt8:   jitc_var_reduce_scalar<uint8_t> (size, &value); break;
                case VarType::Int16:   jitc_var_reduce_scalar<int16_t> (size, &value); break;
                case VarType::UInt16:  jitc_var_reduce_scalar<uint16_t>(size, &value); break;
                case VarType::Int32:   jitc_var_reduce_scalar<int32_t> (size, &value); break;
                case VarType::UInt32:  jitc_var_reduce_scalar<uint32_t>(size, &value); break;
                case VarType::Int64:   jitc_var_reduce_scalar<int64_t> (size, &value); break;
                case VarType::UInt64:  jitc_var_reduce_scalar<uint64_t>(size, &value); break;
                case VarType::Float16: jitc_var_reduce_scalar<half>    (size, &value); break;
                case VarType::Float32: jitc_var_reduce_scalar<float>   (size, &value); break;
                case VarType::Float64: jitc_var_reduce_scalar<double>  (size, &value); break;
                default: jitc_raise("jit_var_reduce(): unsupported operand type!");
            }
        } else if (size != 1 && reduce_op == ReduceOp::Mul) {
            jitc_raise("jit_var_reduce(): ReduceOp::Mul is not supported for vector values!");
        }

        return jitc_var_literal(backend, vt, &value, 1, 0);
    }

    jitc_log(Debug, "jit_var_reduce(r%u, reduce_op=%s)", index,
             red_name[(int) reduce_op]);

    if (jitc_var_eval(index))
        v = jitc_var(index);

    uint8_t *values = (uint8_t *) v->data;
    uint32_t size = v->size;

    void *data =
        jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                : AllocType::HostAsync,
                    (size_t) type_size[(int) vt]);
    jitc_reduce(backend, vt, reduce_op, values, size, data);
    return jitc_var_mem_map(backend, vt, data, 1, 1);
}

uint32_t jitc_var_reduce_dot(uint32_t index_1,
                             uint32_t index_2) {
    if (index_1 == 0 && index_2 == 0)
        return 0;

    if ((index_1 == 0) != (index_2 == 0))
        jitc_raise("jitc_var_reduce_dot(): one of the operands is empty!");

    const Variable *v1 = jitc_var(index_1),
                   *v2 = jitc_var(index_2);

    VarType vt = (VarType) v1->type;
    JitBackend backend = (JitBackend) v1->backend;

    if (v1->backend != v2->backend)
        jitc_raise("jitc_var_reduce_dot(): incompatible backends!");

    if (v1->type != v2->type)
        jitc_raise("jitc_var_reduce_dot(): incompatible types!");

    if (jitc_is_float(v1) && v1->size == v2->size &&
        (v1->is_evaluated() || v2->is_evaluated())) {
        uint32_t size = v1->size;

        // Fast path
        bool eval = jitc_var_eval(index_1);
        eval |= jitc_var_eval(index_2);
        if (eval) {
            v1 = jitc_var(index_1);
            v2 = jitc_var(index_2);
        }

        void *ptr_1 = v1->data,
             *ptr_2 = v2->data,
             *data = jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                             : AllocType::HostAsync,
                                 (size_t) type_size[(int) vt]);

        jitc_reduce_dot(backend, vt, ptr_1, ptr_2, size, data);

        return jitc_var_mem_map(backend, vt, data, 1, 1);
    } else {
        // General case
        Ref tmp = steal(jitc_var_mul(index_1, index_2));
        return jitc_var_reduce(backend, vt, ReduceOp::Add, tmp);
    }
}

uint32_t jitc_var_prefix_sum(uint32_t index, bool exclusive) {
    if (!index)
        return index;

    const Variable *v = jitc_var(index);
    VarType vt = (VarType) v->type;
    JitBackend backend = (JitBackend) v->backend;
    uint32_t size = v->size;

    if (v->is_literal()) {
        Ref ctr = steal(jitc_var_counter(backend, size, true));
        if (vt != VarType::UInt32)
            ctr = steal(jitc_var_cast(ctr, vt, false));

        if (!exclusive) {
            Ref one = steal(jitc_var_literal(backend, vt, &type_one[(int) vt], 1, 0));
            ctr = steal(jitc_var_add(one, ctr));
        }

        return jitc_var_mul(ctr, index);
    } else if (!v->is_evaluated() || v->is_dirty()) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }

    const void *data_in = v->data;

    void *data_out =
        jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                : AllocType::HostAsync,
                    (size_t) type_size[(int) vt] * size);

    Ref result = steal(jitc_var_mem_map(backend, vt, data_out, size, 1));

    jitc_prefix_sum(backend, vt, exclusive, data_in, size, data_out);
    return result.release();
}


std::pair<uint32_t, uint32_t> jitc_var_expand(uint32_t index, ReduceOp reduce_op) {
    Variable *v = jitc_var(index);
    VarType vt = (VarType) v->type;

    uint32_t tsize = ::type_size[v->type],
             workers = pool_size() + 1,
             size = v->size;

    // 1 cache line per worker for scalar targets, otherwise be a bit more reasonable
    uint32_t replication_per_worker = size == 1u ? (64u / tsize) : 1u,
             index_scale = replication_per_worker * size;

    if (workers == 1) {
        jitc_var_inc_ref(index);
        return { index, 1 };
    }

    if (v->reduce_op == (uint32_t) reduce_op) {
        jitc_var_inc_ref(index);
        return { index, index_scale };
    }

    size_t new_size = size * (size_t) replication_per_worker * (size_t) workers;
    if (new_size > 0xffffffffull)
        jitc_raise("jitc_var_expand(): scatter with drjit.ReduceMode.Expand is "
                   "not possible, as this would expand the array size beyond 4 "
                   "billion entries!");

    uint64_t identity = jitc_reduce_identity(reduce_op, vt);

    Ref dst;
    void *dst_addr = nullptr;
    dst = steal(jitc_var_literal(JitBackend::LLVM, vt, &identity, new_size, 0));
    dst = steal(jitc_var_data(dst, false, &dst_addr));

    v = jitc_var(index);
    if (!v->is_literal() || v->literal != identity) {
        void *src_addr = nullptr;
        Ref src = steal(jitc_var_data(index, false, &src_addr));
        jitc_memcpy_async(JitBackend::LLVM, dst_addr, src_addr,
                          size * tsize);
    }

    Variable *v2 = jitc_var(dst);
    v2->reduce_op = (uint32_t) reduce_op;
    v2->size = size;

    jitc_log(Debug, "jit_var_expand(): %s r%u[%zu] = expand(r%u, factor=%zu)",
             type_name[(int) vt], (uint32_t) dst, new_size, index,
             new_size / size);

    return { dst.release(), index_scale };
}

void jitc_var_reduce_expanded(uint32_t index) {
    Variable *v = jitc_var(index);

    if ((ReduceOp) v->reduce_op == ReduceOp::Identity)
        return;

    uint32_t workers = pool_size() + 1,
             tsize = type_size[v->type],
             size = v->size;

    // 1 cache line per worker for scalar targets, otherwise be a bit more reasonable
    uint32_t replication_per_worker = size == 1u ? (64u / tsize) : 1u;

    jitc_reduce_expanded(
        (VarType) v->type,
        (ReduceOp) v->reduce_op,
        v->data,
        replication_per_worker * workers,
        size
    );

    v->reduce_op = (uint32_t) ReduceOp::Identity;
}

/// Return a human-readable summary of registered variables
const char *jitc_var_whos() {
    var_buffer.clear();
    var_buffer.put("\n  ID       Type       Status     Refs       Size      Storage   Label");
    var_buffer.put("\n  ========================================================================\n");

    size_t mem_size_evaluated = 0,
           mem_size_unevaluated = 0,
           variable_counter = 0;

    for (size_t index = 1; index < state.variables.size(); ++index) {
        const Variable &v = state.variables[index];
        if (v.ref_count == 0 && v.ref_count_se == 0)
            continue;

        size_t mem_size = (size_t) v.size * (size_t) type_size[v.type];

        var_buffer.fmt("  %-8zu %s %-5s ", index,
                       (JitBackend) v.backend == JitBackend::CUDA ? "cuda"
                                                                  : "llvm",
                       type_name_short[v.type]);

        if (v.is_literal()) {
            var_buffer.put("const.     ");
        } else if (v.is_evaluated()) {
            auto it = state.alloc_used.find((uintptr_t) v.data);
            if (unlikely(it == state.alloc_used.end())) {
                if (!v.retain_data)
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

        const char *label = jitc_var_label((uint32_t) index);

        var_buffer.fmt(" %3u %10u %12s   %s\n", (uint32_t) v.ref_count,
                       v.size, jitc_mem_string(mem_size), label ? label : "");

        if (v.is_evaluated())
            mem_size_evaluated += mem_size;
        else
            mem_size_unevaluated += mem_size;

        variable_counter++;
    }
    if (variable_counter == 0)
        var_buffer.put("                       -- No variables registered --\n");

    constexpr size_t LVNBucketSize = sizeof(tsl::detail_robin_hash::bucket_entry<LVNMap::value_type, false>);

    var_buffer.put("  ========================================================================\n\n");
    var_buffer.put("  JIT compiler\n");
    var_buffer.put("  ============\n");
    var_buffer.fmt("   - Storage           : %s on device, ",
               jitc_mem_string(mem_size_evaluated));
    var_buffer.fmt("%s unevaluated.\n",
               jitc_mem_string(mem_size_unevaluated));
    var_buffer.fmt("   - Variables created : %zu (peak: %zu, table size: %s).\n",
               state.variable_counter, state.variables.size(),
               jitc_mem_string(
                   state.variables.capacity() * sizeof(Variable)+
                   state.lvn_map.bucket_count() * LVNBucketSize));
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
    var_buffer.clear();
    var_buffer.put("digraph {\n"
                   "    rankdir=TB;\n"
                   "    graph [dpi=50 fontname=Consolas];\n"
                   "    node [shape=record fontname=Consolas];\n"
                   "    edge [fontname=Consolas];\n");

    size_t current_hash = 0, current_depth = 1;

    for (size_t index = 1; index < state.variables.size(); ++index) {
        const Variable &v = state.variables[index];
        if (v.ref_count == 0 && v.ref_count_se == 0)
            continue;

        const char *label = nullptr;
        if (v.extra)
            label = state.extra[v.extra].label;

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

        var_buffer.put(' ', 4 * current_depth);
        var_buffer.put_u32((uint32_t) index);
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

        if (v.is_literal()) {
            var_buffer.put("Literal: ");
            jitc_value_print(&v, true);
            color = "gray90";
        } else if (v.is_evaluated()) {
            if (v.is_dirty()) {
                var_buffer.put("Evaluated (dirty)");
                color = "salmon";
            } else {
                var_buffer.put("Evaluated");
                color = "lightblue2";
            }
        } else {
            const char *name = var_kind_name[v.kind];
            var_buffer.put(name, strlen(name));
        }

        if (v.symbolic && !color)
            color = "yellow";
        if (labeled && !color)
            color = "wheat";

        var_buffer.fmt("|{Type: %s %s|Size: %u}|{r%zu|Refs: %u}}",
            (JitBackend) v.backend == JitBackend::CUDA ? "cuda" : "llvm",
            type_name_short[v.type], v.size, index,
            (uint32_t) v.ref_count);

        var_buffer.put("}\"");
        if (color)
            var_buffer.fmt(" fillcolor=%s style=filled", color);
        var_buffer.put("];\n");
    }

    for (size_t i = current_depth - 1; i > 0; --i) {
        var_buffer.put(' ', 4 * i);
        var_buffer.put("}\n");
    }

    for (size_t index = 1; index < state.variables.size(); ++index) {
        const Variable &v = state.variables[index];
        if (v.ref_count == 0 && v.ref_count_se == 0)
            continue;

        int n_dep = 0;
        for (uint32_t i = 0; i < 4; ++i)
            n_dep += v.dep[i] ? 1 : 0;

        uint32_t edge_index = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            if (!v.dep[i])
                continue;

            var_buffer.fmt("    %u -> %zu", v.dep[i], index);
            bool special = i == 3 && v.dep[2] == 0;
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
    }

    var_buffer.put(
        "    subgraph cluster_legend {\n"
        "        label=\"Legend\";\n"
        "        l5 [style=filled fillcolor=yellow label=\"Symbolic\"];\n"
        "        l4 [style=filled fillcolor=yellowgreen label=\"Special\"];\n"
        "        l3 [style=filled fillcolor=salmon label=\"Dirty\"];\n"
        "        l2 [style=filled fillcolor=lightblue2 label=\"Evaluated\"];\n"
        "        l1 [style=filled fillcolor=wheat label=\"Labeled\"];\n"
        "        l0 [style=filled fillcolor=gray90 label=\"Constant\"];\n"
        "    }\n"
        "}\n");

    return var_buffer.get();
}

// Intense internal instrumentation to catch undefined behavior
#if defined(DRJIT_SANITIZE_INTENSE)
void jitc_sanitation_checkpoint() {
    std::vector<Variable> variables_copy(state.variables);
    state.variables = std::move(variables_copy);
}
#endif
