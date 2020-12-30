/*
    src/var.cpp -- Operations for creating and querying variables

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "var.h"
#include "internal.h"
#include "log.h"
#include "eval.h"
#include "util.h"


/// Descriptive names for the various variable types
const char *var_type_name[(int) VarType::Count] {
    "void",   "bool",  "int8",   "uint8",   "int16",   "uint16",  "int32",
    "uint32", "int64", "uint64", "pointer", "float16", "float32", "float64"
};

/// Descriptive names for the various variable types (extra-short version)
const char *var_type_name_short[(int) VarType::Count] {
    "vd ", "msk", "i8",  "u8",  "i16", "u16", "i32",
    "u32", "i64", "u64", "ptr", "f16", "f32", "f64"
};

/// CUDA PTX type names
const char *var_type_name_ptx[(int) VarType::Count] {
    "???", "pred", "s8",  "u8",  "s16", "u16", "s32",
    "u32", "s64",  "u64", "u64", "f16", "f32", "f64"
};

/// CUDA PTX type names (binary view)
const char *var_type_name_ptx_bin[(int) VarType::Count] {
    "???", "pred", "b8",  "b8",  "b16", "b16", "b32",
    "b32", "b64",  "b64", "b64", "b16", "b32", "b64"
};

/// LLVM IR type names (does not distinguish signed vs unsigned)
const char *var_type_name_llvm[(int) VarType::Count] {
    "???", "i1",  "i8",  "i8",   "i16",   "i16",   "i32",
    "i32", "i64", "i64", "i64", "half", "float", "double"
};

/// Double size integer arrays for mulhi()
const char *var_type_name_llvm_big[(int) VarType::Count] {
    "???", "???",  "i16",  "i16", "i32", "i32", "i64",
    "i64", "i128", "i128", "???", "???", "???", "???"
};

/// Abbreviated LLVM IR type names
const char *var_type_name_llvm_abbrev[(int) VarType::Count] {
    "???", "i1",  "i8",  "i8",  "i16", "i16", "i32",
    "i32", "i64", "i64", "i64", "f16", "f32", "f64"
};

/// LLVM IR type names (binary view)
const char *var_type_name_llvm_bin[(int) VarType::Count] {
    "???", "i1",  "i8",  "i8",  "i16", "i16", "i32",
    "i32", "i64", "i64", "i64", "i16", "i32", "i64"
};

/// LLVM/CUDA register name prefixes
const char *var_type_prefix[(int) VarType::Count] {
    "%u", "%p", "%b", "%b", "%w", "%w", "%r",
    "%r", "%rd", "%rd", "%rd", "%h", "%f", "%d"
};

/// Maps types to byte sizes
const uint32_t var_type_size[(int) VarType::Count] {
    0, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 2, 4, 8
};

/// String version of the above
const char *var_type_size_str[(int) VarType::Count] {
    "0", "1", "1", "1", "2", "2", "4",
    "4", "8", "8", "8", "2", "4", "8"
};

/// Temporary string buffer for miscellaneous variable-related tasks
Buffer var_buffer(0);

/// Cleanup handler, called when the internal/external reference count reaches zero
void jitc_var_free(uint32_t index, Variable *v) {
    jitc_trace("jit_var_free(%u)", index);

    if (v->data) {
        // Release GPU memory
        if (!v->retain_data)
            jitc_free(v->data);
    } else {
        // Unevaluated variable, drop from CSE cache
        jitc_cse_drop(index, v);
    }

    // Free IR string if needed
    if (unlikely(v->free_stmt))
        free(v->stmt);

    uint32_t dep[4];
    memcpy(dep, v->dep, sizeof(uint32_t) * 4);
    bool extra = v->extra;

    // Remove from hash table. 'v' should not be accessed from here on.
    state.variables.erase(index);

    // Decrease reference count of dependencies
    for (int i = 0; i < 4; ++i)
        jitc_var_dec_ref_int(dep[i]);

    if (unlikely(extra)) {
        auto it = state.extra.find(index);
        if (it == state.extra.end())
            jitc_fail("jit_var_free(): entry in 'extra' hash table not found!");
        Extra extra = it.value();
        state.extra.erase(it);

        // Free descriptive label
        free(extra.label);

        // Notify callback that the variable is being freed
        if (extra.free_callback) {
            unlock_guard guard(state.mutex);
            extra.free_callback(extra.callback_payload);
        }

        // Decrease reference counts of extra references if needed
        if (extra.dep_count) {
            for (uint32_t i = 0; i < extra.dep_count; ++i)
                jitc_var_dec_ref_int(extra.dep[i]);
            free(extra.dep);
        }

        // If jitc_vcall() was invoked on this variable, free bucket list
        if (extra.vcall_bucket_count) {
            for (uint32_t i = 0; i < extra.vcall_bucket_count; ++i)
                jitc_var_dec_ref_ext(extra.vcall_buckets[i].index);
            jitc_free(extra.vcall_buckets);
        }
    }
}

/// Access a variable by ID, terminate with an error if it doesn't exist
Variable *jitc_var(uint32_t index) {
    auto it = state.variables.find(index);
    if (unlikely(it == state.variables.end()))
        jitc_fail("jit_var(%u): unknown variable!", index);
    return &it.value();
}

/// Increase the external reference count of a given variable
void jitc_var_inc_ref_ext(uint32_t index, Variable *v) noexcept(true) {
    v->ref_count_ext++;
    jitc_trace("jit_var_inc_ref_ext(%u): %u", index, v->ref_count_ext);
}

/// Increase the external reference count of a given variable
void jitc_var_inc_ref_ext(uint32_t index) noexcept(true) {
    if (index != 0)
        jitc_var_inc_ref_ext(index, jitc_var(index));
}

/// Increase the internal reference count of a given variable
void jitc_var_inc_ref_int(uint32_t index, Variable *v) noexcept(true) {
    v->ref_count_int++;
    jitc_trace("jit_var_inc_ref_int(%u): %u", index, v->ref_count_int);
}

/// Increase the internal reference count of a given variable
void jitc_var_inc_ref_int(uint32_t index) noexcept(true) {
    if (index != 0)
        jitc_var_inc_ref_int(index, jitc_var(index));
}

/// Decrease the external reference count of a given variable
void jitc_var_dec_ref_ext(uint32_t index, Variable *v) noexcept(true) {
    if (unlikely(v->ref_count_ext == 0))
        jitc_fail("jit_var_dec_ref_ext(): variable %u has no external references!", index);

    jitc_trace("jit_var_dec_ref_ext(%u): %u", index, v->ref_count_ext - 1);
    v->ref_count_ext--;

    if (v->ref_count_ext == 0 && v->ref_count_int == 0)
        jitc_var_free(index, v);
}

/// Decrease the external reference count of a given variable
void jitc_var_dec_ref_ext(uint32_t index) noexcept(true) {
    if (index != 0)
        jitc_var_dec_ref_ext(index, jitc_var(index));
}

/// Decrease the internal reference count of a given variable
void jitc_var_dec_ref_int(uint32_t index, Variable *v) noexcept(true) {
    if (unlikely(v->ref_count_int == 0))
        jitc_fail("jit_var_dec_ref_int(): variable %u has no internal references!", index);

    jitc_trace("jit_var_dec_ref_int(%u): %u", index, v->ref_count_int - 1);
    v->ref_count_int--;

    if (v->ref_count_ext == 0 && v->ref_count_int == 0)
        jitc_var_free(index, v);
}

/// Decrease the internal reference count of a given variable
void jitc_var_dec_ref_int(uint32_t index) noexcept(true) {
    if (index != 0)
        jitc_var_dec_ref_int(index, jitc_var(index));
}

/// Remove a variable from the cache used for common subexpression elimination
void jitc_cse_drop(uint32_t index, const Variable *v) {
    VariableKey key(*v);
    CSECache &cache = thread_state(v->backend)->cse_cache;
    auto it = cache.find(key);
    if (it != cache.end() && it.value() == index)
        cache.erase(it);
}

/// Query the pointer variable associated with a given variable
void *jitc_var_ptr(uint32_t index) {
    return index == 0u ? nullptr : jitc_var(index)->data;
}

/// Query the size of a given variable
uint32_t jitc_var_size(uint32_t index) {
    return jitc_var(index)->size;
}

/// Query the type of a given variable
VarType jitc_var_type(uint32_t index) {
    return (VarType) jitc_var(index)->type;
}

/// Query the descriptive label associated with a given variable
const char *jitc_var_label(uint32_t index) {
    ExtraMap::iterator it = state.extra.find(index);
    return it != state.extra.end() ? it.value().label : nullptr;
}

/// Assign a descriptive label to a given variable
void jitc_var_set_label(uint32_t index, const char *label) {
    Variable *v = jitc_var(index);

    jitc_log(Debug, "jit_var_set_label(%u): \"%s\"", index,
            label ? label : "(null)");

    v->extra = true;
    Extra &extra = state.extra[index];
    free(extra.label);
    extra.label = label ? strdup(label) : nullptr;
}


/// Append the given variable to the instruction trace and return its ID
uint32_t jitc_var_new(Variable &v, bool disable_cse) {
    ThreadState *ts = thread_state(v.backend);

    disable_cse |= v.data || (VarType) v.type == VarType::Void;

    // Check if this exact statement already exists ..
    CSECache::iterator key_it;
    bool cse_key_inserted = false;
    if (!disable_cse)
        std::tie(key_it, cse_key_inserted) =
            ts->cse_cache.try_emplace(VariableKey(v), 0);

    uint32_t index;
    Variable *vo;

    if (likely(disable_cse || cse_key_inserted)) {
        // .. nope, it is new.
        VariableMap::iterator var_it;
        bool var_inserted;
        do {
            index = state.variable_index++;

            if (unlikely(index == 0)) // overflow
                index = state.variable_index++;

            std::tie(var_it, var_inserted) =
                state.variables.try_emplace(index, v);
        } while (!var_inserted);

        if (cse_key_inserted)
            key_it.value() = index;

        vo = &var_it.value();
    } else {
        // .. found a match! Deallocate 'v'.
        if (v.free_stmt)
            free(v.stmt);

        for (int i = 0; i < 4; ++i)
            jitc_var_dec_ref_int(v.dep[i]);

        index = key_it.value();
        vo = jitc_var(index);
    }

    jitc_var_inc_ref_ext(index, vo);

    if (unlikely(std::max(state.log_level_stderr, state.log_level_callback) >=
                 LogLevel::Debug)) {
        var_buffer.clear();
        var_buffer.fmt("jit_var_new(%u", index);

        uint32_t n_dep = 0;
        for (int i = 0; i < 4; ++i) {
            if (v.dep[i])
                n_dep = i + 1;
        }
        for (uint32_t i = 0; i < n_dep; ++i)
            var_buffer.fmt("%s%u", i == 0 ? " <- " : ", ", v.dep[i]);
        var_buffer.fmt("): %s[%u] = ", var_type_name[v.type], v.size);

        if (v.literal) {
            float f; double d;
            switch ((VarType) v.type) {
                case VarType::Float32:
                    memcpy(&f, &v.value, sizeof(float));
                    var_buffer.fmt("%g", (double) f);
                    break;

                case VarType::Float64:
                    memcpy(&d, &v.value, sizeof(double));
                    var_buffer.fmt("%g", d);
                    break;

                case VarType::Int8:
                case VarType::Int16:
                case VarType::Int32:
                case VarType::Int64:
                    var_buffer.fmt("%lli", (long long) v.value);
                    break;

                default:
                    var_buffer.fmt("%llu", (long long) v.value);
                    break;
            }
        } else if (v.data) {
            var_buffer.fmt(ENOKI_PTR, (uintptr_t) v.data);
        } else if (v.stmt) {
            var_buffer.put(v.stmt, strlen(v.stmt));
        }

        if (!disable_cse && !cse_key_inserted)
            var_buffer.put(" (reused)");

        jitc_log(Debug, "%s", var_buffer.get());
    }

    return index;
}

uint32_t jitc_var_new_literal(JitBackend backend, VarType type,
                              const void *value, uint32_t size,
                              int eval) {
    if (unlikely(size == 0))
        return 0;

    if (likely(eval == 0)) {
        Variable v;
        memcpy(&v.value, value, var_type_size[(uint32_t) type]);
        v.type = (uint32_t) type;
        v.size = size;
        v.literal = 1;
        v.backend = (uint32_t) backend;

#if 0
        if (dep) {
            /* A literal variable (especially a pointer to some memory region) can
               specify an optional dependency to keep that memory region alive. The
               last entry (v.dep[3]) is strategically chosen as jitc_var_traverse()
               will ignore it given that preceding entries (v.dep[0-2]) are all
               zero, keeping the referenced variable from being merged into
               programs that make use of this literal. */
            v.dep[3] = dep;
            jitc_var_inc_ref_int(dep);
        }
#endif

        return jitc_var_new(v);
    } else {
        uint32_t isize = var_type_size[(int) type];
        void *data =
            jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                    : AllocType::HostAsync,
                        size * (size_t) isize);
        jitc_memset_async(backend, data, size, isize, value);
        return jitc_var_mem_map(backend, type, data, size, 1);
    }
}

uint32_t jitc_var_new_counter(JitBackend backend, uint32_t size) {
    Variable v;
    v.stmt = backend == JitBackend::CUDA ? (char *) "mov.u32 $r0, %r0"
                                         : jitc_llvm_counter_str;
    v.size = size;
    v.type = (uint32_t) VarType::UInt32;
    v.backend = (uint32_t) backend;
    return jitc_var_new(v);
}

uint32_t jitc_var_new_stmt(JitBackend backend, VarType vt, const char *stmt,
                          int stmt_static, uint32_t n_dep,
                          const uint32_t *dep) {
    uint32_t size = n_dep == 0 ? 1 : 0;
    bool dirty = false, uninitialized = false;
    Variable *v[4] { };

    if (unlikely(n_dep > 4))
        jitc_fail("jit_var_new_stmt(): 0-4 dependent variables supported!");

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (likely(dep[i])) {
            Variable *vi = jitc_var(dep[i]);
            size = std::max(size, vi->size);
            dirty |= vi->dirty;
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
        jitc_raise("jit_var_new_stmt(): arithmetic involving an "
                  "uninitialized variable!");
    }

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (v[i]->size != size && v[i]->size != 1)
            jitc_raise("jit_var_new_stmt(): arithmetic involving  arrays of "
                      "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(backend));
        for (uint32_t i = 0; i < n_dep; ++i)
            v[i] = jitc_var(dep[i]);
    }

    Variable v2;
    for (uint32_t i = 0; i < n_dep; ++i) {
        v2.dep[i] = dep[i];
        jitc_var_inc_ref_int(dep[i], v[i]);
    }
    v2.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v2.size = size;
    v2.type = (uint32_t) vt;
    v2.backend = (uint32_t) backend;
    v2.free_stmt = stmt_static == 0;

    return jitc_var_new(v2);
}

void jitc_var_set_free_callback(uint32_t index, void (*callback)(void *), void *payload) {
    Variable *v = jitc_var(index);

    jitc_log(Debug, "jit_var_set_callback(%u): " ENOKI_PTR " (" ENOKI_PTR ")",
            index, (uintptr_t) callback, (uintptr_t) payload);

    v->extra = true;
    Extra &extra = state.extra[index];
    if (unlikely(extra.free_callback))
        jitc_fail("jit_var_set_free_callback(): a callback was already set!");
    extra.free_callback = callback;
    extra.callback_payload = payload;
}

/// Query the current (or future, if not yet evaluated) allocation flavor of a variable
AllocType jitc_var_alloc_type(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (v->data)
        return jitc_malloc_type(v->data);

    return (JitBackend) v->backend == JitBackend::CUDA ? AllocType::Device
                                                       : AllocType::HostAsync;
}

/// Query the device associated with a variable
int jitc_var_device(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (v->data)
        return jitc_malloc_device(v->data);

    return thread_state(v->backend)->device;
}

/// Mark a variable as a scatter operation that writes to 'target'
void jitc_var_mark_side_effect(uint32_t index, uint32_t target) {
    Variable *v = jitc_var(index);
    jitc_log(Debug, "jit_var_mark_side_effect(%u, %u)", index, target);

    v->side_effect = true;

    ThreadState *ts = thread_state(v->backend);
    ts->side_effects.push_back(index);

    if (target) // Mark variable as dirty
        jitc_var(target)->dirty = true;
}

/// Return a human-readable summary of the contents of a variable
const char *jitc_var_str(uint32_t index) {
    const Variable *v = jitc_var(index);

    if (!v->literal && (!v->data || v->dirty)) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }

    size_t size            = v->size,
           isize           = var_type_size[v->type],
           limit_remainder = std::min(5u, (state.print_limit + 3) / 4) * 2;

    uint8_t dst[8] { };

    if (v->literal)
        memcpy(dst, &v->value, isize);

    var_buffer.clear();
    var_buffer.putc('[');
    for (uint32_t i = 0; i < size; ++i) {
        if (size > state.print_limit && i == limit_remainder / 2) {
            var_buffer.fmt(".. %zu skipped .., ", size - limit_remainder);
            i = (uint32_t) (size - limit_remainder / 2 - 1);
            continue;
        }

        if (!v->literal) {
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
    var_buffer.putc(']');
    return var_buffer.get();
}

/// Schedule a variable \c index for future evaluation via \ref jit_eval()
int jitc_var_schedule(uint32_t index) {
    auto it = state.variables.find(index);
    if (unlikely(it == state.variables.end()))
        jitc_raise("jit_var_schedule(%u): unknown variable!", index);
    Variable *v = &it.value();

    if (!v->data) {
        thread_state(v->backend)->scheduled.push_back(index);
        jitc_log(Debug, "jit_var_schedule(%u)", index);
        return 1;
    } else if (v->dirty) {
        return 1;
    }

    return 0;
}

/// Evaluate a literal constant variable
void jitc_var_eval_literal(uint32_t index, Variable *v) {
    jitc_log(Debug,
            "jit_var_eval_literal(%u): writing %s literal of size %u",
            index, var_type_name[v->type], v->size);

    jitc_cse_drop(index, v);

    JitBackend backend = (JitBackend) v->backend;
    uint32_t isize = var_type_size[v->type];
    v->data = jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                      : AllocType::HostAsync,
                          (size_t) v->size * (size_t) isize);

    jitc_memset_async(backend, v->data, v->size, isize, &v->value);

    v->literal = 0;
    v->stmt = nullptr;
}

/// Evaluate the variable \c index right away if it is unevaluated/dirty.
int jitc_var_eval(uint32_t index) {
    auto it = state.variables.find(index);
    if (unlikely(it == state.variables.end()))
        jitc_raise("jit_var_eval(%u): unknown variable!", index);
    Variable *v = &it.value();

    if (!v->data || v->dirty) {
        ThreadState *ts = thread_state(v->backend);

        if (!v->data) {
            if (v->literal) {
                /* If 'v' is a constant, initialize it directly instead of
                   generating code to do so.. */
                jitc_var_eval_literal(index, v);
                return 1;
            } else {
                ts->scheduled.push_back(index);
            }
        }

        jitc_eval(ts);
        v = jitc_var(index);

        if (unlikely(v->dirty))
            jitc_raise("jit_var_eval(): element remains dirty after evaluation!");
        else if (unlikely(!v->data))
            jitc_raise("jit_var_eval(): invalid/uninitialized variable!");

        return 1;
    }

    return 0;
}

/// Read a single element of a variable and write it to 'dst'
void jitc_var_read(uint32_t index, uint32_t offset, void *dst) {
    const Variable *v = jitc_var(index);

    if (!v->literal && (!v->data || v->dirty)) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }

    if (v->size == 1)
        offset = 0;
    else if (unlikely(offset >= v->size))
        jitc_raise("jit_var_read(): attempted to access entry %u in an array of "
                  "size %u!", offset, v->size);

    size_t isize = var_type_size[v->type];
    if (v->literal)
        memcpy(dst, &v->value, isize);
    else
        jitc_memcpy((JitBackend) v->backend, dst,
                    (const uint8_t *) v->data + offset * isize, isize);
}

/// Reverse of jitc_var_read(). Copy 'dst' to a single element of a variable
void jitc_var_write(uint32_t index, uint32_t offset, const void *src) {
    jitc_var_eval(index);

    Variable *v = jitc_var(index);
    if (unlikely(offset >= v->size))
        jitc_raise("jit_var_write(): attempted to access entry %u in an array of "
                  "size %u!", offset, v->size);

    uint32_t isize = var_type_size[v->type];
    uint8_t *dst = (uint8_t *) v->data + (size_t) offset * isize;
    jitc_poke((JitBackend) v->backend, dst, src, isize);
}

/// Register an existing variable with the JIT compiler
uint32_t jitc_var_mem_map(JitBackend backend, VarType type, void *ptr,
                          uint32_t size, int free) {
    if (unlikely(size == 0 || ptr == nullptr))
        return 0;

    Variable v;
    v.type = (uint32_t) type;
    v.backend = (uint32_t) backend;
    v.data = ptr;
    v.size = size;
    v.retain_data = free == 0;

    if (backend == JitBackend::LLVM) {
        uintptr_t align =
            std::min(64u, jitc_llvm_vector_width * var_type_size[(int) type]);
        v.unaligned = uintptr_t(ptr) % align != 0;
    }

    return jitc_var_new(v);
}

/// Copy a memory region onto the device and return its variable index
uint32_t jitc_var_mem_copy(JitBackend backend, AllocType atype, VarType vtype,
                          const void *ptr, uint32_t size) {
    ThreadState *ts = thread_state(backend);

    size_t total_size = (size_t) size * (size_t) var_type_size[(int) vtype];
    void *target_ptr;

    if (backend == JitBackend::CUDA) {
        target_ptr = jitc_malloc(AllocType::Device, total_size);

        scoped_set_context guard(ts->context);
        if (atype == AllocType::HostAsync) {
            jitc_fail("jit_var_mem_copy(): copy from HostAsync to GPU memory not supported!");
        } else if (atype == AllocType::Host) {
            void *host_ptr = jitc_malloc(AllocType::HostPinned, total_size);
            memcpy(host_ptr, ptr, total_size);
            cuda_check(cuMemcpyAsync((CUdeviceptr) target_ptr,
                                     (CUdeviceptr) host_ptr, total_size,
                                     ts->stream));
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
            memcpy(target_ptr, ptr, total_size);
            target_ptr = jitc_malloc_migrate(target_ptr, AllocType::HostAsync, 1);
        } else {
            jitc_fail("jit_var_mem_copy(): copy from GPU to HostAsync memory not supported!");
        }
    }

    uint32_t index = jitc_var_mem_map(backend, vtype, target_ptr, size, true);
    jitc_log(Debug, "jit_var_mem_copy(%u, size=%u)", index, size);
    return index;
}

uint32_t jitc_var_copy(uint32_t index) {
    if (index == 0)
        return 0;

    Variable *v = jitc_var(index);
    if (v->dirty) {
        jitc_var_eval(index);
        v = jitc_var(index);
    }

    uint32_t index_old = index;
    if (v->data) {
        JitBackend backend = (JitBackend) v->backend;
        index = jitc_var_mem_copy(backend,
                                  backend == JitBackend::CUDA
                                      ? AllocType::Device
                                      : AllocType::HostAsync,
                                  (VarType) v->type, v->data, v->size);
    } else {
        Variable v2 = *v;
        v2.ref_count_int = 0;
        v2.ref_count_ext = 0;
        v2.extra = 0;

        if (v2.free_stmt)
            v2.stmt = strdup(v2.stmt);

        index = jitc_var_new(v2, true);
    }
    jitc_log(Debug, "jit_var_copy(%u <- %u)", index, index_old);
    return index;
}

uint32_t jitc_var_resize(uint32_t index, uint32_t size) {
    if (index == 0 && size == 0)
        return 0;

    Variable *v = jitc_var(index);
    if (v->size == size) {
        jitc_var_inc_ref_ext(index, v);
        return index; // Nothing to do
    } else if (v->size != 1) {
        jitc_raise("jit_var_resize(): variable %u must be scalar!", index);
    }

    jitc_log(Debug, "jit_var_resize(%u, size=%u -> %u)", index, v->size, size);

    if (!v->data && v->ref_count_int == 0 && v->ref_count_ext == 1) {
        jitc_var_inc_ref_ext(index, v);
        jitc_cse_drop(index, v);
        v->size = size;
        return index;
    } else if (v->literal) {
        return jitc_var_new_literal((JitBackend) v->backend, (VarType) v->type,
                                    &v->value, size, 0);
    } else {
        Variable v2;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.size = size;
        v2.dep[0] = index;
        jitc_var_inc_ref_int(index, v);
        const char *stmt;
        if ((JitBackend) v->backend == JitBackend::CUDA)
            stmt = "mov.$t0 $r0, $r1";
        else
            stmt = ((VarType) v->type == VarType::Float32 ||
                    (VarType) v->type == VarType::Float64)
                ?  "$r0 = fadd <$w x $t0> $r1, zeroinitializer"
                :  "$r0 = add <$w x $t0> $r1, zeroinitializer";
        v2.stmt = (char *) stmt;
        return jitc_var_new(v2);
    }
}


/// Migrate a variable to a different flavor of memory
uint32_t jitc_var_migrate(uint32_t src_index, AllocType dst_type) {
    if (src_index == 0)
        return 0;

    jitc_var_eval(src_index);

    Variable *v = jitc_var(src_index);
    auto it = state.alloc_used.find(v->data);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_var_migrate(): Cannot resolve pointer to actual allocation!");
    AllocInfo ai = it.value();

    uint32_t dst_index = src_index;

    void *src_ptr = v->data,
         *dst_ptr = jitc_malloc_migrate(src_ptr, dst_type, 0);

    if (src_ptr != dst_ptr) {
        Variable v2 = *v;
        v2.data = dst_ptr;
        v2.retain_data = false;
        v2.ref_count_int = 0;
        v2.ref_count_ext = 0;
        v2.extra = 0;
        dst_index = jitc_var_new(v2);
    } else {
        jitc_var_inc_ref_ext(dst_index, v);
    }

    jitc_log(Debug, "jit_var_migrate(%u -> %u, " ENOKI_PTR " -> " ENOKI_PTR ", %s -> %s)",
            src_index, dst_index, (uintptr_t) src_ptr, (uintptr_t) dst_ptr,
            alloc_type_name[ai.type], alloc_type_name[(int) dst_type]);

    return dst_index;
}

/// Return a human-readable summary of registered variables
const char *jitc_var_whos() {
    var_buffer.clear();
    var_buffer.put("\n  ID        Type       Status       E/I Refs  Entries     Storage    Label");
    var_buffer.put("\n  ========================================================================\n");

    std::vector<uint32_t> indices;
    indices.reserve(state.variables.size());
    for (const auto& it : state.variables)
        indices.push_back(it.first);
    std::sort(indices.begin(), indices.end());

    size_t mem_size_evaluated = 0,
           mem_size_saved = 0,
           mem_size_unevaluated = 0;

    for (uint32_t index: indices) {
        const Variable *v = jitc_var(index);
        size_t mem_size = (size_t) v->size * (size_t) var_type_size[v->type];

        var_buffer.fmt("  %-9u %s %3s   ", index,
                       (JitBackend) v->backend == JitBackend::CUDA ? "cuda"
                                                                   : "llvm",
                       var_type_name_short[v->type]);

        if (v->literal) {
            var_buffer.put("literal    ");
        } else if (v->data) {
            auto it = state.alloc_used.find(v->data);
            if (unlikely(it == state.alloc_used.end())) {
                if (!v->retain_data)
                    jitc_raise("jit_var_whos(): Cannot resolve pointer to actual allocation!");
                else
                    var_buffer.put("mapped mem.");
            } else {
                AllocInfo ai = it.value();

                if ((AllocType) ai.type == AllocType::Device) {
                    var_buffer.fmt("device %-4i", (int) ai.device);
                } else {
                    const char *tname = alloc_type_name_short[ai.type];
                    var_buffer.put(tname, strlen(tname));
                }
            }
        } else {
            var_buffer.put("unevaluated");
        }

        size_t sz = var_buffer.fmt("  %u / %u", v->ref_count_ext, v->ref_count_int);
        const char *label = jitc_var_label(index);

        var_buffer.fmt("%*s%-12u%-8s   %s\n", 12 - (int) sz, "", v->size,
                   jitc_mem_string(mem_size), label ? label : "");

        if (v->data)
            mem_size_evaluated += mem_size;
        else if (v->ref_count_ext == 0)
            mem_size_saved += mem_size;
        else
            mem_size_unevaluated += mem_size;
    }
    if (indices.empty())
        var_buffer.put("                       -- No variables registered --\n");

    var_buffer.put("  ========================================================================\n\n");
    var_buffer.put("  JIT compiler\n");
    var_buffer.put("  ============\n");
    var_buffer.fmt("   - Memory usage (evaluated)   : %s.\n",
               jitc_mem_string(mem_size_evaluated));
    var_buffer.fmt("   - Memory usage (unevaluated) : %s.\n",
               jitc_mem_string(mem_size_unevaluated));
    var_buffer.fmt("   - Memory usage (saved)       : %s.\n",
               jitc_mem_string(mem_size_saved));
    var_buffer.fmt("   - Kernel launches            : %zu (%zu cache hits, "
               "%zu soft, %zu hard misses).\n\n",
               state.kernel_launches, state.kernel_hits,
               state.kernel_soft_misses, state.kernel_hard_misses);

    var_buffer.put("  Memory allocator\n");
    var_buffer.put("  ================\n");
    for (int i = 0; i < (int) AllocType::Count; ++i)
        var_buffer.fmt("   - %-20s: %s/%s used (peak: %s).\n",
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
    var_buffer.put("digraph {\n");
    var_buffer.put("  graph [dpi=50];\n");
    var_buffer.put("  node [shape=record fontname=Consolas];\n");
    var_buffer.put("  edge [fontname=Consolas];\n");

    char stmt_buf[64];
    for (uint32_t index: indices) {
        const Variable *v = jitc_var(index);

        const char *color = "";
        const char *stmt = v->stmt;
        if (v->literal) {
            switch (v->type) {
                case (int) VarType::Float32: {
                        float f;
                        memcpy(&f, &v->value, sizeof(float));
                        snprintf(stmt_buf, strlen(stmt_buf), "literal: %g", f);
                    }
                    break;
                case (int) VarType::Float64: {
                        double f;
                        memcpy(&f, &v->value, sizeof(double));
                        snprintf(stmt_buf, strlen(stmt_buf), "literal: %g", f);
                    }
                    break;
                default:
                    snprintf(stmt_buf, strlen(stmt_buf), "literal: 0x%llx",
                             (unsigned long long) v->value);
                    break;
            }
            color = " fillcolor=wheat style=filled";
            stmt = stmt_buf;
        } else if (v->data) {
            color = " fillcolor=salmon style=filled";
            stmt = "[evaluated array]";
        } else if (v->side_effect) {
            color = " fillcolor=cornflowerblue style=filled";
        }

        char *out = (char *) malloc(strlen(stmt) * 2 + 1),
             *ptr = out;
        for (int j = 0; ; ++j) {
            if (stmt[j] == '$' && stmt[j + 1] == 'n') {
                *ptr++='\\';
                continue;
            } else if (stmt[j] == '<' || stmt[j] == '>') {
                *ptr++='\\';
            }
            *ptr++ = stmt[j];
            if (stmt[j] == '\0')
                break;
        }

        const char *label = jitc_var_label(index);
        var_buffer.fmt(
            "  %u [label=\"{%s%s%s%s%s|{Type: %s %s|Size: %u}|{ID "
            "#%u|E:%u|I:%u}}\"%s];\n",
            index, out, label ? "|Label: \\\"" : "", label ? label : "",
            label ? "\\\"" : "", v->dirty ? "| ** DIRTY **" : "",
            (JitBackend) v->backend == JitBackend::CUDA ? "cuda" : "llvm",
            var_type_name_short[v->type], v->size, index, v->ref_count_ext,
            v->ref_count_int, color);

        free(out);

        for (uint32_t i = 0; i< 4; ++i) {
            if (v->dep[i])
                var_buffer.fmt("  %u -> %u [label=\" %u\"];\n", v->dep[i], index, i + 1);
        }
    }
    var_buffer.put("}\n");
    return var_buffer.get();
}
