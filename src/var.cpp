#include "var.h"
#include "internal.h"
#include "log.h"
#include "eval.h"

/// Return the size of a given variable type
size_t jit_type_size(VarType type) {
    switch (type) {
        case VarType::UInt8:
        case VarType::Int8:
        case VarType::Bool:    return 1;
        case VarType::UInt16:
        case VarType::Int16:   return 2;
        case VarType::UInt32:
        case VarType::Int32:
        case VarType::Float32: return 4;
        case VarType::UInt64:
        case VarType::Int64:
        case VarType::Pointer:
        case VarType::Float64: return 8;
        default: jit_fail("jit_type_size(): invalid type!");
    }
}

/// Return the readable name for the given variable type
const char *jit_type_name(VarType type) {
    switch (type) {
        case VarType::Int8:    return "i8 "; break;
        case VarType::UInt8:   return "u8 "; break;
        case VarType::Int16:   return "i16"; break;
        case VarType::UInt16:  return "u16"; break;
        case VarType::Int32:   return "i32"; break;
        case VarType::UInt32:  return "u32"; break;
        case VarType::Int64:   return "i64"; break;
        case VarType::UInt64:  return "u64"; break;
        case VarType::Float16: return "f16"; break;
        case VarType::Float32: return "f32"; break;
        case VarType::Float64: return "f64"; break;
        case VarType::Bool:    return "msk"; break;
        case VarType::Pointer: return "ptr"; break;
        default: jit_fail("jit_type_name(): invalid type!");
    }
}

/// Access a variable by ID, terminate with an error if it doesn't exist
Variable *jit_var(uint32_t index) {
    auto it = state.variables.find(index);
    if (unlikely(it == state.variables.end()))
        jit_fail("jit_var(%u): unknown variable!", index);
    return &it.value();
}

/// Remove a variable from the cache used for common subexpression elimination
void jit_cse_drop(uint32_t index, const Variable *v) {
    if (!v->stmt)
        return;

    VariableKey key(*v);
    CSECache::iterator it = state.cse_cache.find(key);
    if (it != state.cse_cache.end() && it.value() == index)
        state.cse_cache.erase(it);
}

/// Cleanup handler, called when the internal/external reference count reaches zero
void jit_var_free(uint32_t index, Variable *v) {
    jit_log(Trace, "jit_var_free(%u)", index);
    jit_cse_drop(index, v);

    uint32_t dep[3], extra_dep = v->extra_dep;
    memcpy(dep, v->dep, sizeof(uint32_t) * 3);

    // Release GPU memory
    if (likely(v->data && !v->retain_data))
        jit_free(v->data);

    // Free strings
    if (unlikely(v->free_strings)) {
        free(v->stmt);
        free(v->label);
    }

    if (unlikely(v->direct_pointer)) {
        auto it = state.variable_from_ptr.find(v->data);
        if (unlikely(it == state.variable_from_ptr.end()))
            jit_fail("jit_var_free(): direct pointer not found!");
        state.variable_from_ptr.erase(it);
    }

    // Remove from hash table ('v' invalid from here on)
    state.variables.erase(index);

    // Decrease reference count of dependencies
    for (int i = 0; i < 3; ++i)
        jit_var_int_ref_dec(dep[i]);

    jit_var_ext_ref_dec(extra_dep);
}

/// Increase the external reference count of a given variable
void jit_var_ext_ref_inc(uint32_t index, Variable *v) {
    v->ref_count_ext++;
    jit_log(Trace, "jit_var_ext_ref_inc(%u): %u", index, v->ref_count_ext);
}

/// Increase the external reference count of a given variable
void jit_var_ext_ref_inc(uint32_t index) {
    if (index != 0)
        jit_var_ext_ref_inc(index, jit_var(index));
}

/// Increase the internal reference count of a given variable
void jit_var_int_ref_inc(uint32_t index, Variable *v) {
    v->ref_count_int++;
    jit_log(Trace, "jit_var_int_ref_inc(%u): %u", index, v->ref_count_int);
}

/// Increase the internal reference count of a given variable
void jit_var_int_ref_inc(uint32_t index) {
    if (index != 0)
        jit_var_int_ref_inc(index, jit_var(index));
}

/// Decrease the external reference count of a given variable
void jit_var_ext_ref_dec(uint32_t index) {
    if (index == 0 || state.variables.empty())
        return;
    Variable *v = jit_var(index);

    if (unlikely(v->ref_count_ext == 0))
        jit_fail("jit_var_ext_ref_dec(): variable %u has no external references!", index);

    jit_log(Trace, "jit_var_ext_ref_dec(%u): %u", index, v->ref_count_ext - 1);
    v->ref_count_ext--;

    if (v->ref_count_ext == 0) {
        active_stream->todo.erase(index);

        if (v->ref_count_int == 0)
            jit_var_free(index, v);
    }
}

/// Decrease the internal reference count of a given variable
void jit_var_int_ref_dec(uint32_t index) {
    if (index == 0 || state.variables.empty())
        return;
    Variable *v = jit_var(index);

    if (unlikely(v->ref_count_int == 0))
        jit_fail("jit_var_int_ref_dec(): variable %u has no internal references!", index);

    jit_log(Trace, "jit_var_int_ref_dec(%u): %u", index, v->ref_count_int - 1);
    v->ref_count_int--;

    if (v->ref_count_ext == 0 && v->ref_count_int == 0)
        jit_var_free(index, v);
}

/// Append the given variable to the instruction trace and return its ID
static std::pair<uint32_t, Variable *> jit_trace_append(Variable &v) {
    CSECache::iterator key_it;
    bool key_inserted = false;

    if (v.stmt) {
        if (v.type != VarType::Float32) {
            /// Strip .ftz modifiers from non-float PTX statements
            char *offset = strstr(v.stmt, ".ftz");
            if (offset) {
                if (!v.free_strings) {
                    /* Need to modify the instruction, allocate memory
                       if not already available */
                    v.free_strings = true;
                    v.stmt = strdup(v.stmt);
                    offset = strstr(v.stmt, ".ftz");
                }
                strcat(offset, offset + 4);
            }
        }

        // Check if this exact statement already exists ..
        std::tie(key_it, key_inserted) =
            state.cse_cache.try_emplace(VariableKey(v), 0);
    }

    uint32_t index;
    Variable *v_out;

    if (key_inserted || v.stmt == nullptr) {
        // .. nope, it is new.
        index = state.variable_index++;

        VariableMap::iterator var_it;
        bool var_inserted;
        std::tie(var_it, var_inserted) = state.variables.try_emplace(index, v);

        if (unlikely(!var_inserted))
            jit_fail("jit_trace_append(): could not append instruction!");

        if (key_inserted)
            key_it.value() = index;

        v_out = &var_it.value();
    } else {
        // .. found a match! Deallocate 'v'.
        if (v.free_strings)
            free(v.stmt);

        index = key_it.value();
        v_out = jit_var(index);
    }

    return std::make_pair(index, v_out);
}

/// Query the pointer variable associated with a given variable
void *jit_var_ptr(uint32_t index) {
    return jit_var(index)->data;
}

/// Query the size of a given variable
size_t jit_var_size(uint32_t index) {
    return jit_var(index)->size;
}

/// Set the size of a given variable (if possible, otherwise throw)
uint32_t jit_var_set_size(uint32_t index, size_t size, int copy) {
    Variable *v = jit_var(index);
    if (v->size == size)
        return index;

    if (v->data != nullptr || v->ref_count_int > 0) {
        if (v->size == 1 && copy != 0) {
            uint32_t index_new =
                jit_trace_append_1(v->type, "mov.$t1 $r1, $r2", 1, index);
            jit_var(index_new)->size = size;
            jit_var_ext_ref_dec(index);
            return index_new;
        }

        jit_raise("cuda_var_set_size(): attempted to resize variable %u,"
                  "which was already allocated (current size = %zu, "
                  "requested size = %zu)",
                  index, v->size, size);
    }

    jit_log(Debug, "jit_var_set_size(%u): %zu", index, size);

    VariableKey key(*v), key_new(*v);
    v->size = key_new.size = (uint32_t) size;

    auto it = state.cse_cache.find(key);
    if (it != state.cse_cache.end() && it.value() == index) {
        state.cse_cache.erase(it);
        state.cse_cache.try_emplace(key_new, index);
    }

    return index;
}

/// Query the descriptive label associated with a given variable
const char *jit_var_label(uint32_t index) {
    return jit_var(index)->label;
}

/// Assign a descriptive label to a given variable
void jit_var_label_set(uint32_t index, const char *label_) {
    Variable *v = jit_var(index);
    char *label = label_ ? strdup(label_) : nullptr;
    char *stmt  = v->stmt ? strdup(v->stmt) : nullptr;

    if (v->free_strings) {
        free(v->stmt);
        free(v->label);
    }

    v->label        = label;
    v->stmt         = stmt;
    v->free_strings = true;

    jit_log(Debug, "jit_v_label_set(%u): \"%s\"", index,
            label ? label : "(null)");
}

/// Append a variable to the instruction trace (no operands)
uint32_t jit_trace_append_0(VarType type, const char *stmt, int stmt_static) {
    Stream *stream = jit_get_stream("jit_trace_append_0");

    Variable v;
    v.type = type;
    v.size = 1;
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.tsize = 1;
    v.free_strings = stmt_static != 0;

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u): %s%s",
            index, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_ext_ref_inc(index, vo);
    stream->todo.insert(index);

    return index;
}

/// Append a variable to the instruction trace (1 operand)
uint32_t jit_trace_append_1(VarType type, const char *stmt,
                            int stmt_static, uint32_t arg1) {
    Stream *stream = jit_get_stream("jit_trace_append_1");

    if (unlikely(arg1 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    Variable *v1 = jit_var(arg1);

    Variable v;
    v.type = type;
    v.size = v1->size;
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.dep[0] = arg1;
    v.tsize = 1 + v1->tsize;

    if (unlikely(v1->dirty)) {
        jit_eval();
        v1 = jit_var(arg1);
        v.tsize = 2;
    }

    jit_var_int_ref_inc(arg1, v1);

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u <- %u): %s%s",
            index, arg1, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_ext_ref_inc(index, vo);
    stream->todo.insert(index);

    return index;
}

/// Append a variable to the instruction trace (2 operands)
uint32_t jit_trace_append_2(VarType type, const char *stmt, int stmt_static,
                            uint32_t arg1, uint32_t arg2) {
    Stream *stream = jit_get_stream("jit_trace_append_2");

    if (unlikely(arg1 == 0 || arg2 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    Variable *v1 = jit_var(arg1),
             *v2 = jit_var(arg2);

    Variable v;
    v.type = type;
    v.size = std::max(v1->size, v2->size);
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.dep[0] = arg1;
    v.dep[1] = arg2;
    v.tsize = 1 + v1->tsize + v2->tsize;
    v.free_strings = stmt_static != 0;

    if (unlikely((v1->size != 1 && v1->size != v.size) ||
                 (v2->size != 1 && v2->size != v.size))) {
        jit_raise(
            "jit_trace_append(): arithmetic involving arrays of incompatible "
            "size (%zu and %zu). The instruction was \"%s\".",
            v1->size, v2->size, stmt);
    } else if (unlikely(v1->dirty || v2->dirty)) {
        jit_eval();
        v1 = jit_var(arg1);
        v2 = jit_var(arg2);
        v.tsize = 3;
    }

    jit_var_int_ref_inc(arg1, v1);
    jit_var_int_ref_inc(arg2, v2);

    if (strstr(stmt, "ld.global")) {
        v.extra_dep = state.scatter_gather_operand;
        jit_var_ext_ref_inc(v.extra_dep);
    }

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u <- %u, %u): %s%s",
            index, arg1, arg2, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_ext_ref_inc(index, vo);
    stream->todo.insert(index);

    return index;
}

/// Append a variable to the instruction trace (3 operands)
uint32_t jit_trace_append_3(VarType type, const char *stmt, int stmt_static,
                            uint32_t arg1, uint32_t arg2, uint32_t arg3) {
    Stream *stream = jit_get_stream("jit_trace_append_3");

    if (unlikely(arg1 == 0 || arg2 == 0 || arg3 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    Variable *v1 = jit_var(arg1),
             *v2 = jit_var(arg2),
             *v3 = jit_var(arg3);

    Variable v;
    v.type = type;
    v.size = std::max({ v1->size, v2->size, v3->size });
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.dep[0] = arg1;
    v.dep[1] = arg2;
    v.dep[2] = arg3;
    v.tsize = 1 + v1->tsize + v2->tsize + v3->tsize;
    v.free_strings = stmt_static != 0;

    if (unlikely((v1->size != 1 && v1->size != v.size) ||
                 (v2->size != 1 && v2->size != v.size) ||
                 (v3->size != 1 && v3->size != v.size))) {
        jit_raise(
            "jit_trace_append(): arithmetic involving arrays of incompatible "
            "size (%zu, %zu, and %zu). The instruction was \"%s\".",
            v1->size, v2->size, v3->size, stmt);
    } else if (unlikely(v1->dirty || v2->dirty || v3->dirty)) {
        jit_eval();
        v1 = jit_var(arg1);
        v2 = jit_var(arg2);
        v3 = jit_var(arg3);
        v.tsize = 4;
    }

    jit_var_int_ref_inc(arg1, v1);
    jit_var_int_ref_inc(arg2, v2);
    jit_var_int_ref_inc(arg3, v3);

    if ((strstr(stmt, "st.global") || strstr(stmt, "atom.global.add")) &&
        state.scatter_gather_operand != 0) {
        v.extra_dep = state.scatter_gather_operand;
        jit_var_ext_ref_inc(v.extra_dep);
    }

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u <- %u, %u, %u): %s%s",
            index, arg1, arg2, arg3, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_ext_ref_inc(index, vo);
    stream->todo.insert(index);

    return index;
}

/// Register an existing variable with the JIT compiler
uint32_t jit_var_register(VarType type, void *ptr,
                          size_t size, int free) {
    if (unlikely(size == 0))
        jit_raise("jit_var_register: size must be > 0!");

    Variable v;
    v.type = type;
    v.data = ptr;
    v.size = (uint32_t) size;
    v.retain_data = free == 0;
    v.tsize = 1;

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_var_register(%u): " PTR ", size=%zu, free=%i",
            index, ptr, size, (int) free);

    jit_var_ext_ref_inc(index, vo);

    return index;
}

/// Register pointer literal as a special variable within the JIT compiler
uint32_t jit_var_register_ptr(const void *ptr) {
    auto it = state.variable_from_ptr.find(ptr);
    if (it != state.variable_from_ptr.end()) {
        uint32_t index = it.value();
        jit_var_ext_ref_inc(index);
        return index;
    }

    Variable v;
    v.type = VarType::Pointer;
    v.data = (void *) ptr;
    v.size = 1;
    v.tsize = 0;
    v.retain_data = true;
    v.direct_pointer = true;

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_var_register_ptr(%u): " PTR, index, ptr);

    jit_var_ext_ref_inc(index, vo);
    state.variable_from_ptr[ptr] = index;
    return index;
}

/// Copy a memory region onto the device and return its variable index
uint32_t jit_var_copy_to_device(VarType type,
                                const void *ptr,
                                size_t size) {
    Stream *stream = jit_get_stream("jit_var_copy_to_device");

    size_t total_size = size * jit_type_size(type);

    void *host_ptr   = jit_malloc(AllocType::HostPinned, total_size),
         *device_ptr = jit_malloc(AllocType::Device, total_size);

    memcpy(host_ptr, ptr, total_size);
    cuda_check(cuMemcpyAsync(device_ptr, host_ptr, total_size, stream->handle));

    jit_free(host_ptr);
    uint32_t index = jit_var_register(type, device_ptr, size, true);
    jit_log(Debug, "jit_var_copy_to_device(%u, %zu)", index, size);
    return index;
}

/// Migrate a variable to a different flavor of memory
void jit_var_migrate(uint32_t index, AllocType type) {
    if (index == 0)
        return;

    Variable *v = jit_var(index);
    if (v->data == nullptr || v->dirty) {
        jit_eval();
        v = jit_var(index);
    }

    jit_log(Debug, "jit_var_migrate(%u, " PTR "): %s", index, v->data,
            alloc_type_names[(int) type]);

    v->data = jit_malloc_migrate(v->data, type);
}

/// Indicate that evaluation of the given variable causes side effects
void jit_var_mark_side_effect(uint32_t index) {
    jit_log(Debug, "jit_var_mark_side_effect(%u)", index);
    jit_var(index)->side_effect = true;
}

/// Mark variable as dirty, e.g. because of pending scatter operations
void jit_var_mark_dirty(uint32_t index) {
    jit_log(Debug, "jit_var_mark_dirty(%u)", index);
    Variable *v = jit_var(index);
    v->dirty = true;

    /* The contents of this variable no longer match up with its description,
       hence we cannot use it as part of common subexpression elimination */
    jit_cse_drop(index, v);
}

/// Inform the JIT that the next scatter/gather references var. 'index'
void jit_set_scatter_gather_operand(uint32_t index, bool gather) {
    if (index == 0)
        return;
    Variable *v = jit_var(index);
    if (v->data == nullptr || (gather && v->dirty))
        jit_eval();
    state.scatter_gather_operand = index;
}

/// Return a human-readable summary of registered variables
const char *jit_var_whos() {
    buffer.clear();
    buffer.put("\n  ID        Type   E/I Refs   Size        Memory     Ready    Label");
    buffer.put("\n  =================================================================\n");

    std::vector<uint32_t> indices;
    indices.reserve(state.variables.size());
    for (const auto& it : state.variables)
        indices.push_back(it.first);
    std::sort(indices.begin(), indices.end());

    size_t mem_size_scheduled = 0,
           mem_size_ready = 0,
           mem_size_arith = 0;

    for (uint32_t index: indices) {
        const Variable *v = jit_var(index);
        size_t mem_size = v->size * jit_type_size(v->type);

        buffer.fmt("  %-9u %s    ", index, jit_type_name(v->type));
        size_t sz = buffer.fmt("%u / %u", v->ref_count_ext, v->ref_count_int);
        buffer.fmt("%*s%-12u%-12s[%c]     %s\n", 11 - sz, "", v->size,
                   jit_mem_string(mem_size), v->data ? 'x' : ' ',
                   v->label ? v->label : "");

        if (v->data) {
            mem_size_ready += mem_size;
        } else {
            if (v->ref_count_ext == 0)
                mem_size_arith += mem_size;
            else
                mem_size_scheduled += mem_size;
        }
    }

    buffer.put("  =================================================================\n\n");
    buffer.put("  JIT compiler\n");
    buffer.put("  ============\n");
    buffer.fmt("   - Memory usage (ready)     : %s.\n",
               jit_mem_string(mem_size_ready));
    buffer.fmt("   - Memory usage (scheduled) : %s + %s = %s.\n",
               std::string(jit_mem_string(mem_size_ready)).c_str(),
               std::string(jit_mem_string(mem_size_scheduled)).c_str(),
               std::string(jit_mem_string(mem_size_ready + mem_size_scheduled)).c_str());
    buffer.fmt("   - Memory savings           : %s.\n\n",
               jit_mem_string(mem_size_arith));

    buffer.put("  Memory allocator\n");
    buffer.put("  ================\n");
    for (int i = 0; i < 5; ++i)
        buffer.fmt("   - %-20s: %s used (max. %s).\n",
                   alloc_type_names[i],
                   std::string(jit_mem_string(state.alloc_usage[i])).c_str(),
                   std::string(jit_mem_string(state.alloc_watermark[i])).c_str());

    return buffer.get();
}

/// Return a human-readable summary of the contents of a variable
const char *jit_var_str(uint32_t index) {
    const Variable *v = jit_var(index);
    Stream *stream = jit_get_stream("jit_var_str");

    if (v->data == nullptr || v->dirty)
        jit_eval();

    uint32_t size = v->size,
             isize = jit_type_size(v->type),
             limit_thresh = 20,
             limit_remainder = 10;

    buffer.clear();
    buffer.put("[");
    uint8_t dst[8];
    const uint8_t *src = (const uint8_t *) v->data;
    for (uint32_t i = 0; i < size; ++i) {
        if (size > limit_thresh && i == limit_remainder / 2) {
            buffer.fmt(".. %u skipped .., ", size - limit_remainder);
            i = size - limit_remainder / 2 - 1;
            continue;
        }
        const uint8_t *src_offset = src + i * isize;
        cuda_check(cuMemcpyAsync(dst, src_offset, isize, stream->handle));
        /* Temporarily release the lock while synchronizing */ {
            unlock_guard(state.mutex);
            cuda_check(cuStreamSynchronize(stream->handle));
        }

        const char *comma = i + 1 < size ? ", " : "";
        switch (v->type) {
            case VarType::Bool:    buffer.fmt("%"   PRIu8  "%s", *(( uint8_t *) dst), comma); break;
            case VarType::Int8:    buffer.fmt("%"   PRId8  "%s", *((  int8_t *) dst), comma); break;
            case VarType::UInt8:   buffer.fmt("%"   PRIu8  "%s", *(( uint8_t *) dst), comma); break;
            case VarType::Int16:   buffer.fmt("%"   PRId16 "%s", *(( int16_t *) dst), comma); break;
            case VarType::UInt16:  buffer.fmt("%"   PRIu16 "%s", *((uint16_t *) dst), comma); break;
            case VarType::Int32:   buffer.fmt("%"   PRId32 "%s", *(( int32_t *) dst), comma); break;
            case VarType::UInt32:  buffer.fmt("%"   PRIu32 "%s", *((uint32_t *) dst), comma); break;
            case VarType::Int64:   buffer.fmt("%"   PRId64 "%s", *(( int64_t *) dst), comma); break;
            case VarType::UInt64:  buffer.fmt("%"   PRIu64 "%s", *((uint64_t *) dst), comma); break;
            case VarType::Pointer: buffer.fmt("0x%" PRIx64 "%s", *((uint64_t *) dst), comma); break;
            case VarType::Float32: buffer.fmt("%g%s", *((float *) dst), comma); break;
            case VarType::Float64: buffer.fmt("%g%s", *((double *) dst), comma); break;
            default: jit_fail("jit_var_str(): invalid type!");
        }
    }
    buffer.put("]");
    return buffer.get();
}
