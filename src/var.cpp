#include "var.h"
#include "internal.h"
#include "log.h"
#include "eval.h"

/// Descriptive names for the various variable types
const char *var_type_name[(int) VarType::Count]{
    "invalid", "int8",   "uint8",   "int16",   "uint16",  "int32", "uint32",
    "int64",   "uint64", "float16", "float32", "float64", "mask",  "pointer"
};

/// Descriptive names for the various variable types (extra-short version)
const char *var_type_name_short[(int) VarType::Count]{
    "???", "i8", "u8", "i16", "u16", "i32", "u32",
    "i64", "u64", "f16", "f32", "f64", "msk", "ptr"
};

/// CUDA PTX type names
const char *var_type_name_ptx[(int) VarType::Count]{
    "???", "s8", "u8", "s16", "u16", "s32", "u32",
    "s64", "u64", "f16", "f32", "f64", "pred", "u64"
};

/// CUDA PTX type names (binary view)
const char *var_type_name_ptx_bin[(int) VarType::Count]{
    "???", "b8", "b8", "b16", "b16", "b32", "b32",
    "b64", "b64", "b16", "b32", "b64", "pred", "b64"
};

/// LLVM IR type names (does not distinguish signed vs unsigned)
const char *var_type_name_llvm[(int) VarType::Count]{
    "???", "i8", "i8", "i16", "i16", "i32", "i32",
    "i64", "i64", "half", "float", "double", "i1", "i64"
};

/// Abbreviated LLVM IR type names
const char *var_type_name_llvm_abbrev[(int) VarType::Count]{
    "???", "i8", "i8", "i16", "i16", "i32", "i32",
    "i64", "i64", "f16", "f32", "f64", "i1", "i64"
};

/// LLVM IR type names (binary view)
const char *var_type_name_llvm_bin[(int) VarType::Count]{
    "???", "i8", "i8", "i16", "i16", "i32", "i32",
    "i64", "i64", "i16", "i32", "i64", "i1", "i64"
};

/// LLVM/CUDA register name prefixes
const char *var_type_prefix[(int) VarType::Count]{
    "", "%b", "%b", "%w", "%w", "%r", "%r",
    "%rd", "%rd", "%h", "%f", "%d", "%p", "%rd"
};

/// Maps types to byte sizes
const uint32_t var_type_size[(int) VarType::Count]{
    (uint32_t) -1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8, 1, 8
};

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
    auto it = state.cse_cache.find(key);
    if (it != state.cse_cache.end() && it.value() == index)
        state.cse_cache.erase(it);
}

/// Cleanup handler, called when the internal/external reference count reaches zero
void jit_var_free(uint32_t index, Variable *v) {
    jit_trace("jit_var_free(%u)", index);

    jit_cse_drop(index, v);

    uint32_t dep[3], extra_dep = v->extra_dep;
    memcpy(dep, v->dep, sizeof(uint32_t) * 3);

    // Release GPU memory
    if (likely(v->data && !v->retain_data))
        jit_free(v->data);

    // Free strings
    if (unlikely(v->free_stmt))
        free(v->stmt);

    // Free descriptive label, if needed
    if (unlikely(v->has_label)) {
        auto it = state.labels.find(index);
        if (unlikely(it == state.labels.end()))
            jit_fail("jit_var_free(): label not found!");
        free(it.value());
        state.labels.erase(it);
    }

    // Free reverse pointer table entry, if needed
    if (unlikely(v->direct_pointer)) {
        auto it = state.variable_from_ptr.find(v->data);
        if (unlikely(it == state.variable_from_ptr.end()))
            jit_fail("jit_var_free(): direct pointer not found!");
        state.variable_from_ptr.erase(it);
    }

    // Remove from hash table (careful: 'v' invalid from here on)
    state.variables.erase(index);

    // Decrease reference count of dependencies
    for (int i = 0; i < 3; ++i)
        jit_var_dec_ref_int(dep[i]);

    jit_var_dec_ref_ext(extra_dep);
}

/// Increase the external reference count of a given variable
void jit_var_inc_ref_ext(uint32_t index, Variable *v) {
    v->ref_count_ext++;
    jit_trace("jit_var_inc_ref_ext(%u): %u", index, v->ref_count_ext);
}

/// Increase the external reference count of a given variable
void jit_var_inc_ref_ext(uint32_t index) {
    if (index != 0)
        jit_var_inc_ref_ext(index, jit_var(index));
}

/// Increase the internal reference count of a given variable
void jit_var_inc_ref_int(uint32_t index, Variable *v) {
    v->ref_count_int++;
    jit_trace("jit_var_inc_ref_int(%u): %u", index, v->ref_count_int);
}

/// Increase the internal reference count of a given variable
void jit_var_inc_ref_int(uint32_t index) {
    if (index != 0)
        jit_var_inc_ref_int(index, jit_var(index));
}

/// Decrease the external reference count of a given variable
void jit_var_dec_ref_ext(uint32_t index, Variable *v) {
    if (unlikely(v->ref_count_ext == 0))
        jit_fail("jit_var_dec_ref_ext(): variable %u has no external references!", index);

    jit_trace("jit_var_dec_ref_ext(%u): %u", index, v->ref_count_ext - 1);
    v->ref_count_ext--;

    if (v->ref_count_ext == 0 && v->ref_count_int == 0)
        jit_var_free(index, v);
}

/// Decrease the external reference count of a given variable
void jit_var_dec_ref_ext(uint32_t index) {
    if (index != 0)
        jit_var_dec_ref_ext(index, jit_var(index));
}

/// Decrease the internal reference count of a given variable
void jit_var_dec_ref_int(uint32_t index, Variable *v) {
    if (unlikely(v->ref_count_int == 0))
        jit_fail("jit_var_dec_ref_int(): variable %u has no internal references!", index);

    jit_trace("jit_var_dec_ref_int(%u): %u", index, v->ref_count_int - 1);
    v->ref_count_int--;

    if (v->ref_count_ext == 0 && v->ref_count_int == 0)
        jit_var_free(index, v);
}

/// Decrease the internal reference count of a given variable
void jit_var_dec_ref_int(uint32_t index) {
    if (index != 0)
        jit_var_dec_ref_int(index, jit_var(index));
}

/// Append the given variable to the instruction trace and return its ID
static std::pair<uint32_t, Variable *> jit_trace_append(Variable &v) {
    CSECache::iterator key_it;
    bool key_inserted = false;

    if (v.stmt) {
        if (v.type != (uint32_t) VarType::Float32) {
            /// Strip .ftz modifiers from non-float PTX statements
            char *offset = strstr(v.stmt, ".ftz");
            if (offset) {
                if (!v.free_stmt) {
                    /* Need to modify the instruction, allocate memory
                       if not already available */
                    v.free_stmt = true;
                    v.stmt = strdup(v.stmt);
                    offset = strstr(v.stmt, ".ftz");
                }
                do {
                    if ((offset[0] = offset[4]) == '\0')
                        break;
                    offset++;
                } while (true);
            }
        }

        // Check if this exact statement already exists ..
        std::tie(key_it, key_inserted) =
            state.cse_cache.try_emplace(VariableKey(v), 0);
    }

    uint32_t index;
    Variable *v_out;

    if (likely(key_inserted || v.stmt == nullptr)) {
        // .. nope, it is new.

        VariableMap::iterator var_it;
        bool var_inserted;
        do {
            index = state.variable_index++;

            if (unlikely(index == 0)) // overflow
                index = state.variable_index++;

            std::tie(var_it, var_inserted) =
                state.variables.try_emplace(index, v);

            if (likely(var_inserted))
                break;
        } while (true);

        if (key_inserted)
            key_it.value() = index;

        v_out = &var_it.value();
    } else {
        // .. found a match! Deallocate 'v'.
        if (v.free_stmt)
            free(v.stmt);

        for (int i = 0; i< 3; ++i)
            jit_var_dec_ref_int(v.dep[i]);
        jit_var_dec_ref_int(v.extra_dep);

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
            uint32_t index_new = jit_trace_append_1(
                (VarType) v->type, "mov.$t1 $r1, $r2", 1, index);
            jit_var(index_new)->size = size;
            jit_var_dec_ref_ext(index);
            return index_new;
        }

        jit_raise("jit_var_set_size(): attempted to resize variable %u,"
                  "which was already allocated (current size = %u, "
                  "requested size = %u)",
                  index, v->size, (uint32_t) size);
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
    auto it = state.labels.find(index);
    return it != state.labels.end() ? it.value() : nullptr;
}

/// Assign a descriptive label to a given variable
void jit_var_set_label(uint32_t index, const char *label_) {
    Variable *v = jit_var(index);
    char *label = label_ ? strdup(label_) : nullptr;

    jit_log(Debug, "jit_var_set_label(%u): \"%s\"", index,
            label ? label : "(null)");

    if (v->has_label) {
        auto it = state.labels.find(index);
        if (it == state.labels.end())
            jit_fail("jit_var_set_label(): previous label not found!");
        free(it.value());
        it.value() = label;
    } else {
        state.labels[index] = label;
        v->has_label = true;
    }
}

/// Append a variable to the instruction trace (no operands)
uint32_t jit_trace_append_0(VarType type, const char *stmt, int stmt_static) {
    Stream *stream = active_stream;

    Variable v;
    v.type = (uint32_t) type;
    v.size = 1;
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.tsize = 1;
    v.free_stmt = stmt_static == 0;
    v.cuda = stream != nullptr;

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u): %s%s",
            index, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_inc_ref_ext(index, vo);

    auto &todo = stream ? stream->todo : state.todo_host;
    todo.push_back(index);

    return index;
}

/// Append a variable to the instruction trace (1 operand)
uint32_t jit_trace_append_1(VarType type, const char *stmt,
                            int stmt_static, uint32_t op1) {
    Stream *stream = active_stream;

    if (unlikely(op1 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    Variable *v1 = jit_var(op1);

    Variable v;
    v.type = (uint32_t) type;
    v.size = v1->size;
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.dep[0] = op1;
    v.tsize = 1 + v1->tsize;
    v.free_stmt = stmt_static == 0;
    v.cuda = stream != nullptr;

    if (unlikely(v1->pending_scatter)) {
        jit_eval();
        v1 = jit_var(op1);
        v.tsize = 2;
    }

    jit_var_inc_ref_int(op1, v1);

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u <- %u): %s%s",
            index, op1, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_inc_ref_ext(index, vo);

    auto &todo = stream ? stream->todo : state.todo_host;
    todo.push_back(index);

    return index;
}

/// Append a variable to the instruction trace (2 operands)
uint32_t jit_trace_append_2(VarType type, const char *stmt, int stmt_static,
                            uint32_t op1, uint32_t op2) {
    Stream *stream = active_stream;

    if (unlikely(op1 == 0 || op2 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    Variable *v1 = jit_var(op1),
             *v2 = jit_var(op2);

    Variable v;
    v.type = (uint32_t) type;
    v.size = std::max(v1->size, v2->size);
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.dep[0] = op1;
    v.dep[1] = op2;
    v.tsize = 1 + v1->tsize + v2->tsize;
    v.free_stmt = stmt_static == 0;
    v.cuda = stream != nullptr;

    if (unlikely((v1->size != 1 && v1->size != v.size) ||
                 (v2->size != 1 && v2->size != v.size))) {
        jit_raise(
            "jit_trace_append(): arithmetic involving arrays of incompatible "
            "size (%u and %u). The instruction was \"%s\".",
            v1->size, v2->size, stmt);
    } else if (unlikely(v1->pending_scatter || v2->pending_scatter)) {
        jit_eval();
        v1 = jit_var(op1);
        v2 = jit_var(op2);
        v.tsize = 3;
    }

    jit_var_inc_ref_int(op1, v1);
    jit_var_inc_ref_int(op2, v2);

    if (state.scatter_gather_operand != 0 &&
        (strstr(stmt, "ld.global") || strstr(stmt, "gather"))) {
        v.extra_dep = state.scatter_gather_operand;
        jit_var_inc_ref_ext(v.extra_dep);
    }

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u <- %u, %u): %s%s",
            index, op1, op2, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_inc_ref_ext(index, vo);

    auto &todo = stream ? stream->todo : state.todo_host;
    todo.push_back(index);

    return index;
}

/// Append a variable to the instruction trace (3 operands)
uint32_t jit_trace_append_3(VarType type, const char *stmt, int stmt_static,
                            uint32_t op1, uint32_t op2, uint32_t op3) {
    Stream *stream = active_stream;

    if (unlikely(op1 == 0 || op2 == 0 || op3 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    Variable *v1 = jit_var(op1),
             *v2 = jit_var(op2),
             *v3 = jit_var(op3);

    Variable v;
    v.type = (uint32_t) type;
    v.size = std::max({ v1->size, v2->size, v3->size });
    v.stmt = stmt_static ? (char *) stmt : strdup(stmt);
    v.dep[0] = op1;
    v.dep[1] = op2;
    v.dep[2] = op3;
    v.tsize = 1 + v1->tsize + v2->tsize + v3->tsize;
    v.free_stmt = stmt_static == 0;
    v.cuda = stream != nullptr;

    if (unlikely((v1->size != 1 && v1->size != v.size) ||
                 (v2->size != 1 && v2->size != v.size) ||
                 (v3->size != 1 && v3->size != v.size))) {
        jit_raise(
            "jit_trace_append(): arithmetic involving arrays of incompatible "
            "size (%u, %u, and %u). The instruction was \"%s\".",
            v1->size, v2->size, v3->size, stmt);
    } else if (unlikely(v1->pending_scatter || v2->pending_scatter || v3->pending_scatter)) {
        jit_eval();
        v1 = jit_var(op1);
        v2 = jit_var(op2);
        v3 = jit_var(op3);
        v.tsize = 4;
    }

    jit_var_inc_ref_int(op1, v1);
    jit_var_inc_ref_int(op2, v2);
    jit_var_inc_ref_int(op3, v3);

    if (state.scatter_gather_operand != 0 &&
        (strstr(stmt, "st.global") || strstr(stmt, "atom.global.add") ||
         strstr(stmt, "scatter"))) {
        v.extra_dep = state.scatter_gather_operand;
        jit_var_inc_ref_ext(v.extra_dep);
    }

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_trace_append(%u <- %u, %u, %u): %s%s",
            index, op1, op2, op3, vo->stmt,
            vo->ref_count_int + vo->ref_count_ext == 0 ? "" : " (reused)");

    jit_var_inc_ref_ext(index, vo);

    auto &todo = stream ? stream->todo : state.todo_host;
    todo.push_back(index);

    return index;
}

/// Register an existing variable with the JIT compiler
uint32_t jit_var_map(VarType type, void *ptr, size_t size, int free) {
    if (unlikely(size == 0))
        jit_raise("jit_var_map: size must be nonzero!");

    Variable v;
    v.type = (uint32_t) type;
    v.data = ptr;
    v.size = (uint32_t) size;
    v.retain_data = free == 0;
    v.tsize = 1;
    v.cuda = active_stream != nullptr;

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_var_map(%u): " ENOKI_PTR ", size=%zu, free=%i",
            index, (uintptr_t) ptr, size, (int) free);

    jit_var_inc_ref_ext(index, vo);

    return index;
}

/// Copy a memory region onto the device and return its variable index
uint32_t jit_var_copy(VarType type, const void *ptr, size_t size) {
    Stream *stream = active_stream;

    size_t total_size = size * (size_t) var_type_size[(int) type];
    void *target_ptr;

    if (stream) {
        target_ptr = jit_malloc(AllocType::Device, total_size);
        void *host_ptr = jit_malloc(AllocType::HostPinned, total_size);
        memcpy(host_ptr, ptr, total_size);
        cuda_check(cuMemcpyAsync((CUdeviceptr) target_ptr,
                                 (CUdeviceptr) host_ptr, total_size,
                                 stream->handle));
        jit_free(host_ptr);
    } else {
        target_ptr = jit_malloc(AllocType::Host, total_size);
        memcpy(target_ptr, ptr, total_size);
    }

    uint32_t index = jit_var_map(type, target_ptr, size, true);
    jit_log(Debug, "jit_var_copy(%u, size=%zu)", index, size);
    return index;
}

/// Register pointer literal as a special variable within the JIT compiler
uint32_t jit_var_copy_ptr(const void *ptr) {
    auto it = state.variable_from_ptr.find(ptr);
    if (it != state.variable_from_ptr.end()) {
        uint32_t index = it.value();
        jit_var_inc_ref_ext(index);
        return index;
    }

    Variable v;
    v.type = (uint32_t) VarType::Pointer;
    v.data = (void *) ptr;
    v.size = 1;
    v.tsize = 0;
    v.retain_data = true;
    v.direct_pointer = true;
    v.cuda = active_stream != nullptr;

    uint32_t index; Variable *vo;
    std::tie(index, vo) = jit_trace_append(v);
    jit_log(Debug, "jit_var_copy_ptr(%u): " ENOKI_PTR, index, (uintptr_t) ptr);

    jit_var_inc_ref_ext(index, vo);
    state.variable_from_ptr[ptr] = index;
    return index;
}

/// Migrate a variable to a different flavor of memory
void jit_var_migrate(uint32_t index, AllocType type) {
    if (index == 0)
        return;

    Variable *v = jit_var(index);
    if (v->data == nullptr || v->pending_scatter) {
        jit_eval();
        v = jit_var(index);
    }

    jit_log(Debug, "jit_var_migrate(%u, " ENOKI_PTR "): %s", index,
            (uintptr_t) v->data, alloc_type_name[(int) type]);

    v->data = jit_malloc_migrate(v->data, type);
}

/// Indicate that evaluation of the given variable causes side effects
void jit_var_mark_side_effect(uint32_t index) {
    jit_log(Debug, "jit_var_mark_side_effect(%u)", index);
    jit_var(index)->side_effect = true;
}

/// Mark variable as dirty due to pending scatter operations
void jit_var_mark_dirty(uint32_t index) {
    jit_log(Debug, "jit_var_mark_dirty(%u)", index);
    Variable *v = jit_var(index);
    v->pending_scatter = true;
}

/// Inform the JIT that the next scatter/gather references var. 'index'
void jit_set_scatter_gather_operand(uint32_t index, int gather) {
    jit_log(Trace, "jit_set_scatter_gather_operand(index=%u, gather=%u)", index, gather);
    if (index) {
        Variable *v = jit_var(index);
        if (v->data == nullptr || (gather && v->pending_scatter))
            jit_eval();
    }
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
        size_t mem_size = (size_t) v->size * (size_t) var_type_size[(int) v->type];

        buffer.fmt("  %-9u %3s    ", index, var_type_name_short[(int) v->type]);
        size_t sz = buffer.fmt("%u / %u", v->ref_count_ext, v->ref_count_int);
        const char *label = jit_var_label(index);

        buffer.fmt("%*s%-12u%-12s[%c]     %s\n", 11 - (int) sz, "", v->size,
                   jit_mem_string(mem_size), v->data ? 'x' : ' ',
                   label ? label : "");

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
                   alloc_type_name[i],
                   std::string(jit_mem_string(state.alloc_usage[i])).c_str(),
                   std::string(jit_mem_string(state.alloc_watermark[i])).c_str());

    return buffer.get();
}

/// Return a human-readable summary of the contents of a variable
const char *jit_var_str(uint32_t index) {
    Stream *stream = active_stream;
    bool cuda = stream != nullptr;
    const Variable *v = jit_var(index);

    if (v->cuda != cuda)
        jit_raise("jit_var_str(): attempted to stringify a %s variable "
                  "while the %s backend was activated! You must invoke "
                  "jit_device_set() before!",
                  v->cuda ? "CUDA" : "LLVM", cuda ? "CUDA" : "LLVM");

    if (unlikely(v->data == nullptr || v->pending_scatter)) {
        jit_eval();
        v = jit_var(index);
    }

    if (unlikely(v->pending_scatter))
        jit_raise("jit_var_str(): element remains dirty after jit_eval()!");
    else if (unlikely(!v->data))
        jit_raise("jit_var_str(): invalid/uninitialized variable!");

    size_t size            = v->size,
           isize           = var_type_size[(int) v->type],
           limit_thresh    = 20,
           limit_remainder = 10;

    uint8_t dst[8];
    const uint8_t *src = (const uint8_t *) v->data;

    buffer.clear();
    buffer.putc('[');
    for (uint32_t i = 0; i < size; ++i) {
        if (size > limit_thresh && i == limit_remainder / 2) {
            buffer.fmt(".. %zu skipped .., ", size - limit_remainder);
            i = size - limit_remainder / 2 - 1;
            continue;
        }

        const uint8_t *src_offset = src + i * isize;

        if (cuda) {
            // Temporarily release the lock while synchronizing
            unlock_guard guard(state.mutex);
            cuda_check(cuMemcpyAsync((CUdeviceptr) dst,
                                     (CUdeviceptr) src_offset, isize,
                                     stream->handle));
            cuda_check(cuStreamSynchronize(stream->handle));
        } else {
            memcpy(dst, src_offset, isize);
        }

        const char *comma = i + 1 < size ? ", " : "";
        switch ((VarType) v->type) {
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
    buffer.putc(']');
    return buffer.get();
}

/// Call jit_eval() only if the variable 'index' requires evaluation
void jit_var_eval(uint32_t index) {
    Variable *v = jit_var(index);
    if (v->data == nullptr || v->pending_scatter)
        jit_eval();
}

/// Read a single element of a variable and write it to 'dst'
void jit_var_read(uint32_t index, size_t offset, void *dst) {
    Stream *stream = active_stream;
    bool cuda = stream != nullptr;
    Variable *v = jit_var(index);

    if (unlikely(v->cuda != cuda))
        jit_fail("jit_var_write(): attempted to read from a %s variable "
                 "while the %s backend was activated! You must invoke "
                 "jit_device_set() before!",
                 v->cuda ? "CUDA" : "LLVM", cuda ? "CUDA" : "LLVM");

    if (unlikely(v->data == nullptr || v->pending_scatter)) {
        jit_eval();
        v = jit_var(index);
    }

    if (unlikely(v->pending_scatter))
        jit_raise("jit_var_read(): element remains dirty after jit_eval()!");
    else if (unlikely(!v->data))
        jit_raise("jit_var_read(): invalid/uninitialized variable!");

    if (v->size == 1)
        offset = 0;

    uint32_t isize = var_type_size[(int) v->type];
    const uint8_t *src = (const uint8_t *) v->data + offset * isize;

    if  (cuda) {
        // Temporarily release the lock while synchronizing
        unlock_guard guard(state.mutex);
        cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, isize,
                                 stream->handle));
        cuda_check(cuStreamSynchronize(stream->handle));
    } else {
        memcpy(dst, src, isize);
    }
}

/// Reverse of jit_var_read(). Copy 'dst' to a single element of a variable
void jit_var_write(uint32_t index, size_t offset, const void *src) {
    Stream *stream = active_stream;
    bool cuda = stream != nullptr;
    Variable *v = jit_var(index);

    if (unlikely(v->cuda != cuda))
        jit_raise("jit_var_write(): attempted to write to a %s variable "
                  "while the %s backend was activated! You must invoke "
                  "jit_device_set() before!",
                  v->cuda ? "CUDA" : "LLVM", cuda ? "CUDA" : "LLVM");

    if (unlikely(v->data == nullptr || v->pending_scatter)) {
        jit_eval();
        v = jit_var(index);
    }

    if (unlikely(v->pending_scatter))
        jit_raise("jit_var_write(): element remains dirty after jit_eval()!");
    else if (unlikely(!v->data))
        jit_raise("jit_var_write(): invalid/uninitialized variable!");

    if (v->size == 1)
        offset = 0;

    uint32_t isize = var_type_size[(int) v->type];
    uint8_t *dst = (uint8_t *) v->data + offset * isize;

    if (cuda)
        cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, isize,
                                 stream->handle));
    else
        memcpy(dst, src, isize);
}
