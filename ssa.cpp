#include "ssa.h"
#include "jit.h"
#include "log.h"

// Access a variable by ID, terminate with an error if it doesn't exist
Variable *jit_var(uint32_t index) {
    auto it = state.variables.find(index);
    if (it == state.variables.end())
        jit_fail("jit_var(%u): unknown variable!", index);
    return &it.value();
}

// Create a new variable of the given type
Variable* jit_var_new(uint32_t type, const char *cmd_, size_t size) {
    char *cmd = strdup(cmd_);

#if defined(ENOKI_CUDA)
    if (type != EnokiType::Float32) {
        char *offset = strstr(cmd, ".ftz");
        if (offset)
            strcat(offset, offset + 4);
    }
#endif

    Variable *var = &state.variables[state.variable_index++];
    var->cmd = cmd;
    var->size = (uint32_t) size;
    return var;
}

// Cleanup handler, called when the internal/external reference count reaches zero
void jit_var_free(uint32_t index, Variable *v) {
    jit_log(Trace, "jit_var_free(%u) = %p.", index, v->data);

    uint32_t dep[3], extra_dep = v->extra_dep;
    memcpy(dep, v->dep, sizeof(uint32_t) * 3);

    // Release GPU memory
    if (v->free_variable && v->data)
        jit_free(v->data);

    // Free strings
    free(v->cmd);
    free(v->label);

    if (v->direct_pointer) {
        auto it = state.ptr_map.find(v->data);
        if (it == state.ptr_map.end())
            jit_fail("jit_var_free(): direct pointer not found!");
        state.ptr_map.erase(it);
    }

    // Remove from hash table ('v' invalid from here on)
    state.variables.erase(index);

    // Decrease reference count of dependencies
    for (int i = 0; i < 3; ++i)
        jit_dec_ref_int(dep[i]);

    jit_dec_ref_ext(extra_dep);
}

/// Increase the external reference count of a given variable
void jit_inc_ref_ext(uint32_t index) {
    if (index == 0)
        return;
    Variable *v = jit_var(index);
    v->ref_count_ext++;
    jit_log(Trace, "jit_inc_ref_ext(%u) -> %u", index, v->ref_count_ext);
}


/// Increase the internal reference count of a given variable
void jit_inc_ref_int(uint32_t index) {
    if (index == 0)
        return;
    Variable *v = jit_var(index);
    v->ref_count_int++;
    jit_log(Trace, "jit_inc_ref_int(%u) -> %u", index, v->ref_count_int);
}

/// Decrease the external reference count of a given variable
void jit_dec_ref_ext(uint32_t index) {
    if (index == 0 || state.variables.empty())
        return;
    Variable *v = jit_var(index);

    if (unlikely(v->ref_count_ext == 0))
        jit_fail("jit_dec_ref_ext(): variable %u has no external references!", index);

    jit_log(Trace, "jit_dec_ref_ext(%u) -> %u", index, v->ref_count_ext - 1);
    v->ref_count_ext--;

    if (v->ref_count_ext == 0 && !v->side_effect)
        state.live.erase(index);

    if (v->ref_count_ext == 0 && v->ref_count_int == 0)
        jit_var_free(index, v);
}

/// Decrease the internal reference count of a given variable
void jit_dec_ref_int(uint32_t index) {
    if (index == 0 || state.variables.empty())
        return;
    Variable *v = jit_var(index);

    if (unlikely(v->ref_count_int == 0))
        jit_fail("jit_dec_ref_int(): variable %u has no internal references!", index);

    jit_log(Trace, "jit_dec_ref_int(%u) -> %u", index, v->ref_count_int - 1);
    v->ref_count_int--;

    if (v->ref_count_ext == 0 && v->ref_count_int == 0)
        jit_var_free(index, v);
}

/// Query the pointer variable associated with a given variable
void *jit_var_ptr(uint32_t index) {
    return jit_var(index)->data;
}

/// Query the size of a given variable
size_t jit_var_size(uint32_t index) {
    return jit_var(index)->size;
}

uint32_t jit_trace_append(uint32_t type, const char *cmd) {
    uint32_t idx = state.variable_index;
    jit_log(Debug, "jit_trace_append(%u): %s.", idx, cmd);
    jit_var_new(type, cmd, 1);
    jit_inc_ref_ext(idx);
    state.live.insert(idx);
    return idx;
}

uint32_t jit_trace_append(uint32_t type, const char *cmd,
                          uint32_t arg1) {
    if (unlikely(arg1 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    const Variable *v1 = jit_var(arg1);

    size_t tree_size = v1->tree_size,
           size      = v1->size;

    if (v1->dirty) {
        jit_eval();
        tree_size = 1;
    }

    uint32_t idx = state.variable_index;
    jit_log(Debug, "jit_trace_append(%u <- %u): %s.",
            idx, arg1, cmd);

    Variable *v = jit_var_new(type, cmd, size);

    v->tree_size += tree_size;
    v->dep[0] = arg1;

    jit_inc_ref_ext(idx);
    jit_inc_ref_int(arg1);

    state.live.insert(idx);

    return idx;
}

uint32_t jit_trace_append(uint32_t type, const char *cmd,
                          uint32_t arg1, uint32_t arg2) {
    if (unlikely(arg1 == 0 || arg2 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    const Variable *v1 = jit_var(arg1),
                   *v2 = jit_var(arg2);

    size_t tree_size = v1->tree_size + v2->tree_size,
           size      = std::max(v1->size, v2->size);
    if (unlikely((v1->size != 1 && v1->size != size) ||
                 (v2->size != 1 && v2->size != size)))
        jit_raise(
            "jit_trace_append(): arithmetic involving arrays of incompatible "
            "size (%zu and %zu). The instruction was \"%s\".",
            v1->size, v2->size, cmd);

    if (v1->dirty || v2->dirty) {
        jit_eval();
        tree_size = 2;
    }

    uint32_t idx = state.variable_index;
    jit_log(Debug, "jit_trace_append(%u <- %u, %u): %s.", idx,
            arg1, arg2, cmd);

    Variable *v = jit_var_new(type, cmd, size);

    v->tree_size += tree_size;
    v->dep[0] = arg1;
    v->dep[1] = arg2;

    jit_inc_ref_ext(idx);
    jit_inc_ref_int(arg1);
    jit_inc_ref_int(arg2);

    state.live.insert(idx);

    return idx;
}

uint32_t jit_trace_append(uint32_t type, const char *cmd,
                          uint32_t arg1, uint32_t arg2, uint32_t arg3) {
    if (unlikely(arg1 == 0 || arg2 == 0 || arg3 == 0))
        jit_raise("jit_trace_append(): arithmetic involving "
                  "uninitialized variable!");

    const Variable *v1 = jit_var(arg1),
                   *v2 = jit_var(arg2),
                   *v3 = jit_var(arg3);

    size_t tree_size = v1->tree_size + v2->tree_size + v3->tree_size;
    size_t size = std::max({ v1->size, v2->size, v3->size });
    if (unlikely((v1->size != 1 && v1->size != size) ||
                 (v2->size != 1 && v2->size != size) ||
                 (v3->size != 1 && v3->size != size)))
        jit_raise(
            "jit_trace_append(): arithmetic involving arrays of incompatible "
            "size (%zu, %zu, and %zu). The instruction was \"%s\".",
            v1->size, v2->size, v3->size, cmd);

    if (v1->dirty || v2->dirty || v3->dirty) {
        jit_eval();
        tree_size = 3;
    }

    uint32_t idx = state.variable_index;
    jit_log(Debug, "jit_trace_append(%u <- %u, %u, %u): %s.", idx,
            arg1, arg2, arg3, cmd);

    Variable *v = jit_var_new(type, cmd, size);

    v->tree_size += tree_size;
    v->dep[0] = arg1;
    v->dep[1] = arg2;
    v->dep[2] = arg3;

    jit_inc_ref_ext(idx);
    jit_inc_ref_int(arg1);
    jit_inc_ref_int(arg2);
    jit_inc_ref_int(arg3);

    state.live.insert(idx);

#if defined(ENOKI_CUDA)
    if (strstr(cmd, "st.global") || strstr(cmd, "atom.global.add")) {
        v->extra_dep = state.scatter_gather_operand;
        jit_inc_ref_ext(v->extra_dep);
    }
#endif

    return idx;
}

void jit_eval() {

}
