#include "ssa.h"
#include "jit.h"
#include "log.h"

// Internal: access a variable by ID
Variable &jit_var(uint32_t index) {
    auto it = state.variables.find(index);
    if (it == state.variables.end())
        jit_fail("jit_variable(%u): referenced unknown variable!", index);
    return it.value();
}

// Internal: create a new variable of the given type
Variable& jit_var_new(uint32_t type, const char *cmd, size_t size) {
    cmd = strdup(cmd);
    if (type != EnokiType::Float32) {
        char *offset = strstr(cmd, ".ftz");
        if (offset)
            strcat(offset, offset + 4);
    }
    // return state.variables.emplace(state.variable_index++, type).first->second;
}

void jit_var_free(uint32_t index, Variable &v) {
    jit_log(Trace, "jit_var_free(%u) = %p", index, v.data);

    if (v.direct_pointer) {
        auto it = state.ptr_map.find(v.data);
        if (it == state.ptr_map.end())
            jit_fail("jit_var_free(): internal error: direct pointer not found!");
        state.ptr_map.erase(it);
    }

    uint32_t dep[3], extra_dep = v.extra_dep;
    memcpy(dep, v.dep, sizeof(uint32_t) * 3);

    for (int i = 0; i < 3; ++i)
        jit_dec_ref_int(dep[i]);

    jit_dec_ref_ext(extra_dep);
    state.variables.erase(index); // invokes Variable destructor + cudaFree().
}

void jit_inc_ref_ext(uint32_t index) {
    if (index == 0)
        return;
    Variable &v = jit_var(index);
    v.ref_count_ext++;
    jit_log(Trace, "jit_inc_ref_ext(%u) -> %u", index, v.ref_count_ext);
}


/// Public: increase the internal reference count of a given variable
void jit_inc_ref_int(uint32_t index) {
    if (index == 0)
        return;
    Variable &v = jit_var(index);
    v.ref_count_int++;
    jit_log(Trace, "jit_inc_ref_int(%u) -> %u", index, v.ref_count_int);
}

/// Public: decrease the external reference count of a given variable
void jit_dec_ref_ext(uint32_t index) {
    if (index == 0 || state.variables.empty())
        return;
    Variable &v = jit_var(index);

    if (unlikely(v.ref_count_ext == 0))
        jit_fail("jit_dec_ref_ext(): variable %u has no external references!", index);

    jit_log(Trace, "jit_dec_ref_ext(%u) -> %u", index, v.ref_count_ext - 1);
    v.ref_count_ext--;

    if (v.ref_count_ext == 0 && !v.side_effect)
        state.live.erase(index);

    if (v.ref_count_ext == 0 && v.ref_count_int == 0)
        jit_var_free(index, v);
}

ENOKI_EXPORT void jit_dec_ref_int(uint32_t index) {
    if (index == 0 || state.variables.empty())
        return;
    Variable &v = jit_var(index);

    if (unlikely(v.ref_count_int == 0))
        jit_fail("jit_dec_ref_int(): variable %u has no internal references!", index);

    jit_log(Trace, "jit_dec_ref_int(%u) -> %u", index, v.ref_count_int - 1);
    v.ref_count_int--;

    if (v.ref_count_ext == 0 && v.ref_count_int == 0)
        jit_var_free(index, v);
}

void *jit_var_ptr(uint32_t index) {
    return jit_var(index).data;
}

size_t jit_var_size(uint32_t index) {
    return jit_var(index).size;
}

void jit_strip_ftz(Variable &v) {
}

uint32_t jit_trace_append(uint32_t type, const char *cmd) {
    uint32_t idx = state.variable_index;

    jit_log(Debug, "jit_trace_append(%u): %s", idx, cmd);
    Variable &v = jit_var_new(type, cmd, 1);
    v.subtree_size = 1;
    jit_inc_ref_ext(idx);
    state.live.insert(idx);
    return idx;
}
