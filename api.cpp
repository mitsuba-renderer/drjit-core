#include "jit.h"
#include "ssa.h"
#include <thread>

void jitc_init() {
    lock_guard guard(state.mutex);
    jit_init();
}

void jitc_init_async() {
    std::thread([]() { jitc_init(); }).detach();
}

void jitc_shutdown() {
    lock_guard guard(state.mutex);
    jit_shutdown();
}

uint32_t jitc_device_count() {
    lock_guard guard(state.mutex);
    return state.devices.size();
}

void jitc_set_context(uint32_t device, uint32_t stream) {
    lock_guard guard(state.mutex);
    jit_set_context(device, stream);
}

void jitc_device_sync() {
    jit_device_sync(); // no synchronization needed.
}

void *jitc_malloc(AllocType type, size_t size) {
    lock_guard guard(state.mutex);
    return jit_malloc(type, size);
}

void jitc_free(void *ptr) {
    lock_guard guard(state.mutex);
    jit_free(ptr);
}

void jitc_malloc_trim() {
    lock_guard guard(state.mutex);
    jit_malloc_trim();
}

void jitc_inc_ref_ext(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_inc_ref_ext(index);
}

void jitc_dec_ref_ext(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_dec_ref_ext(index);
}

void jitc_inc_ref_int(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_inc_ref_int(index);
}

void jitc_dec_ref_int(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_dec_ref_int(index);
}

void *jitc_var_ptr(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_ptr(index);
}

size_t jitc_var_size(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_size(index);
}

uint32_t jitc_trace_append(uint32_t type, const char *cmd) {
    lock_guard guard(state.mutex);
    return jit_trace_append(type, cmd);
}

uint32_t jitc_trace_append(uint32_t type, const char *cmd, uint32_t arg1) {
    lock_guard guard(state.mutex);
    return jit_trace_append(type, cmd, arg1);
}

uint32_t jitc_trace_append(uint32_t type, const char *cmd, uint32_t arg1,
                           uint32_t arg2) {
    lock_guard guard(state.mutex);
    return jit_trace_append(type, cmd, arg1, arg2);
}

uint32_t jitc_trace_append(uint32_t type, const char *cmd, uint32_t arg1,
                           uint32_t arg2, uint32_t arg3) {
    lock_guard guard(state.mutex);
    return jit_trace_append(type, cmd, arg1, arg2, arg3);
}
