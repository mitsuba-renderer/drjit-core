#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include <thread>

void jitc_init() {
    lock_guard guard(state.mutex);
    jit_init();
}

void jitc_init_async() {
    state.mutex.lock();
    std::thread([]() { jit_init(); state.mutex.unlock(); }).detach();
}

void jitc_shutdown() {
    lock_guard guard(state.mutex);
    jit_shutdown();
}

void jitc_log_buffer_enable(int value) {
    lock_guard guard(state.mutex);
    state.log_to_buffer = value != 0;
}

char *jitc_log_buffer() {
    lock_guard guard(state.mutex);
    return jit_log_buffer();
}

uint32_t jitc_log_level_set() {
    lock_guard guard(state.mutex);
    return state.log_level;
}

void jitc_log_level_set(uint32_t log_level) {
    lock_guard guard(state.mutex);
    state.log_level = log_level;
}

int32_t jitc_device_count() {
    lock_guard guard(state.mutex);
    return (int32_t) state.devices.size();
}

void jitc_device_set(int32_t device, uint32_t stream) {
    lock_guard guard(state.mutex);
    jit_device_set(device, stream);
}

void jitc_parallel_dispatch_set(int enable) {
    lock_guard guard(state.mutex);
    state.parallel_dispatch = enable != 0;
}

int jitc_parallel_dispatch() {
    lock_guard guard(state.mutex);
    return state.parallel_dispatch ? 1 : 0;
}

void jitc_sync_stream() {
    lock_guard guard(state.mutex);
    jit_sync_stream();
}

void jitc_sync_device() {
    lock_guard guard(state.mutex);
    jit_sync_device();
}

void *jitc_malloc(AllocType type, size_t size) {
    lock_guard guard(state.mutex);
    return jit_malloc(type, size);
}

void jitc_free(void *ptr) {
    lock_guard guard(state.mutex);
    jit_free(ptr);
}

void* jitc_malloc_migrate(void *ptr, AllocType type) {
    lock_guard guard(state.mutex);
    return jit_malloc_migrate(ptr, type);
}

void jitc_malloc_trim() {
    lock_guard guard(state.mutex);
    jit_malloc_trim(false);
}

void jitc_var_inc_ref_ext(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_inc_ref_ext(index);
}

void jitc_var_dec_ref_ext(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_dec_ref_ext(index);
}

void jitc_var_inc_ref_int(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_inc_ref_int(index);
}

void jitc_var_dec_ref_int(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_dec_ref_int(index);
}

void *jitc_var_ptr(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_ptr(index);
}

size_t jitc_var_size(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_size(index);
}

uint32_t jitc_var_set_size(uint32_t index, size_t size, int copy) {
    lock_guard guard(state.mutex);
    return jit_var_set_size(index, size, copy);
}

const char *jitc_var_label(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_label(index);
}

void jitc_var_label_set(uint32_t index, const char *label) {
    lock_guard guard(state.mutex);
    jit_var_label_set(index, label);
}

uint32_t jitc_var_register(VarType type, void *ptr, size_t size, int free) {
    lock_guard guard(state.mutex);
    return jit_var_register(type, ptr, size, free);
}

uint32_t jitc_var_register_ptr(const void *ptr) {
    lock_guard guard(state.mutex);
    return jit_var_register_ptr(ptr);
}

uint32_t jitc_var_copy_to_device(VarType type,
                                 const void *value,
                                 size_t size) {
    lock_guard guard(state.mutex);
    return jit_var_copy_to_device(type, value, size);
}

uint32_t jitc_trace_append_0(VarType type, const char *cmd) {
    lock_guard guard(state.mutex);
    return jit_trace_append_0(type, cmd);
}

uint32_t jitc_trace_append_1(VarType type, const char *cmd, uint32_t arg1) {
    lock_guard guard(state.mutex);
    return jit_trace_append_1(type, cmd, arg1);
}

uint32_t jitc_trace_append_2(VarType type, const char *cmd, uint32_t arg1,
                            uint32_t arg2) {
    lock_guard guard(state.mutex);
    return jit_trace_append_2(type, cmd, arg1, arg2);
}

uint32_t jitc_trace_append_3(VarType type, const char *cmd, uint32_t arg1,
                             uint32_t arg2, uint32_t arg3) {
    lock_guard guard(state.mutex);
    return jit_trace_append_3(type, cmd, arg1, arg2, arg3);
}

void jitc_var_migrate(uint32_t index, AllocType type) {
    lock_guard guard(state.mutex);
    jit_var_migrate(index, type);
}

void jitc_var_mark_side_effect(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_mark_side_effect(index);
}

void jitc_var_mark_dirty(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_mark_side_effect(index);
}

void jitc_set_scatter_gather_operand(uint32_t index, bool gather) {
    lock_guard guard(state.mutex);
    jit_set_scatter_gather_operand(index, gather);
}

const char *jitc_var_whos() {
    lock_guard guard(state.mutex);
    return jit_var_whos();
}

void jitc_eval() {
    lock_guard guard(state.mutex);
    jit_eval();
}

void jitc_eval_var(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_eval_var(index);
}
