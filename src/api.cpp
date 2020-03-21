#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "util.h"
#include <thread>

void jitc_init(int llvm, int cuda) {
    lock_guard guard(state.mutex);
    jit_init(llvm, cuda);
}

void jitc_init_async(int llvm, int cuda) {
    state.mutex.lock();
    std::thread([llvm, cuda]() {
        jit_init(llvm, cuda);
        state.mutex.unlock();
    }).detach();
}

int jitc_has_llvm() {
    return (int) state.has_llvm;
}

int jitc_has_cuda() {
    return (int) state.has_cuda;
}

void jitc_shutdown(int light) {
    lock_guard guard(state.mutex);
    jit_shutdown(light);
}

void jitc_log_set_stderr(LogLevel level) {
    lock_guard guard(state.mutex);
    state.log_level_stderr = level;
}

LogLevel jitc_log_stderr() {
    lock_guard guard(state.mutex);
    return state.log_level_stderr;
}

void jitc_set_log_callback(LogLevel level, LogCallback callback) {
    lock_guard guard(state.mutex);
    state.log_level_callback = callback ? level : Disable;
    state.log_callback = callback;
}

LogLevel jitc_log_callback() {
    lock_guard guard(state.mutex);
    return state.log_level_callback;
}

void jitc_log(LogLevel level, const char* fmt, ...) {
    lock_guard guard(state.mutex);
    va_list args;
    va_start(args, fmt);
    jit_vlog(level, fmt, args);
    va_end(args);
}

void jitc_raise(const char* fmt, ...) {
    lock_guard guard(state.mutex);
    va_list args;
    va_start(args, fmt);
    jit_vraise(fmt, args);
    va_end(args);
}

void jitc_fail(const char* fmt, ...) {
    lock_guard guard(state.mutex);
    va_list args;
    va_start(args, fmt);
    jit_vfail(fmt, args);
    va_end(args);
}

int32_t jitc_device_count() {
    lock_guard guard(state.mutex);
    return (int32_t) state.devices.size();
}

void jitc_device_set(int32_t device, uint32_t stream) {
    lock_guard guard(state.mutex);
    jit_device_set(device, stream);
}

void jitc_parallel_set_dispatch(int enable) {
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

void jitc_malloc_prefetch(void *ptr, int device) {
    lock_guard guard(state.mutex);
    jit_malloc_prefetch(ptr, device);
}

uint32_t jitc_malloc_to_id(void *ptr) {
    lock_guard guard(state.mutex);
    return jit_malloc_to_id(ptr);
}

void *jitc_malloc_from_id(uint32_t id) {
    lock_guard guard(state.mutex);
    return jit_malloc_from_id(id);
}

void jitc_var_inc_ref_ext(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_inc_ref_ext(index);
}

void jitc_var_dec_ref_ext(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_dec_ref_ext(index);
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

void jitc_var_set_label(uint32_t index, const char *label) {
    lock_guard guard(state.mutex);
    jit_var_set_label(index, label);
}

uint32_t jitc_var_map(VarType type, void *ptr, size_t size, int free) {
    lock_guard guard(state.mutex);
    return jit_var_map(type, ptr, size, free);
}

uint32_t jitc_var_copy_ptr(const void *ptr) {
    lock_guard guard(state.mutex);
    return jit_var_copy_ptr(ptr);
}

uint32_t jitc_var_copy(VarType type,
                                 const void *value,
                                 size_t size) {
    lock_guard guard(state.mutex);
    return jit_var_copy(type, value, size);
}

uint32_t jitc_trace_append_0(VarType type, const char *stmt, int copy_stmt) {
    lock_guard guard(state.mutex);
    return jit_trace_append_0(type, stmt, copy_stmt);
}

uint32_t jitc_trace_append_1(VarType type, const char *stmt, int copy_stmt,
                             uint32_t arg1) {
    lock_guard guard(state.mutex);
    return jit_trace_append_1(type, stmt, copy_stmt, arg1);
}

uint32_t jitc_trace_append_2(VarType type, const char *stmt, int copy_stmt,
                             uint32_t arg1, uint32_t arg2) {
    lock_guard guard(state.mutex);
    return jit_trace_append_2(type, stmt, copy_stmt, arg1, arg2);
}

uint32_t jitc_trace_append_3(VarType type, const char *stmt, int copy_stmt,
                             uint32_t arg1, uint32_t arg2, uint32_t arg3) {
    lock_guard guard(state.mutex);
    return jit_trace_append_3(type, stmt, copy_stmt, arg1, arg2, arg3);
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
    jit_var_mark_dirty(index);
}

void jitc_set_scatter_gather_operand(uint32_t index, int gather) {
    lock_guard guard(state.mutex);
    jit_set_scatter_gather_operand(index, gather);
}

const char *jitc_var_whos() {
    lock_guard guard(state.mutex);
    return jit_var_whos();
}

const char *jitc_var_str(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_str(index);
}

void jitc_var_read(uint32_t index, size_t offset, void *dst) {
    lock_guard guard(state.mutex);
    jit_var_read(index, offset, dst);
}

void jitc_var_write(uint32_t index, size_t offset, const void *src) {
    lock_guard guard(state.mutex);
    jit_var_write(index, offset, src);
}

void jitc_eval() {
    lock_guard guard(state.mutex);
    jit_eval();
}

void jitc_var_eval(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_eval(index);
}

void jitc_fill(VarType type, void *ptr, size_t size, const void *src) {
    lock_guard guard(state.mutex);
    jit_fill(type, ptr, size, src);
}

void jitc_reduce(VarType type, ReductionType rtype,
                 const void *ptr, size_t size, void *out) {
    lock_guard guard(state.mutex);
    jit_reduce(type, rtype, ptr, size, out);
}

void jitc_llvm_set_target(const char *target_cpu,
                          const char *target_features,
                          int vector_width) {
    lock_guard guard(state.mutex);
    jit_llvm_set_target(target_cpu, target_features, vector_width);
}
