/*
    src/api.cpp -- C -> C++ API locking wrappers

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "util.h"
#include "registry.h"
#include "llvm_api.h"
#include <thread>
#include <condition_variable>

void jitc_init(int llvm, int cuda) {
    lock_guard guard(state.mutex);
    jit_init(llvm, cuda);
}

void jitc_init_async(int llvm, int cuda) {
    /// Probably overkill for a simple wait flag..
    struct Sync {
        bool flag = false;
        std::mutex mutex;
        std::condition_variable cv;
    };

    std::shared_ptr<Sync> sync = std::make_shared<Sync>();
    lock_guard guard(sync->mutex);

    std::thread([llvm, cuda, sync]() {
        lock_guard guard2(state.mutex);
        {
            lock_guard guard3(sync->mutex);
            sync->flag = true;
            sync->cv.notify_one();
        }
        jit_init(llvm, cuda);
    }).detach();

    while (!sync->flag)
        sync->cv.wait(guard);
}

int jitc_has_llvm() {
    lock_guard guard(state.mutex);
    return (int) state.has_llvm;
}

int jitc_has_cuda() {
    lock_guard guard(state.mutex);
    return (int) (state.has_cuda && !state.devices.empty());
}

void jitc_shutdown(int light) {
    lock_guard guard(state.mutex);
    jit_shutdown(light);
}

void jitc_set_log_level_stderr(LogLevel level) {
    /// Allow changing this variable without acquiring a lock
    state.log_level_stderr = level;
}

LogLevel jitc_log_level_stderr() {
    /// Allow reading this variable without acquiring a lock
    return state.log_level_stderr;
}

void jitc_set_log_level_callback(LogLevel level, LogCallback callback) {
    lock_guard guard(state.mutex);
    state.log_level_callback = callback ? level : Disable;
    state.log_callback = callback;
}

LogLevel jitc_log_level_callback() {
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

void jitc_set_flags(uint32_t flags) {
    jit_set_flags(flags);
}

uint32_t jitc_flags() {
    return jit_flags();
}

void jitc_set_flag(JitFlag flag) {
    jit_set_flags(jit_flags() | (uint32_t) flag);
}

void jitc_unset_flag(JitFlag flag) {
    jit_set_flags(jit_flags() & ~(uint32_t) flag);
}

uint32_t jitc_side_effect_counter(int cuda) {
    lock_guard guard(state.mutex);
    return thread_state(cuda)->side_effect_counter;
}

void* jitc_cuda_stream() {
    lock_guard guard(state.mutex);
    return jit_cuda_stream();
}

void* jitc_cuda_context() {
    lock_guard guard(state.mutex);
    return jit_cuda_context();
}

int jitc_cuda_device_count() {
    lock_guard guard(state.mutex);
    return (int) state.devices.size();
}

void jitc_cuda_set_device(int device) {
    lock_guard guard(state.mutex);
    jit_cuda_set_device(device);
}

int jitc_cuda_device() {
    lock_guard guard(state.mutex);
    return thread_state(true)->device;
}

int jitc_cuda_device_raw() {
    lock_guard guard(state.mutex);
    return state.devices[thread_state(true)->device].id;
}

/// Return the compute capability of the current device (e.g. '52')
int jitc_cuda_compute_capability() {
    lock_guard guard(state.mutex);

    return state.devices[thread_state(true)->device].compute_capability;
}

void jitc_cuda_set_target(uint32_t ptx_version,
                          uint32_t compute_capability) {
    lock_guard guard(state.mutex);
    ThreadState *ts = thread_state(true);
    ts->ptx_version = ptx_version;
    ts->compute_capability = compute_capability;
}

void jitc_llvm_set_target(const char *target_cpu,
                          const char *target_features,
                          uint32_t vector_width) {
    lock_guard guard(state.mutex);
    jit_llvm_set_target(target_cpu, target_features, vector_width);
}

const char *jitc_llvm_target_cpu() {
    lock_guard guard(state.mutex);
    return jit_llvm_target_cpu;
}

const char *jitc_llvm_target_features() {
    lock_guard guard(state.mutex);
    return jit_llvm_target_features;
}

int jitc_llvm_version_major() {
    lock_guard guard(state.mutex);
    return jit_llvm_version_major;
}

int jitc_llvm_if_at_least(uint32_t vector_width, const char *feature) {
    lock_guard guard(state.mutex);
    return jit_llvm_if_at_least(vector_width, feature);
}

uint32_t jitc_llvm_active_mask() {
    lock_guard guard(state.mutex);
    return jit_llvm_active_mask();
}

void jitc_llvm_active_mask_push(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_llvm_active_mask_push(index);
}

void jitc_llvm_active_mask_pop() {
    lock_guard guard(state.mutex);
    jit_llvm_active_mask_pop();
}

void jitc_sync_thread() {
    lock_guard guard(state.mutex);
    jit_sync_thread();
}

void jitc_sync_device() {
    lock_guard guard(state.mutex);
    jit_sync_device();
}

void jitc_sync_all_devices() {
    lock_guard guard(state.mutex);
    jit_sync_all_devices();
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
    jit_malloc_trim(false);
}

void jitc_malloc_prefetch(void *ptr, int device) {
    lock_guard guard(state.mutex);
    jit_malloc_prefetch(ptr, device);
}

enum AllocType jitc_malloc_type(void *ptr) {
    lock_guard guard(state.mutex);
    return jit_malloc_type(ptr);
}

int jitc_malloc_device(void *ptr) {
    lock_guard guard(state.mutex);
    return jit_malloc_device(ptr);
}

void *jitc_malloc_migrate(void *ptr, AllocType type, int move) {
    lock_guard guard(state.mutex);
    return jit_malloc_migrate(ptr, type, move);
}

enum AllocType jitc_var_alloc_type(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_alloc_type(index);
}

int jitc_var_device(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_device(index);
}

void jitc_var_inc_ref_ext_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    lock_guard guard(state.mutex);
    jit_var_inc_ref_ext(index);
}

void jitc_var_dec_ref_ext_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    lock_guard guard(state.mutex);
    jit_var_dec_ref_ext(index);
}

uint32_t jitc_var_ext_ref(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var(index)->ref_count_ext;
}

uint32_t jitc_var_int_ref(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var(index)->ref_count_int;
}

void *jitc_var_ptr(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_ptr(index);
}

uint32_t jitc_var_size(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.mutex);
    return jit_var_size(index);
}

VarType jitc_var_type(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_type(index);
}

uint32_t jitc_var_set_size(uint32_t index, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_var_set_size(index, size);
}

const char *jitc_var_label(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_label(index);
}

void jitc_var_set_label(uint32_t index, const char *label) {
    if (index == 0)
        return;
    lock_guard guard(state.mutex);
    jit_var_set_label(index, label);
}

void jitc_var_set_free_callback(uint32_t index, void (*callback)(void *),
                                void *payload) {
    lock_guard guard(state.mutex);
    jit_var_set_free_callback(index, callback, payload);
}

uint32_t jitc_var_map_mem(int cuda, VarType type, void *ptr, uint32_t size, int free) {
    lock_guard guard(state.mutex);
    return jit_var_map_mem(cuda, type, ptr, size, free);
}

uint32_t jitc_var_copy_ptr(int cuda, const void *ptr, uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_copy_ptr(cuda, ptr, index);
}

uint32_t jitc_var_copy_mem(int cuda, AllocType atype, VarType vtype,
                           const void *value, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_var_copy_mem(cuda, atype, vtype, value, size);
}

uint32_t jitc_var_copy_var(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_copy_var(index);
}

uint32_t jitc_var_new_0(int cuda, VarType type, const char *stmt, int stmt_static,
                        uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_var_new_0(cuda, type, stmt, stmt_static, size);
}

uint32_t jitc_var_new_1(int cuda, VarType type, const char *stmt,
                        int stmt_static, uint32_t arg1) {
    lock_guard guard(state.mutex);
    return jit_var_new_1(cuda, type, stmt, stmt_static, arg1);
}

uint32_t jitc_var_new_2(int cuda, VarType type, const char *stmt,
                        int stmt_static, uint32_t arg1, uint32_t arg2) {
    lock_guard guard(state.mutex);
    return jit_var_new_2(cuda, type, stmt, stmt_static, arg1, arg2);
}

uint32_t jitc_var_new_3(int cuda, VarType type, const char *stmt,
                        int stmt_static, uint32_t arg1, uint32_t arg2,
                        uint32_t arg3) {
    lock_guard guard(state.mutex);
    return jit_var_new_3(cuda, type, stmt, stmt_static, arg1, arg2, arg3);
}

uint32_t jitc_var_new_4(int cuda, VarType type, const char *stmt,
                        int stmt_static, uint32_t arg1, uint32_t arg2,
                        uint32_t arg3, uint32_t arg4) {
    lock_guard guard(state.mutex);
    return jit_var_new_4(cuda, type, stmt, stmt_static, arg1, arg2, arg3, arg4);
}

uint32_t jitc_var_new_literal(int cuda, VarType type, uint64_t value,
                              uint32_t size, int eval) {
    lock_guard guard(state.mutex);
    return jit_var_new_literal(cuda, type, value, size, eval);
}

uint32_t jitc_var_migrate(uint32_t index, AllocType type) {
    lock_guard guard(state.mutex);
    return jit_var_migrate(index, type);
}

void jitc_var_mark_scatter(uint32_t index, uint32_t target) {
    lock_guard guard(state.mutex);
    jit_var_mark_scatter(index, target);
}

int jitc_var_is_literal_zero(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_is_literal_zero(index);
}

int jitc_var_is_literal_one(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_is_literal_one(index);
}

const char *jitc_var_whos() {
    lock_guard guard(state.mutex);
    return jit_var_whos();
}

const char *jitc_var_graphviz() {
    lock_guard guard(state.mutex);
    return jit_var_graphviz();
}

const char *jitc_var_str(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_str(index);
}

void jitc_var_read(uint32_t index, uint32_t offset, void *dst) {
    lock_guard guard(state.mutex);
    jit_var_read(index, offset, dst);
}

void jitc_var_write(uint32_t index, uint32_t offset, const void *src) {
    lock_guard guard(state.mutex);
    jit_var_write(index, offset, src);
}

void jitc_eval() {
    lock_guard guard(state.mutex);
    jit_eval();
}

int jitc_var_eval(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.mutex);
    return jit_var_eval(index);
}

int jitc_var_schedule(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.mutex);
    return jit_var_schedule(index);
}

/// Enable/disable common subexpression elimination
void jitc_set_cse(int cuda, int value) {
    lock_guard guard(state.mutex);
    thread_state(cuda)->enable_cse = value != 0;
}

/// Return whether or not common subexpression elimination is enabled
int jitc_cse(int cuda) {
    lock_guard guard(state.mutex);
    return thread_state(cuda)->enable_cse;
}

void jitc_memset_async(int cuda, void *ptr, uint32_t size, uint32_t isize,
                       const void *src) {
    lock_guard guard(state.mutex);
    jit_memset_async(cuda, ptr, size, isize, src);
}

void jitc_memcpy(int cuda, void *dst, const void *src, size_t size) {
    lock_guard guard(state.mutex);
    jit_memcpy(cuda, dst, src, size);
}

void jitc_memcpy_async(int cuda, void *dst, const void *src, size_t size) {
    lock_guard guard(state.mutex);
    jit_memcpy_async(cuda, dst, src, size);
}

void jitc_reduce(int cuda, VarType type, ReductionType rtype, const void *ptr,
                 uint32_t size, void *out) {
    lock_guard guard(state.mutex);
    jit_reduce(cuda, type, rtype, ptr, size, out);
}

void jitc_scan_u32(int cuda, const uint32_t *in, uint32_t size, uint32_t *out) {
    lock_guard guard(state.mutex);
    jit_scan_u32(cuda, in, size, out);
}

uint32_t jitc_compress(int cuda, const uint8_t *in, uint32_t size, uint32_t *out) {
    lock_guard guard(state.mutex);
    return jit_compress(cuda, in, size, out);
}

uint8_t jitc_all(int cuda, uint8_t *values, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_all(cuda, values, size);
}

uint8_t jitc_any(int cuda, uint8_t *values, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_any(cuda, values, size);
}

uint32_t jitc_mkperm(int cuda, const uint32_t *values, uint32_t size,
                     uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {
    lock_guard guard(state.mutex);
    return jit_mkperm(cuda, values, size, bucket_count, perm, offsets);
}

void jitc_block_copy(int cuda, enum VarType type, const void *in, void *out,
                     uint32_t size, uint32_t block_size) {
    lock_guard guard(state.mutex);
    jit_block_copy(cuda, type, in, out, size, block_size);
}

void jitc_block_sum(int cuda, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    lock_guard guard(state.mutex);
    jit_block_sum(cuda, type, in, out, size, block_size);
}

uint32_t jitc_registry_put(const char *domain, void *ptr) {
    lock_guard guard(state.mutex);
    return jit_registry_put(domain, ptr);
}

void jitc_registry_remove(void *ptr) {
    lock_guard guard(state.mutex);
    jit_registry_remove(ptr);
}

uint32_t jitc_registry_get_id(const void *ptr) {
    lock_guard guard(state.mutex);
    return jit_registry_get_id(ptr);
}

const char *jitc_registry_get_domain(const void *ptr) {
    lock_guard guard(state.mutex);
    return jit_registry_get_domain(ptr);
}

void *jitc_registry_get_ptr(const char *domain, uint32_t id) {
    lock_guard guard(state.mutex);
    return jit_registry_get_ptr(domain, id);
}

uint32_t jitc_registry_get_max(const char *domain) {
    lock_guard guard(state.mutex);
    return jit_registry_get_max(domain);
}

void jitc_registry_trim() {
    lock_guard guard(state.mutex);
    jit_registry_trim();
}

void jitc_registry_set_attr(void *self, const char *name, const void *value,
                            size_t size) {
    lock_guard guard(state.mutex);
    jit_registry_set_attr(self, name, value, size);
}

const void *jitc_registry_attr_data(const char *domain, const char *name) {
    lock_guard guard(state.mutex);
    return jit_registry_attr_data(domain, name);
}

VCallBucket *jitc_vcall(int cuda, const char *domain, uint32_t index,
                        uint32_t *bucket_count_out) {
    lock_guard guard(state.mutex);
    return jit_vcall(cuda, domain, index, bucket_count_out);
}

const char *jitc_eval_ir(int cuda,
                         const uint32_t *in, uint32_t n_in,
                         const uint32_t *out, uint32_t n_out,
                         uint32_t n_side_effects,
                         uint64_t *hash_out) {
    lock_guard guard(state.mutex);
    return jit_eval_ir(cuda, in, n_in, out, n_out, n_side_effects, hash_out);
}

uint32_t jitc_eval_ir_var(int cuda,
                          const uint32_t *in, uint32_t n_in,
                          const uint32_t *out, uint32_t n_out,
                          uint32_t n_side_effects,
                          uint64_t *hash_out) {
    lock_guard guard(state.mutex);
    return jit_eval_ir_var(cuda, in, n_in, out, n_out, n_side_effects, hash_out);
}
