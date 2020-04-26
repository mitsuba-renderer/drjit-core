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
#include <thread>

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
    std::unique_lock<std::mutex> guard(sync->mutex);

    std::thread([llvm, cuda, sync]() {
        lock_guard guard2(state.mutex);
        {
            lock_guard guard2(sync->mutex);
            sync->flag = true;
            sync->cv.notify_one();
        }
        jit_init(llvm, cuda);
    }).detach();

    while (!sync->flag)
        sync->cv.wait(guard);
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

int32_t jitc_device_count() {
    lock_guard guard(state.mutex);
    return (int32_t) state.devices.size();
}

void jitc_set_device(int32_t device, uint32_t stream) {
    lock_guard guard(state.mutex);
    jit_set_device(device, stream);
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

void jitc_set_parallel_dispatch(int enable) {
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

void *jitc_malloc_migrate(void *ptr, AllocType type) {
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

void jitc_var_inc_ref_ext(uint32_t index) {
    if (index == 0)
        return;
    lock_guard guard(state.mutex);
    jit_var_inc_ref_ext(index);
}

void jitc_var_dec_ref_ext(uint32_t index) {
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

const char *jitc_var_label(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_label(index);
}

void jitc_var_set_label(uint32_t index, const char *label) {
    lock_guard guard(state.mutex);
    jit_var_set_label(index, label);
}

uint32_t jitc_var_map(VarType type, void *ptr, uint32_t size, int free) {
    lock_guard guard(state.mutex);
    return jit_var_map(type, ptr, size, free);
}

uint32_t jitc_var_copy_ptr(const void *ptr, uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_copy_ptr(ptr, index);
}

uint32_t jitc_var_copy(AllocType atype, VarType vtype, const void *value,
                       uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_var_copy(atype, vtype, value, size);
}

uint32_t jitc_var_new_0(VarType type, const char *stmt, int stmt_static, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_var_new_0(type, stmt, stmt_static, size);
}

uint32_t jitc_var_new_1(VarType type, const char *stmt, int stmt_static,
                        uint32_t arg1) {
    lock_guard guard(state.mutex);
    return jit_var_new_1(type, stmt, stmt_static, arg1);
}

uint32_t jitc_var_new_2(VarType type, const char *stmt, int stmt_static,
                        uint32_t arg1, uint32_t arg2) {
    lock_guard guard(state.mutex);
    return jit_var_new_2(type, stmt, stmt_static, arg1, arg2);
}

uint32_t jitc_var_new_3(VarType type, const char *stmt, int stmt_static,
                        uint32_t arg1, uint32_t arg2, uint32_t arg3) {
    lock_guard guard(state.mutex);
    return jit_var_new_3(type, stmt, stmt_static, arg1, arg2, arg3);
}

uint32_t jitc_var_new_4(VarType type, const char *stmt, int stmt_static,
                        uint32_t arg1, uint32_t arg2, uint32_t arg3,
                        uint32_t arg4) {
    lock_guard guard(state.mutex);
    return jit_var_new_4(type, stmt, stmt_static, arg1, arg2, arg3, arg4);
}

void jitc_var_migrate(uint32_t index, AllocType type) {
    lock_guard guard(state.mutex);
    jit_var_migrate(index, type);
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

void jitc_var_eval(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_eval(index);
}

void jitc_var_schedule(uint32_t index) {
    lock_guard guard(state.mutex);
    jit_var_schedule(index);
}

void jitc_memset_async(void *ptr, uint32_t size, uint32_t isize, const void *src) {
    lock_guard guard(state.mutex);
    jit_memset_async(ptr, size, isize, src);
}

void jitc_memcpy(void *dst, const void *src, size_t size) {
    lock_guard guard(state.mutex);
    jit_memcpy(dst, src, size);
}

void jitc_memcpy_async(void *dst, const void *src, size_t size) {
    lock_guard guard(state.mutex);
    jit_memcpy_async(dst, src, size);
}

void jitc_reduce(VarType type, ReductionType rtype,
                 const void *ptr, uint32_t size, void *out) {
    lock_guard guard(state.mutex);
    jit_reduce(type, rtype, ptr, size, out);
}

void jitc_scan_u32(const uint32_t *in, uint32_t size, uint32_t *out) {
    lock_guard guard(state.mutex);
    jit_scan_u32(in, size, out);
}

void jitc_compress(const uint8_t *in, uint32_t size, uint32_t *out, uint32_t *count_out) {
    lock_guard guard(state.mutex);
    jit_compress(in, size, out, count_out);
}

uint8_t jitc_all(uint8_t *values, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_all(values, size);
}

uint8_t jitc_any(uint8_t *values, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_any(values, size);
}

uint32_t jitc_mkperm(const uint32_t *values, uint32_t size,
                     uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {
    lock_guard guard(state.mutex);
    return jit_mkperm(values, size, bucket_count, perm, offsets);
}

void jitc_block_copy(enum VarType type, const void *in, void *out,
                     uint32_t size, uint32_t block_size) {
    lock_guard guard(state.mutex);
    jit_block_copy(type, in, out, size, block_size);
}

void jitc_block_sum(enum VarType type, const void *in, void *out,
                     uint32_t size, uint32_t block_size) {
    lock_guard guard(state.mutex);
    jit_block_sum(type, in, out, size, block_size);
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

VCallBucket *jitc_vcall(const char *domain, uint32_t index,
                        uint32_t *bucket_count_out) {
    lock_guard guard(state.mutex);
    return jit_vcall(domain, index, bucket_count_out);
}
