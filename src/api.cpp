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

void jitc_init(int llvm, int cuda) {
    lock_guard guard(state.mutex);
    jit_init(llvm, cuda, nullptr);
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

    Stream **stream = &active_stream;

    std::thread([llvm, cuda, sync, stream]() {
        lock_guard guard2(state.mutex);
        {
            lock_guard_t<std::mutex> guard2(sync->mutex);
            sync->flag = true;
            sync->cv.notify_one();
        }
        jit_init(llvm, cuda, stream);
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

uint32_t jitc_device() {
    lock_guard guard(state.mutex);
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_device(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function!");
    return stream->device;
}

uint32_t jitc_stream() {
    lock_guard guard(state.mutex);
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_device(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function!");
    return stream->stream;
}

void jitc_set_eval_enabled(int enable) {
    lock_guard guard(state.mutex);
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_eval_enabled(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function!");
    stream->eval_enabled = enable != 0;
}

int jitc_eval_enabled() {
    lock_guard guard(state.mutex);
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_eval_enabled(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function!");
    return stream->eval_enabled ? 1 : 0;
}

uint32_t jitc_side_effect_counter() {
    lock_guard guard(state.mutex);

    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_side_effect_counter(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function!");
    return stream->side_effect_counter;
}

void* jitc_cuda_stream() {
    lock_guard guard(state.mutex);
    return jit_cuda_stream();
}

void* jitc_cuda_context() {
    lock_guard guard(state.mutex);
    return jit_cuda_context();
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

void jitc_set_parallel_dispatch(int enable) {
    lock_guard guard(state.mutex);
    jit_set_parallel_dispatch(enable != 0);
}

int jitc_parallel_dispatch() {
    lock_guard guard(state.mutex);
    return jit_parallel_dispatch() ? 1 : 0;
}

void jitc_sync_stream() {
    lock_guard guard(state.mutex);
    jit_sync_stream();
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

uint32_t jitc_var_map_mem(VarType type, int cuda, void *ptr, uint32_t size, int free) {
    lock_guard guard(state.mutex);
    return jit_var_map_mem(type, cuda, ptr, size, free);
}

uint32_t jitc_var_copy_ptr(const void *ptr, uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_copy_ptr(ptr, index);
}

uint32_t jitc_var_copy_mem(AllocType atype, VarType vtype, int cuda,
                           const void *value, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_var_copy_mem(atype, vtype, cuda, value, size);
}

uint32_t jitc_var_copy_var(uint32_t index) {
    lock_guard guard(state.mutex);
    return jit_var_copy_var(index);
}

uint32_t jitc_var_new_0(VarType type, const char *stmt, int stmt_static,
                        int cuda, uint32_t size) {
    lock_guard guard(state.mutex);
    return jit_var_new_0(type, stmt, stmt_static, cuda, size);
}

uint32_t jitc_var_new_1(VarType type, const char *stmt, int stmt_static,
                        int cuda, uint32_t arg1) {
    lock_guard guard(state.mutex);
    return jit_var_new_1(type, stmt, stmt_static, cuda, arg1);
}

uint32_t jitc_var_new_2(VarType type, const char *stmt, int stmt_static,
                        int cuda, uint32_t arg1, uint32_t arg2) {
    lock_guard guard(state.mutex);
    return jit_var_new_2(type, stmt, stmt_static, cuda, arg1, arg2);
}

uint32_t jitc_var_new_3(VarType type, const char *stmt, int stmt_static,
                        int cuda, uint32_t arg1, uint32_t arg2, uint32_t arg3) {
    lock_guard guard(state.mutex);
    return jit_var_new_3(type, stmt, stmt_static, cuda, arg1, arg2, arg3);
}

uint32_t jitc_var_new_4(VarType type, const char *stmt, int stmt_static,
                        int cuda, uint32_t arg1, uint32_t arg2, uint32_t arg3,
                        uint32_t arg4) {
    lock_guard guard(state.mutex);
    return jit_var_new_4(type, stmt, stmt_static, cuda, arg1, arg2, arg3, arg4);
}

uint32_t jitc_var_new_literal(VarType type, int cuda, uint64_t value,
                              uint32_t size, int eval) {
    lock_guard guard(state.mutex);
    return jit_var_new_literal(type, cuda, value, size, eval);
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
void jitc_set_cse(int value) {
    lock_guard guard(state.mutex);
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_set_cse(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function!");
    stream->enable_cse = value != 0;
}

/// Return whether or not common subexpression elimination is enabled
int jitc_cse() {
    lock_guard guard(state.mutex);
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_set_cse(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function!");
    return stream->enable_cse ? 1 : 0;
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

uint32_t jitc_compress(const uint8_t *in, uint32_t size, uint32_t *out) {
    lock_guard guard(state.mutex);
    return jit_compress(in, size, out);
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

void jitc_registry_set_attr(void *self, const char *name, const void *value,
                            size_t size) {
    lock_guard guard(state.mutex);
    jit_registry_set_attr(self, name, value, size);
}

const void *jitc_registry_attr_data(const char *domain, const char *name) {
    lock_guard guard(state.mutex);
    return jit_registry_attr_data(domain, name);
}

VCallBucket *jitc_vcall(const char *domain, uint32_t index,
                        uint32_t *bucket_count_out) {
    lock_guard guard(state.mutex);
    return jit_vcall(domain, index, bucket_count_out);
}

