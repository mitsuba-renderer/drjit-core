/*
    src/history.cpp -- Global log of captured kernel launches (kernel history)

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "history.h"
#include "internal.h"
#include "log.h"
#include "eval.h"
#include "strbuf.h"

#if defined(DRJIT_ENABLE_CUDA)
#  include "cuda.h"
#endif
#if defined(DRJIT_ENABLE_METAL)
#  include "metal.h"
#endif

/// Doubly linked list of all live entries
static KernelHistoryEntryImpl *registry_head = nullptr;

/// Wait for the operation to finish and convert the backend-specific timing
/// state into a millisecond value.
static void finalize_timing(KernelHistoryEntryImpl *e) {
    if (e->timed)
        return;

#if defined(DRJIT_ENABLE_CUDA)
    if (e->event_start) {
        scoped_set_context guard((CUcontext) e->cuda_context);
        CUevent event_end = (CUevent) e->event_end;
        {
            unlock_guard guard_2(state.lock);
            cuda_check(cuEventSynchronize(event_end));
        }
        if (!e->timed) {
            cuda_check(cuEventElapsedTime(&e->execution_time,
                                          (CUevent) e->event_start, event_end));
            // The events are destroyed together with the entry (see
            // release_timing()) so that concurrent waiters stay valid
            e->timed = true;
        }
        return;
    }
#endif

#if defined(DRJIT_ENABLE_METAL)
    if (e->command_buffer) {
        void *cb = e->command_buffer;
        jitc_metal_history_retain(cb);
        float time;
        {
            unlock_guard guard(state.lock);
            time = jitc_metal_history_wait(cb);
        }
        jitc_metal_history_release(cb);
        if (!e->timed) {
            e->execution_time = time;
            jitc_metal_history_release(e->command_buffer);
            e->command_buffer = nullptr;
            e->timed = true;
        }
        return;
    }
#endif

    if (e->task) {
        Task *task = (Task *) e->task;
        task_retain(task);
        {
            // task_wait() may execute other queued work inline on this
            // thread; shield the thread's LLVM state as in jitc_sync_thread()
            ThreadState *ts = thread_state_llvm;
            if (ts) {
                scoped_reset_thread_state ts_guard(ts);
                unlock_guard guard(state.lock);
                task_wait(task);
            } else {
                unlock_guard guard(state.lock);
                task_wait(task);
            }
        }
        if (!e->timed) {
            e->execution_time = (float) task_time(task);
            task_release((Task *) e->task);
            e->task = nullptr;
            e->timed = true;
        }
        task_release(task);
        return;
    }

    e->timed = true;
}

/// Release timing state of an entry that was never queried, without waiting
/// for the associated operation
static void release_timing(KernelHistoryEntryImpl *e) {
#if defined(DRJIT_ENABLE_CUDA)
    if (e->event_start) {
        scoped_set_context guard((CUcontext) e->cuda_context);
        cuda_check(cuEventDestroy((CUevent) e->event_start));
        cuda_check(cuEventDestroy((CUevent) e->event_end));
        e->event_start = e->event_end = e->cuda_context = nullptr;
    }
#endif

#if defined(DRJIT_ENABLE_METAL)
    if (e->command_buffer) {
        jitc_metal_history_release(e->command_buffer);
        e->command_buffer = nullptr;
    }
#endif

    if (e->task) {
        task_release((Task *) e->task);
        e->task = nullptr;
    }
}

static void entry_dec_ref(KernelHistoryEntryImpl *e) {
    if (--e->ref_count)
        return;

    if (e->reg_prev)
        e->reg_prev->reg_next = e->reg_next;
    else
        registry_head = e->reg_next;
    if (e->reg_next)
        e->reg_next->reg_prev = e->reg_prev;

    release_timing(e);
    free(e->source);
    delete e;
}

/// Empty the log, dropping its reference to each entry
static void log_trim() {
    KernelHistoryLog &log = state.kernel_history;
    log.base += log.entries.size();
    for (KernelHistoryEntryImpl *e : log.entries)
        entry_dec_ref(e);
    log.entries.clear();
}

/// Snapshot the log entries with sequence numbers >= start
static KernelHistory *snapshot_since(uint64_t start) {
    KernelHistoryLog &log = state.kernel_history;
    size_t first = start > log.base ? (size_t) (start - log.base) : 0,
           n     = log.entries.size();

    KernelHistory *h = new KernelHistory();
    if (first < n) {
        h->entries.reserve(n - first);
        for (size_t i = first; i < n; ++i) {
            KernelHistoryEntryImpl *e = log.entries[i];
            e->ref_count++;
            h->entries.push_back(e);
        }
    }
    return h;
}

static KernelHistoryEntryImpl *entry_check(KernelHistory *h, size_t i,
                                           const char *func) {
    if (!h || i >= h->entries.size())
        jitc_raise("%s(): entry index %zu is out of bounds!", func, i);
    return h->entries[i];
}

KernelHistoryEntryImpl *
jitc_kernel_history_append(const KernelHistoryEntry &meta) {
    KernelHistoryEntryImpl *e = new KernelHistoryEntryImpl();
    e->meta = meta;

    e->reg_next = registry_head;
    if (registry_head)
        registry_head->reg_prev = e;
    registry_head = e;

    state.kernel_history.entries.push_back(e);
    return e;
}

uint64_t jitc_kernel_history_begin() {
    KernelHistoryLog &log = state.kernel_history;
    if (log.scopes++ == 0)
        log_trim();
    return log.base + log.entries.size();
}

KernelHistory *jitc_kernel_history_end(uint64_t start) {
    KernelHistoryLog &log = state.kernel_history;
    if (log.scopes == 0)
        jitc_raise("jit_kernel_history_end(): no open capture region!");

    KernelHistory *h = snapshot_since(start);
    if (--log.scopes == 0)
        log_trim();
    return h;
}

KernelHistory *jitc_kernel_history_view(uint64_t start) {
    return snapshot_since(start);
}

void jitc_kernel_history_free(KernelHistory *h) {
    if (!h)
        return;
    for (KernelHistoryEntryImpl *e : h->entries)
        entry_dec_ref(e);
    delete h;
}

/// Convert an internal millisecond value into nanoseconds for reporting
static uint64_t time_ns(float ms) { return (uint64_t) ((double) ms * 1e6); }

uint64_t jitc_kernel_history_query(KernelHistory *h, size_t i,
                                   KernelHistoryField field) {
    KernelHistoryEntryImpl *e = entry_check(h, i, "jit_kernel_history_query");
    const KernelHistoryEntry &m = e->meta;

    switch (field) {
        case KernelHistoryField::Backend:        return (uint32_t) m.backend;
        case KernelHistoryField::Type:           return (uint32_t) m.type;
        case KernelHistoryField::RecordingMode:  return (uint32_t) m.recording_mode;
        case KernelHistoryField::HashLow:        return m.hash[0];
        case KernelHistoryField::HashHigh:       return m.hash[1];
        case KernelHistoryField::Size:           return m.size;
        case KernelHistoryField::InputCount:     return m.input_count;
        case KernelHistoryField::OutputCount:    return m.output_count;
        case KernelHistoryField::OperationCount: return m.operation_count;
        case KernelHistoryField::UsesOptix:      return m.uses_optix ? 1 : 0;
        case KernelHistoryField::CacheHit:       return m.cache_hit ? 1 : 0;
        case KernelHistoryField::CacheDisk:      return m.cache_disk ? 1 : 0;
        case KernelHistoryField::CodegenTime:    return time_ns(m.codegen_time);
        case KernelHistoryField::BackendTime:    return time_ns(m.backend_time);

        case KernelHistoryField::ExecutionTime:
            finalize_timing(e);
            return time_ns(e->execution_time);

        default:
            jitc_raise("jit_kernel_history_query(): unknown field!");
    }
}

const char *jitc_kernel_history_source(KernelHistory *h, size_t i) {
    KernelHistoryEntryImpl *e = entry_check(h, i, "jit_kernel_history_source");
    if (e->source_resolved)
        return e->source;
    e->source_resolved = true;

    const KernelHistoryEntry &m = e->meta;
    if (m.type != KernelType::JIT)
        return nullptr;

    KernelKey key(XXH128_hash_t{ m.hash[0], m.hash[1] }, m.cache_device,
                  m.cache_flags);
    auto it = state.kernel_cache.find(key);
    if (it == state.kernel_cache.end())
        return nullptr;

    const Kernel &kernel = it.value();
    if (!kernel.src)
        return nullptr;

#if defined(DRJIT_ENABLE_METAL)
    if (jitc_is_metal(m.backend)) {
        StringBuffer tmp(kernel.src_size + 1024);
        jitc_metal_format(kernel.src, kernel.src_size, tmp);
        e->source = (char *) malloc_check(tmp.size() + 1);
        memcpy(e->source, tmp.get(), tmp.size() + 1);
        return e->source;
    }
#endif

    e->source = (char *) malloc_check(kernel.src_size + 1);
    memcpy(e->source, kernel.src, kernel.src_size + 1);
    return e->source;
}

void jitc_kernel_history_clear() {
    if (state.kernel_history.scopes == 0)
        log_trim();
}

void jitc_kernel_history_shutdown() {
    // Drop the backend timing handles (e.g. CUDA events) while the backends
    // still exist
    for (KernelHistoryEntryImpl *e = registry_head; e; e = e->reg_next)
        release_timing(e);
    log_trim();
    state.kernel_history.scopes = 0;
}
