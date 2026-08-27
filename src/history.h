/*
    src/history.h -- Global log of captured kernel launches (kernel history)

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <vector>

/// Metadata describing a single operation captured by the kernel history
struct KernelHistoryEntry {
    /// Jit backend, for which the kernel was compiled
    JitBackend backend;

    /// Kernel type
    KernelType type;

    /// Was this kernel recorded or replayed by a frozen function?
    KernelRecordingMode recording_mode;

    /// Stores the low/high 64 bits of the 128-bit hash kernel identifier
    /// (JIT-compiled kernels only, zero otherwise)
    uint64_t hash[2];

    /// Remaining fields of the kernel cache key (see jitc_kernel_history_source())
    int cache_device;
    uint64_t cache_flags;

    /// Does the kernel contain any OptiX (ray tracing) operations?
    int uses_optix;

    /// Whether the kernel was reused from the kernel cache
    int cache_hit;

    /// Whether the kernel was loaded from the cache on disk
    int cache_disk;

    /// Launch width / number of array entries that were processed
    uint32_t size;

    /// Number of input arrays
    uint32_t input_count;

    /// Number of output arrays + side effects
    uint32_t output_count;

    /// Number of IR operations (JIT-compiled kernels only)
    uint32_t operation_count;

    /// Time (ms) spent generating the kernel intermediate representation
    float codegen_time;

    /// Time (ms) spent compiling the kernel (\c 0 if \c cache_hit is \c true)
    float backend_time;
};

/**
 * \brief One operation captured by the kernel history
 *
 * Entries are reference-counted: the global log (\ref State::kernel_history)
 * and any number of snapshots (\ref KernelHistory) share them. All fields are
 * protected by \c state.lock.
 *
 * The expensive parts of an entry resolve lazily on first access: \ref
 * jitc_kernel_history_query() converts the backend-specific timing state
 * (CUDA events, a Metal command buffer, a nanothread task handle) into a
 * millisecond value, and \ref jitc_kernel_history_source() fetches the source
 * of JIT-compiled kernels from the in-memory kernel cache.
 */
struct KernelHistoryEntryImpl {
    /// Metadata captured at launch time
    KernelHistoryEntry meta{};

    /// CUDA: events bracketing the launch, and the context that owns them
    void *event_start = nullptr, *event_end = nullptr, *cuda_context = nullptr;

    /// Metal: retained id<MTLCommandBuffer> executing the operation
    void *command_buffer = nullptr;

    /// LLVM: retained nanothread task handle
    void *task = nullptr;

    /// Execution time (ms), valid once 'timed' is set
    float execution_time = 0.f;
    bool timed = false;

    /// Cached copy of the kernel source, valid once 'source_resolved' is set
    char *source = nullptr;
    bool source_resolved = false;

    /// Reference count: one for the log, plus one per snapshot
    uint32_t ref_count = 1;

    /// Membership links of the live-entry registry (see history.cpp)
    KernelHistoryEntryImpl *reg_prev = nullptr, *reg_next = nullptr;
};

/// Opaque snapshot type of the public API: a set of shared history entries
struct KernelHistory {
    std::vector<KernelHistoryEntryImpl *> entries;
};

/**
 * \brief Global log of kernel history entries
 *
 * Kernel launches append entries while \c JitFlag::KernelHistory is set.
 * \c base holds the absolute sequence number of <tt>entries[0]</tt>; the
 * public API identifies capture regions via such sequence numbers, which
 * makes overlapping regions (nested scopes, multiple threads) well-defined.
 * The log is emptied whenever the number of open regions drops to zero.
 */
struct KernelHistoryLog {
    std::vector<KernelHistoryEntryImpl *> entries;
    uint64_t base = 0;
    uint32_t scopes = 0;
};

// The functions below require that state.lock is held. See the documentation
// of their jit_*() counterparts in drjit-core/jit.h for details.

/// Append an entry to the log; the caller may attach timing state afterwards
extern KernelHistoryEntryImpl *
jitc_kernel_history_append(const KernelHistoryEntry &meta);

extern uint64_t jitc_kernel_history_begin();
extern KernelHistory *jitc_kernel_history_end(uint64_t start);
extern KernelHistory *jitc_kernel_history_view(uint64_t start);
extern void jitc_kernel_history_free(KernelHistory *h);
extern uint64_t jitc_kernel_history_query(KernelHistory *h, size_t i,
                                          KernelHistoryField field);
extern const char *jitc_kernel_history_source(KernelHistory *h, size_t i);
extern void jitc_kernel_history_clear();

/// Release the timing state of all live entries and empty the log. Called by
/// jitc_shutdown() before the backends owning this state disappear.
extern void jitc_kernel_history_shutdown();
