/*
    src/unit.h -- shared infrastructure for assembling compilation units

    Dr.Jit splits each kernel into a set of separately compiled compilation
    units: a main entry point and a set of callables. Compilation units can call
    each other through the dispatch table but otherwise do not reference each
    other's symbols. This header and ``unit.cpp`` implement a pool of
    ``UnitBuilder`` instances that enable buffer reuse and minimize dynamic
    memory allocation during code generation of the compilation units.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "strbuf.h"
#include "hash.h"
#include "log.h"
#include <tsl/robin_set.h>
#include <nanothread/nanothread.h>
#include <algorithm>
#include <vector>

enum class JitBackend : uint32_t;

/// A compilation unit under construction
struct UnitBuilder {
    /// The unit's complete source (prologue + preamble + body + epilogue)
    StringBuffer src;

    /// Per-unit deduplication of preamble entries
    tsl::robin_set<XXH128_hash_t, XXH128Hasher, XXH128Eq> registered;

    /// Content hash of ``src``
    XXH128_hash_t unit_hash { 0, 0 };

    void clear() {
        src.clear();
        registered.clear();
    }
};

/// A retained indirect-callable unit. After jitc_unit_finalize(), the vector
/// below is sorted by body hash, and position determines the callable index.
struct CallableUnit {
    /// The callable's body hash determining the 'func_<hash>' symbol name
    XXH128_hash_t hash;

    /// The unit holding its definition
    UnitBuilder *unit;
};

/// Backend prologue shared by every unit of the kernel being assembled
extern StringBuffer unit_prologue;

/// Backend epilogue shared by every unit
extern StringBuffer unit_epilogue;

/// The kernel entry unit.
extern UnitBuilder *unit_entry;

/// Indirect-callable units, appended in registration order and sorted into
/// callable-index order by jitc_unit_finalize()
extern std::vector<CallableUnit> callable_units;

/// Reset the unit builders before starting to assemble a kernel
extern void jitc_unit_reset();

/// Begin assembling an indirect callable in a fresh unit
extern UnitBuilder *jitc_unit_push();

/// Is a callable with the given body hash already registered in this kernel?
extern bool jitc_unit_callable_known(XXH128_hash_t hash);

/// Retain the active unit (the ``buffer`` data starting at ``body_start``).
/// The caller already provides a hash.
extern void jitc_unit_pop_keep(UnitBuilder *unit, XXH128_hash_t hash,
                               size_t body_start);

/// Discard the active callable unit (a duplicate) and rewind ``buffer``
extern void jitc_unit_pop_discard(UnitBuilder *unit, size_t body_start);

/// Move the preamble entry (a global declaration or direct callable) rendered
/// between ``start`` and the tail of ``buffer`` into the active unit's
/// preamble and rewind the buffer. Drops duplicate entries.
extern void jitc_unit_capture_preamble(size_t start);

/// As above, but with a caller-supplied hash. Returns true if the entry was new.
extern bool jitc_unit_capture_preamble(XXH128_hash_t hash, size_t start);

/// Sort the registered callables by body hash and assign callable indices
extern void jitc_unit_finalize(JitBackend backend);

/// Return the callable index assigned to the given body hash (post-finalize)
extern uint32_t jitc_unit_callable_index(XXH128_hash_t hash);

/// Finish and hash the entry unit, then combine it with the callable hashes
/// into the returned kernel identifier. Afterwards, all unit text lives in
/// the pooled builders, and ``buffer`` holds nothing of value.
extern XXH128_hash_t jitc_unit_finish_kernel();

/// Concatenate the sources of all units verbatim, separated by horizontal
/// rules, for inspection via PrintIR and the kernel history. The returned
/// pointer is valid until the next call.
extern const char *jitc_unit_materialize_print(size_t *size_out);

/// Return unit ``i`` of the current kernel: index 0 is the entry unit,
/// followed by the callables in callable-index order.
extern UnitBuilder *jitc_unit_at(size_t i);

/// Common fields of one per-unit compilation job: the kernel entry point at
/// index 0, followed by the indirect callables in callable-index order.
///
/// The backends release 'state.lock' during the (potentially long)
/// compilation so that other threads can keep using the API. 'source'
/// points into the unit's pooled 'src' buffer, which stays valid
/// throughout: 'state.eval_lock' remains held for the entire eval, so at
/// most one kernel compiles at a time, and only assembly (excluded by that
/// lock) touches the builder pool.
struct UnitCompileJob {
    XXH128_hash_t unit_hash;
    char symbol[64];
    const char *source = nullptr;
    size_t source_size = 0;
};

/// Fill the common fields of unit ``i``'s compilation job and return the unit
extern UnitBuilder *jitc_unit_job_init(size_t i, UnitCompileJob &job);

/// Invoke ``work(index)`` for every entry of ``indices`` concurrently on the
/// nanothread pool, issuing the largest jobs (as reported by ``size(index)``)
/// first, and wait for completion.
template <typename SizeFn, typename WorkFn>
void jitc_unit_compile_parallel(std::vector<uint32_t> &indices, SizeFn &&size,
                                WorkFn &&work) {
    std::sort(indices.begin(), indices.end(),
              [&](uint32_t a, uint32_t b) { return size(a) > size(b); });

    struct Payload {
        WorkFn &work;
        const uint32_t *indices;
    } payload { work, indices.data() };

    Task *task = task_submit_dep(
        nullptr, nullptr, 0, (uint32_t) indices.size(),
        [](uint32_t i, void *p) {
            Payload *pl = (Payload *) p;
            pl->work(pl->indices[i]);
        },
        &payload, 0, nullptr, /* always_async = */ 1);
    task_wait_exclusive(task);
    task_release(task);
    jitc_log_flush();
}

/// A compiled per-unit artifact held by the unit cache. CUDA entries are
/// empty and only record that the driver's own cache holds the unit.
struct UnitArtifact {
    void *ptr[2];   // LLVM: resource tracker, OptiX: module, Metal: retained library + function
    uint64_t value; // LLVM: entry point address
    uint32_t size;  // LLVM: object file size
};

/// Look up the artifact of a unit compiled earlier, keyed by content hash
/// and a backend-specific salt (device identity, compile options). The
/// cache pins artifacts until jitc_unit_cache_flush().
extern bool jitc_unit_cache_lookup(JitBackend backend, XXH128_hash_t hash,
                                   uint64_t salt, UnitArtifact &out);

/// Publish a freshly compiled artifact. If a concurrent compile got there
/// first, 'artifact' is released and replaced by the cached one. 'release'
/// (may be null) is also used to evict the entry later.
extern void jitc_unit_cache_insert(JitBackend backend, XXH128_hash_t hash,
                                   uint64_t salt, UnitArtifact &artifact,
                                   void (*release)(UnitArtifact &));

/// Evict the artifacts of one backend (or all, backend = -1). Must run after
/// the kernels referencing them have been freed.
extern void jitc_unit_cache_flush(int backend = -1);
