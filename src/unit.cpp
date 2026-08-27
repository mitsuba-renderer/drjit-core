/*
    src/unit.cpp -- Compilation units assembled during code generation

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "unit.h"
#include "eval.h"
#include "internal.h"
#include "log.h"
#include <mutex>

StringBuffer unit_prologue { 256 };
StringBuffer unit_epilogue { 256 };

UnitBuilder *unit_entry = nullptr;
std::vector<CallableUnit> callable_units;

/// Pool of unit builders, reused across kernels. Only grows.
static std::vector<UnitBuilder *> unit_pool;
static uint32_t unit_pool_used = 0;

/// Stack of active units; the entry unit sits at the bottom
static std::vector<UnitBuilder *> unit_stack;

/// Kernel-level registry of indirect callables: body hash -> callable index
/// (assigned in jitc_unit_finalize(), zero until then)
static tsl::robin_map<XXH128_hash_t, uint32_t, XXH128Hasher, XXH128Eq>
    callable_registry;

/// Scratch buffer into which jitc_unit_materialize_print() concatenates the
/// unit sources
static StringBuffer unit_scratch;

static UnitBuilder *jitc_unit_alloc() {
    UnitBuilder *unit;
    if (unit_pool_used < unit_pool.size()) {
        unit = unit_pool[unit_pool_used];
        unit->clear();
    } else {
        unit = new UnitBuilder();
        unit_pool.push_back(unit);
    }
    unit_pool_used++;
    return unit;
}

void jitc_unit_reset() {
    unit_pool_used = 0;
    unit_stack.clear();
    callable_registry.clear();
    callable_units.clear();
    unit_prologue.clear();
    unit_epilogue.clear();
    unit_entry = jitc_unit_alloc();
    unit_stack.push_back(unit_entry);
}

UnitBuilder *jitc_unit_push() {
    UnitBuilder *unit = jitc_unit_alloc();
    unit_stack.push_back(unit);
    return unit;
}

bool jitc_unit_callable_known(XXH128_hash_t hash) {
    return callable_registry.find(hash) != callable_registry.end();
}

/// Finish the unit's source by appending its body and the epilogue to the
/// prologue/preamble already in ``src``, then hash the result
static void jitc_unit_finish(UnitBuilder *unit, const char *body,
                             size_t body_size) {
    StringBuffer &src = unit->src;
    // The prologue is placed by whichever write to 'src' comes first: the
    // initial preamble append, or this point for units without a preamble
    if (src.size() == 0)
        src.put(unit_prologue.get(), unit_prologue.size());
    src.put(body, body_size);
    src.put(unit_epilogue.get(), unit_epilogue.size());
    unit->unit_hash = XXH128(src.get(), src.size(), 0);
}

void jitc_unit_pop_keep(UnitBuilder *unit, XXH128_hash_t hash,
                        size_t body_start) {
    jitc_assert(unit_stack.back() == unit,
                "jitc_unit_pop_keep(): unit stack corruption!");
    unit_stack.pop_back();

    jitc_unit_finish(unit, buffer.get() + body_start,
                     buffer.size() - body_start);
    buffer.rewind_to(body_start);

    callable_registry.emplace(hash, 0);
    callable_units.push_back({ hash, unit });
}

void jitc_unit_pop_discard(UnitBuilder *unit, size_t body_start) {
    jitc_assert(unit_stack.back() == unit,
                "jitc_unit_pop_discard(): unit stack corruption!");
    unit_stack.pop_back();
    buffer.rewind_to(body_start);

    // Reclaim the pool slot when the discarded unit is still on top. This is
    // expected to always hold (a duplicate's nested callables assemble as
    // duplicates too, so nothing newer can have been retained). If it ever
    // does not, the slot merely stays allocated until the next kernel.
    if (unit_pool[unit_pool_used - 1] == unit)
        unit_pool_used--;
}

/// Append '[ptr, ptr+length)' to the active unit's preamble under 'hash'
static bool jitc_unit_preamble_append(XXH128_hash_t hash, const char *ptr,
                                      size_t length) {
    UnitBuilder *unit = unit_stack.back();
    if (!unit->registered.insert(hash).second)
        return false;

    StringBuffer &src = unit->src;
    if (src.size() == 0)
        src.put(unit_prologue.get(), unit_prologue.size());

    src.put(ptr, length);
    src.put('\n');
    return true;
}

void jitc_unit_capture_preamble(size_t start) {
    jitc_unit_capture_preamble(
        XXH128(buffer.get() + start, buffer.size() - start, 0), start);
}

bool jitc_unit_capture_preamble(XXH128_hash_t hash, size_t start) {
    bool result = jitc_unit_preamble_append(hash, buffer.get() + start,
                                            buffer.size() - start);
    buffer.rewind_to(start);
    return result;
}

void jitc_unit_finalize(JitBackend backend) {
    // Order by body hash rather than registration order, which can be
    // non-deterministic in programs that use Dr.Jit with parallelization
    std::sort(callable_units.begin(), callable_units.end(),
              [](const CallableUnit &a, const CallableUnit &b) {
                  return std::tie(a.hash.high64, a.hash.low64) <
                         std::tie(b.hash.high64, b.hash.low64);
              });

    // LLVM and CUDA reserve entry 0 of the dispatch table
    bool one_based =
        jitc_is_llvm(backend) || (jitc_is_cuda(backend) && !uses_optix);
    uint32_t base = one_based ? 1 : 0;
    for (uint32_t i = 0; i < (uint32_t) callable_units.size(); ++i)
        callable_registry[callable_units[i].hash] = base + i;
}

uint32_t jitc_unit_callable_index(XXH128_hash_t hash) {
    auto it = callable_registry.find(hash);
    if (unlikely(it == callable_registry.end()))
        jitc_fail("jitc_unit_callable_index(): could not find callable!");
    return it->second;
}

XXH128_hash_t jitc_unit_finish_kernel() {
    // Finish and hash the entry unit. Its body (the current ``buffer``
    // contents) still holds the '^^^' name placeholder that jitc_assemble()
    // patches afterwards.
    jitc_unit_finish(unit_entry, buffer.get(), buffer.size());

    // A kernel without callables consists of the entry unit alone
    if (callable_units.empty())
        return unit_entry->unit_hash;

    XXH3_state_t xs;
    XXH3_128bits_reset(&xs);
    XXH3_128bits_update(&xs, &unit_entry->unit_hash, sizeof(XXH128_hash_t));
    for (const CallableUnit &cu : callable_units)
        XXH3_128bits_update(&xs, &cu.unit->unit_hash, sizeof(XXH128_hash_t));
    return XXH3_128bits_digest(&xs);
}

const char *jitc_unit_materialize_print(size_t *size_out) {
    static const char separator[] =
        "\n============================================="
        "===================================\n\n";

    StringBuffer &out = unit_scratch;
    size_t n_units = 1 + callable_units.size();

    out.clear();
    for (size_t i = 0; i < n_units; ++i) {
        if (i > 0)
            out.put(separator);
        const UnitBuilder *unit = jitc_unit_at(i);
        out.put(unit->src.get(), unit->src.size());
    }

    if (size_out)
        *size_out = out.size();
    return out.get();
}

// ============================================================================
//  In-memory cache of compiled per-unit artifacts
// ============================================================================

struct UnitCacheKey {
    XXH128_hash_t hash;
    uint64_t salt;
    uint32_t backend;

    bool operator==(const UnitCacheKey &k) const {
        return hash.low64 == k.hash.low64 && hash.high64 == k.hash.high64 &&
               salt == k.salt && backend == k.backend;
    }
};

struct UnitCacheKeyHasher {
    size_t operator()(const UnitCacheKey &k) const {
        return (size_t) k.hash.low64 + (size_t) k.salt + k.backend;
    }
};

struct UnitCacheEntry {
    UnitArtifact artifact;
    void (*release)(UnitArtifact &);
};

static tsl::robin_map<UnitCacheKey, UnitCacheEntry, UnitCacheKeyHasher>
    unit_cache;

/// Guards ``unit_cache``. Never held while acquiring ``state.lock``.
static std::mutex unit_cache_mutex;

bool jitc_unit_cache_lookup(JitBackend backend, XXH128_hash_t hash,
                            uint64_t salt, UnitArtifact &out) {
    std::lock_guard<std::mutex> guard(unit_cache_mutex);
    auto it = unit_cache.find({ hash, salt, (uint32_t) backend });
    if (it == unit_cache.end())
        return false;
    out = it->second.artifact;
    return true;
}

void jitc_unit_cache_insert(JitBackend backend, XXH128_hash_t hash,
                            uint64_t salt, UnitArtifact &artifact,
                            void (*release)(UnitArtifact &)) {
    std::lock_guard<std::mutex> guard(unit_cache_mutex);
    auto [it, inserted] = unit_cache.try_emplace(
        UnitCacheKey { hash, salt, (uint32_t) backend },
        UnitCacheEntry { artifact, release });
    if (!inserted) {
        // A concurrent compile beat us to it: use its artifact
        if (release)
            release(artifact);
        artifact = it->second.artifact;
    }
}

void jitc_unit_cache_flush(int backend) {
    std::lock_guard<std::mutex> guard(unit_cache_mutex);
    for (auto it = unit_cache.begin(); it != unit_cache.end(); ) {
        if (backend != -1 && it->first.backend != (uint32_t) backend) {
            ++it;
            continue;
        }
        UnitCacheEntry entry = it->second;
        if (entry.release)
            entry.release(entry.artifact);
        it = unit_cache.erase(it);
    }
}

UnitBuilder *jitc_unit_at(size_t i) {
    return i == 0 ? unit_entry : callable_units[i - 1].unit;
}

/// Write unit ``i``'s symbol name (``drjit_<hash>`` / ``func_<hash>``) to ``buf``
static void jitc_unit_symbol(size_t i, char *buf, size_t size) {
    if (i == 0)
        snprintf(buf, size, "%s", kernel_name);
    else
        snprintf(buf, size, "func_%016llx%016llx",
                 (unsigned long long) callable_units[i - 1].hash.high64,
                 (unsigned long long) callable_units[i - 1].hash.low64);
}

UnitBuilder *jitc_unit_job_init(size_t i, UnitCompileJob &job) {
    UnitBuilder *unit = jitc_unit_at(i);
    job.unit_hash = unit->unit_hash;
    job.source = unit->src.get();
    job.source_size = unit->src.size();
    jitc_unit_symbol(i, job.symbol, sizeof(job.symbol));
    return unit;
}
