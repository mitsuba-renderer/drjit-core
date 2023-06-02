/*
    src/init.cpp -- Initialization and shutdown of the core parts of DrJit

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "init.h"
#include "log.h"
#include "registry.h"
#include "state.h"
#include "profiler.h"
#include "thread_state.h"
#include "backends.h"
#include <sys/stat.h>

#if defined(DRJIT_ENABLE_ITTNOTIFY)
__itt_domain *drjit_domain = __itt_domain_create("drjit");
#endif

static_assert(
    sizeof(VariableKey) == 9 * sizeof(uint32_t),
    "VariableKey: incorrect size, likely an issue with padding/packing!");

static_assert(
    sizeof(tsl::detail_robin_hash::bucket_entry<VariableMap::value_type, false>) == 64,
    "VariableMap: incorrect bucket size, likely an issue with padding/packing!");

static ProfilerRegion profiler_region_init("jit_init");

#if defined(_WIN32)
extern float timer_frequency_scale = 0;
#endif

static void jitc_init_temp_path();

/// Initialize core data structures of the JIT compiler
void jitc_init(uint32_t backends) {
    ProfilerPhase profiler(profiler_region_init);

    if (!state.temp_path)
        jitc_init_temp_path();

#if defined(_WIN32)
    // Initialize frequency scale for performance counters
    LARGE_INTEGER timer_frequency;
    QueryPerformanceFrequency(&timer_frequency);
    timer_frequency_scale = 1e6f / timer_frequency.QuadPart;
#endif

    // Enumerate CUDA devices and collect suitable ones
    jitc_log(Info, "jit_init(): detecting devices ..");

    // Filter backends that are not available on the current platform
    uint32_t available_backends = 0;

#if defined(DRJIT_ENABLE_LLVM)
    available_backends |= (1 << (uint32_t) JitBackend::LLVM);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    available_backends |= (1 << (uint32_t) JitBackend::CUDA);
#endif

#if defined(DRJIT_ENABLE_METAL)
    available_backends |= (1 << (uint32_t) JitBackend::Metal);
#endif

    // Filter backends that aren't available or which are already initialized
    backends &= available_backends;
    backends &= ~state.backends;

    if (!backends)
        return;

    // Now, try enabling the rest
#if defined(DRJIT_ENABLE_LLVM)
    if ((backends & (1 << (uint32_t) JitBackend::LLVM)) && jitc_llvm_init())
        state.backends |= 1 << (uint32_t) JitBackend::LLVM;
#endif

#if defined(DRJIT_ENABLE_CUDA)
    if ((backends & (1 << (uint32_t) JitBackend::CUDA)) && jitc_cuda_init())
        state.backends |= 1 << (uint32_t) JitBackend::CUDA;
#endif

#if defined(DRJIT_ENABLE_METAL)
    if ((backends & (1 << (uint32_t) JitBackend::Metal)) && jitc_metal_init())
        state.backends |= 1 << (uint32_t) JitBackend::Metal;
#endif

    state.variable_index = 1;
    state.variable_watermark = 0;

    state.kernel_hard_misses = state.kernel_soft_misses = 0;
    state.kernel_hits = state.kernel_launches = 0;
}

/// Release all resources used by the JIT compiler, and report reference leaks.
void jitc_shutdown(int light) {
    // Synchronize with everything
    for (ThreadState *ts : state.tss)
        ts->sync();

    if (!state.kernel_cache.empty()) {
        jitc_log(Info, "jit_shutdown(): releasing %zu kernel%s ..",
                state.kernel_cache.size(),
                state.kernel_cache.size() > 1 ? "s" : "");

        for (auto &v : state.kernel_cache) {
            jitc_kernel_free(v.first.device, v.second);
            free(v.first.str);
        }

        state.kernel_cache.clear();
    }

    state.kernel_history.clear();

    if (!state.tss.empty()) {
        jitc_log(Info, "jit_shutdown(): releasing %zu thread state%s ..",
                state.tss.size(), state.tss.size() > 1 ? "s" : "");

        for (ThreadState *ts : state.tss) {
            for (uint32_t index : ts->side_effects)
                jitc_var_dec_ref(index);

            if (!ts->prefix_stack.empty()) {
                for (char *s : ts->prefix_stack)
                    free(s);
                jitc_log(Warn,
                         "jit_shutdown(): leaked %zu prefix stack entries.",
                         ts->prefix_stack.size());
                free(ts->prefix);
            }

            if (!ts->mask_stack.empty())
                jitc_log(Warn, "jit_shutdown(): leaked %zu active masks!",
                         ts->mask_stack.size());

            delete ts;
        }

        if (state.variables.empty() && !state.lvn_map.empty()) {
            for (auto &kv: state.lvn_map)
                jitc_log(Warn,
                        " - id=%u: size=%u, type=%s, dep=[%u, "
                        "%u, %u, %u]",
                        kv.second, kv.first.size, type_name[kv.first.type],
                        kv.first.dep[0], kv.first.dep[1], kv.first.dep[2],
                        kv.first.dep[3]);

            jitc_log(Warn, "jit_shutdown(): detected a common subexpression "
                           "elimination cache leak (see above).");
        }

        state.tss.clear();
    }

    if (std::max(state.log_level_stderr, state.log_level_callback) >= LogLevel::Warn) {
        uint32_t n_leaked = 0;
        for (auto &var : state.variables) {
            if (n_leaked == 0)
                jitc_log(Warn, "jit_shutdown(): detected variable leaks:");
            if (n_leaked < 10)
                jitc_log(Warn,
                         " - variable r%u is still being referenced! "
                         "(ref=%u, ref_se=%u, type=%s, size=%u, "
                         "stmt=\"%s\", dep=[%u, %u, %u, %u])",
                         var.first,
                         (uint32_t) var.second.ref_count,
                         (uint32_t) var.second.ref_count_se,
                         type_name[var.second.type],
                         var.second.size,
                         var.second.is_literal()
                             ? "<value>"
                             : (var.second.stmt ? var.second.stmt : "<null>"),
                         var.second.dep[0], var.second.dep[1],
                         var.second.dep[2], var.second.dep[3]);
            else if (n_leaked == 10)
                jitc_log(Warn, " - (skipping remainder)");
            ++n_leaked;
        }

        if (n_leaked > 0)
            jitc_log(Warn, "jit_shutdown(): %u variables are still referenced!", n_leaked);

        if (state.variables.empty() && !state.extra.empty()) {
            jitc_log(Warn,
                    "jit_shutdown(): %zu 'extra' records were not cleaned up:",
                    state.extra.size());
            n_leaked = 0;
            for (const auto &kv : state.extra) {
                jitc_log(Warn, "- variable r%u", kv.first);
                if (++n_leaked == 10) {
                    jitc_log(Warn, " - (skipping remainder)");
                    break;
                }
            }
        }
    }

    jitc_registry_shutdown();
    jitc_malloc_shutdown();

    jitc_log(Info, "jit_shutdown(light=%u): done", (uint32_t) light);

#if defined(DRJIT_ENABLE_LLVM)
    thread_state_llvm = nullptr;
    if (!light)
        jitc_llvm_shutdown();
#endif

#if defined(DRJIT_ENABLE_OPTIX)
    if (!light)
        jitc_optix_shutdown();
#endif

#if defined(DRJIT_ENABLE_CUDA)
    thread_state_cuda = nullptr;
    if (!light)
        jitc_cuda_shutdown();
#endif

#if defined(DRJIT_ENABLE_METAL)
    thread_state_metal = nullptr;
    if (!light)
        jitc_metal_shutdown();
#endif

    state.backends = 0;
}

/// Abstractions around UTF-8 on Linux/macOS and UTF-16 on Windows
#if !defined(_WIN32)
#  define uchar        char
#  define ustr(x)      x
#  define ugetenv      getenv
#  define umkdir(x)    mkdir(x, 0700)
#  define ustat_desc   struct stat
#  define ustat        stat
#  define usnprintf    snprintf
#  define ustrdup      strdup
#else
#  define uchar        wchar_t
#  define ustr(x)      L##x
#  define ugetenv      _wgetenv
#  define umkdir(x)     _wmkdir(x)
#  define ustat_desc   struct _stat
#  define ustat        _wstat
#  define usnprintf    _snwprintf
#  define ustrdup      _wcsdup
#endif

/// Create a temporary directory for Dr.Jit on Windows
static void jitc_init_temp_path() {
    uchar *path = ugetenv(ustr("DRJIT_CACHE_DIR"));
    uchar temp[512];

    // Set a path in the home/temp directory
    if (!path) {
#if !defined(_WIN32)
        const uchar *base = ugetenv(ustr("HOME")), *fmt = ustr("%s/.drjit");
        bool fail = !base;
#else
        const uchar *fmt = ustr("%sdrjit");
        uchar base[512];
        bool fail = GetTempPathW(sizeof(base) / sizeof(uchar), base) == 0;
#endif

        if (fail)
            jitc_fail("jit_init(): could not obtain a base path");

        usnprintf(temp, sizeof(temp) / sizeof(uchar), fmt, base);
        path = temp;
    }

    state.temp_path = ustrdup(path);

    ustat_desc st { };
    int rv = ustat(path, &st);

    if (rv == -1) {
#if !defined(_WIN32)
        char *path_u8 = path;
#else
        char path_u8[512];
        wcstombs(path_u8, path, sizeof(path_u8));
#endif

        jitc_log(Info, "jit_init(): creating directory \"%s\" ..", path_u8);
        if (umkdir(state.temp_path) == -1)
            jitc_fail("jit_init(): creation of directory \"%s\" failed: %s",
                path_u8, strerror(errno));
    }
}

