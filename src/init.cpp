/*
    src/init.cpp -- Initialization and shutdown of the core parts of DrJit

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "drjit-core/jit.h"
#include "internal.h"
#include "cuda_ts.h"
#include "llvm_ts.h"
#include "malloc.h"
#include "internal.h"
#include "log.h"
#include "registry.h"
#include "var.h"
#include "profile.h"
#include "strbuf.h"
#include <sys/stat.h>

#include "nvtx_api.h"

#if defined(DRJIT_ENABLE_OPTIX)
#  include "optix_api.h"
#  include "optix.h"
#endif

#if defined(_WIN32)
#  include <windows.h>
#  include <direct.h>
#else
#  include <glob.h>
#  include <dlfcn.h>
#endif

State state;

#if !defined(_WIN32)
  char* jitc_temp_path = nullptr;
#else
  wchar_t* jitc_temp_path = nullptr;
#endif

#if defined(_MSC_VER)
  __declspec(thread) ThreadState* thread_state_cuda = nullptr;
  __declspec(thread) ThreadState* thread_state_llvm = nullptr;
  __declspec(thread) uint32_t jitc_flags_v = (uint32_t) JitFlag::Default;
  __declspec(thread) JitBackend default_backend = JitBackend::None;
#else
  __thread ThreadState* thread_state_cuda = nullptr;
  __thread ThreadState* thread_state_llvm = nullptr;
  __thread uint32_t jitc_flags_v = (uint32_t) JitFlag::Default;
  __thread JitBackend default_backend = JitBackend::None;
#endif

#if defined(DRJIT_ENABLE_ITTNOTIFY)
__itt_domain *drjit_domain = __itt_domain_create("drjit");
#endif

static ProfilerRegion profiler_region_init("jit_init");

#if defined(_WIN32)
extern float timer_frequency_scale;
#endif

/// Initialize core data structures of the JIT compiler
void jitc_init(uint32_t backends) {
    ProfilerPhase profiler(profiler_region_init);

#if defined(_WIN32)
    // Initialize frequency scale for performance counters
    LARGE_INTEGER timer_frequency;
    QueryPerformanceFrequency(&timer_frequency);
    timer_frequency_scale = 1e6f / timer_frequency.QuadPart;
#endif

#if defined(__APPLE__)
    backends &= ~(uint32_t) JitBackend::CUDA;
#endif

    if ((backends & ~state.backends) == 0)
        return;

#if !defined(_WIN32)
    char temp_path[512];
    snprintf(temp_path, sizeof(temp_path), "%s/.drjit", getenv("HOME"));
    struct stat st = {};
    int rv = stat(temp_path, &st);
    size_t temp_path_size = (strlen(temp_path) + 1) * sizeof(char);
    jitc_temp_path = (char*) malloc(temp_path_size);
    memcpy(jitc_temp_path, temp_path, temp_path_size);
#else
    wchar_t temp_path_w[512];
    char temp_path[512];
    if (GetTempPathW(sizeof(temp_path_w) / sizeof(wchar_t), temp_path_w) == 0)
        jitc_fail("jit_init(): could not obtain path to temporary directory!");
    wcsncat(temp_path_w, L"drjit", sizeof(temp_path) / sizeof(wchar_t));
    struct _stat st = {};
    int rv = _wstat(temp_path_w, &st);
    size_t temp_path_size = (wcslen(temp_path_w) + 1) * sizeof(wchar_t);
    jitc_temp_path = (wchar_t*) malloc(temp_path_size);
    memcpy(jitc_temp_path, temp_path_w, temp_path_size);
    wcstombs(temp_path, temp_path_w, sizeof(temp_path));
#endif

    if (rv == -1) {
        jitc_log(Info, "jit_init(): creating directory \"%s\" ..", temp_path);
#if !defined(_WIN32)
        if (mkdir(temp_path, 0700) == -1 && errno != EEXIST)
#else
        if (_wmkdir(temp_path_w) == -1 && errno != EEXIST)
#endif
            jitc_fail("jit_init(): creation of directory \"%s\" failed: %s",
                temp_path, strerror(errno));
    }

    // Enumerate CUDA devices and collect suitable ones
    jitc_log(Info, "jit_init(): detecting devices ..");

    if ((backends & ~state.backends) == 0)
        return;

    if ((backends & (uint32_t) JitBackend::LLVM) && jitc_llvm_init())
        state.backends |= (uint32_t) JitBackend::LLVM;

    if ((backends & (uint32_t) JitBackend::CUDA) && jitc_cuda_init())
        state.backends |= (uint32_t) JitBackend::CUDA;

    state.variable_counter = 0;
    state.kernel_hard_misses = state.kernel_soft_misses = 0;
    state.kernel_hits = state.kernel_launches = 0;
    jitc_nvtx_init();
}

void* jitc_cuda_stream() {
    return (void*) thread_state(JitBackend::CUDA)->stream;
}

void* jitc_cuda_context() {
    return (void*) thread_state(JitBackend::CUDA)->context;
}

void jitc_cuda_push_context(void* ctx) {
    cuda_check(cuCtxPushCurrent((CUcontext) ctx));
}

void* jitc_cuda_pop_context() {
    CUcontext out;
    cuda_check(cuCtxPopCurrent(&out));
    return out;
}

/// Release all resources used by the JIT compiler, and report reference leaks.
void jitc_shutdown(int light) {
    // Synchronize with everything
    for (ThreadState *ts : state.tss) {
        if (ts->backend == JitBackend::CUDA) {
            scoped_set_context guard(ts->context);
            cuda_check(cuStreamSynchronize(ts->stream));
        }
        if (!ts->mask_stack.empty() && state.leak_warnings)
            jitc_log(Warn, "jit_shutdown(): leaked %zu active masks!",
                     ts->mask_stack.size());
    }

    if (jitc_task) {
        Task *task = jitc_task;
        task_wait(task);
        if (jitc_task == task) {
            task_release(task);
            jitc_task = nullptr;
        }
    }

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

    // CUDA: Try to already free some memory asynchronously (faster)
    if (thread_state_cuda && thread_state_cuda->memory_pool) {
        ThreadState *ts = thread_state_cuda;
        scoped_set_context guard2(ts->context);

        lock_guard guard(state.alloc_free_lock);
        for (auto it = state.alloc_free.begin(); it != state.alloc_free.end(); ++it) {
            auto [size, type, device] = alloc_info_decode(it->first);
            (void) device;

            if (type != AllocType::Device)
                continue;

            std::vector<void *> entries;
            entries.swap(it.value());
            state.alloc_allocated[(int) type] -= size * entries.size();

            for (void *ptr : entries)
                cuda_check(cuMemFreeAsync((CUdeviceptr) ptr, ts->stream));
        }
    }

#if defined(DRJIT_ENABLE_OPTIX)
    // Free the default OptiX shader binding table and pipeline (ref counting)
    if (state.optix_default_sbt_index) {
        jitc_var_dec_ref(state.optix_default_sbt_index);
        state.optix_default_sbt_index = 0;
    }
#endif

    if (!state.tss.empty()) {
        jitc_log(Info, "jit_shutdown(): releasing %zu thread state%s ..",
                state.tss.size(), state.tss.size() > 1 ? "s" : "");

        for (ThreadState *ts : state.tss) {
            for (uint32_t index : ts->side_effects)
                jitc_var_dec_ref(index);

            if (ts->backend == JitBackend::CUDA && ts->stream) {
                scoped_set_context guard(ts->context);
                cuda_check(cuStreamSynchronize(ts->stream));
            }

            if (!ts->prefix_stack.empty()) {
                for (char *s : ts->prefix_stack)
                    free(s);
                if (state.leak_warnings)
                    jitc_log(Warn,
                             "jit_shutdown(): leaked %zu prefix stack entries.",
                             ts->prefix_stack.size());
                free(ts->prefix);
            }

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

            if (state.leak_warnings)
                jitc_log(Warn, "jit_shutdown(): detected a common subexpression "
                               "elimination cache leak (see above).");
        }

        pool_destroy();
        state.tss.clear();
    }

    thread_state_llvm = nullptr;
    thread_state_cuda = nullptr;

    if (std::max(state.log_level_stderr, state.log_level_callback) >= LogLevel::Warn &&
        state.leak_warnings) {
        size_t n_leaked = state.variables.size() - state.unused_variables.size() - 1;

        if (n_leaked > 0) {
            jitc_log(Warn, "jit_shutdown(): detected %zu variable leaks:", n_leaked);
            n_leaked = 0;

            for (size_t i = 1; i < state.variables.size(); ++i) {
                const Variable &v = state.variables[i];
                if (v.ref_count == 0 && v.ref_count_se == 0)
                    continue;

                if (n_leaked < 50) {
                    buffer.clear();
                    buffer.fmt("%s r%zu[%u] = %s(", type_name[v.type], i,
                               v.size, var_kind_name[v.kind]);
                    bool prev_dep = false;
                    for (int j = 0; j < 4; ++j) {
                        if (!v.dep[j])
                            continue;
                        if (prev_dep)
                            buffer.put(", ");
                        buffer.fmt("r%u", v.dep[j]);
                        prev_dep = true;
                    }
                    buffer.put(")");
                    if (v.ref_count)
                        buffer.fmt(", refs=%u", v.ref_count);
                    if (v.ref_count_se)
                        buffer.fmt(", refs_se=%u", v.ref_count_se);
                    jitc_log(Warn, " - %s", buffer.get());
                } else if (n_leaked == 50) {
                    jitc_log(Warn, " - (skipping remainder)");
                }
                ++n_leaked;
            }
        }

        if (n_leaked == 0 && state.extra.size() != state.unused_extra.size() + 1) {
            jitc_log(Warn,
                    "jit_shutdown(): %zu 'extra' records were not cleaned up:",
                    state.extra.size() - state.unused_extra.size() - 1);
        }
    }

    jitc_registry_shutdown();
    jitc_malloc_shutdown();
    jitc_nvtx_shutdown();

    jitc_log(Info, "jit_shutdown(light=%u): done", (uint32_t) light);

    if (light == 0) {
        jitc_llvm_shutdown();
        jitc_cuda_shutdown();
#if defined(DRJIT_ENABLE_OPTIX)
        jitc_optix_api_shutdown();
#endif
    }

    free(jitc_temp_path);
    jitc_temp_path = nullptr;

    state.backends = 0;
}


ThreadState *jitc_init_thread_state(JitBackend backend) {
    ThreadState *ts;

    if (backend == JitBackend::CUDA) {
        ts = new CUDAThreadState();
        if ((state.backends & (uint32_t) JitBackend::CUDA) == 0) {
            delete ts;

            if (jitc_cuda_cuinit_result == CUDA_ERROR_NOT_INITIALIZED) {
                jitc_raise(
                    "jit_init_thread_state(): the CUDA backend hasn't been "
                    "initialized. Make sure to call jit_init(JitBackend::CUDA) "
                    "to properly initialize this backend.");
            } else if (jitc_cuda_cuinit_result != CUDA_SUCCESS) {
                const char *msg = nullptr;
                cuGetErrorString(jitc_cuda_cuinit_result, &msg);
                jitc_raise("jit_cuda_init(): the CUDA backend is not available "
                           "because cuInit() failed.\nThere are two common "
                           "explanations for this type of failure:\n\n 1. your "
                           "computer simply does not contain a graphics card "
                           "that supports CUDA.\n\n 2. your CUDA kernel module "
                           "and CUDA library are out of sync. Try to see if "
                           "you\n    can run a utility like 'nvidia-smi'. If "
                           "not, a reboot will likely fix this\n    issue. "
                           "Otherwise reinstall your graphics driver. \n\n "
                           "The specific error message produced by cuInit was\n"
                           "   \"%s\"", msg);
            } else {
#if defined(_WIN32)
              const char *cuda_fname = "nvcuda.dll";
#elif defined(__linux__)
              const char *cuda_fname = "libcuda.so";
#else
              const char *cuda_fname = "libcuda.dylib";
#endif
              jitc_raise(
                  "jit_init_thread_state(): the CUDA backend is inactive "
                  "because it has not been initialized via jit_init(), or "
                  "because the CUDA driver library (\"%s\") could not be "
                  "found! Set the DRJIT_LIBCUDA_PATH environment variable to "
                  "specify its path.",
                  cuda_fname);
            }
        }

        if (state.devices.empty()) {
            delete ts;
            jitc_raise("jit_init_thread_state(): the CUDA backend is inactive "
                       "because no compatible CUDA devices were found on your "
                       "system.");
        }

        Device &device = state.devices[0];
        ts->device = 0;
        ts->context = device.context;
        ts->compute_capability = device.compute_capability;
        ts->ptx_version = device.ptx_version;
        ts->memory_pool = device.memory_pool;
        ts->stream = device.stream;
        ts->event = device.event;
        ts->sync_stream_event = device.sync_stream_event;
        thread_state_cuda = ts;
    } else {
        ts = new LLVMThreadState();
        if ((state.backends & (uint32_t) JitBackend::LLVM) == 0) {
            delete ts;
            #if defined(_WIN32)
                const char *llvm_fname = "LLVM-C.dll";
            #elif defined(__linux__)
                const char *llvm_fname  = "libLLVM.so";
            #else
                const char *llvm_fname  = "libLLVM.dylib";
            #endif

            jitc_raise("jit_init_thread_state(): the LLVM backend is inactive "
                       "because the LLVM shared library (\"%s\") could not be "
                       "found! Set the DRJIT_LIBLLVM_PATH environment "
                       "variable to specify its path.",
                       llvm_fname);
        }
        thread_state_llvm = ts;
        ts->device = -1;
    }

    ts->backend = backend;
    ts->scope = ++state.scope_ctr;
    state.tss.push_back(ts);
    return ts;
}

void jitc_cuda_set_device(int device_id) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    if (ts->device == device_id)
        return;

    if ((size_t) device_id >= state.devices.size())
        jitc_raise("jit_cuda_set_device(%i): must be in the range 0..%i!",
                  device_id, (int) state.devices.size() - 1);

    jitc_log(Info, "jit_cuda_set_device(%i)", device_id);

    Device &device = state.devices[device_id];

    if (ts->stream) {
        scoped_set_context guard(ts->context);
        cuda_check(cuStreamSynchronize(ts->stream));
    }

    /* Associate with new context */ {
        ts->context = device.context;
        ts->device = device_id;
        ts->compute_capability = device.compute_capability;
        ts->ptx_version = device.ptx_version;
        ts->memory_pool = device.memory_pool;
        ts->stream = device.stream;
        ts->event = device.event;
        ts->sync_stream_event = device.sync_stream_event;
    }
}

void jitc_sync_thread(ThreadState *ts) {
    if (!ts)
        return;

    if (jitc_flag(JitFlag::ForbidSynchronization))
        jitc_raise("Attempted to synchronize in a context, where "
                   "synchronization was explicitly forbidden!");

    if (ts->backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        CUstream stream = ts->stream;
        unlock_guard guard_2(state.lock);
        cuda_check(cuStreamSynchronize(stream));
    } else {
        Task *task = jitc_task;
        if (!task)
            return;

        /* task_wait allows other tasks from the thread pool to be
         * started on this thread while we wait.
         *
         * However we don't want the ThreadState dynamic internal
         * variables (e.g. scheduled, side_effects, mask_stack) to
         * be shared across these tasks
         */
        scoped_reset_thread_state ts_guard(ts);
        {
            unlock_guard guard(state.lock);
            task_wait(task);
        }
        // Clear 'jitc_task' if no work was added in the meantime
        if (task == jitc_task) {
            jitc_task = nullptr;
            task_release(task);
        }
    }
}

/// Wait for all computation on the current stream to finish
void jitc_sync_thread() {
    jitc_sync_thread(thread_state_cuda);
    jitc_sync_thread(thread_state_llvm);
}

/// Wait for all computation on the current device to finish
void jitc_sync_device() {
    ThreadState *ts = thread_state_cuda;
    if (ts) {
        /* Release lock while synchronizing */ {
            unlock_guard guard(state.lock);
            scoped_set_context guard2(ts->context);
            cuda_check(cuCtxSynchronize());
        }
    }

    if (thread_state_llvm) {
        std::vector<ThreadState *> tss = state.tss;
        // Release lock while synchronizing */
        for (ThreadState *ts_2 : tss) {
            if (ts_2->backend == JitBackend::LLVM)
                jitc_sync_thread(ts_2);
        }
    }
}

/// Wait for all computation on *all devices* to finish
void jitc_sync_all_devices() {
    std::vector<ThreadState *> tss = state.tss;
    for (ThreadState *ts : tss)
        jitc_sync_thread(ts);
}

static void jitc_rebuild_prefix(ThreadState *ts) {
    free(ts->prefix);

    if (!ts->prefix_stack.empty()) {
        size_t size = 1;
        for (const char *s : ts->prefix_stack)
            size += strlen(s) + 1;
        ts->prefix = (char *) malloc(size);
        char *p = ts->prefix;

        for (const char *s : ts->prefix_stack) {
            size_t len = strlen(s);
            memcpy(p, s, len);
            p += len;
            *p++ = '/';
        }
        *p++ = '\0';
    } else {
        ts->prefix = nullptr;
    }
}

void jitc_prefix_push(JitBackend backend, const char *label) {
    if (strchr(label, '\n') || strchr(label, '/'))
        jitc_raise("jit_prefix_push(): invalid string (may not contain newline "
                   "or '/' characters)");

    ThreadState *ts = thread_state(backend);
    ts->prefix_stack.push_back(strdup(label));
    jitc_rebuild_prefix(ts);
}

void jitc_prefix_pop(JitBackend backend) {
    ThreadState *ts = thread_state(backend);
    auto &stack = ts->prefix_stack;
    if (stack.empty())
        jitc_raise("jit_prefix_pop(): stack underflow!");
    free(stack.back());
    stack.pop_back();
    jitc_rebuild_prefix(ts);
}

/// Glob for a shared library and try to load the most recent version
void *jitc_find_library(const char *fname, const char *glob_pat,
                        const char *env_var) {
#if !defined(_WIN32)
    const char* env_var_val = env_var ? getenv(env_var) : nullptr;
    if (env_var_val != nullptr && strlen(env_var_val) == 0)
        env_var_val = nullptr;

    void* handle = dlopen(env_var_val ? env_var_val : fname, RTLD_LAZY);

    if (!handle) {
        if (env_var_val) {
            jitc_log(Warn, "jit_find_library(): Unable to load \"%s\": %s!",
                    env_var_val, dlerror());
            return nullptr;
        }

        glob_t g;
        if (glob(glob_pat, GLOB_BRACE, nullptr, &g) == 0) {
            const char *chosen = nullptr;
            if (g.gl_pathc > 1) {
                jitc_log(Info, "jit_find_library(): Multiple versions of "
                              "%s were found on your system!\n", fname);
                std::sort(g.gl_pathv, g.gl_pathv + g.gl_pathc,
                          [](const char *a, const char *b) {
                              while (a != nullptr && b != nullptr) {
                                  while (*a == *b && *a != '\0' && !isdigit(*a)) {
                                      ++a; ++b;
                                  }
                                  if (isdigit(*a) && isdigit(*b)) {
                                      char *ap, *bp;
                                      int ai = strtol(a, &ap, 10);
                                      int bi = strtol(b, &bp, 10);
                                      if (ai != bi)
                                          return ai < bi;
                                      a = ap;
                                      b = bp;
                                  } else {
                                      return strcmp(a, b) < 0;
                                  }
                              }
                              return false;
                          });
                uint32_t counter = 1;
                for (int j = 0; j < 2; ++j) {
                    for (size_t i = 0; i < g.gl_pathc; ++i) {
                        struct stat buf;
                        // Skip symbolic links at first
                        if (j == 0 && (lstat(g.gl_pathv[i], &buf) || S_ISLNK(buf.st_mode)))
                            continue;
                        jitc_log(Info, " %u. \"%s\"", counter++, g.gl_pathv[i]);
                        chosen = g.gl_pathv[i];
                    }
                    if (chosen)
                        break;
                }
                jitc_log(Info,
                        "\nChoosing the last one. Specify a path manually "
                        "using the environment\nvariable '%s' to override this "
                        "behavior.\n", env_var);
            } else if (g.gl_pathc == 1) {
                chosen = g.gl_pathv[0];
            }
            if (chosen)
                handle = dlopen(chosen, RTLD_LAZY);
            globfree(&g);
        }
    }
#else
    (void) glob_pat;

    wchar_t buf[1024];
    mbstowcs(buf, env_var, sizeof(buf) / sizeof(wchar_t));

    const wchar_t* env_var_val = env_var ? _wgetenv(buf) : nullptr;
    if (env_var_val != nullptr && wcslen(env_var_val) == 0)
        env_var_val = nullptr;

    mbstowcs(buf, fname, sizeof(buf) / sizeof(wchar_t));
    void* handle = (void *) LoadLibraryW(env_var_val ? env_var_val : buf);
#endif

    return handle;
}

void jitc_set_flags(uint32_t new_flags) {
    if (new_flags & (uint32_t) JitFlag::KernelHistory) {
        // Must leave this on, since kernels may terminate outside of
        // the KernelHistory capture region
        pool_set_profile(true);
    }

    jitc_flags_v = new_flags;
}

uint32_t jitc_flags() {
    return jitc_flags_v;
}

void jitc_set_flag(JitFlag flag, int enable) {
    uint32_t flags = jitc_flags();

    if (enable)
        flags |= (uint32_t) flag;
    else
        flags &= ~(uint32_t) flag;

    jitc_set_flags(flags);
}

int jitc_flag(JitFlag flag) {
    return (jitc_flags() & (uint32_t) flag) ? 1 : 0;
}

/// ==========================================================================

KernelHistory::KernelHistory() : m_data(nullptr), m_size(0), m_capacity(0) { }

KernelHistory::~KernelHistory() { free(m_data); }

void KernelHistory::append(const KernelHistoryEntry &value) {
    /* Expand kernel history buffer if necessary. There should always be
       enough memory for an additional end-of-list marker at the end */

    if (m_size + 2 > m_capacity) {
        m_capacity = (m_size + 2) * 2;
        void *tmp = malloc_check(m_capacity * sizeof(KernelHistoryEntry));
        memcpy(tmp, m_data, m_size * sizeof(KernelHistoryEntry));
        free(m_data);
        m_data = (KernelHistoryEntry *) tmp;
    }

    m_data[m_size++] = value;
    memset(m_data + m_size, 0, sizeof(KernelHistoryEntry));
}

KernelHistoryEntry *KernelHistory::get() {
    KernelHistoryEntry *data = m_data;

    for (size_t i = 0; i < m_size; i++) {
        KernelHistoryEntry &k = data[i];
        if (k.backend == JitBackend::CUDA) {
            cuEventElapsedTime(&k.execution_time,
                               (CUevent) k.event_start,
                               (CUevent) k.event_end);
            cuEventDestroy((CUevent) k.event_start);
            cuEventDestroy((CUevent) k.event_end);
            k.event_start = k.event_end = 0;
        } else {
            task_wait((Task *) k.task);
            k.execution_time = (float) task_time((Task *) k.task);
            task_release((Task *) k.task);
            k.task = nullptr;
        }
    }

    m_data = nullptr;
    m_size = m_capacity = 0;

    return data;
}

void KernelHistory::clear() {
    if (m_size == 0)
        return;

    for (size_t i = 0; i < m_size; i++) {
        KernelHistoryEntry &k = m_data[i];
        if (k.backend == JitBackend::CUDA) {
            cuEventDestroy((CUevent) k.event_start);
            cuEventDestroy((CUevent) k.event_end);
        } else {
            task_release((Task *) k.task);
        }
    }

    free(m_data);
    m_data = nullptr;
    m_size = m_capacity = 0;
}

/// Default implementations of ThreadState functions
ThreadState::~ThreadState() { }
void ThreadState::barrier() { }
void ThreadState::reset_state() {
    scheduled.clear();
    side_effects.clear();
    side_effects_symbolic.clear();
    mask_stack.clear();
    prefix_stack.clear();
    record_stack.clear();
    prefix = nullptr;
    scope = 2;
    call_self_value = 0;
    call_self_index = 0;
#if defined(DRJIT_ENABLE_OPTIX)
    optix_pipeline = nullptr;
    optix_sbt = nullptr;
#endif
}
void ThreadState::notify_free(const void *) { }
void ThreadState::notify_expand(uint32_t) { }
void ThreadState::notify_opaque_width(uint32_t, uint32_t) {}
void ThreadState::notify_init_undefined(uint32_t) {}
// TODO: rename to block_reduce_bool
void ThreadState::block_reduce_bool(uint8_t *values, uint32_t size,
                                    uint8_t *out, ReduceOp op) {
    /* When \c size is not a multiple of 4, the implementation will initialize
       up to 3 bytes beyond the end of the supplied range so that an efficient
       32 bit reduction algorithm can be used. This is fine for allocations made
       using
       \ref jit_malloc(), which allow for this. */

    uint32_t size_4   = ceil_div(size, 4),
             trailing = size_4 * 4 - size;

    jitc_log(Debug, "jit_%s(" DRJIT_PTR ", size=%u)",
             op == ReduceOp::Or ? "any" : "all", (uintptr_t) values, size);

    if (trailing) {
        bool filler = op == ReduceOp::Or ? false : true;
        memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    block_reduce(VarType::UInt32, op, size_4, size_4, values, out);
}
