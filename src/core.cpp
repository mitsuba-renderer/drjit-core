#include "core.h"
#include "state.h"
#include "thread_state.h"
#include "log.h"
#include "backends.h"

#include <nanothread/nanothread.h>

#if defined(_WIN32)
#  include <windows.h>
#  include <direct.h>
#else
#  include <glob.h>
#  include <dlfcn.h>
#endif

#include <sys/stat.h>

NB_TLS uint32_t jitc_flags_v = (uint32_t) JitFlag::Default;

State state;

JIT_MALLOC void* malloc_check(size_t size) {
    void *ptr = malloc(size);
    if (unlikely(!ptr)) {
        fprintf(stderr, "malloc_check(): failed to allocate %zu bytes!", size);
        abort();
    }
    return ptr;
}

JIT_MALLOC void* malloc_check_zero(size_t size) {
    void *ptr = malloc_check(size);
    memset(ptr, 0, size);
    return ptr;
}

JIT_MALLOC void* realloc_check(void *orig, size_t size) {
    void *ptr = realloc(orig, size);
    if (unlikely(!ptr)) {
        fprintf(stderr, "realloc_check(): could not resize memory region to %zu bytes!", size);
        abort();
    }
    return ptr;
}

ThreadState *jitc_init_thread_state(JitBackend backend) {
    ThreadState *ts;
    switch (backend) {
#if defined(DRJIT_ENABLE_LLVM)
        case JitBackend::LLVM:
            ts = jitc_llvm_thread_state_new();
            thread_state_llvm = ts;
            break;
#endif

#if defined(DRJIT_ENABLE_CUDA)
        case JitBackend::CUDA:
            ts = jitc_cuda_thread_state_new();
            thread_state_cuda = ts;
            break;
#endif

#if defined(DRJIT_ENABLE_METAL)
        case JitBackend::Metal:
            ts = jitc_metal_thread_state_new();
            thread_state_metal = ts;
            break;
#endif

        default:
            jitc_fail("jitc_thread_state_new(): unhandled backend!");
    }

    ts->backend = backend;
    ts->scope = ++state.scope_ctr;
    state.tss.push_back(ts);
    return ts;
}

void jitc_sync_thread(ThreadState *ts) {
    if (!ts)
        return;

    unlock_guard guard(state.lock);
    ts->sync();
}

/// Wait for all computation on the current stream to finish
void jitc_sync_thread() {
#if defined(DRJIT_ENABLE_LLVM)
    jitc_sync_thread(thread_state_llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    jitc_sync_thread(thread_state_cuda);
#endif

#if defined(DRJIT_ENABLE_METAL)
    jitc_sync_thread(thread_state_metal);
#endif
}

/// Wait for all computation on the current device to finish
void jitc_sync_device() {
    // There is no longer a distrinction between sync_thread and sync_device
    jitc_sync_thread();
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

void jitc_set_flags(uint32_t flags) {
    jitc_flags_v = flags;

#if defined(DRJIT_ENABLE_LLVM)
    pool_set_profile(int(flags & (uint32_t) JitFlag::KernelHistory));
#endif
}

uint32_t jitc_flags() {
    return jitc_flags_v;
}

void jitc_kernel_free(JitBackend backend, int device_id, const Kernel &kernel) {
    switch (backend) {
#if defined(DRJIT_ENABLE_LLVM)
        case JitBackend::LLVM:
            jitc_llvm_kernel_free(kernel);
            break;
#endif

#if defined(DRJIT_ENABLE_CUDA)
        case JitBackend::CUDA:
            jitc_cuda_kernel_free(device_id, kernel);
            break;
#endif

        default:
            jitc_fail("jit_kernel_free(): unhandled backend!");
    }
}

void jitc_flush_kernel_cache() {
    jitc_log(Info, "jit_flush_kernel_cache(): releasing %zu kernel%s ..",
            state.kernel_cache.size(),
            state.kernel_cache.size() > 1 ? "s" : "");

    for (auto &v : state.kernel_cache) {
        jitc_kernel_free(v.first.device, v.second);
        free(v.first.str);
    }

    state.kernel_cache.clear();
}
