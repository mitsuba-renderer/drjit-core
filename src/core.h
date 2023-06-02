#pragma once

#include <drjit-core/jit.h>
#include <utility>
#include <string>
#include <errno.h>
#include <inttypes.h>

#if !defined(likely)
#  if !defined(_MSC_VER)
#    define likely(x)   __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#  else
#    define unlikely(x) x
#    define likely(x) x
#  endif
#endif

#if defined(_WIN32)
#  define dlsym(ptr, name) GetProcAddress((HMODULE) ptr, name)
#endif

#define DRJIT_PTR "<0x%" PRIxPTR ">"

/* Some forward declarations */
struct State;
struct Variable;
struct Kernel;
class ThreadState;

#if defined(__linux__)
#include <pthread.h>
using Lock = pthread_spinlock_t;

// Danger zone: the drjit-core locks are held for an extremely short amount of
// time and normally uncontended. Switching to a spin lock cuts tracing time 8-10%
inline void lock_init(Lock &lock) { pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE); }
inline void lock_destroy(Lock &lock) { pthread_spin_destroy(&lock); }
inline void lock_acquire(Lock &lock) { pthread_spin_lock(&lock); }
inline void lock_release(Lock &lock) { pthread_spin_unlock(&lock); }
#elif defined(__APPLE__)
#include <os/lock.h>

using Lock = os_unfair_lock_s;
inline void lock_init(Lock &lock) { lock = OS_UNFAIR_LOCK_INIT; }
inline void lock_destroy(Lock &) { }
inline void lock_acquire(Lock &lock) { os_unfair_lock_lock(&lock);  }
inline void lock_release(Lock &lock) { os_unfair_lock_unlock(&lock); }
#else
#include <mutex>

using Lock = std::mutex;
inline void lock_init(Lock &) { }
inline void lock_destroy(Lock &) { }
inline void lock_acquire(Lock &lock) { lock.lock(); }
inline void lock_release(Lock &lock) { lock.unlock(); }
#endif

/// RAII helper for scoped lock acquisition
class lock_guard {
public:
    lock_guard(Lock &lock) : m_lock(lock) { lock_acquire(m_lock); }
    ~lock_guard() { lock_release(m_lock); }

    lock_guard(const lock_guard &) = delete;
    lock_guard &operator=(const lock_guard &) = delete;
private:
    Lock &m_lock;
};

/// RAII helper for scoped lock release
class unlock_guard {
public:
    unlock_guard(Lock &lock) : m_lock(lock) { lock_release(m_lock); }
    ~unlock_guard() { lock_acquire(m_lock); }
    unlock_guard(const unlock_guard &) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;
private:
    Lock &m_lock;
};

namespace detail {
    template <typename Func> struct scope_guard {
        scope_guard(const Func &func) : func(func) { }
        scope_guard(const scope_guard &) = delete;
        scope_guard(scope_guard &&g) : func(std::move(g.func)) {}
        scope_guard &operator=(const scope_guard &) = delete;
        scope_guard &operator=(scope_guard &&g) { func = std::move(g.func); }
    private:
        Func func;
    };
};

template <class Func>
detail::scope_guard<Func> scope_guard(const Func &func) {
    return detail::scope_guard<Func>(func);
}


template <size_t> struct uint_with_size { };
template <> struct uint_with_size<1> { using type = uint8_t; };
template <> struct uint_with_size<2> { using type = uint16_t; };
template <> struct uint_with_size<4> { using type = uint32_t; };
template <> struct uint_with_size<8> { using type = uint64_t; };

template <typename T>
using uint_with_size_t = typename uint_with_size<sizeof(T)>::type;

extern void jitc_var_dec_ref(uint32_t) noexcept(true);
extern void jitc_var_inc_ref(uint32_t) noexcept(true);

struct Ref {
    friend Ref steal(uint32_t);
    friend Ref borrow(uint32_t);

    Ref() : index(0) { }
    ~Ref() { jitc_var_dec_ref(index); }

    Ref(Ref &&r) : index(r.index) { r.index = 0; }

    Ref &operator=(Ref &&r) {
        jitc_var_dec_ref(index);
        index = r.index;
        r.index = 0;
        return *this;
    }

    Ref(const Ref &) = delete;
    Ref &operator=(const Ref &) = delete;

    operator uint32_t() const { return index; }
    uint32_t release() {
        uint32_t value = index;
        index = 0;
        return value;
    }
    void reset() {
        jitc_var_dec_ref(index);
        index = 0;
    }

private:
    uint32_t index = 0;
};

inline Ref steal(uint32_t index) {
    Ref r;
    r.index = index;
    return r;
}

inline Ref borrow(uint32_t index) {
    Ref r;
    r.index = index;
    jitc_var_inc_ref(index);
    return r;
}

inline uint32_t log2i_ceil(uint32_t x) {
    if (x <= 1)
        return 0;
    x -= 1;
#if defined(_MSC_VER)
    unsigned long y;
    _BitScanReverse(&y, (unsigned long)x);
    return 1u + (uint32_t) y;
#else
    return 32u - __builtin_clz(x);
#endif
}

extern JIT_MALLOC void* malloc_check(size_t size);

extern JIT_MALLOC void* malloc_check_zero(size_t size);

extern JIT_MALLOC void* realloc_check(void *orig, size_t size) ;


/// Wait for all computation on the current stream to finish
extern void jitc_sync_thread();

/// Wait for all computation on the current stream to finish
extern void jitc_sync_thread(ThreadState *stream);

/// Wait for all computation on the current device to finish
extern void jitc_sync_device();

/// Wait for all computation on *all devices* to finish
extern void jitc_sync_all_devices();

/// Search for a shared library and dlopen it if possible
void *jitc_find_library(const char *fname, const char *glob_pat,
                        const char *env_var);

/// Push a new label onto the prefix stack
extern void jitc_prefix_push(JitBackend backend, const char *label);

/// Pop a label from the prefix stack
extern void jitc_prefix_pop(JitBackend backend);

/// Free all kernels that are currently cached
extern void jitc_flush_kernel_cache();

/// Free a specific kernel from the device
extern void jitc_kernel_free(int device_id, const Kernel &kernel);
