#pragma once

#include <drjit-core/macros.h>
#include <atomic>
#include <cstdint>
#include <cstdio>   // The assertions below report through stderr and abort()
#include <cstdlib>

#if defined(_MSC_VER)
#  include <intrin.h>
#  define LOCK_UNLIKELY(x) (x)
#else
#  define LOCK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// Annotations so that TSAN handles the lock below like a system lock
#if defined(__SANITIZE_THREAD__)
#  define DRJIT_TSAN 1
#elif defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define DRJIT_TSAN 1
#  endif
#endif

#if defined(DRJIT_TSAN)
#  include <sanitizer/tsan_interface.h>
#else
#  define __tsan_mutex_create(...)      ((void) 0)
#  define __tsan_mutex_destroy(...)     ((void) 0)
#  define __tsan_mutex_pre_lock(...)    ((void) 0)
#  define __tsan_mutex_post_lock(...)   ((void) 0)
#  define __tsan_mutex_pre_unlock(...)  ((void) 0)
#  define __tsan_mutex_post_unlock(...) ((void) 0)
#endif

/*
 * The recursive locks below have a fast path that skips the atomic transaction
 * when the current thread already holds the lock. It reads `lock.owner`
 * without holding the lock and races with other threads' stores.
 *
 * The code uses a relaxed atomic to make this race-free in the C++ memory
 * model. Relaxed ordering suffices because the load only decides whether to
 * take the fast path. The actual synchronization comes from the following
 * compare-exchange operations. The only value that affects control flow is
 * whether or not the stored ID matches the current thread's ID, which can
 * only be observed if this thread wrote it while holding the lock. The
 * ``recursion_count`` fields needs no atomic because it is only touched by the
 * owning thread under the lock.
 */

// Fast unique (non-zero) token identifying the current thread.
// pthread_self()/std::this_thread::get_id() are relatively costly function
// calls; the variants below compile to one or two machine instructions.
// They are based on mimalloc's _mi_prim_thread_id(), see
// https://github.com/microsoft/mimalloc/blob/master/include/mimalloc/prim.h
#if defined(__linux__)

inline uintptr_t thread_id() { return (uintptr_t) __builtin_thread_pointer(); }

#elif defined(__APPLE__)

// macOS/Darwin doesn't expose an official API to read the thread identifier. See
// https://github.com/apple/darwin-xnu/blob/main/libsyscall/os/tsd.h
inline uintptr_t thread_id() {
    #if defined(__aarch64__)
        uintptr_t tsd;
        __asm__("mrs %0, tpidrro_el0\nbic %0, %0, #7" : "=r" (tsd));
        return tsd;
    #else
        void *self;
        __asm__("movq %%gs:0, %0" : "=r" (self));
        return (uintptr_t) self;
    #endif
}

#elif defined(_M_X64) || defined(_M_ARM64)

// TEB pointer; mimalloc calls NtCurrentTeb(), which is expanded here to
// avoid a windows.h dependency. See
//
//  - https://github.com/mingw-w64/mingw-w64/blob/master/mingw-w64-headers/include/winnt.h
//  - https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions
//
// For the ARM64 and Intel-compatible implementations of the functions.
inline uintptr_t thread_id() {
    #if defined(_M_ARM64)
        return (uintptr_t) __getReg(18);
    #else
        return (uintptr_t) __readgsqword(0x30);
    #endif
}

#else

// Unknown OS/processor: the address of a thread-local variable is unique per thread
inline uintptr_t thread_id() {
    static thread_local char id;
    return (uintptr_t) &id;
}

#endif

/// CPU hint that the current thread is in a spin-wait loop
inline void lock_pause() {
    #if defined(_MSC_VER)
        #if defined(_M_ARM64)
            __yield();
        #else
            _mm_pause();
        #endif
    #elif defined(__aarch64__)
        __asm__ __volatile__("yield");
    #else
        __builtin_ia32_pause();
    #endif
}

struct Lock;

/**
 * \brief Flag to request postponed delivery of log message
 *
 * Mixed Dr.Jit+Python workloads use both the Dr.Jit lock and the GIL. To avoid
 * deadlocks when delivering log messages to Python via callbacks, we postpone
 * these callbacks and deliver them after leaving the outermost critical
 * section.
 *
 * The flag below, stored in the topmost bit of ``Lock::recursion_count``,
 * indicates the need to perform this postponed delivery.
 */
static constexpr size_t LOCK_PENDING = (size_t) 1 << (8 * sizeof(size_t) - 1);

/// Indicate that `lock`, which the caller holds, owes deferred work
inline void lock_set_pending(Lock &lock);

/**
 * \brief Fully release `lock` and carry out the deferred work
 *
 * Invoked by \ref lock_release() following the last release of a lock marked
 * by \ref lock_set_pending(). Implemented in src/log.cpp.
 */
extern JIT_EXPORT void lock_release_pending(Lock &lock) noexcept;

#if !defined(DRJIT_USE_STD_MUTEX)

// The drjit-core locks are held for an extremely short amount of time and are
// normally uncontended, so a fully inlined spin lock beats the system
// primitives, which cost a cross-library call on both acquire and release
// (pthread_spin_lock on Linux, os_unfair_lock on macOS, SRWLOCK on Windows).
struct Lock {
    // Zero while the lock is free, otherwise the thread_id() of the owner.
    std::atomic<uintptr_t> owner;
    size_t recursion_count;
};

inline void lock_init(Lock &lock) {
    lock.owner.store(0, std::memory_order_relaxed);
    lock.recursion_count = 0;
    __tsan_mutex_create(&lock, 0);
}

inline void lock_destroy(Lock &lock) {
    __tsan_mutex_destroy(&lock, 0);
    (void) lock;
}

// Only the outermost acquire/release is annotated: recursive acquisitions are
// pure bookkeeping in this wrapper, so TSAN sees a plain non-recursive mutex.
inline void lock_acquire(Lock &lock) {
    uintptr_t self = thread_id(),
              prev = lock.owner.load(std::memory_order_relaxed);

    // Recursive locking is the rarer of the two cases
    if (LOCK_UNLIKELY(prev == self)) {
#if !defined(NDEBUG)
        // The locking system is designed around the property that thread_id()
        // can never return zero. Debug builds verify that this holds true.
        if (LOCK_UNLIKELY(prev == 0)) {
            fprintf(stderr, "lock_acquire(): thread_id() returned zero!\n");
            abort();
        }
#endif
        lock.recursion_count++;
        return;
    }

    __tsan_mutex_pre_lock(&lock, 0);

    // Mirrors glibc's pthread_spin_lock (sysdeps/nptl/pthread_spin_lock.c):
    // spin on plain loads and retry with a CAS.
    if (prev != 0 || !lock.owner.compare_exchange_strong(
                          prev, self, std::memory_order_acquire,
                          std::memory_order_relaxed)) {
        do {
            while (lock.owner.load(std::memory_order_relaxed) != 0)
                lock_pause();
            prev = 0;
        } while (!lock.owner.compare_exchange_weak(prev, self,
                                                   std::memory_order_acquire,
                                                   std::memory_order_relaxed));
    }

    // Also clears LOCK_PENDING, which cannot be set while the lock is free
    lock.recursion_count = 1;
    __tsan_mutex_post_lock(&lock, 0, 0);
}

inline void lock_release(Lock &lock) {
#if !defined(NDEBUG)
    /* An unmatched release (e.g. an unlock_guard on a lock that the current
       thread does not hold) corrupts the recursion bookkeeping and can
       silently unlock another thread's critical section -- a bug that later
       manifests as a hard-to-debug deadlock. Catch it at the source. */
    if (lock.owner.load(std::memory_order_relaxed) != thread_id() ||
        (lock.recursion_count & ~LOCK_PENDING) == 0) {
        fprintf(stderr, "lock_release(): unmatched release (owner=%p, self=%p, "
                "count=%zu)!\n", (void *) lock.owner.load(),
                (void *) thread_id(), lock.recursion_count & ~LOCK_PENDING);
        abort();
    }
#endif
    lock.recursion_count--;
    if (lock.recursion_count == 0) {
        __tsan_mutex_pre_unlock(&lock, 0);
        lock.owner.store(0, std::memory_order_release);
        __tsan_mutex_post_unlock(&lock, 0);
    } else if (LOCK_UNLIKELY(lock.recursion_count == LOCK_PENDING)) {
        lock_release_pending(lock); // Outermost release, log messages waiting
    }
}

#else

#if defined(_WIN32)
#include <shared_mutex>
struct Lock {
    std::shared_mutex lock; // Based on the faster Win7 SRWLOCK
    std::atomic<uintptr_t> owner;
    size_t recursion_count;
};
#else
#include <mutex>
struct Lock {
    std::mutex lock;
    std::atomic<uintptr_t> owner;
    size_t recursion_count;
};
#endif

inline void lock_init(Lock &lock) {
    lock.owner.store(0, std::memory_order_relaxed);
    lock.recursion_count = 0;
}

inline void lock_destroy(Lock &) {}
inline void lock_acquire(Lock &lock) {
    uintptr_t self = thread_id();
    if (lock.owner.load(std::memory_order_relaxed) == self) {
        lock.recursion_count++;
        return;
    }

    lock.lock.lock();
    lock.owner.store(self, std::memory_order_relaxed);
    lock.recursion_count = 1;
}
inline void lock_release(Lock &lock) {
    lock.recursion_count--;
    if (lock.recursion_count == 0) {
        lock.owner.store(0, std::memory_order_relaxed);
        lock.lock.unlock();
    } else if (LOCK_UNLIKELY(lock.recursion_count == LOCK_PENDING)) {
        lock_release_pending(lock); // Outermost release, log messages waiting
    }
}

#endif

inline void lock_set_pending(Lock &lock) {
    lock.recursion_count |= LOCK_PENDING;
}

extern void jitc_sanitation_checkpoint();

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

/**
 * \brief RAII helper for scoped lock release
 *
 * Temporarily release the Dr.Jit lock.
 *
 * This is just a temporary release. We still delay postponed log delivery
 * until leaving the outer lock scope.
 */
class unlock_guard {
public:
    unlock_guard(Lock &lock) : m_lock(lock) {
        m_pending = lock.recursion_count & LOCK_PENDING;
        lock.recursion_count &= ~LOCK_PENDING;
        lock_release(m_lock);
    }

    ~unlock_guard() {
        lock_acquire(m_lock);
        m_lock.recursion_count |= m_pending;
        #if defined(DRJIT_SANITIZE_INTENSE)
            jitc_sanitation_checkpoint();
        #endif
    }

    unlock_guard(const unlock_guard &) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;
private:
    Lock &m_lock;
    size_t m_pending;
};
