#pragma once

#include <atomic>

/*
 * The recursive locks below have a fast path that skips the atomic transaction
 * when the current thread already holds the lock. This reads `lock.owner`
 * without holding the lock, racing with other threads' stores.
 *
 * The code below uses a relaxed atomic to make this race-free in the C++ memory
 * model. Relaxed ordering suffices because `owner` is only a same-thread
 * recursion hint; all synchronization comes from the underlying lock. The only
 * value that affects control flow is "equal to my own id", which can only be
 * observed if this thread wrote it while holding the lock. `recursion_count`
 * needs no atomic: it is only touched by the owning thread under the lock.
 */

#if defined(__linux__) && !defined(DRJIT_USE_STD_MUTEX)
#include <pthread.h>

// Opaque per-thread token used to detect recursive lock acquisition.
// Prefer __builtin_thread_pointer() over the slower pthread_self() if available.
#if defined(__has_builtin) && __has_builtin(__builtin_thread_pointer)
using lock_thread_id_t = void *; // __builtin_thread_pointer() returns void *
inline lock_thread_id_t lock_thread_id() { return __builtin_thread_pointer(); }
#else
using lock_thread_id_t = pthread_t;
inline lock_thread_id_t lock_thread_id() { return pthread_self(); }
#endif

struct Lock {
    pthread_spinlock_t lock;
    std::atomic<lock_thread_id_t> owner;
    size_t recursion_count;
};

// The drjit-core locks are held for an extremely short amount of
// time and normally uncontended. Switching to a spin lock cuts tracing time 8-10%
inline void lock_init(Lock &lock) {
    pthread_spin_init(&lock.lock, PTHREAD_PROCESS_PRIVATE);
    lock.owner.store(0, std::memory_order_relaxed);
    lock.recursion_count = 0;
}
inline void lock_destroy(Lock &lock) { pthread_spin_destroy(&lock.lock); }
inline void lock_acquire(Lock &lock) {
    lock_thread_id_t self = lock_thread_id();
    if (lock.owner.load(std::memory_order_relaxed) == self) {
        lock.recursion_count++;
        return;
    }

    pthread_spin_lock(&lock.lock);
    lock.owner.store(self, std::memory_order_relaxed);
    lock.recursion_count = 1;
}
inline void lock_release(Lock &lock) {
    lock.recursion_count--;
    if (lock.recursion_count == 0) {
        lock.owner.store(0, std::memory_order_relaxed);
        pthread_spin_unlock(&lock.lock);
    }
}
#elif defined(__APPLE__) && !defined(DRJIT_USE_STD_MUTEX)
#include <os/lock.h>
#include <pthread.h>

struct Lock {
    os_unfair_lock_s lock;
    std::atomic<pthread_t> owner;
    size_t recursion_count;
};

inline void lock_init(Lock &lock) {
    lock.lock = OS_UNFAIR_LOCK_INIT;
    lock.owner.store((pthread_t) -1, std::memory_order_relaxed);
    lock.recursion_count = 0;
}
inline void lock_destroy(Lock &) {}
inline void lock_acquire(Lock &lock) {
    pthread_t self = pthread_self();
    if (pthread_equal(lock.owner.load(std::memory_order_relaxed), self)) {
        lock.recursion_count++;
        return;
    }

    os_unfair_lock_lock(&lock.lock);
    lock.owner.store(self, std::memory_order_relaxed);
    lock.recursion_count = 1;
}
inline void lock_release(Lock &lock) {
    lock.recursion_count--;
    if (lock.recursion_count == 0) {
        lock.owner.store((pthread_t) -1, std::memory_order_relaxed);
        os_unfair_lock_unlock(&lock.lock);
    }
}
#else
#include <thread>
#if defined(_WIN32)
#include <shared_mutex>
struct Lock {
    std::shared_mutex lock; // Based on the faster Win7 SRWLOCK
    std::atomic<std::thread::id> owner;
    size_t recursion_count;
};
#else
#include <mutex>
struct Lock {
    std::mutex lock;
    std::atomic<std::thread::id> owner;
    size_t recursion_count;
};
#endif

inline void lock_init(Lock &lock) {
    lock.owner.store(std::thread::id(), std::memory_order_relaxed);
    lock.recursion_count = 0;
}

inline void lock_destroy(Lock &) {}
inline void lock_acquire(Lock &lock) {
    std::thread::id self = std::this_thread::get_id();
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
        lock.owner.store(std::thread::id(), std::memory_order_relaxed);
        lock.lock.unlock();
    }
}
#endif

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

/// RAII helper for scoped lock release
class unlock_guard {
public:
    unlock_guard(Lock &lock) : m_lock(lock) { lock_release(m_lock); }
    ~unlock_guard() {
        lock_acquire(m_lock);
        #if defined(DRJIT_SANITIZE_INTENSE)
            jitc_sanitation_checkpoint();
        #endif
    }
    unlock_guard(const unlock_guard &) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;
private:
    Lock &m_lock;
};
