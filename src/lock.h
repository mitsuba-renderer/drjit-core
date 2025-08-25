#pragma once

#if defined(__linux__) && !defined(DRJIT_USE_STD_MUTEX)
#include <pthread.h>

/*
 * This recursive spinlock improves the efficiency of @dr.freeze and operations
 * that acquire a lock recursively to avoid repeated atomic memory transactions
 * when accessing JIT variables.
 *
 * The code contains unprotected reads/writes of members (`lock.owner`,
 * `lock.recursion_count`) that are likely to be flagged as data races when using
 * thread sanitizers or similar infrastructure. Here is a rationale on why this is
 * safe even on architectures with weakly ordered memory semantics like ARM:
 *
 * The check `lock.owner == self` in lock_acquire avoids taking out the lock if
 * the current thread already holds it. However, the read from `lock.owner` is a
 * data race. Other threads might be reading/writing this value, and there are no
 * guarantees of how these operations are ordered with respect to each other.
 *
 * However, only one possible value of this field matters, which is when
 * `lock.owner` matches the value of `pthread_self()`. This value can only be read
 * if the current thread wrote it, which means that it is still holding the lock.
 * This means that the body of the conditionals is effectively part of a previous
 * critical section.
 */

struct Lock {
    pthread_spinlock_t lock;
    pthread_t owner;
    size_t recursion_count;
};

// Danger zone: the drjit-core locks are held for an extremely short amount of
// time and normally uncontended. Switching to a spin lock cuts tracing time 8-10%
inline void lock_init(Lock &lock) {
    pthread_spin_init(&lock.lock, PTHREAD_PROCESS_PRIVATE);
    lock.owner           = 0;
    lock.recursion_count = 0;
}
inline void lock_destroy(Lock &lock) { pthread_spin_destroy(&lock.lock); }
inline void lock_acquire(Lock &lock) {
    pthread_t self = pthread_self();
    if (lock.owner == self) {
        lock.recursion_count++;
        return;
    }

    pthread_spin_lock(&lock.lock);
    lock.owner           = self;
    lock.recursion_count = 1;
}
inline void lock_release(Lock &lock) {
    lock.recursion_count--;
    if (lock.recursion_count == 0) {
        lock.owner = (pthread_t) -1;
        pthread_spin_unlock(&lock.lock);
    }
}
#elif defined(__APPLE__) && !defined(DRJIT_USE_STD_MUTEX)
#include <os/lock.h>

struct Lock {
    os_unfair_lock_s lock;
    pthread_t owner;
    size_t recursion_count;
};

inline void lock_init(Lock &lock) {
    lock.lock            = OS_UNFAIR_LOCK_INIT;
    lock.owner           = (pthread_t) -1;
    lock.recursion_count = 0;
}
inline void lock_destroy(Lock &) {}
inline void lock_acquire(Lock &lock) {
    pthread_t self = pthread_self();
    if (pthread_equal(lock.owner, self)) {
        lock.recursion_count++;
        return;
    }

    os_unfair_lock_lock(&lock.lock);
    lock.owner           = self;
    lock.recursion_count = 1;
}
inline void lock_release(Lock &lock) {
    lock.recursion_count--;
    if (lock.recursion_count == 0) {
        lock.owner = (pthread_t) -1;
        os_unfair_lock_unlock(&lock.lock);
    }
}
#else
#include <thread>
#if defined(_WIN32)
#include <shared_mutex>
struct Lock {
    std::shared_mutex lock; // Based on the faster Win7 SRWLOCK
    std::thread::id owner;
    size_t recursion_count;
};
#else
#include <mutex>
struct Lock {
    std::mutex lock;
    std::thread::id owner;
    size_t recursion_count;
};
#endif

inline void lock_init(Lock &lock) {
    lock.owner           = std::thread::id();
    lock.recursion_count = 0;
}

inline void lock_destroy(Lock &) {}
inline void lock_acquire(Lock &lock) {
    std::thread::id self = std::this_thread::get_id();
    if (lock.owner == self) {
        lock.recursion_count++;
        return;
    }

    lock.lock.lock();
    lock.owner = self;
    lock.recursion_count = 1;
}
inline void lock_release(Lock &lock) {
    lock.recursion_count--;
    if (lock.recursion_count == 0) {
        lock.owner = std::thread::id();
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
