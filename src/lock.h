#pragma once

#if defined(__linux__) && !defined(DRJIT_USE_STD_MUTEX)
#include <pthread.h>
using Lock = pthread_spinlock_t;

// Danger zone: the drjit-core locks are held for an extremely short amount of
// time and normally uncontended. Switching to a spin lock cuts tracing time 8-10%
inline void lock_init(Lock &lock) { pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE); }
inline void lock_destroy(Lock &lock) { pthread_spin_destroy(&lock); }
inline void lock_acquire(Lock &lock) { pthread_spin_lock(&lock); }
inline void lock_release(Lock &lock) { pthread_spin_unlock(&lock); }
#elif defined(__APPLE__) && !defined(DRJIT_USE_STD_MUTEX)
#include <os/lock.h>

using Lock = os_unfair_lock_s;
inline void lock_init(Lock &lock) { lock = OS_UNFAIR_LOCK_INIT; }
inline void lock_destroy(Lock &) { }
inline void lock_acquire(Lock &lock) { os_unfair_lock_lock(&lock);  }
inline void lock_release(Lock &lock) { os_unfair_lock_unlock(&lock); }
#else
#if defined(_WIN32)
#include <shared_mutex>
using Lock = std::shared_mutex; // Based on the faster Win7 SRWLOCK
#else
#include <mutex>
using Lock = std::mutex;
#endif

inline void lock_init(Lock &) { }
inline void lock_destroy(Lock &) { }
inline void lock_acquire(Lock &lock) { lock.lock(); }
inline void lock_release(Lock &lock) { lock.unlock(); }
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
