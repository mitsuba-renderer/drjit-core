#pragma once

#include <mutex>

#if !defined(likely)
#  if !defined(_MSC_VER)
#    define likely(x)   __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#  else
#    define unlikely(x) x
#    define likely(x) x
#  endif
#endif

using lock_guard = std::unique_lock<std::mutex>;

/// RAII helper for *unlocking* a mutex
class unlock_guard {
public:
    unlock_guard(std::mutex &mutex) : m_mutex(mutex) {
        m_mutex.unlock();
    }

    ~unlock_guard() { m_mutex.lock(); }
    unlock_guard(const unlock_guard &) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;
private:
    std::mutex &m_mutex;
};

