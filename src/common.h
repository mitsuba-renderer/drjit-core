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

#if defined(_WIN32)
#  define dlsym(ptr, name) GetProcAddress((HMODULE) ptr, name)
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

extern void jitc_var_dec_ref_ext(uint32_t) noexcept(true);
extern void jitc_var_inc_ref_ext(uint32_t) noexcept(true);

struct Ref {
    friend Ref steal(uint32_t);
    friend Ref borrow(uint32_t);

    Ref() : index(0) { }
    ~Ref() { jitc_var_dec_ref_ext(index); }

    Ref(Ref &&r) : index(r.index) { r.index = 0; }

    Ref &operator=(Ref &&r) {
        jitc_var_dec_ref_ext(index);
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
        jitc_var_dec_ref_ext(index);
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
    jitc_var_inc_ref_ext(index);
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
