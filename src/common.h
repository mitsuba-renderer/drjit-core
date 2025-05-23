#pragma once

#include <cstdint>
#include <utility>
#include <string>
#include <errno.h>
#include "lock.h"

#if defined(_MSC_VER)
#  pragma warning (disable:4201) // nonstandard extension used: nameless struct/union
#endif

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

/// Helper function for intense internal sanitation instrumentation
#if defined(DRJIT_SANITIZE_INTENSE)
  extern void jitc_sanitation_checkpoint();
#endif

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

extern void jitc_var_dec_ref(uint32_t) noexcept;
extern void jitc_var_inc_ref(uint32_t) noexcept;

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

inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

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

// Precomputed integer division helper
// Based on libidivide (https://github.com/ridiculousfish/libdivide)
struct divisor {
    uint32_t magic;
    uint32_t shift;
    uint32_t value;

    divisor() = default;

    divisor(uint32_t value) : magic(0), shift(0), value(value) {
        for (uint32_t tmp = value; tmp > 1; tmp >>= 1)
            shift++;

        if (value & (value - 1)) {
            uint64_t w = 1ull << (shift + 32);

            uint32_t w_div = (uint32_t) (w / (uint64_t) value),
                     w_rem = (uint32_t) (w % (uint64_t) value);

            // Overflows in the following code are expected
            magic = w_div * 2 + 1;
            uint32_t w_rem_2 = w_rem * 2;
            if (w_rem_2 >= value || w_rem_2 < w_rem)
                ++magic;
        }
    }

    uint32_t mulhi(uint32_t a, uint32_t b) const {
        return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32);
    }

    void div_rem(uint32_t input, uint32_t *out_div, uint32_t *out_rem) const {
        uint32_t div = 0, rem = 0;

        if (!magic) {
            div = input >> shift;
            rem = input - (div << shift);
        } else {
            uint32_t hi = mulhi(input, magic);
            div = (((input - hi) >> 1) + hi) >> shift;
            rem = input - div * value;
        }

        *out_div = div;
        *out_rem = rem;
    }
};
