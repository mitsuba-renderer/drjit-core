/*
    enoki-jit/util.h -- Utility routines shared by llvm.h and cuda.h

    This file provides helper traits that are needed by the C++ array
    wrappers defined in in 'enoki/jitvar.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/traits.h>
#include <cstring>

NAMESPACE_BEGIN(enoki)

template <size_t Size> struct uint_with_size { };
template <> struct uint_with_size<1> { using type = uint8_t; };
template <> struct uint_with_size<2> { using type = uint16_t; };
template <> struct uint_with_size<4> { using type = uint32_t; };
template <> struct uint_with_size<8> { using type = uint64_t; };

template <typename T> using uint_with_size_t = typename uint_with_size<sizeof(T)>::type;

struct void_t { };

/// Reinterpret the binary represesentation of a data type
template<typename Target, typename Source> Target memcpy_cast(const Source &source) {
    static_assert(sizeof(Source) == sizeof(Target), "memcpy_cast: sizes did not match!");
    Target target;
    std::memcpy(&target, &source, sizeof(Target));
    return target;
}

template <typename Value> Value sign_mask() {
    using UInt = uint_with_size_t<Value>;
    return memcpy_cast<Value>(UInt(1) << (sizeof(UInt) * 8 - 1));
}

template <typename Value> Value sign_mask_neg() {
    using UInt = uint_with_size_t<Value>;
    return memcpy_cast<Value>(~(UInt(1) << (sizeof(UInt) * 8 - 1)));
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
void jitc_schedule(Args&&... args) {
    bool unused[] = { (jitc_schedule(args), false)..., false };
    (void) unused;
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 0)> = 0>
void jitc_eval(Args&&... args) {
    jitc_schedule(args...);
    if (sizeof...(Args) > 0)
        ::jitc_eval();
}

using ssize_t = typename std::make_signed<size_t>::type;

inline int clz(uint32_t value) {
#if !defined(_MSC_VER)
    return __builtin_clz(value);
#else
    int lz = 32;
    while (value) {
        value >>= 1;
        lz -= 1;
    }
    return result;
#endif
}

inline int clz(uint64_t value) {
#if !defined(_MSC_VER)
    return __builtin_clz(value);
#else
    int lz = 64;
    while (value) {
        value >>= 1;
        lz -= 1;
    }
    return result;
#endif
}

NAMESPACE_END(enoki)
