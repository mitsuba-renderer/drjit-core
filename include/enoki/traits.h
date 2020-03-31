/*
    enoki/traits.h -- C++ type traits for analyzing variable types

    This file provides helper traits that are needed by the C++ array
    wrappers defined in in 'enoki/jitvar.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <type_traits>
#include <cstring>
#include "jit.h"

template <bool Value> using enable_if_t = typename std::enable_if<Value, int>::type;

template <typename T, typename = int> struct var_type {
    static constexpr VarType value = VarType::Invalid;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral<T>::value && sizeof(T) == 1>> {
    static constexpr VarType value =
        std::is_signed<T>::value ? VarType::Int8 : VarType::UInt8;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral<T>::value && sizeof(T) == 2>> {
    static constexpr VarType value =
        std::is_signed<T>::value ? VarType::Int16 : VarType::UInt16;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral<T>::value && sizeof(T) == 4>> {
    static constexpr VarType value =
        std::is_signed<T>::value ? VarType::Int32 : VarType::UInt32;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral<T>::value && sizeof(T) == 8>> {
    static constexpr VarType value =
        std::is_signed<T>::value ? VarType::Int64 : VarType::UInt64;
};

template <typename T> struct var_type<T, enable_if_t<std::is_enum<T>::value>> {
    static constexpr VarType value = var_type<typename std::underlying_type<T>::type>::value;
};

template <typename T> struct var_type<T, enable_if_t<std::is_floating_point<T>::value && sizeof(T) == 2>> {
    static constexpr VarType value = VarType::Float16;
};

template <> struct var_type<float> {
    static constexpr VarType value = VarType::Float32;
};

template <> struct var_type<double> {
    static constexpr VarType value = VarType::Float64;
};

template <> struct var_type<bool> {
    static constexpr VarType value = VarType::Bool;
};

template <typename T> struct var_type<T *> {
    static constexpr VarType value = VarType::Pointer;
};

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
