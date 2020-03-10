/*
    enoki/traits.h -- C++ type traits for analyzing variable types

    This file provides helper traits that are needed by the C++ array
    wrappers defined in in 'enoki/jitvar.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <type_traits>
#include "jit.h"

template <bool Value> using enable_if_t = std::enable_if_t<Value, int>;

template <typename T, typename = int> struct var_type {
    static constexpr VarType value = VarType::Invalid;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral_v<T> && sizeof(T) == 1>> {
    static constexpr VarType value =
        std::is_signed_v<T> ? VarType::Int8 : VarType::UInt8;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral_v<T> && sizeof(T) == 2>> {
    static constexpr VarType value =
        std::is_signed_v<T> ? VarType::Int16 : VarType::UInt16;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral_v<T> && sizeof(T) == 4>> {
    static constexpr VarType value =
        std::is_signed_v<T> ? VarType::Int32 : VarType::UInt32;
};

template <typename T> struct var_type<T, enable_if_t<std::is_integral_v<T> && sizeof(T) == 8>> {
    static constexpr VarType value =
        std::is_signed_v<T> ? VarType::Int64 : VarType::UInt64;
};

template <typename T> struct var_type<T, enable_if_t<std::is_enum_v<T>>> {
    static constexpr VarType value = var_type<std::underlying_type_t<T>>::value;
};

template <typename T> struct var_type<T, enable_if_t<std::is_floating_point_v<T> && sizeof(T) == 2>> {
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

template <typename T> constexpr VarType var_type_v = var_type<T>::value;

template <size_t Size> struct uint_with_size { };
template <> struct uint_with_size<1> { using type = uint8_t; };
template <> struct uint_with_size<2> { using type = uint16_t; };
template <> struct uint_with_size<4> { using type = uint32_t; };
template <> struct uint_with_size<8> { using type = uint64_t; };

template <typename T> using uint_with_size_t = typename uint_with_size<sizeof(T)>::type;
