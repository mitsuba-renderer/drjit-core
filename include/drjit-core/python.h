/*
    drjit-core/python.h: type casters for Dr.Jit nanostl and half types

    Copyright (c) 2022 Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/detail/nb_list.h>
#include <drjit-core/nanostl.h>
#include <drjit-core/half.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename... Ts> struct type_caster<drjit::tuple<Ts...>> {
    static constexpr size_t N  = sizeof...(Ts),
                            N1 = N > 0 ? N : 1;

    using Value = drjit::tuple<Ts...>;
    using Indices = std::make_index_sequence<N>;

    static constexpr auto Name =
        const_name("tuple[") +
        const_name<N == 0>(const_name("()"), concat(make_caster<Ts>::Name...)) +
        const_name("]");

    /// This caster constructs instances on the fly (otherwise it would not be
    /// able to handle tuples containing references_). Because of this, only the
    /// `operator Value()` cast operator is implemented below, and the type
    /// alias below informs users of this class of this fact.
    template <typename T> using Cast = Value;

    bool from_python(handle src, uint32_t flags,
                     cleanup_list *cleanup) noexcept {
        return from_python_impl(src, flags, cleanup, Indices{});
    }

    template <size_t... Is>
    bool from_python_impl(handle src, uint32_t flags, cleanup_list *cleanup,
                          std::index_sequence<Is...>) noexcept {
        (void) src; (void) flags; (void) cleanup;

        PyObject *temp; // always initialized by the following line
        PyObject **o = NB_CALL(seq_get_with_size)(src.ptr(), N, &temp);

        temp_ref = steal(temp);

        return (o && ... &&
                drjit::get<Is>(casters).from_python(
                    o[Is], flags_for_local_caster<Ts>(flags), cleanup));
    }

    template <typename T>
    static handle from_cpp(T&& value, rv_policy policy,
                           cleanup_list *cleanup) noexcept {
        return from_cpp_impl((forward_t<T>) value, policy, cleanup, Indices{});
    }

    template <typename T>
    static handle from_cpp(T *value, rv_policy policy,
                           cleanup_list *cleanup) noexcept {
        if (!value)
            return none().release();
        return from_cpp_impl(*value, policy, cleanup, Indices{});
    }

    template <typename T, size_t... Is>
    static handle from_cpp_impl(T &&value, rv_policy policy,
                                cleanup_list *cleanup,
                                std::index_sequence<Is...>) noexcept {
        (void) value; (void) policy; (void) cleanup;

        PyObject *items[N1] { };

        (void) (... && ((items[Is] = make_caster<Ts>::from_cpp(
                             forward_like_<T>(drjit::get<Is>(value)), policy,
                             cleanup).ptr()) != nullptr));

        return NB_CALL(tuple_new)(items, N);
    }

    template <typename T>
    bool can_cast() const noexcept { return can_cast_impl(Indices{}); }

    explicit operator Value() { return cast_impl(Indices{}); }

    template <size_t... Is>
    bool can_cast_impl(std::index_sequence<Is...>) const noexcept {
        return (drjit::get<Is>(casters).template can_cast<Ts>() && ...);
    }

    template <size_t... Is> Value cast_impl(std::index_sequence<Is...>) {
        return Value(drjit::get<Is>(casters).operator cast_t<Ts>()...);
    }

    drjit::tuple<make_caster<Ts>...> casters;
    object temp_ref;
};

template <> struct type_caster<drjit::half> {
    bool from_python(handle src, uint32_t flags, cleanup_list *) noexcept {
        float f;
        bool success = NB_CALL(load_f32)(NB_CTX, src.ptr(), flags, &f);
        value = drjit::half(f);
        return success;
    }

    static handle from_cpp(drjit::half src, rv_policy, cleanup_list *) noexcept {
        return PyFloat_FromDouble((double) src);
    }

    NB_TYPE_CASTER(drjit::half, const_name("float"))
};

template <> struct type_caster<drjit::string> {
    NB_TYPE_CASTER(drjit::string, const_name("str"))

    bool from_python(handle src, uint32_t, cleanup_list *) noexcept {
        if (!PyUnicode_Check(src.ptr()))
            return false;
        Py_ssize_t size;
        const char *str = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
        if (!str) {
            PyErr_Clear();
            return false;
        }
        value = drjit::string(str, (size_t) size);
        return true;
    }

    static handle from_cpp(const drjit::string &value, rv_policy,
                           cleanup_list *) noexcept {
        return PyUnicode_FromStringAndSize(value.begin(), value.size());
    }
};

template <typename Type>
struct type_caster<drjit::vector<Type>> : list_caster<drjit::vector<Type>, Type> { };

NAMESPACE_END(detail)

template <> struct ndarray_traits <drjit::half> {
    static constexpr bool is_complex = false;
    static constexpr bool is_float   = true;
    static constexpr bool is_bool    = false;
    static constexpr bool is_int     = false;
    static constexpr bool is_signed  = true;
};

NAMESPACE_END(NB_NAMESPACE)
