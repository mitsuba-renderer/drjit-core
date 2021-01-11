/*
    enoki-jit/containers.h -- Tiny self-contained unique_ptr/vector/tuple

    unique_ptr/vector/tuple are used by the Enoki parent project and some test
    cases in this repository. Unfortunately, the std::... versions of these
    containers pull in ~800KB / 31K LOC of headers into *every compile unit*,
    which is insane. This file satisifies all needs with < 5KB and 170 LOC.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>
#include <utility>

NAMESPACE_BEGIN(enoki)

template <typename T> struct ek_unique_ptr {
    using Type = std::remove_extent_t<T>;

    ek_unique_ptr() = default;
    ek_unique_ptr(const ek_unique_ptr &) = delete;
    ek_unique_ptr &operator=(const ek_unique_ptr &) = delete;
    ek_unique_ptr(Type *data) : m_data(data) { }
    ek_unique_ptr(ek_unique_ptr &&other) : m_data(other.m_data) {
        other.m_data = nullptr;
    }
    ~ek_unique_ptr() { reset(); }

    ek_unique_ptr &operator=(ek_unique_ptr &&other) {
        reset();
        m_data = other.m_data;
        other.m_data = nullptr;
        return *this;
    }

    void reset() noexcept(true) {
        if constexpr (std::is_array_v<T>)
            delete[] m_data;
        else
            delete m_data;
        m_data = nullptr;
    }

    Type& operator[](size_t index) { return m_data[index]; }
    const Type& operator[](size_t index) const { return m_data[index]; }

    Type* get() { return m_data; }
    const Type* get() const { return m_data; }
    Type* operator->() { return m_data; }
    const Type* operator->() const { return m_data; }

    Type* release () {
        Type *tmp = m_data;
        m_data = nullptr;
        return tmp;
    }

protected:
    Type *m_data = nullptr;
};

template <typename T> struct ek_vector {
    ek_vector() = default;
    ek_vector(const ek_vector &) = delete;
    ek_vector &operator=(const ek_vector &) = delete;
    ek_vector(ek_vector &&) = default;
    ek_vector &operator=(ek_vector &&) = default;
    ek_vector(size_t size, const T &value)
        : m_data(new T[size]), m_size(size), m_capacity(size) {
        for (size_t i = 0; i < size; ++i)
            m_data[i] = value;
    }

    void push_back(const T &value) {
        if (m_size >= m_capacity)
            expand();
        m_data[m_size++] = value;
    }

    void clear() { m_size = 0; }
    size_t size() const { return m_size; }
    T *data() { return m_data.get(); }
    const T *data() const { return m_data.get(); }

    void expand() {
        size_t capacity_new = m_capacity == 0 ? 1 : (m_capacity * 2);
        ek_unique_ptr<T[]> data_new(new T[capacity_new]);
        for (size_t i = 0; i < m_size; ++i)
            data_new[i] = m_data[i];
        m_data = std::move(data_new);
        m_capacity = capacity_new;
    }

    T &operator[](size_t i) { return m_data[i]; }
    const T &operator[](size_t i) const { return m_data[i]; }

protected:
    ek_unique_ptr<T[]> m_data;
    size_t m_size = 0;
    size_t m_capacity = 0;
};

struct ek_index_vector : ek_vector<uint32_t> {
    using Base = ek_vector<uint32_t>;
    using Base::Base;
    using Base::operator=;

    ek_index_vector(size_t size) : Base(size, 0) { }
    ~ek_index_vector() { clear(); }

    void push_back(uint32_t value) {
        jit_var_inc_ref_ext_impl(value);
        Base::push_back(value);
    }

    void clear() {
        for (size_t i = 0; i < size(); ++i)
            jit_var_dec_ref_ext_impl(operator[](i));
        Base::clear();
    }
};

// Tiny self-contained tuple to avoid having to import 1000s of LOC from <tuple>
template <typename... Ts> struct ek_tuple;
template <> struct ek_tuple<> {
    template <size_t> using type = void;
};

template <typename T, typename... Ts> struct ek_tuple<T, Ts...> : ek_tuple<Ts...> {
    using Base = ek_tuple<Ts...>;

    ek_tuple() = default;
    ek_tuple(const ek_tuple &) = default;
    ek_tuple(ek_tuple &&) = default;
    ek_tuple& operator=(ek_tuple &&) = default;
    ek_tuple& operator=(const ek_tuple &) = default;

    ek_tuple(const T& value, const Ts&... ts)
        : Base(ts...), value(value) { }

    ek_tuple(T&& value, Ts&&... ts)
        : Base(std::move(ts)...), value(std::move(value)) { }

    template <size_t I> auto& get() {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I> const auto& get() const {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I>
    using type =
        std::conditional_t<I == 0, T, typename Base::template type<I - 1>>;

private:
    T value;
};

template <typename... Ts> ek_tuple(Ts &&...) -> ek_tuple<std::decay_t<Ts>...>;

NAMESPACE_END(enoki)
