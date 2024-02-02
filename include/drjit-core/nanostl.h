/*
    drjit-core/nanostl.h -- Tiny self-contained implementations of a subset
    of the STL providing: std::unique_ptr, std::vector, and std::tuple

    unique_ptr/vector/tuple are used throughout Dr.Jit. However, the official
    ``std::`` versions of these containers pull in a truly astonishing amount
    of header code--e.g., over 2 megabytes for std::vector<..> on libc++. This
    imposes an unacceptable compile-time cost on every compilation unit. For
    this reason, Dr.Jit maintains a parallel "nano" version of those STL
    containers with only the subset of features we need. This version requires
    less than 10 KiB of header code.

    Related issue: https://github.com/llvm/llvm-project/issues/80196

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <utility>
#include <cstdlib>
#include "macros.h"

NAMESPACE_BEGIN(drjit)

template <typename T>
using forward_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, T &&>;

template <typename T> struct unique_ptr {
    using Type = std::remove_extent_t<T>;

    unique_ptr() = default;
    unique_ptr(const unique_ptr &) = delete;
    unique_ptr &operator=(const unique_ptr &) = delete;
    unique_ptr(Type *data) : m_data(data) { }
    unique_ptr(unique_ptr &&other) : m_data(other.m_data) {
        other.m_data = nullptr;
    }
    template <typename E>
    unique_ptr(unique_ptr<E> &&other) : m_data(other.release()) {}
    ~unique_ptr() { reset(); }

    unique_ptr &operator=(unique_ptr &&other) {
        reset(other.m_data);
        other.m_data = nullptr;
        return *this;
    }

    template <typename E> unique_ptr &operator=(unique_ptr<E> &&other) {
        reset(other.release());
        return *this;
    }

    void reset(Type *p = nullptr) noexcept(true) {
        if constexpr (std::is_array_v<T>)
            delete[] m_data;
        else
            delete m_data;
        m_data = p;
    }

    Type& operator[](size_t index) { return m_data[index]; }
    const Type& operator[](size_t index) const { return m_data[index]; }
    bool operator!() const { return m_data == nullptr; }
    operator bool() const { return m_data != nullptr; }

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

template <class T, class... Ts> unique_ptr<T> make_unique(Ts&&...args) {
    return unique_ptr<T>(new T((forward_t<Ts>) args...));
};

template <typename T> struct vector {
    vector() = default;
    vector(const vector &v) { operator=(v); }
    vector(vector &&v) { operator=(std::move(v)); }

    vector(size_t size)
        : m_data(new T[size]), m_size(size), m_capacity(size) { }

    vector(size_t size, const T &value)
        : m_data(new T[size]), m_size(size), m_capacity(size) {
        for (size_t i = 0; i < size; ++i)
            m_data[i] = value;
    }

    vector(const T *start, const T *end) {
        m_size = m_capacity = end - start;
        m_data = new T[end - start];
        for (size_t i = 0; i < m_size; ++i)
            m_data[i] = start[i];
    }

    vector &operator=(const vector &v) {
        m_data = new T[v.m_size];
        m_size = v.m_size;
        m_capacity = v.m_size;
        for (size_t i = 0; i < m_size; ++i)
            m_data[i] = v.m_data[i];
        return *this;
    }

    vector &operator=(vector &&v) {
        m_data = std::move(v.m_data);
        m_size = v.m_size;
        m_capacity = v.m_capacity;
        v.m_size = v.m_capacity = 0;
        return *this;
    }

    void push_back(const T &value) {
        if (m_size == m_capacity)
            expand();
        m_data[m_size++] = value;
    }

    void resize(size_t size, T value = T()) {
        reserve(size);
        for (size_t i = m_size; i < size; ++i)
            m_data[i] = value;
        m_size = size;
    }

    void swap(vector &v) {
        std::swap(m_data, v.m_data);
        std::swap(m_size, v.m_size);
        std::swap(m_capacity, v.m_capacity);
    }

    void reserve(size_t size) {
        if (size <= m_capacity)
            return;

        unique_ptr<T[]> data_new(new T[size]);
        for (size_t i = 0; i < m_size; ++i)
            data_new[i] = m_data[i];

        m_data = std::move(data_new);
        m_capacity = size;
    }

    void expand() { reserve(m_capacity ? m_capacity * 2 : 1); }

    void clear() { m_size = 0; }
    size_t size() const { return m_size; }
    T *data() { return m_data.get(); }
    const T *data() const { return m_data.get(); }
    bool empty() const { return m_size == 0; }

    T &operator[](size_t i) { return m_data[i]; }
    const T &operator[](size_t i) const { return m_data[i]; }
    T *begin() { return m_data.get(); }
    T *end() { return m_data.get() + m_size; }
    const T *begin() const { return m_data.get(); }
    const T *end() const { return m_data.get() + m_size; }

protected:
    unique_ptr<T[]> m_data;
    size_t m_size = 0;
    size_t m_capacity = 0;
};

// Tiny self-contained tuple to avoid having to import 1000s of LOC from <tuple>
template <typename... Ts> struct tuple;
template <> struct tuple<> {
    template <size_t> using type = void;
    static constexpr size_t Size = 0;
};

template <typename T, typename... Ts> struct tuple<T, Ts...> : tuple<Ts...> {
    static constexpr size_t Size = 1 + sizeof...(Ts);
    static constexpr size_t IsDrJitTuple = true;
    using Base = tuple<Ts...>;

    tuple() = default;
    tuple(const tuple &) = default;
    tuple(tuple &&) = default;
    tuple& operator=(tuple &&) = default;
    tuple& operator=(const tuple &) = default;

    template <typename... Ts2> tuple(const tuple<Ts2...> &a) : Base(a.base()), value(a.value) { }
    template <typename... Ts2> tuple(tuple<Ts2...> &&a) : Base(std::move(a.base())), value(a.value) { }
    template <typename... Ts2> tuple &operator=(const tuple<Ts2...> &a) {
        value = a.value;
        Base::operator=(a.base());
        return *this;
    }

    template <typename... Ts2> tuple &operator=(tuple<Ts2...> &&a) {
        value = std::move(a.value);
        Base::operator=(std::move(a.base()));
        return *this;
    }

    template <typename A, typename... As, std::enable_if_t<sizeof...(As) == sizeof...(Ts), int> = 0>
    JIT_INLINE tuple(A &&a, As &&...as)
        : Base((forward_t<As>) as...), value((forward_t<A>) a) { }

    JIT_INLINE Base &base() { return *this; }
    JIT_INLINE const Base &base() const { return *this; }

    template <size_t I>
    using type =
        std::conditional_t<I == 0, T, typename Base::template type<I - 1>>;

    T value;
};

template <typename... Ts> tuple(Ts &&...) -> tuple<std::decay_t<Ts>...>;

template <typename... Ts> drjit::tuple<std::decay_t<Ts>...> make_tuple(Ts&&... args) {
    return { (forward_t<Ts>) args ... };
}

template <typename... Ts> drjit::tuple<Ts&...> tie(Ts&... args) {
    return { args ... };
}

template <size_t I, typename... Args> JIT_INLINE auto &get(const drjit::tuple<Args...> &t) {
    if constexpr (I == 0)
        return t.value;
    else
        return drjit::get<I - 1>((const typename drjit::tuple<Args...>::Base &) t);
}

template <size_t I, typename... Args> JIT_INLINE auto &get(drjit::tuple<Args...> &t) {
    if constexpr (I == 0)
        return t.value;
    else
        return drjit::get<I - 1>((typename drjit::tuple<Args...>::Base &) t);
}

template <size_t I, typename... Args> JIT_INLINE auto&& get(drjit::tuple<Args...> &&t) {
    if constexpr (I == 0)
        return std::move(t.value);
    else
        return drjit::get<I - 1>((typename drjit::tuple<Args...>::Base &&) t);
}

NAMESPACE_END(drjit)

// Support for C++17 structured bindings
template <typename... Ts>
struct std::tuple_size<drjit::tuple<Ts...>>
    : std::integral_constant<size_t, sizeof...(Ts)> { };

template <size_t I, typename... Ts>
struct std::tuple_element<I, drjit::tuple<Ts...>> {
    using type = typename drjit::tuple<Ts...>::template type<I>;
};
