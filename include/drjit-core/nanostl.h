/*
    drjit-core/nanostl.h -- Tiny self-contained implementations of a subset
    of the STL providing: std::unique_ptr, std::vector, and std::tuple

    unique_ptr/vector/tuple/string are used throughout Dr.Jit. However, the
    official ``std::`` versions of these containers pull in a truly astonishing
    amount of header code--e.g., over 2 megabytes for std::vector<..> on
    libc++. This imposes an unacceptable compile-time cost on every compilation
    unit. For this reason, Dr.Jit maintains a parallel "nano" version of those
    STL containers with only the subset of features we need. This version
    requires less than 20 KiB of header code.

    Related issue: https://github.com/llvm/llvm-project/issues/80196

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <cstdlib>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <new>
#include "macros.h"
#if defined(_MSC_VER)
#  include <cstring>
#endif

NAMESPACE_BEGIN(drjit)

template <bool Cond> using enable_if_t = std::enable_if_t<Cond, int>;

template <typename T>
using forward_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, T &&>;

template <typename T> struct unique_ptr {
    using Type = std::remove_extent_t<T>;

public:
    unique_ptr() = default;
    unique_ptr(const unique_ptr &) = delete;
    unique_ptr &operator=(const unique_ptr &) = delete;
    unique_ptr(Type *data) : m_data(data) { }
    unique_ptr(unique_ptr &&other) noexcept : m_data(other.m_data) {
        other.m_data = nullptr;
    }
    template <typename E>
    unique_ptr(unique_ptr<E> &&other) : m_data(other.release()) {}
    ~unique_ptr() { reset(); }

    unique_ptr &operator=(unique_ptr &&other) noexcept {
        reset(other.m_data);
        other.m_data = nullptr;
        return *this;
    }

    template <typename E> unique_ptr &operator=(unique_ptr<E> &&other) {
        reset(other.release());
        return *this;
    }

    void reset(Type *p = nullptr) noexcept {
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

template <typename T, typename ... Ts> unique_ptr<T> make_unique(Ts&&...args) {
    return unique_ptr<T>(new T((forward_t<Ts>) args...));
};

template <typename T> struct vector {
public:
    vector() = default;
    vector(const vector &v)
        : m_data(alloc(v.size())), m_capacity(v.size()) {
        for (size_t i = 0; i < v.size(); ++i) {
            new (&m_data[i]) T(v[i]);
            m_size++;
        }
    }
    vector(vector &&v) noexcept
        : m_data(v.m_data), m_size(v.m_size), m_capacity(v.m_capacity) {
        v.m_data = nullptr;
        v.m_size = v.m_capacity = 0;
    }

    ~vector() {
        release();
    }

    void clear() {
        for (size_t i = 0; i < m_size; ++i)
            m_data[i].~T();
        m_size = 0;
    }

    vector(size_t size) : m_data(alloc(size)), m_capacity(size) {
        for (size_t i = 0; i < size; ++i) {
            new (&m_data[m_size]) T();
            m_size++;
        }
    }

    vector(size_t size, const T &value)
        : m_data(alloc(size)), m_capacity(size) {
        for (size_t i = 0; i < size; ++i) {
            new (&m_data[i]) T(value);
            m_size++;
        }
    }

    vector(const T *start, const T *end)
        : m_data(alloc(end - start)), m_capacity(end - start) {
        for (size_t i = 0; i < m_capacity; ++i) {
            new (&m_data[i]) T(start[i]);
            m_size++;
        }
    }

    vector &operator=(const vector &v) {
        return operator=(vector(v));
    }

    vector &operator=(vector &&v) noexcept {
        release();
        m_data = v.m_data;
        m_size = v.m_size;
        m_capacity = v.m_capacity;
        v.m_size = v.m_capacity = 0;
        v.m_data = nullptr;
        return *this;
    }

    void push_back(const T &value) {
        if (m_size == m_capacity)
            expand();
        new (&m_data[m_size]) T(value);
        m_size++;
    }

    void push_back(T &&value) {
        if (m_size == m_capacity)
            expand();
        new (&m_data[m_size]) T(std::move(value));
        m_size++;
    }

    template <typename... Args> void emplace_back(Args &&...args) {
        if (m_size == m_capacity)
            expand();
        new (&m_data[m_size]) T(std::forward<Args>(args)...);
        m_size++;
    }

    bool operator==(const vector &s) const {
        if (m_size != s.m_size)
            return false;
        if (m_size == 0)
            return true;
        for (size_t i = 0; i < s.m_size; ++i) {
            if (m_data[i] != s.m_data[i])
                return false;
        }
        return true;
    }

    bool operator!=(const vector &s) const { return !operator==(s); }

    void pop_back() {
        back().~T();
        --m_size;
    }

    T &back() { return m_data[m_size - 1]; }
    const T &back() const { return m_data[m_size - 1]; }

    void resize(size_t size, T value = T()) {
        reserve(size);
        if (m_size > size) {
            while (m_size != size)
                m_data[--m_size].~T();
        } else {
            while (m_size != size) {
                new (&m_data[m_size]) T(value);
                m_size++;
            }
        }
    }

    void swap(vector &v) {
        std::swap(m_data, v.m_data);
        std::swap(m_size, v.m_size);
        std::swap(m_capacity, v.m_capacity);
    }

    void reserve(size_t size) {
        if (size <= m_capacity)
            return;

        vector tmp;
        tmp.m_data = alloc(size);
        tmp.m_capacity = size;
        for (size_t i = 0; i < m_size; ++i) {
            new (&tmp.m_data[i]) T(std::move(m_data[i]));
            tmp.m_size++;
        }
        operator=(std::move(tmp));
    }

    void expand() { reserve(m_capacity ? m_capacity * 2 : 1); }

    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }
    T *data() { return m_data; }
    const T *data() const { return m_data; }
    bool empty() const { return m_size == 0; }

    T &operator[](size_t i) { return m_data[i]; }
    const T &operator[](size_t i) const { return m_data[i]; }
    T *begin() { return m_data; }
    T *end() { return m_data + m_size; }
    const T *begin() const { return m_data; }
    const T *end() const { return m_data + m_size; }

protected:
    static T * alloc(size_t size) {
        if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
            return (T*) operator new[](sizeof(T) * size, (std::align_val_t) alignof(T));
        else
            return (T*) operator new[](sizeof(T) * size);
    }

    void release() noexcept {
        clear();
        #if defined (__cpp_sized_deallocation)
            if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
                operator delete[]((void *) m_data, sizeof(T) * m_capacity, (std::align_val_t) alignof(T));
            else
                operator delete[]((void *) m_data, sizeof(T) * m_capacity);
        #else
            if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
                operator delete[]((void *) m_data, (std::align_val_t) alignof(T));
            else
                operator delete[]((void *) m_data);
        #endif
    }

protected:
    T* m_data = nullptr;
    size_t m_size = 0;
    size_t m_capacity = 0;
};


template <typename... Ts> struct tuple;
template <> struct tuple<> {
    template <size_t> using type = void;
    static constexpr size_t Size = 0;
};

template <typename T, typename... Ts> struct tuple<T, Ts...> : tuple<Ts...> {
    using Base = tuple<Ts...>;
    static constexpr size_t Size = 1 + sizeof...(Ts);
    template <size_t I>
    using type = std::conditional_t<I == 0, T, typename Base::template type<I - 1>>;

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

namespace detail {
    // Type rait to prune all occurrences of references and 'const'
    template <typename T> struct strip_cr { using type = T; };
    template <typename T> using strip_cr_t = typename strip_cr<T>::type;
    template <typename T> struct strip_cr<const T> { using type = strip_cr_t<T>; };
    template <typename T, size_t Size> struct strip_cr<T[Size]> { using type = strip_cr_t<T>[Size]; };
    template <typename T, size_t Size> struct strip_cr<const T[Size]> { using type = strip_cr_t<T>[Size]; };
    template <typename T> struct strip_cr<T*> { using type = strip_cr_t<T>*; };
    template <typename T> struct strip_cr<T&> { using type = strip_cr_t<T>; };
    template <typename T> struct strip_cr<T&&> { using type = strip_cr_t<T>; };

    template <typename, typename = int> struct formatter;
    template <typename T> using make_formatter = formatter<strip_cr_t<T>>;
};

class string {
public:
    string() = default;
    string(string &&s) noexcept { operator=(std::move(s)); }
    string(const char *value, size_t size) noexcept {
        if (size) {
            m_data = unique_ptr<char[]>(new char[size + 1]);
            m_capacity = m_size = size;
            JIT_BUILTIN(memcpy)(m_data.get(), value, size);
            m_data[size] = '\0';
        }
    }

    string(const char *s) { assign(s); }
    template <typename T> explicit string(const T &v) { assign(v); }
    string(const string &v) { assign(v); }

    template <typename T> string &operator=(const T &v) { return assign(v); }
    string &operator=(const string &v) { return assign(v); }
    string &operator=(string &&v) noexcept {
        m_data = std::move(v.m_data);
        m_size = v.m_size;
        m_capacity = v.m_capacity;
        v.m_size = v.m_capacity = 0;
        return *this;
    }

    template <typename T> string &assign(const T &value) {
        clear();
        put(value);
        return *this;
    }

    bool operator==(const string &s) const {
        return m_size == s.m_size &&
               (m_size == 0 ||
                JIT_BUILTIN(memcmp)(m_data.get(), s.m_data.get(), m_size) == 0);
    }

    bool operator!=(const string &s) const { return !operator==(s); }

    void put_unchecked(const char *value, size_t size) {
        if (!size)
            return;
        #if defined(NDEBUG)
        if (m_size + size > m_capacity)
            abort();
        #endif

        JIT_BUILTIN(memcpy)(m_data.get() + m_size, value, size);
        m_size += size;
        m_data[m_size] = '\0';
    }

    void swap(string &s) {
        std::swap(m_data, s.m_data);
        std::swap(m_size, s.m_size);
        std::swap(m_capacity, s.m_capacity);
    }

    void reserve(size_t capacity) {
        if (capacity > m_capacity)
            grow<false>(capacity - m_size);
    }

    void resize(size_t size, char ch = '\0') {
        reserve(size);
        if (m_data) {
            for (size_t i = m_size; i < size; ++i)
                m_data[i] = ch;
            m_data[size] = '\0';
        }
        m_size = size;
    }

    void clear() {
        if (m_data)
            m_data[0] = '\0';
        m_size = 0;
    }

    template <typename T> string operator+(const T &v) const & {
        string r;
        size_t s1 = size(), s2 = detail::make_formatter<T>::bound(0, v);
        r.grow(s1 + s2);
        r.put_unchecked(m_data.get(), s1);
        detail::make_formatter<T>::format(r, 0, s2, v);
        return r;
    }

    template <typename T> friend string operator+(const T &v, const string &s) {
        string r;
        size_t s1 = detail::make_formatter<T>::bound(0, v), s2 = s.size();
        r.grow(s1 + s2);
        detail::make_formatter<T>::format(r, 0, s1, v);
        r.put_unchecked(s.m_data.get(), s2);
        return r;
    }

    template <typename T> string operator+(const T &v) && {
        string r(std::move(*this));
        r.put(v);
        return r;
    }

    string &iput(size_t) { return *this; }
    template <typename T> string &iput(size_t indent, const T &value) {
        size_t bound = detail::make_formatter<T>::bound(indent, value);
        grow(bound);
        detail::make_formatter<T>::format(*this, indent, bound, value);
        return *this;
    }

    template <typename... Ts> string &iput(size_t indent, const Ts &...args) {
        size_t bounds[sizeof...(Ts)];
        int ctr = 0;
        ((bounds[ctr++] = detail::make_formatter<Ts>::bound(indent, args)), ...);
        size_t bound = 0;
        for (size_t i : bounds)
            bound += i;
        grow(bound);
        ctr = 0;
        (detail::make_formatter<Ts>::format(*this, indent, bounds[ctr++], args), ...);
        return *this;
    }

    template <typename... Ts> string &put(const Ts &...args) {
        return iput(0, args...);
    }

    string& indent(size_t amount) {
        if (amount) {
            grow(amount);
            JIT_BUILTIN(memset)(m_data.get() + m_size, ' ', amount);
            m_size += amount;
            m_data[m_size] = '\0';
        }
        return *this;
    }

    template <typename T> string &operator+=(const T &s) { return put(s); }

    char *data() { return m_data.get(); }
    const char *data() const { return m_data.get(); }
    char *begin() { return m_data.get(); }
    char *end() { return m_data.get() + m_size; }
    const char *begin() const { return m_data.get(); }
    const char *end() const { return m_data.get() + m_size; }
    const char *c_str() const { return m_data ? m_data.get() : &m_terminator; }

    char &operator[](size_t i) { return begin()[i]; }
    const char &operator[](size_t i) const { return begin()[i]; }

    size_t capacity() const { return m_capacity; }
    size_t size() const { return m_size; }
    size_t length() const { return m_size; }
    size_t emtpy() const { return m_size == 0; }

private:
    template <bool EnsureMinGrowth = true> void grow(size_t size) {
        size_t needed_capacity = m_size + size;
        if (needed_capacity > m_capacity) {
            if constexpr (EnsureMinGrowth) {
                size_t min_capacity = m_capacity * 2;
                if (needed_capacity < min_capacity)
                    needed_capacity = min_capacity;
            }
            m_capacity = needed_capacity;
            unique_ptr<char[]> new_data(new char[needed_capacity + 1]);
            if (m_size)
                JIT_BUILTIN(memcpy)(new_data.get(), m_data.get(), m_size);
            new_data[m_size] = '\0';
            m_data = std::move(new_data);
        }
    }
private:
    static inline const char m_terminator = '\0';
    unique_ptr<char[]> m_data;
    size_t m_capacity = 0;
    size_t m_size = 0;
};

NAMESPACE_BEGIN(detail)

template <typename T> struct formatter<T, enable_if_t<std::is_integral_v<T>>> {
    static constexpr size_t MaxSize =
        (std::is_signed_v<T> ? 1 : 0) + int(sizeof(T) * 5 + 1) / 2;
    using UInt = std::make_unsigned_t<T>;

    static size_t bound(size_t, T) { return MaxSize; }
    static void format(string &s, size_t, size_t, T value) {
        const char *digits = "0123456789";
        bool is_negative = false;
        UInt value_u = UInt(value);

        if constexpr (std::is_signed_v<T>) {
            if (value < 0) {
                value_u = UInt(-value);
                is_negative = true;
            }
        }

        char buf[MaxSize];
        size_t offset = MaxSize;
        do {
            buf[--offset] = digits[value_u % 10];
            value_u /= 10;
        } while (value_u);

        if (is_negative)
            buf[--offset] = '-';

        s.put_unchecked(buf + offset, MaxSize - offset);
    }
};

template <typename T_> struct formatter<T_, enable_if_t<std::is_pointer_v<T_>>> {
    using T = std::add_pointer_t<const std::remove_pointer_t<T_>>;
    static constexpr size_t MaxSize = 2 + 2 * sizeof(T);
    static size_t bound(size_t, T) { return MaxSize; }
    static void format(string &s, size_t, size_t, T value_) {
        uintptr_t value = (uintptr_t) value_;
        const char *digits = "0123456789abcdef";

        char buf[MaxSize];
        size_t offset = MaxSize;
        do {
            buf[--offset] = digits[value % 16];
            value /= 16;
        } while (value);

        buf[--offset] = 'x';
        buf[--offset] = '0';

        s.put_unchecked(buf + offset, MaxSize - offset);
    }
};

template <size_t Size> struct formatter<char[Size]> {
    static size_t bound(size_t, const char *) { return Size - 1; }
    static void format(string &s, size_t, size_t, const char (&value)[Size]) {
        s.put_unchecked(value, Size - 1);
    }
};

template <> struct formatter<char> {
    static size_t bound(size_t, char) { return 1; }
    static void format(string &s, size_t, size_t, char value) {
        s.put_unchecked(&value, 1);
    }
};

template <> struct formatter<char *> {
    static size_t bound(size_t, const char *s) { return JIT_BUILTIN(strlen)(s); }
    static void format(string &s, size_t, size_t bound, const char *value) {
        s.put_unchecked(value, bound);
    }
};

template<> struct formatter<drjit::string> {
    static size_t bound(size_t, const drjit::string &s) { return s.size(); }
    static void format(string &s, size_t, size_t bound, const string &value) {
        s.put_unchecked(value.begin(), bound);
    }
};

template <typename T> struct formatter<T, enable_if_t<std::is_floating_point_v<T>>> {
    static constexpr size_t MaxSize = 20;
    static size_t bound(size_t, T) { return MaxSize; }
    static void format(string &s, size_t, size_t, T value) {
        char buf[MaxSize + 1];
        size_t size = JIT_BUILTIN(snprintf)(buf, MaxSize + 1, "%g", (double) value);
        s.put_unchecked(buf, size > MaxSize ? MaxSize : size);
    }
};


struct dummy_string {
    template <typename... Ts> dummy_string &iput(size_t indent, const Ts &...args) {
        m_size += (detail::formatter<Ts>::bound(indent, args) + ...);
        return *this;
    }
    template <typename... Ts> dummy_string &put(const Ts &...args) {
        (iput(0, args), ...);
        return *this;
    }
    dummy_string& indent(size_t amount) { m_size += amount; return *this; }
    void put_unchecked(const char *, size_t size) { m_size += size; }
    size_t size() { return m_size; }
    size_t m_size = 0;
};

template <> struct formatter<bool> {
    static size_t bound(size_t, bool) { return 1; }
    static void format(string &s, size_t, size_t, bool value) {
        char c = value ? '1' : '0';
        s.put_unchecked(&c, 1);
    }
};

NAMESPACE_END(detail)

template <typename Stream, typename Sentry = typename Stream::sentry>
Stream &operator<<(Stream &stream, const string &s) {
    stream.write(s.begin(), s.size());
    return stream;
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
