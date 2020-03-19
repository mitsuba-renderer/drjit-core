/*
    enoki/cuda.h -- Simple C++ array class with operator overloading (CUDA)

    This library implements convenient wrapper class around the C API in
    'enoki/jit.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/jit.h>
#include <enoki/traits.h>
#include <cstring>
#include <cstdio>

template <typename Value_>
struct CUDAArray {
    using Value = Value_;
    static constexpr VarType Type = var_type<Value>::value;

    CUDAArray() = default;

    ~CUDAArray() { jitc_var_ext_ref_dec(m_index); }

    CUDAArray(const CUDAArray &a) : m_index(a.m_index) {
        jitc_var_ext_ref_inc(m_index);
    }

    CUDAArray(CUDAArray &&a) : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> CUDAArray(const CUDAArray<T> &v) {
        const char *op;

        if (std::is_floating_point<T>::value && std::is_integral<Value>::value)
            op = "cvt.rzi.$t0.$t1 $r0, $r1";
        else if (std::is_integral<T>::value && std::is_floating_point<Value>::value)
            op = "cvt.rn.$t0.$t1 $r0, $r1";
        else
            op = "cvt.$t0.$t1 $r0, $r1";

        m_index = jitc_trace_append_1(Type, op, 1, v.index());
    }

    CUDAArray(Value value) {
        const char *fmt = nullptr;

        switch (Type) {
            case VarType::Float16:
                fmt = "mov.$t0 $r0, %04x";
                break;

            case VarType::Float32:
                fmt = "mov.$t0 $r0, 0f%08x";
                break;

            case VarType::Float64:
                fmt = "mov.$t0 $r0, 0d%016llx";
                break;

            case VarType::Bool:
                fmt = "mov.$t0 $r0, %i";
                break;

            case VarType::Int8:
            case VarType::UInt8:
                fmt = "mov.$t0 $r0, 0x%02x";
                break;

            case VarType::Int16:
            case VarType::UInt16:
                fmt = "mov.$t0 $r0, 0x%04x";
                break;

            case VarType::Int32:
            case VarType::UInt32:
                fmt = "mov.$t0 $r0, 0x%08x";
                break;

            case VarType::Pointer:
            case VarType::Int64:
            case VarType::UInt64:
                fmt = "mov.$t0 $r0, 0x%016llx";
                break;

            default:
                fmt = "<<invalid format during cast>>";
                break;
        }

        uint_with_size_t<Value> value_uint;
        char value_str[32];
        memcpy(&value_uint, &value, sizeof(Value));
        snprintf(value_str, 32, fmt, value_uint);

        m_index = jitc_trace_append_0(Type, value_str, 0);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    CUDAArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jitc_var_copy(Type, data, sizeof...(Args));
    }

    CUDAArray &operator=(const CUDAArray &a) {
        jitc_var_ext_ref_inc(a.m_index);
        jitc_var_ext_ref_dec(m_index);
        m_index = a.m_index;
        return *this;
    }

    CUDAArray &operator=(CUDAArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    CUDAArray operator+(const CUDAArray &v) const {
        const char *op = std::is_floating_point<Value>::value
            ? "add.rn.ftz.$t0 $r0, $r1, $r2"
            : "add.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray operator-(const CUDAArray &v) const {
        const char *op = std::is_floating_point<Value>::value
            ? "sub.rn.ftz.$t0 $r0, $r1, $r2"
            : "sub.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray operator*(const CUDAArray &v) const {
        const char *op = std::is_floating_point<Value>::value
            ? "mul.rn.ftz.$t0 $r0, $r1, $r2"
            : "mul.lo.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray operator/(const CUDAArray &v) const {
        const char *op = std::is_floating_point<Value>::value
            ? "div.rn.ftz.$t0 $r0, $r1, $r2"
            : "div.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray& operator+=(const CUDAArray &v) {
        return operator=(*this + v);
    }

    CUDAArray& operator-=(const CUDAArray &v) {
        return operator=(*this - v);
    }

    CUDAArray& operator*=(const CUDAArray &v) {
        return operator=(*this * v);
    }

    CUDAArray& operator/=(const CUDAArray &v) {
        return operator=(*this / v);
    }

    CUDAArray operator-() const {
        return from_index(
            jitc_trace_append_1(Type, "neg.ftz.$t0 $r0, $r1", 1, m_index));
    }

    friend CUDAArray fmadd(const CUDAArray &a, const CUDAArray &b,
                           const CUDAArray &c) {
        const char *op = std::is_floating_point<Value>::value
            ? "fma.rn.ftz.$t0 $r0, $r1, $r2, $r3"
            : "mad.lo.$t0 $r0, $r1, $r2, $r3";

        return CUDAArray::from_index(
            jitc_trace_append_3(Type, op, 1, a.index(), b.index(), c.index()));
    }

    friend CUDAArray fmsub(const CUDAArray &a, const CUDAArray &b,
                           const CUDAArray &c) {
        return fmadd(a, b, -c);
    }

    friend CUDAArray fnmadd(const CUDAArray &a, const CUDAArray &b,
                            const CUDAArray &c) {
        return fmadd(-a, b, c);
    }

    friend CUDAArray fnmsub(const CUDAArray &a, const CUDAArray &b,
                            const CUDAArray &c) {
        return fmadd(-a, b, -c);
    }

    static CUDAArray empty(size_t size) {
        size_t byte_size = size * sizeof(Value);
        void *ptr = jitc_malloc(AllocType::Device, byte_size);
        return from_index(jitc_var_map(Type, ptr, size, 1));
    }

    static CUDAArray zero(size_t size) {
        if (size == 1) {
            return CUDAArray(0);
        } else {
            uint8_t value = 0;
            size_t byte_size = size * sizeof(Value);
            void *ptr = jitc_malloc(AllocType::Device, byte_size);
            jitc_fill(VarType::UInt8, ptr, byte_size, &value);
            return from_index(jitc_var_map(Type, ptr, size, 1));
        }
    }

    static CUDAArray full(Value value, size_t size) {
        if (size == 1) {
            return CUDAArray(value);
        } else {
            size_t byte_size = size * sizeof(Value);
            void *ptr = jitc_malloc(AllocType::Device, byte_size);
            jitc_fill(Type, ptr, size, &value);
            return from_index(jitc_var_map(Type, ptr, size, 1));
        }
    }

    static CUDAArray arange(size_t size) {
        return arange(0, (size_t) size, 1);
    }

    static CUDAArray arange(ssize_t start, ssize_t stop, ssize_t step) {
        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);

        using UInt32 = CUDAArray<uint32_t>;
        UInt32 index = UInt32::from_index(
            jitc_trace_append_0(VarType::UInt32, "mov.u32, $r0, %r0", 1));
        jitc_var_set_size(index.index(), size, false);

        if (start == 0 && step == 1)
            return index;
        else
            return fmadd(CUDAArray(index), CUDAArray((Value) step), CUDAArray((Value) start));
    }

    CUDAArray eval() const {
        jitc_var_eval(m_index);
        return *this;
    }

    bool valid() const { return m_index != 0; }

    size_t size() const {
        return jitc_var_size(m_index);
    }

    uint32_t index() const {
        return m_index;
    }

    const char *str() {
        return jitc_var_str(m_index);
    }

    const Value *data() const {
        return jitc_var_ptr(m_index);
    }

    Value *data() {
        return jitc_var_ptr(m_index);
    }

    static CUDAArray from_index(uint32_t index) {
        CUDAArray result;
        result.m_index = index;
        return result;
    }

protected:
    uint32_t m_index = 0;
};


template <typename Value> CUDAArray<Value> hsum(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Add, v.data(), v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hprod(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Mul, v.data(), v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hmax(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Max, v.data(), v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hmin(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Min, v.data(), v.size(),
                result.data());
    return result;
}
