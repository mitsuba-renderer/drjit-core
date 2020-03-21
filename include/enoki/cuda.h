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

template <typename Value_> struct CUDAArray;

template <typename Value>
CUDAArray<Value> select(const CUDAArray<bool> &m, const CUDAArray<Value> &a,
                        const CUDAArray<Value> &b);

template <typename Value_>
struct CUDAArray {
    using Value = Value_;
    static constexpr VarType Type = var_type<Value>::value;
    static constexpr bool IsCUDA = true;

    CUDAArray() = default;

    ~CUDAArray() { jitc_var_dec_ref_ext(m_index); }

    CUDAArray(const CUDAArray &a) : m_index(a.m_index) {
        jitc_var_inc_ref_ext(m_index);
    }

    CUDAArray(CUDAArray &&a) : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> CUDAArray(const CUDAArray<T> &v) {
        const char *op;

        if (std::is_floating_point<Value>::value &&
            std::is_floating_point<T>::value && sizeof(Value) > sizeof(T))
            op = "cvt.$t0.$t1 $r0, $r1";
        else if (std::is_floating_point<Value>::value)
            op = "cvt.rn.ftz.$t0.$t1 $r0, $r1";
        else if (std::is_floating_point<T>::value && std::is_integral<Value>::value)
            op = "cvt.rzi.ftz.$t0.$t1 $r0, $r1";
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
        jitc_var_inc_ref_ext(a.m_index);
        jitc_var_dec_ref_ext(m_index);
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

    CUDAArray<bool> operator>(const CUDAArray &a) const {
        const char *op = std::is_signed<Value>::value
                             ? "setp.gt.$t1 $r0, $r1, $r2"
                             : "setp.hi.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, m_index, a.index()));
    }

    CUDAArray<bool> operator>=(const CUDAArray &a) const {
        const char *op = std::is_signed<Value>::value
                             ? "setp.ge.$t1 $r0, $r1, $r2"
                             : "setp.hs.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, m_index, a.index()));
    }


    CUDAArray<bool> operator<(const CUDAArray &a) const {
        const char *op = std::is_signed<Value>::value
                             ? "setp.lt.$t1 $r0, $r1, $r2"
                             : "setp.lo.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, m_index, a.index()));
    }

    CUDAArray<bool> operator<=(const CUDAArray &a) const {
        const char *op = std::is_signed<Value>::value
                             ? "setp.le.$t1 $r0, $r1, $r2"
                             : "setp.lo.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, m_index, a.index()));
    }

    friend CUDAArray<bool> eq(const CUDAArray &a, const CUDAArray &b) {
        const char *op = !std::is_same<Value, bool>::value
            ? "setp.eq.$t1 $r0, $r1, $r2" :
              "xor.$t1 $r0, $r1, $r2$n"
              "not.$t1 $r0, $r0";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, a.index(), b.index()));
    }

    friend CUDAArray<bool> neq(const CUDAArray &a, const CUDAArray &b) {
        const char *op = !std::is_same<Value, bool>::value
            ? "setp.ne.$t1 $r0, $r1, $r2" :
              "xor.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, a.index(), b.index()));
    }

    CUDAArray operator-() const {
        return from_index(
            jitc_trace_append_1(Type, "neg.ftz.$t0 $r0, $r1", 1, m_index));
    }

    CUDAArray operator~() const {
        return from_index(
            jitc_trace_append_1(Type, "not.$b0 $r0, $r1", 1, m_index));
    }

    CUDAArray operator|(const CUDAArray &a) const {
        return from_index(jitc_trace_append_2(Type, "or.$b0 $r0, $r1, $r2", 1,
                                              m_index, a.index()));
    }

    CUDAArray operator&(const CUDAArray &a) const {
        return from_index(jitc_trace_append_2(Type, "and.$b0 $r0, $r1, $r2", 1,
                                              m_index, a.index()));
    }

    CUDAArray operator^(const CUDAArray &a) const {
        return from_index(jitc_trace_append_2(Type, "xor.$b0 $r0, $r1, $r2", 1,
                                              m_index, a.index()));
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

    CUDAArray& operator|=(const CUDAArray &v) {
        return operator=(*this | v);
    }

    CUDAArray& operator&=(const CUDAArray &v) {
        return operator=(*this & v);
    }

    CUDAArray& operator^=(const CUDAArray &v) {
        return operator=(*this ^ v);
    }

    friend CUDAArray sqrt(const CUDAArray &a) {
        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "sqrt.rn.ftz.$t0 $r0, $r1", 1, a.index()));
    }

    friend CUDAArray abs(const CUDAArray &a) {
        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "abs.ftz.$t0 $r0, $r1", 1, a.index()));
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
            jitc_trace_append_0(VarType::UInt32, "mov.u32 $r0, $i", 1));
        jitc_var_set_size(index.index(), size, false);

        if (start == 0 && step == 1)
            return index;
        else
            return fmadd(CUDAArray(index), CUDAArray((Value) step),
                         CUDAArray((Value) start));
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
        return (const Value *) jitc_var_ptr(m_index);
    }

    Value *data() {
        return (Value *) jitc_var_ptr(m_index);
    }

    static CUDAArray from_index(uint32_t index) {
        CUDAArray result;
        result.m_index = index;
        return result;
    }

protected:
    uint32_t m_index = 0;
};

template <typename Value>
void set_label(const CUDAArray<Value> &a, const char *label) {
    jitc_var_set_label(a.index(), label);
}

template <typename Value>
CUDAArray<Value> select(const CUDAArray<bool> &m, const CUDAArray<Value> &t,
                        const CUDAArray<Value> &f) {
    if (!std::is_same<Value, bool>::value) {
        return CUDAArray<Value>::from_index(jitc_trace_append_3(
            CUDAArray<Value>::Type, "selp.$t0 $r0, $r1, $r2, $r3", 1, t.index(),
            f.index(), m.index()));
    } else {
        return (m & t) | (~m & f);
    }
}

template <typename OutArray, size_t Stride = sizeof(typename OutArray::Value),
          typename Index, typename std::enable_if<OutArray::IsCUDA, int>::type = 0>
static OutArray gather(const void *ptr, const CUDAArray<Index> &index,
                       const CUDAArray<bool> &mask = true) {
    using Value = typename OutArray::Value;

    if (sizeof(Index) != 4) {
        /* Prefer 32 bit index arithmetic, 64 bit multiplies are
           emulated and thus very expensive on NVIDIA GPUs.. */
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int32_t, uint32_t>::type;
        return gather<OutArray, Stride>(ptr, CUDAArray<Int>(index), mask);
    }

    const char *mul_op;
    switch (Stride) {
        case 1: mul_op = "add.$t0 $r0, $r1, $r2"; break;
        case 2: mul_op = "mul.wide.$t1 $r0, $r1, 2$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 4: mul_op = "mul.wide.$t1 $r0, $r1, 4$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 8: mul_op = "mul.wide.$t1 $r0, $r1, 8$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        default: jitc_fail("Unsupported stride!");
    }

    using UInt64 = CUDAArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr)),
           addr = UInt64::from_index(jitc_trace_append_2(
               UInt64::Type, mul_op, 1, index.index(), base.index()));

    if (!std::is_same<Value, bool>::value) {
        return OutArray::from_index(jitc_trace_append_2(
            OutArray::Type,
            "@$r2 ld.global.$t0 $r0, [$r1]$n"
            "@!$r2 mov.$b0 $r0, 0", 1,
            addr.index(), mask.index()));
    } else {
        return neq(OutArray::from_index(jitc_trace_append_2(
            OutArray::Type,
            "@$r2 ld.global.u8 $r0, [$r1]$n"
            "@!$r2 mov.$b0 $r0, 0", 1,
            addr.index(), mask.index())), 0u);
    }
}

template <size_t Stride_ = 0, typename Value, typename Index>
static void scatter(void *ptr, const CUDAArray<Value> &value,
                    const CUDAArray<Index> &index,
                    const CUDAArray<bool> &mask = true) {
    if (sizeof(Index) != 4) {
        /* Prefer 32 bit index arithmetic, 64 bit multiplies are
           emulated and thus very expensive on NVIDIA GPUs.. */
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int32_t, uint32_t>::type;
        scatter<Stride_>(ptr, value, CUDAArray<Int>(index), mask);
        return;
    }

    constexpr size_t Stride = Stride_ != 0 ? Stride_ : sizeof(Value);

    const char *mul_op;
    switch (Stride) {
        case 1: mul_op = "add.$t0 $r0, $r1, $r2"; break;
        case 2: mul_op = "mul.wide.$t1 $r0, $r1, 2$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 4: mul_op = "mul.wide.$t1 $r0, $r1, 4$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 8: mul_op = "mul.wide.$t1 $r0, $r1, 8$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        default: jitc_fail("Unsupported stride!");
    }

    using UInt64 = CUDAArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr)),
           addr = UInt64::from_index(jitc_trace_append_2(
               UInt64::Type, mul_op, 1, index.index(), base.index()));

    uint32_t var;
    if (!std::is_same<Value, bool>::value) {
        var = jitc_trace_append_3(VarType::Invalid,
                                  "@$r3 st.global.$t2 [$r1], $r2", 1,
                                  addr.index(), value.index(), mask.index());
    } else {
        var = jitc_trace_append_3(VarType::Invalid,
                                  "selp.u32 $r0, 1, 0, $r2$n"
                                  "@$r3 st.global.u8 [$r1], $r0", 1,
                                  addr.index(), value.index(), mask.index());
    }

    jitc_var_mark_side_effect(var);
}

template <typename Array, size_t Stride = sizeof(typename Array::Value),
          typename Index, typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array gather(const Array &src, const CUDAArray<Index> &index,
             const CUDAArray<bool> &mask = true) {

    jitc_set_scatter_gather_operand(src.index(), 1);
    Array result = gather<Array, Stride>(src.data(), index, mask);
    jitc_set_scatter_gather_operand(0, 0);
    return result;
}

template <size_t Stride = 0, typename Value, typename Index>
void scatter(CUDAArray<Value> &dst, const CUDAArray<Value> &value,
             const CUDAArray<Index> &index,
             const CUDAArray<bool> &mask = true) {

    jitc_set_scatter_gather_operand(dst.index(), 0);
    scatter<Stride>(dst.data(), value, index, mask);
    jitc_set_scatter_gather_operand(0, 0);
    jitc_var_mark_dirty(dst.index());
}

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
