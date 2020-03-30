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
#include <vector>

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

        m_index = jitc_trace_append_0(Type, value_str, 0, 1);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    CUDAArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jitc_var_copy(Type, data, (uint32_t) sizeof...(Args));
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
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "add.rn.ftz.$t0 $r0, $r1, $r2"
            : "add.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray operator-(const CUDAArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "sub.rn.ftz.$t0 $r0, $r1, $r2"
            : "sub.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray operator*(const CUDAArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "mul.rn.ftz.$t0 $r0, $r1, $r2"
            : "mul.lo.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray operator/(const CUDAArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "div.rn.ftz.$t0 $r0, $r1, $r2"
            : "div.$t0 $r0, $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    CUDAArray<bool> operator>(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_signed<Value>::value
                             ? "setp.gt.$t1 $r0, $r1, $r2"
                             : "setp.hi.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, m_index, a.index()));
    }

    CUDAArray<bool> operator>=(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_signed<Value>::value
                             ? "setp.ge.$t1 $r0, $r1, $r2"
                             : "setp.hs.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, m_index, a.index()));
    }


    CUDAArray<bool> operator<(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_signed<Value>::value
                             ? "setp.lt.$t1 $r0, $r1, $r2"
                             : "setp.lo.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_trace_append_2(
            CUDAArray<bool>::Type, op, 1, m_index, a.index()));
    }

    CUDAArray<bool> operator<=(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

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

    bool operator==(const CUDAArray &a) const {
        return all(eq(*this, a));
    }

    bool operator!=(const CUDAArray &a) const {
        return any(neq(*this, a));
    }

    CUDAArray operator-() const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        return from_index(
            jitc_trace_append_1(Type, "neg.ftz.$t0 $r0, $r1", 1, m_index));
    }

    CUDAArray operator~() const {
        return from_index(
            jitc_trace_append_1(Type, "not.$b0 $r0, $r1", 1, m_index));
    }

    CUDAArray operator|(const CUDAArray &a) const {
        // Simple constant propagation for masks
        if (std::is_same<Value, bool>::value) {
            if (is_all_true() || a.is_all_false())
                return *this;
            else if (a.is_all_true() || is_all_false())
                return a;
        }

        return from_index(jitc_trace_append_2(Type, "or.$b0 $r0, $r1, $r2", 1,
                                              m_index, a.index()));
    }

    CUDAArray operator&(const CUDAArray &a) const {
        // Simple constant propagation for masks
        if (std::is_same<Value, bool>::value) {
            if (is_all_true() || a.is_all_false())
                return a;
            else if (a.is_all_true() || is_all_false())
                return *this;
        }

        return from_index(jitc_trace_append_2(Type, "and.$b0 $r0, $r1, $r2", 1,
                                              m_index, a.index()));
    }

    CUDAArray operator^(const CUDAArray &a) const {
        // Simple constant propagation for masks
        if (std::is_same<Value, bool>::value) {
            if (is_all_false())
                return a;
            else if (a.is_all_false())
                return *this;
        }

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

    CUDAArray operator&&(const CUDAArray &a) const {
        if (!jitc_is_mask(Type))
            jitc_raise("Unsupported operand type");
        return operator&(a);
    }

    CUDAArray operator||(const CUDAArray &a) const {
        if (!jitc_is_mask(Type))
            jitc_raise("Unsupported operand type");
        return operator|(a);
    }

    CUDAArray operator!() const {
        if (!jitc_is_mask(Type))
            jitc_raise("Unsupported operand type");
        return operator~();
    }

    friend CUDAArray abs(const CUDAArray &a) {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "abs.ftz.$t0 $r0, $r1", 1, a.index()));
    }

    friend CUDAArray sqrt(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "sqrt.rn.ftz.$t0 $r0, $r1", 1, a.index()));
    }

    friend CUDAArray round(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "cvt.rni.$t0.$t0 $r0, $r1", 1, a.index()));
    }

    friend CUDAArray floor(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "cvt.rmi.$t0.$t0 $r0, $r1", 1, a.index()));
    }

    friend CUDAArray ceil(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "cvt.rpi.$t0.$t0 $r0, $r1", 1, a.index()));
    }

    friend CUDAArray trunc(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_trace_append_1(Type,
            "cvt.rzi.$t0.$t0 $r0, $r1", 1, a.index()));
    }

    friend CUDAArray fmadd(const CUDAArray &a, const CUDAArray &b,
                           const CUDAArray &c) {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

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

    CUDAArray& eval() {
        jitc_var_eval(m_index);
        return *this;
    }

    const CUDAArray& eval() const {
        jitc_var_eval(m_index);
        return *this;
    }

    bool valid() const { return m_index != 0; }

    bool is_all_true() const { return (bool) jitc_var_is_all_true(m_index); }
    bool is_all_false() const { return (bool) jitc_var_is_all_false(m_index); }

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

    Value read(uint32_t offset) const {
        Value out;
        jitc_var_read(m_index, offset, &out);
        return out;
    }

    void write(uint32_t offset, Value value) {
        jitc_var_write(m_index, offset, &value);
    }

    static CUDAArray map(void *ptr, size_t size, bool free = false) {
        return from_index(
            jitc_var_map(Type, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static CUDAArray copy(const void *ptr, size_t size) {
        return from_index(jitc_var_copy(Type, ptr, (uint32_t) size));
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

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array empty(size_t size) {
    size_t byte_size = size * sizeof(typename Array::Value);
    void *ptr = jitc_malloc(AllocType::Device, byte_size);
    return Array::from_index(jitc_var_map(Array::Type, ptr, (uint32_t) size, 1));
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array zero(size_t size) {
    if (size == 1) {
        return Array(0);
    } else {
        uint8_t value = 0;
        size_t byte_size = size * sizeof(typename Array::Value);
        void *ptr = jitc_malloc(AllocType::Device, byte_size);
        jitc_fill(VarType::UInt8, ptr, byte_size, &value);
        return Array::from_index(jitc_var_map(Array::Type, ptr, (uint32_t) size, 1));
    }
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array full(typename Array::Value value, size_t size) {
    if (size == 1) {
        return Array(value);
    } else {
        size_t byte_size = size * sizeof(typename Array::Value);
        void *ptr = jitc_malloc(AllocType::Device, byte_size);
        jitc_fill(Array::Type, ptr, size, &value);
        return Array::from_index(jitc_var_map(Array::Type, ptr, (uint32_t) size, 1));
    }
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array arange(ssize_t start, ssize_t stop, ssize_t step) {
    using UInt32 = CUDAArray<uint32_t>;
    using Value = typename Array::Value;

    size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
    UInt32 index = UInt32::from_index(
        jitc_trace_append_0(VarType::UInt32, "mov.u32 $r0, $i", 1, (uint32_t) size));

    if (start == 0 && step == 1)
        return Array(index);
    else
        return fmadd(Array(index), Array((Value) step), Array((Value) start));
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array arange(size_t size) {
    return arange<Array>(0, (size_t) size, 1);
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array linspace(typename Array::Value min, typename Array::Value max, size_t size) {
    using UInt32 = CUDAArray<uint32_t>;
    using Value = typename Array::Value;

    UInt32 index = UInt32::from_index(
        jitc_trace_append_0(VarType::UInt32, "mov.u32 $r0, $i", 1, (uint32_t) size));

    Value step = (max - min) / Value(size - 1);
    return fmadd(Array(index), Array(step), Array(min));
}

template <typename Value>
CUDAArray<Value> select(const CUDAArray<bool> &m,
                        const CUDAArray<Value> &t,
                        const CUDAArray<Value> &f) {
    // Simple constant propagation for masks
    if (m.is_all_true()) {
        return t;
    } else if (m.is_all_false()) {
        return f;
    } else if (!std::is_same<Value, bool>::value) {
        return CUDAArray<Value>::from_index(jitc_trace_append_3(
            CUDAArray<Value>::Type, "selp.$t0 $r0, $r1, $r2, $r3", 1, t.index(),
            f.index(), m.index()));
    } else {
        return (m & t) | (~m & f);
    }
}

template <typename Value>
CUDAArray<Value> min(const CUDAArray<Value> &a, const CUDAArray<Value> &b) {
    return CUDAArray<Value>::from_index(jitc_trace_append_2(
        CUDAArray<Value>::Type, "min.ftz.$t0 $r0, $r1, $r2", 1,
        a.index(), b.index()));
}

template <typename Value>
CUDAArray<Value> max(const CUDAArray<Value> &a, const CUDAArray<Value> &b) {
    return CUDAArray<Value>::from_index(jitc_trace_append_2(
        CUDAArray<Value>::Type, "max.ftz.$t0 $r0, $r1, $r2", 1,
        a.index(), b.index()));
}

template <typename OutArray,
          typename Index, typename std::enable_if<OutArray::IsCUDA, int>::type = 0>
OutArray gather(const void *ptr,
                const CUDAArray<Index> &index,
                const CUDAArray<bool> &mask = true) {
    using Value = typename OutArray::Value;
    constexpr size_t Size = sizeof(Value);

    if (sizeof(Index) != 4 && Size != 1) {
        /* Prefer 32 bit index arithmetic, 64 bit multiplies are
           emulated and thus very expensive on NVIDIA GPUs.. */
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int32_t, uint32_t>::type;
        return gather<OutArray>(ptr, CUDAArray<Int>(index), mask);
    } else if (sizeof(Index) != 8 && Size == 1) {
        // Exception: stride == 1
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int64_t, uint64_t>::type;
        return gather<OutArray>(ptr, CUDAArray<Int>(index), mask);
    }

    const char *mul_op;
    switch (Size) {
        case 1: mul_op = "add.$t0 $r0, $r1, $r2"; break;
        case 2: mul_op = "mul.wide.$t1 $r0, $r1, 2$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 4: mul_op = "mul.wide.$t1 $r0, $r1, 4$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 8: mul_op = "mul.wide.$t1 $r0, $r1, 8$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        default: jitc_fail("CUDAArray::gather(): unsupported type!");
    }

    using UInt64 = CUDAArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr)),
           addr = UInt64::from_index(jitc_trace_append_2(
               UInt64::Type, mul_op, 1, index.index(), base.index()));

    uint32_t var = 0;
    if (mask.is_all_false()) {
        return OutArray(Value(0));
    } else if (mask.is_all_true()) {
        var = jitc_trace_append_1(OutArray::Type,
                                  !std::is_same<Value, bool>::value
                                      ? "ld.global.nc.$t0 $r0, [$r1]"
                                      : "ld.global.nc.u8 %w0, [$r1]$n"
                                        "setp.ne.u16 $r0, %w0, 0",
                                  1, addr.index());
    } else {
        var = jitc_trace_append_2(OutArray::Type,
                                  !std::is_same<Value, bool>::value
                                      ? "@$r2 ld.global.nc.$t0 $r0, [$r1]$n"
                                        "@!$r2 mov.$b0 $r0, 0"
                                      : "@$r2 ld.global.nc.u8 %w0, [$r1]$n"
                                        "@!$r2 mov.u16 %w0, 0$n"
                                        "setp.ne.u16 $r0, %w0, 0",
                                  1, addr.index(), mask.index());
    }

    return OutArray::from_index(var);
}

template <typename Value, typename Index>
CUDAArray<void_t> scatter(void *ptr,
                          const CUDAArray<Value> &value,
                          const CUDAArray<Index> &index,
                          const CUDAArray<bool> &mask = true) {
    constexpr size_t Size = sizeof(Value);

    if (sizeof(Index) != 4 && Size != 1) {
        /* Prefer 32 bit index arithmetic, 64 bit multiplies are
           emulated and thus very expensive on NVIDIA GPUs.. */
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int32_t, uint32_t>::type;
        return scatter(ptr, value, CUDAArray<Int>(index), mask);
    } else if (sizeof(Index) != 8 && Size == 1) {
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int64_t, uint64_t>::type;
        return scatter(ptr, value, CUDAArray<Int>(index), mask);
    }

    const char *mul_op;
    switch (Size) {
        case 1: mul_op = "add.$t0 $r0, $r1, $r2"; break;
        case 2: mul_op = "mul.wide.$t1 $r0, $r1, 2$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 4: mul_op = "mul.wide.$t1 $r0, $r1, 4$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        case 8: mul_op = "mul.wide.$t1 $r0, $r1, 8$n"
                         "add.$t0 $r0, $r0, $r2"; break;
        default: jitc_fail("CUDAArray::gather(): unsupported type!");
    }

    using UInt64 = CUDAArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr)),
           addr = UInt64::from_index(jitc_trace_append_2(
               UInt64::Type, mul_op, 1, index.index(), base.index()));

    uint32_t var;
    if (mask.is_all_false()) {
        return CUDAArray<void_t>();
    } else if (mask.is_all_true()) {
        if (!std::is_same<Value, bool>::value) {
            var = jitc_trace_append_2(VarType::Invalid,
                                      "st.global.$t2 [$r1], $r2", 1,
                                      addr.index(), value.index());
        } else {
            var = jitc_trace_append_2(VarType::Invalid,
                                      "selp.u16 %w0, 1, 0, $r2$n"
                                      "st.global.u8 [$r1], %w0", 1,
                                      addr.index(), value.index());
        }
    } else {
        if (!std::is_same<Value, bool>::value) {
            var = jitc_trace_append_3(VarType::Invalid,
                                      "@$r3 st.global.$t2 [$r1], $r2", 1,
                                      addr.index(), value.index(), mask.index());
        } else {
            var = jitc_trace_append_3(VarType::Invalid,
                                      "selp.u16 %w0, 1, 0, $r2$n"
                                      "@$r3 st.global.u8 [$r1], %w0", 1,
                                      addr.index(), value.index(), mask.index());
        }
    }

    jitc_var_inc_ref_ext(var);
    jitc_var_mark_side_effect(var);

    return CUDAArray<void_t>::from_index(var);
}

template <typename Array, typename Index,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array gather(const Array &src, const CUDAArray<Index> &index,
             const CUDAArray<bool> &mask = true) {
    if (mask.is_all_false())
        return Array(typename Array::Value(0));

    jitc_var_eval(src.index());
    Array result = gather<Array>(src.data(), index, mask);
    if (!mask.is_all_false())
        jitc_var_set_extra_dep(result.index(), src.index());
    return result;
}

template <typename Value, typename Index>
CUDAArray<void_t> scatter(CUDAArray<Value> &dst,
                          const CUDAArray<Value> &value,
                          const CUDAArray<Index> &index,
                          const CUDAArray<bool> &mask = true) {
    if (mask.is_all_false())
        return CUDAArray<void_t>();

    if (dst.data() == nullptr)
        jitc_var_eval(dst.index());

    CUDAArray<void_t> result = scatter(dst.data(), value, index, mask);
    jitc_var_set_extra_dep(result.index(), dst.index());
    jitc_var_mark_dirty(dst.index());
    return result;
}

inline bool all(const CUDAArray<bool> &v) {
    if (v.is_all_true()) {
        return true;
    } else if (v.is_all_false()) {
        return false;
    } else {
        v.eval();
        return (bool) jitc_all((uint8_t *) v.data(), (uint32_t) v.size());
    }
}

inline bool any(const CUDAArray<bool> &v) {
    if (v.is_all_true()) {
        return true;
    } else if (v.is_all_false()) {
        return false;
    } else {
        v.eval();
        return (bool) jitc_any((uint8_t *) v.data(), (uint32_t) v.size());
    }
}

inline bool none(const CUDAArray<bool> &v) {
    return !any(v);
}

template <typename Value> CUDAArray<Value> hsum(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Add, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hprod(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Mul, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hmax(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Max, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hmin(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Min, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

inline std::vector<std::pair<uint32_t, CUDAArray<uint32_t>>>
mkperm(const CUDAArray<uint32_t> &v, uint32_t bucket_count) {
    using UInt32 = CUDAArray<uint32_t>;

    size_t size         = v.size(),
           perm_size    = size * sizeof(uint32_t),
           offsets_size = (bucket_count * 3 + 1) * sizeof(uint32_t);

    v.eval();

    uint32_t *offsets = (uint32_t *) jitc_malloc(AllocType::HostPinned, offsets_size),
             *perm    = (uint32_t *) jitc_malloc(AllocType::Device, perm_size);

    jitc_mkperm(v.data(), (uint32_t) size, bucket_count, perm, offsets);

    uint32_t unique_count = offsets[0];
    std::vector<std::pair<uint32_t, CUDAArray<uint32_t>>> result;
    result.reserve(unique_count);

    UInt32 parent =
        UInt32::from_index(jitc_var_map(UInt32::Type, perm, (uint32_t) size, 1));

    for (uint32_t i = 0; i < unique_count; ++i) {
        uint32_t bucket_id     = offsets[i * 3 + 1],
                 bucket_offset = offsets[i * 3 + 2],
                 bucket_size   = offsets[i * 3 + 3];

        uint32_t var_idx =
            jitc_var_map(UInt32::Type, perm + bucket_offset, bucket_size, 0);

        result.emplace_back(bucket_id, UInt32::from_index(var_idx));
    }

    jitc_free(offsets);

    return result;
}
