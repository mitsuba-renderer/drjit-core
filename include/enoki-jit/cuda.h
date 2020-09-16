/*
    enoki-jit/cuda.h -- Simple C++ array class with operator overloading (CUDA)

    This library implements convenient wrapper class around the C API in
    'enoki/jit.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/util.h>
#include <cstdio>
#include <vector>

NAMESPACE_BEGIN(enoki)

template <typename Value_> struct CUDAArray;

template <typename Value>
CUDAArray<Value> select(const CUDAArray<bool> &m,
                        const CUDAArray<Value> &a,
                        const CUDAArray<Value> &b);

template <typename Value_>
struct CUDAArray {
    using Value = Value_;
    static constexpr VarType Type = var_type<Value>::value;
    static constexpr bool IsCUDA = true;
    static constexpr bool IsLLVM = false;

    CUDAArray() = default;

    ~CUDAArray() { jitc_var_dec_ref_ext(m_index); }

    CUDAArray(const CUDAArray &a) : m_index(a.m_index) {
        jitc_var_inc_ref_ext(m_index);
    }

    CUDAArray(CUDAArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> CUDAArray(const CUDAArray<T> &v) {
        const char *op;

        if (std::is_floating_point<Value>::value &&
            std::is_floating_point<T>::value && sizeof(Value) > sizeof(T)) {
            op = "cvt.$t0.$t1 $r0, $r1";
        } else if (std::is_floating_point<Value>::value) {
                op = "cvt.rn.$t0.$t1 $r0, $r1";
        } else if (std::is_floating_point<T>::value && std::is_integral<Value>::value) {
                op = "cvt.rzi.$t0.$t1 $r0, $r1";
        } else {
            if (sizeof(T) == sizeof(Value)) {
                m_index = v.index();
                jitc_var_inc_ref_ext(m_index);
                return;
            }
            op = "cvt.$t0.$t1 $r0, $r1";
        }

        m_index = jitc_var_new_1(Type, op, 1, 1, v.index());
    }

    CUDAArray(Value value) {
        uint64_t tmp = 0;
        memcpy(&tmp, &value, sizeof(Value));
        m_index = jitc_var_new_literal(Type, 1, tmp, 1, 0);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    CUDAArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jitc_var_copy_mem(AllocType::Host, Type, 1, data,
                                (uint32_t) sizeof...(Args));
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

        // Simple constant propagation
        if (is_literal_zero())
            return v;
        else if (v.is_literal_zero())
            return *this;

        const char *op = std::is_same<Value, float>::value
                             ? "add.ftz.$t0 $r0, $r1, $r2"
                             : "add.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray operator-(const CUDAArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_same<Value, float>::value
                             ? "sub.ftz.$t0 $r0, $r1, $r2"
                             : "sub.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray operator*(const CUDAArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        // Simple constant propagation
        if (is_literal_one())
            return v;
        else if (v.is_literal_one())
            return *this;

        const char *op;
        if (std::is_floating_point<Value>::value)
            op = std::is_same<Value, float>::value ? "mul.ftz.$t0 $r0, $r1, $r2"
                                                   : "mul.$t0 $r0, $r1, $r2";
        else
            op = "mul.lo.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray operator/(const CUDAArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        // Simple constant propagation
        if (v.is_literal_one())
            return *this;

        const char *op;
        if (std::is_same<Value, float>::value)
            op = "div.rn.ftz.$t0 $r0, $r1, $r2";
        else if (std::is_same<Value, double>::value)
            op = "div.rn.$t0 $r0, $r1, $r2";
        else
            op = "div.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray operator/(Value value) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        // Simple constant propagation
        if (value == (Value) 1)
            return *this;

        if (std::is_floating_point<Value>::value) {
            return operator*(Value(1) / value);
        } else if ((value & (value - 1)) == 0) {
            int shift = sizeof(Value) * 8 - 1 - clz(uint_with_size_t<Value>(value));
            return operator>>(shift);
        }

        return operator/(CUDAArray(value));
    }

    CUDAArray<bool> operator>(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_signed<Value>::value
                             ? "setp.gt.$t1 $r0, $r1, $r2"
                             : "setp.hi.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }

    CUDAArray<bool> operator>=(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_signed<Value>::value
                             ? "setp.ge.$t1 $r0, $r1, $r2"
                             : "setp.hs.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }


    CUDAArray<bool> operator<(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_signed<Value>::value
                             ? "setp.lt.$t1 $r0, $r1, $r2"
                             : "setp.lo.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }

    CUDAArray<bool> operator<=(const CUDAArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_signed<Value>::value
                             ? "setp.le.$t1 $r0, $r1, $r2"
                             : "setp.ls.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }

    friend CUDAArray<bool> eq(const CUDAArray &a, const CUDAArray &b) {
        const char *op = !std::is_same<Value, bool>::value
            ? "setp.eq.$t1 $r0, $r1, $r2" :
              "xor.$t1 $r0, $r1, $r2$n"
              "not.$t1 $r0, $r0";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, a.index(), b.index()));
    }

    friend CUDAArray<bool> neq(const CUDAArray &a, const CUDAArray &b) {
        const char *op = !std::is_same<Value, bool>::value
            ? "setp.ne.$t1 $r0, $r1, $r2" :
              "xor.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, a.index(), b.index()));
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

        const char *op = std::is_same<Value, float>::value
                             ? "neg.ftz.$t0 $r0, $r1"
                             : "neg.$t0 $r0, $r1";

        return from_index(jitc_var_new_1(Type, op, 1, 1, m_index));
    }

    CUDAArray operator~() const {
        return from_index(jitc_var_new_1(Type, "not.$b0 $r0, $r1", 1, 1, m_index));
    }

    CUDAArray operator|(const CUDAArray &a) const {
        // Simple constant propagation
        if (std::is_same<Value, bool>::value) {
            if (is_literal_one() || a.is_literal_zero())
                return *this;
            else if (a.is_literal_one() || is_literal_zero())
                return a;
        }

        return from_index(jitc_var_new_2(Type, "or.$b0 $r0, $r1, $r2",
                                         1, 1, m_index, a.index()));
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    CUDAArray operator|(const CUDAArray<bool> &m) const {
        // Simple constant propagation
        if (m.is_literal_zero())
            return *this;
        else if (m.is_literal_one())
            return CUDAArray(memcpy_cast<Value>(uint_with_size_t<Value>(-1)));

        return from_index(jitc_var_new_2(Type, "selp.$b0 $r0, -1, $r1, $r2",
                                         1, 1, index(), m.index()));
    }

    CUDAArray operator&(const CUDAArray &a) const {
        // Simple constant propagation
        if (std::is_same<Value, bool>::value) {
            if (is_literal_one() || a.is_literal_zero())
                return a;
            else if (a.is_literal_one() || is_literal_zero())
                return *this;
        }

        return from_index(jitc_var_new_2(Type, "and.$b0 $r0, $r1, $r2",
                                         1, 1, m_index, a.index()));
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    CUDAArray operator&(const CUDAArray<bool> &m) const {
        // Simple constant propagation
        if (m.is_literal_one())
            return *this;
        else if (m.is_literal_zero())
            return CUDAArray(Value(0));

        return from_index(jitc_var_new_2(Type, "selp.$b0 $r0, $r1, 0, $r2",
                                         1, 1, index(), m.index()));
    }

    CUDAArray operator^(const CUDAArray &a) const {
        // Simple constant propagation
        if (std::is_same<Value, bool>::value) {
            if (is_literal_zero())
                return a;
            else if (a.is_literal_zero())
                return *this;
        }

        return from_index(jitc_var_new_2(Type, "xor.$b0 $r0, $r1, $r2",
                                         1, 1, m_index, a.index()));
    }

    CUDAArray operator<<(const CUDAArray<uint32_t> &v) const {
        return from_index(jitc_var_new_2(
            Type, "shl.$b0 $r0, $r1, $r2", 1, 1, index(), v.index()));
    }

    CUDAArray operator>>(const CUDAArray<uint32_t> &v) const {
        const char *op;
        if (std::is_signed<Value>::value)
            op = "shr.$t0 $r0, $r1, $r2";
        else
            op = "shr.$b0 $r0, $r1, $r2";

        return from_index(
            jitc_var_new_2(Type, op, 1, 1, index(), v.index()));
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

    CUDAArray& operator/=(Value v) {
        return operator=(*this / v);
    }

    CUDAArray& operator|=(const CUDAArray &v) {
        return operator=(*this | v);
    }

    CUDAArray& operator&=(const CUDAArray &v) {
        return operator=(*this & v);
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    CUDAArray& operator|=(const CUDAArray<bool> &v) {
        return operator=(*this | v);
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    CUDAArray& operator&=(const CUDAArray<bool> &v) {
        return operator=(*this & v);
    }

    CUDAArray& operator^=(const CUDAArray &v) {
        return operator=(*this ^ v);
    }

    CUDAArray& operator<<=(const CUDAArray<uint32_t> &v) {
        return operator=(*this << v);
    }

    CUDAArray& operator>>=(const CUDAArray<uint32_t> &v) {
        return operator=(*this >> v);
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

        const char *op = std::is_same<Value, float>::value
                             ? "abs.ftz.$t0 $r0, $r1"
                             : "abs.$t0 $r0, $r1";

        return CUDAArray::from_index(
            jitc_var_new_1(Type, op, 1, 1, a.index()));
    }

    friend CUDAArray sqrt(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_same<Value, float>::value
                             ? "sqrt.rn.ftz.$t0 $r0, $r1"
                             : "sqrt.rn.$t0 $r0, $r1";

        return CUDAArray::from_index(jitc_var_new_1(Type, op, 1, 1, a.index()));
    }

    friend CUDAArray round(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_var_new_1(Type,
            "cvt.rni.$t0.$t0 $r0, $r1", 1, 1, a.index()));
    }

    friend CUDAArray floor(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_var_new_1(Type,
            "cvt.rmi.$t0.$t0 $r0, $r1", 1, 1, a.index()));
    }

    friend CUDAArray ceil(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_var_new_1(Type,
            "cvt.rpi.$t0.$t0 $r0, $r1", 1, 1, a.index()));
    }

    friend CUDAArray trunc(const CUDAArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return CUDAArray::from_index(jitc_var_new_1(Type,
            "cvt.rzi.$t0.$t0 $r0, $r1", 1, 1, a.index()));
    }

    friend CUDAArray fmadd(const CUDAArray &a, const CUDAArray &b,
                           const CUDAArray &c) {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        // Simple constant propagation
        if (a.is_literal_one()) {
            return b + c;
        } else if (b.is_literal_one()) {
            return a + c;
        } else if (a.is_literal_zero() || b.is_literal_zero()) {
            return c;
        } else if (c.is_literal_zero()) {
            return a * b;
        }

        const char *op;
        if (std::is_floating_point<Value>::value)
            op = std::is_same<Value, float>::value
                     ? "fma.rn.ftz.$t0 $r0, $r1, $r2, $r3"
                     : "fma.rn.$t0 $r0, $r1, $r2, $r3";
        else
            op = "mad.lo.$t0 $r0, $r1, $r2, $r3";

        return CUDAArray::from_index(
            jitc_var_new_3(Type, op, 1, 1, a.index(), b.index(), c.index()));
    }

    CUDAArray& schedule() {
        jitc_var_schedule(m_index);
        return *this;
    }

    const CUDAArray& schedule() const {
        jitc_var_schedule(m_index);
        return *this;
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

    bool is_literal_one() const { return (bool) jitc_var_is_literal_one(m_index); }
    bool is_literal_zero() const { return (bool) jitc_var_is_literal_zero(m_index); }

    size_t size() const {
        return jitc_var_size(m_index);
    }

    void resize(size_t size) {
        uint32_t index = jitc_var_set_size(m_index, (uint32_t) size);
        jitc_var_dec_ref_ext(m_index);
        m_index = index;
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
        if (jitc_var_int_ref(m_index) > 0) {
            eval();
            *this = from_index(
                jitc_var_copy_mem(AllocType::Device, CUDAArray<Value>::Type, 1,
                              data(), (uint32_t) size()));
        }

        jitc_var_write(m_index, offset, &value);
    }

    static CUDAArray map(void *ptr, size_t size, bool free = false) {
        return from_index(
            jitc_var_map_mem(Type, 1, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static CUDAArray copy(const void *ptr, size_t size) {
        return from_index(jitc_var_copy_mem(AllocType::Host, Type, 1, ptr, (uint32_t) size));
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
    return Array::from_index(jitc_var_map_mem(Array::Type, 1, ptr, (uint32_t) size, 1));
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array zero(size_t size) {
    return Array::from_index(jitc_var_new_literal(Array::Type, 1, 0, (uint32_t) size, 0));
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array full(typename Array::Value value, size_t size) {
    uint64_t tmp = 0;
    memcpy(&tmp, &value, sizeof(typename Array::Value));
    return Array::from_index(jitc_var_new_literal(Array::Type, 1, tmp, (uint32_t) size, 0));
}

template <typename Array,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array arange(ssize_t start, ssize_t stop, ssize_t step) {
    using UInt32 = CUDAArray<uint32_t>;
    using Value = typename Array::Value;

    size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
    UInt32 index = UInt32::from_index(
        jitc_var_new_0(VarType::UInt32, "mov.u32 $r0, $i", 1, 1, (uint32_t) size));

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
        jitc_var_new_0(VarType::UInt32, "mov.u32 $r0, $i", 1, 1, (uint32_t) size));

    Value step = (max - min) / Value(size - 1);
    return fmadd(Array(index), Array(step), Array(min));
}

template <typename Value>
CUDAArray<Value> select(const CUDAArray<bool> &m,
                        const CUDAArray<Value> &t,
                        const CUDAArray<Value> &f) {
    // Simple constant propagation
    if (m.is_literal_one()) {
        return t;
    } else if (m.is_literal_zero()) {
        return f;
    } else if (!std::is_same<Value, bool>::value) {
        return CUDAArray<Value>::from_index(jitc_var_new_3(
            CUDAArray<Value>::Type, "selp.$t0 $r0, $r1, $r2, $r3", 1, 1,
            t.index(), f.index(), m.index()));
    } else {
        return (m & t) | (~m & f);
    }
}

template <typename OutArray, typename ValueIn>
OutArray reinterpret_array(const CUDAArray<ValueIn> &input) {
    using ValueOut = typename OutArray::Value;

    static_assert(
        sizeof(ValueIn) == sizeof(ValueOut),
        "reinterpret_array requires arrays with equal-sized element types!");

    if (std::is_integral<ValueIn>::value != std::is_integral<ValueOut>::value) {
        return OutArray::from_index(jitc_var_new_1(
            OutArray::Type, "mov.$b0 $r0, $r1", 1, 1, input.index()));
    } else {
        jitc_var_inc_ref_ext(input.index());
        return OutArray::from_index(input.index());
    }
}

template <typename Value>
CUDAArray<Value> min(const CUDAArray<Value> &a, const CUDAArray<Value> &b) {
    const char *op = std::is_same<Value, float>::value
                         ? "min.ftz.$t0 $r0, $r1, $r2"
                         : "min.$t0 $r0, $r1, $r2";

    return CUDAArray<Value>::from_index(jitc_var_new_2(
        CUDAArray<Value>::Type, op, 1, 1, a.index(), b.index()));
}

template <typename Value>
CUDAArray<Value> max(const CUDAArray<Value> &a, const CUDAArray<Value> &b) {
    const char *op = std::is_same<Value, float>::value
                         ? "max.ftz.$t0 $r0, $r1, $r2"
                         : "max.$t0 $r0, $r1, $r2";

    return CUDAArray<Value>::from_index(jitc_var_new_2(
        CUDAArray<Value>::Type, op, 1, 1, a.index(), b.index()));
}

template <typename OutArray, typename ValueIn>
OutArray round2int(const CUDAArray<ValueIn> &a) {
    if (!jitc_is_floating_point(CUDAArray<ValueIn>::Type) ||
        !jitc_is_integral(OutArray::Type))
        jitc_raise("Unsupported operand type");

    return OutArray::from_index(jitc_var_new_1(
        OutArray::Type, "cvt.rni.$t0.$t1 $r0, $r1", 1, 1, a.index()));
}

template <typename OutArray, typename ValueIn>
OutArray floor2int(const CUDAArray<ValueIn> &a) {
    if (!jitc_is_floating_point(CUDAArray<ValueIn>::Type) ||
        !jitc_is_integral(OutArray::Type))
        jitc_raise("Unsupported operand type");

    return OutArray::from_index(jitc_var_new_1(
        OutArray::Type, "cvt.rmi.$t0.$t1 $r0, $r1", 1, 1, a.index()));
}

template <typename OutArray, typename ValueIn>
OutArray ceil2int(const CUDAArray<ValueIn> &a) {
    if (!jitc_is_floating_point(CUDAArray<ValueIn>::Type) ||
        !jitc_is_integral(OutArray::Type))
        jitc_raise("Unsupported operand type");

    return OutArray::from_index(jitc_var_new_1(
        OutArray::Type, "cvt.rpi.$t0.$t1 $r0, $r1", 1, 1, a.index()));
}

template <typename OutArray, typename ValueIn>
OutArray trunc2int(const CUDAArray<ValueIn> &a) {
    if (!jitc_is_floating_point(CUDAArray<ValueIn>::Type) ||
        !jitc_is_integral(OutArray::Type))
        jitc_raise("Unsupported operand type");

    return OutArray::from_index(jitc_var_new_1(
        OutArray::Type, "cvt.rzi.$t0.$t1 $r0, $r1", 1, 1, a.index()));
}


namespace detail {
template <typename Array,
          typename Index, typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array gather_impl(const void *src_ptr,
                  uint32_t src_index,
                  const CUDAArray<Index> &index,
                  const CUDAArray<bool> &mask = true) {
    using Value = typename Array::Value;

    if (mask.is_literal_zero()) {
        return Array(Value(0));
    } else if (sizeof(Index) != 4) {
        /* Prefer 32 bit index arithmetic, 64 bit multiplies are
           emulated and thus very expensive on NVIDIA GPUs.. */
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int32_t, uint32_t>::type;
        return gather_impl<Array>(src_ptr, src_index, CUDAArray<Int>(index), mask);
    }

    CUDAArray<void *> base = CUDAArray<void *>::from_index(
        jitc_var_copy_ptr(src_ptr, src_index));

    uint32_t var = 0;
    if (mask.is_literal_one()) {
        var = jitc_var_new_2(Array::Type,
                                  !std::is_same<Value, bool>::value
                                      ? "mul.wide.$t2 %rd3, $r2, $s0$n"
                                        "add.$t1 %rd3, %rd3, $r1$n"
                                        "ld.global.nc.$t0 $r0, [%rd3]"
                                      : "mul.wide.$t2 %rd3, $r2, $s0$n"
                                        "add.$t1 %rd3, %rd3, $r1$n"
                                        "ld.global.nc.u8 %w0, [%rd3]$n"
                                        "setp.ne.u16 $r0, %w0, 0",
                                  1, 1, base.index(), index.index());
    } else {
        var = jitc_var_new_3(Array::Type,
                                  !std::is_same<Value, bool>::value
                                      ? "mul.wide.$t2 %rd3, $r2, $s0$n"
                                        "add.$t1 %rd3, %rd3, $r1$n"
                                        "@$r3 ld.global.nc.$t0 $r0, [$r1]$n"
                                        "@!$r3 mov.$b0 $r0, 0"
                                      : "mul.wide.$t2 %rd3, $r2, $s0$n"
                                        "add.$t1 %rd3, %rd3, $r1$n"
                                        "@$r3 ld.global.nc.u8 %w0, [$r1]$n"
                                        "@!$r3 mov.u16 %w0, 0$n"
                                        "setp.ne.u16 $r0, %w0, 0",
                                  1, 1, base.index(), index.index(), mask.index());
    }

    return Array::from_index(var);
}
}

template <typename Array, typename Index,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array gather(const void *src, const CUDAArray<Index> &index,
             const CUDAArray<bool> &mask = true) {
    if (mask.is_literal_zero())
        return Array(typename Array::Value(0));
    return detail::gather_impl<Array>(src, 0, index, mask);
}

template <typename Array, typename Index,
          typename std::enable_if<Array::IsCUDA, int>::type = 0>
Array gather(const Array &src, const CUDAArray<Index> &index,
             const CUDAArray<bool> &mask = true) {
    if (mask.is_literal_zero())
        return Array(typename Array::Value(0));
    src.eval();
    return detail::gather_impl<Array>(src.data(), src.index(), index, mask);
}

template <typename Value, typename Index>
void scatter(CUDAArray<Value> &dst,
             const CUDAArray<Value> &value,
             const CUDAArray<Index> &index,
             const CUDAArray<bool> &mask = true) {
    if (mask.is_literal_zero()) {
        return;
    } else if (sizeof(Index) != 4) {
        /* Prefer 32 bit index arithmetic, 64 bit multiplies are
           emulated and thus very expensive on NVIDIA GPUs.. */
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int32_t, uint32_t>::type;
        return scatter(dst, value, CUDAArray<Int>(index), mask);
    }

    void *ptr = dst.data();

    if (!ptr) {
        dst.eval();
        ptr = dst.data();
    }

    if (jitc_var_int_ref(dst.index()) > 0) {
        dst = CUDAArray<Value>::from_index(jitc_var_copy_mem(
            AllocType::Device, CUDAArray<Value>::Type, 1, ptr, (uint32_t) dst.size()));
        ptr = dst.data();
    }

    CUDAArray<void *> base = CUDAArray<void *>::from_index(
        jitc_var_copy_ptr(ptr, dst.index()));

    uint32_t var;
    if (mask.is_literal_one()) {
        if (!std::is_same<Value, bool>::value) {
            var = jitc_var_new_3(VarType::Invalid,
                                      "mul.wide.$t3 %rd3, $r3, $s2$n"
                                      "add.$t1 %rd3, %rd3, $r1$n"
                                      "st.global.$t2 [%rd3], $r2",
                                      1, 1, base.index(), value.index(),
                                      index.index());
        } else {
            var = jitc_var_new_3(VarType::Invalid,
                                      "mul.wide.$t3 %rd3, $r3, $s2$n"
                                      "add.$t1 %rd3, %rd3, $r1$n"
                                      "selp.u16 %w0, 1, 0, $r2$n"
                                      "st.global.u8 [%rd3], %w0",
                                      1, 1, base.index(), value.index(),
                                      index.index());
        }
    } else {
        if (!std::is_same<Value, bool>::value) {
            var = jitc_var_new_4(VarType::Invalid,
                                      "mul.wide.$t3 %rd3, $r3, $s2$n"
                                      "add.$t1 %rd3, %rd3, $r1$n"
                                      "@$r4 st.global.$t2 [%rd3], $r2",
                                      1, 1, base.index(), value.index(),
                                      index.index(), mask.index());
        } else {
            var = jitc_var_new_4(VarType::Invalid,
                                      "mul.wide.$t3 %rd3, $r3, $s2$n"
                                      "add.$t1 %rd3, %rd3, $r1$n"
                                      "selp.u16 %w0, 1, 0, $r2$n"
                                      "@$r4 st.global.u8 [%rd3], %w0",
                                      1, 1, base.index(), value.index(),
                                      index.index(), mask.index());
        }
    }

    jitc_var_mark_scatter(var, dst.index());
}

template <typename Value, typename Index>
void scatter_add(CUDAArray<Value> &dst,
                 const CUDAArray<Value> &value,
                 const CUDAArray<Index> &index,
                 const CUDAArray<bool> &mask = true) {
    if (mask.is_literal_zero()) {
        return;
    } else if (sizeof(Index) != 4) {
        /* Prefer 32 bit index arithmetic, 64 bit multiplies are
           emulated and thus very expensive on NVIDIA GPUs.. */
        using Int = typename std::conditional<std::is_signed<Index>::value,
                                              int32_t, uint32_t>::type;
        return scatter(dst, value, CUDAArray<Int>(index), mask);
    }

    void *ptr = dst.data();

    if (!ptr) {
        dst.eval();
        ptr = dst.data();
    }

    if (jitc_var_int_ref(dst.index()) > 0) {
        dst = CUDAArray<Value>::from_index(jitc_var_copy_mem(
            AllocType::Device, CUDAArray<Value>::Type, 1, ptr, (uint32_t) dst.size()));
        ptr = dst.data();
    }

    CUDAArray<void *> base = CUDAArray<void *>::from_index(
        jitc_var_copy_ptr(ptr, dst.index()));

    uint32_t var;
    if (mask.is_literal_one()) {
        var = jitc_var_new_3(VarType::Invalid,
                                  "mul.wide.$t3 %rd3, $r3, $s2$n"
                                  "add.$t1 %rd3, %rd3, $r1$n"
                                  "red.global.add.$t2 [%rd3], $r2",
                                  1, 1, base.index(), value.index(),
                                  index.index());
    } else {
        var = jitc_var_new_4(VarType::Invalid,
                                  "mul.wide.$t3 %rd3, $r3, $s2$n"
                                  "add.$t1 %rd3, %rd3, $r1$n"
                                  "@$r4 red.global.add.$t2 [%rd3], $r2",
                                  1, 1, base.index(), value.index(),
                                  index.index(), mask.index());
    }

    jitc_var_mark_scatter(var, dst.index());
}

template <typename Value>
CUDAArray<Value> copysign(const CUDAArray<Value> &v1,
                          const CUDAArray<Value> &v2) {
    if (!jitc_is_floating_point(CUDAArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return abs(v1) | (CUDAArray<Value>(sign_mask<Value>()) & v2);
}

template <typename Value>
CUDAArray<Value> copysign_neg(const CUDAArray<Value> &v1,
                              const CUDAArray<Value> &v2) {
    if (!jitc_is_floating_point(CUDAArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return abs(v1) | (CUDAArray<Value>(sign_mask<Value>()) & ~v2);
}

template <typename Value>
CUDAArray<Value> mulsign(const CUDAArray<Value> &v1,
                         const CUDAArray<Value> &v2) {
    if (!jitc_is_floating_point(CUDAArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return v1 ^ (CUDAArray<Value>(sign_mask<Value>()) & v2);
}

template <typename Value>
CUDAArray<Value> mulsign_neg(const CUDAArray<Value> &v1,
                             const CUDAArray<Value> &v2) {
    if (!jitc_is_floating_point(CUDAArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return v1 ^ (CUDAArray<Value>(sign_mask<Value>()) & ~v2);
}

inline bool all(const CUDAArray<bool> &v) {
    if (v.is_literal_one()) {
        return true;
    } else if (v.is_literal_zero()) {
        return false;
    } else {
        v.eval();
        return (bool) jitc_all((uint8_t *) v.data(), (uint32_t) v.size());
    }
}

inline bool any(const CUDAArray<bool> &v) {
    if (v.is_literal_one()) {
        return true;
    } else if (v.is_literal_zero()) {
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
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Add, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hprod(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Mul, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hmax(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Max, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> CUDAArray<Value> hmin(const CUDAArray<Value> &v) {
    using Array = CUDAArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Min, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

inline std::vector<std::pair<uint32_t, CUDAArray<uint32_t>>>
mkperm(const CUDAArray<uint32_t> &v, uint32_t bucket_count) {
    using UInt32 = CUDAArray<uint32_t>;

    size_t size         = v.size(),
           perm_size    = size * sizeof(uint32_t),
           offsets_size = (size_t(bucket_count) * 3 + 1) * sizeof(uint32_t);

    v.eval();

    uint32_t *offsets = (uint32_t *) jitc_malloc(AllocType::HostPinned, offsets_size),
             *perm    = (uint32_t *) jitc_malloc(AllocType::Device, perm_size);

    jitc_mkperm(v.data(), (uint32_t) size, bucket_count, perm, offsets);

    uint32_t unique_count = offsets[0];
    std::vector<std::pair<uint32_t, CUDAArray<uint32_t>>> result;
    result.reserve(unique_count);

    UInt32 parent =
        UInt32::from_index(jitc_var_map_mem(UInt32::Type, 1, perm, (uint32_t) size, 1));

    for (uint32_t i = 0; i < unique_count; ++i) {
        uint32_t bucket_id     = offsets[i * 3 + 1],
                 bucket_offset = offsets[i * 3 + 2],
                 bucket_size   = offsets[i * 3 + 3];

        uint32_t var_idx =
            jitc_var_map_mem(UInt32::Type, 1, perm + bucket_offset, bucket_size, 0);

        result.emplace_back(bucket_id, UInt32::from_index(var_idx));
    }

    jitc_free(offsets);

    return result;
}

template <typename Value, typename Array = CUDAArray<Value>>
Array popcnt(const CUDAArray<Value> &a) {
    if (!jitc_is_integral(Array::Type))
        jitc_raise("Unsupported operand type");

    return Array::from_index(
        jitc_var_new_1(Array::Type, "popc.$b0 $r0, $r1", 1, 1, a.index()));
}

template <typename Value, typename Array = CUDAArray<Value>>
Array lzcnt(const CUDAArray<Value> &a) {
    if (!jitc_is_integral(Array::Type))
        jitc_raise("Unsupported operand type");

    return Array::from_index(
        jitc_var_new_1(Array::Type, "clz.$b0 $r0, $r1", 1, 1, a.index()));
}

template <typename Value, typename Array = CUDAArray<Value>>
Array tzcnt(const CUDAArray<Value> &a) {
    if (!jitc_is_integral(Array::Type))
        jitc_raise("Unsupported operand type");

    return Array::from_index(jitc_var_new_1(
        Array::Type, "brev.$b0 $r0, $r1$nclz.$b0 $r0, $r0", 1, 1, a.index()));
}

template <typename T> void jitc_schedule(const CUDAArray<T> &a) {
    a.schedule();
}

inline std::vector<std::pair<void *, CUDAArray<uint32_t>>>
vcall(const char *domain, const CUDAArray<uint32_t> &a) {
    uint32_t bucket_count = 0;
    VCallBucket *buckets = jitc_vcall(domain, a.index(), &bucket_count);

    std::vector<std::pair<void *, CUDAArray<uint32_t>>> result;
    result.reserve(bucket_count);
    for (uint32_t i = 0; i < bucket_count; ++i) {
        jitc_var_inc_ref_ext(buckets[i].index);
        result.emplace_back(buckets[i].ptr,
                            CUDAArray<uint32_t>::from_index(buckets[i].index));
    }

    return result;
}

inline CUDAArray<uint32_t> compress(const CUDAArray<bool> &a) {
    uint32_t *perm = (uint32_t *) jitc_malloc(AllocType::Device, a.size() * sizeof(uint32_t));

    uint32_t size = jitc_compress((const uint8_t *) a.data(), (uint32_t) a.size(), perm);

    return CUDAArray<uint32_t>::from_index(
        jitc_var_map_mem(VarType::UInt32, 1, perm, size, 1));
}

template <typename T>
inline CUDAArray<T> block_copy(const CUDAArray<T> &a, uint32_t block_size) {
    size_t size = a.eval().size();
    CUDAArray<T> output = empty<CUDAArray<T>>(size * block_size);
    jitc_block_copy(CUDAArray<T>::Type, a.data(), output.data(), (uint32_t) size, block_size);
    return output;
}

template <typename T>
inline CUDAArray<T> block_sum(const CUDAArray<T> &a, uint32_t block_size) {
    size_t size = a.eval().size();
    if (size % block_size != 0)
        jitc_raise("block_sum(): array size must be divisible by block size (%u)!", block_size);
    size /= block_size;
    CUDAArray<T> output = empty<CUDAArray<T>>(size);
    jitc_block_sum(CUDAArray<T>::Type, a.data(), output.data(), (uint32_t) size, block_size);
    return output;
}

NAMESPACE_END(enoki)
