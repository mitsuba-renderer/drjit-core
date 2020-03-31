/*
    enoki/llvm.h -- Simple C++ array class with operator overloading (LLVM)

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

template <typename Value_> struct LLVMArray;

template <typename Value>
LLVMArray<Value> select(const LLVMArray<bool> &m,
                        const LLVMArray<Value> &a,
                        const LLVMArray<Value> &b);

template <typename OutArray, typename ValueIn>
OutArray reinterpret_array(const LLVMArray<ValueIn> &input);

template <typename Value_>
struct LLVMArray {
    using Value = Value_;
    static constexpr VarType Type = var_type<Value>::value;
    static constexpr bool IsLLVM = true;

    LLVMArray() = default;

    ~LLVMArray() { jitc_var_dec_ref_ext(m_index); }

    LLVMArray(const LLVMArray &a) : m_index(a.m_index) {
        jitc_var_inc_ref_ext(m_index);
    }

    LLVMArray(LLVMArray &&a) : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> LLVMArray(const LLVMArray<T> &v) {
        static_assert(!std::is_same<T, Value>::value,
                      "Conversion constructor called with arguments that don't "
                      "correspond to a conversion!");

        constexpr bool Signed =
            std::is_signed<T>::value && std::is_signed<Value>::value;

        const char *op;
        if (std::is_floating_point<Value>::value && std::is_integral<T>::value) {
            op = std::is_signed<T>::value ? "$r0 = sitofp <$w x $t1> $r1 to <$w x $t0>"
                                          : "$r0 = uitofp <$w x $t1> $r1 to <$w x $t0>";
        } else if (std::is_integral<Value>::value && std::is_floating_point<T>::value) {
            op = std::is_signed<Value>::value ? "$r0 = fptosi <$w x $t1> $r1 to <$w x $t0>"
                                              : "$r0 = fptoui <$w x $t1> $r1 to <$w x $t0>";
        } else if (std::is_floating_point<T>::value && std::is_floating_point<Value>::value) {
            op = sizeof(T) > sizeof(Value) ? "$r0 = fptrunc <$w x $t1> $r1 to <$w x $t0>"
                                           : "$r0 = fpext <$w x $t1> $r1 to <$w x $t0>";
        } else if (std::is_integral<T>::value && std::is_integral<Value>::value) {
            if (sizeof(T) == sizeof(Value)) {
                m_index = v.index();
                jitc_var_inc_ref_ext(m_index);
                return;
            } else {
                op = sizeof(T) > sizeof(Value)
                         ? "$r0 = trunc <$w x $t1> $r1 to <$w x $t0>"
                         : (Signed ? "$r0 = sext <$w x $t1> $r1 to <$w x $t0>"
                                   : "$r0 = zext <$w x $t1> $r1 to <$w x $t0>");
            }
        }
        else {
            jitc_fail("Unsupported conversion!");
        }

        m_index = jitc_trace_append_1(Type, op, 1, v.index());
    }

    LLVMArray(Value value) {
        uint_with_size_t<Value> value_uint;
        unsigned long long value_ull;

        if (Type == VarType::Float32) {
            double d = (double) value;
            memcpy(&value_ull, &d, sizeof(double));
        }  else {
            memcpy(&value_uint, &value, sizeof(Value));
            value_ull = (unsigned long long) value_uint;
        }

        char value_str[256];
        snprintf(value_str, 256,
            (Type == VarType::Float32 || Type == VarType::Float64) ?
            "$r0_0 = insertelement <$w x $t0> undef, $t0 0x%llx, i32 0$n"
            "$r0 = shufflevector <$w x $t0> $r0_0, <$w x $t0> undef, <$w x i32> zeroinitializer" :
            "$r0_0 = insertelement <$w x $t0> undef, $t0 %llu, i32 0$n"
            "$r0 = shufflevector <$w x $t0> $r0_0, <$w x $t0> undef, <$w x i32> zeroinitializer",
            value_ull);

        m_index = jitc_trace_append_0(Type, value_str, 0, 1);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    LLVMArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jitc_var_copy(Type, data, (uint32_t) sizeof...(Args));
    }

    LLVMArray &operator=(const LLVMArray &a) {
        jitc_var_inc_ref_ext(a.m_index);
        jitc_var_dec_ref_ext(m_index);
        m_index = a.m_index;
        return *this;
    }

    LLVMArray &operator=(LLVMArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    LLVMArray operator+(const LLVMArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fadd <$w x $t0> $r1, $r2"
            : "$r0 = add <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator-(const LLVMArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fsub <$w x $t0> $r1, $r2"
            : "$r0 = sub <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator*(const LLVMArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fmul <$w x $t0> $r1, $r2"
            : "$r0 = mul <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator/(const LLVMArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fdiv <$w x $t0> $r1, $r2"
            : "$r0 = div <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray<bool> operator>(const LLVMArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op;
        if (std::is_integral<Value>::value)
            op = std::is_signed<Value>::value
                     ? "$r0 = icmp sgt <$w x $t1> $r1, $r2"
                     : "$r0 = icmp ugt <$w x $t1> $r1, $r2";
        else
            op = "$r0 = fcmp ogt <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_trace_append_2(
            LLVMArray<bool>::Type, op, 1, m_index, a.index()));
    }

    LLVMArray<bool> operator>=(const LLVMArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op;
        if (std::is_integral<Value>::value)
            op = std::is_signed<Value>::value
                     ? "$r0 = icmp sge <$w x $t1> $r1, $r2"
                     : "$r0 = icmp uge <$w x $t1> $r1, $r2";
        else
            op = "$r0 = fcmp oge <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_trace_append_2(
            LLVMArray<bool>::Type, op, 1, m_index, a.index()));
    }


    LLVMArray<bool> operator<(const LLVMArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op;
        if (std::is_integral<Value>::value)
            op = std::is_signed<Value>::value
                     ? "$r0 = icmp slt <$w x $t1> $r1, $r2"
                     : "$r0 = icmp ult <$w x $t1> $r1, $r2";
        else
            op = "$r0 = fcmp olt <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_trace_append_2(
            LLVMArray<bool>::Type, op, 1, m_index, a.index()));
    }

    LLVMArray<bool> operator<=(const LLVMArray &a) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op;
        if (std::is_integral<Value>::value)
            op = std::is_signed<Value>::value
                     ? "$r0 = icmp sle <$w x $t1> $r1, $r2"
                     : "$r0 = icmp ule <$w x $t1> $r1, $r2";
        else
            op = "$r0 = fcmp ole <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_trace_append_2(
            LLVMArray<bool>::Type, op, 1, m_index, a.index()));
    }

    friend LLVMArray<bool> eq(const LLVMArray &a, const LLVMArray &b) {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = icmp eq <$w x $t1> $r1, $r2"
                             : "$r0 = fcmp oeq <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_trace_append_2(
            LLVMArray<bool>::Type, op, 1, a.index(), b.index()));
    }

    friend LLVMArray<bool> neq(const LLVMArray &a, const LLVMArray &b) {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = icmp ne <$w x $t1> $r1, $r2"
                             : "$r0 = fcmp one <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_trace_append_2(
            LLVMArray<bool>::Type, op, 1, a.index(), b.index()));
    }

    bool operator==(const LLVMArray &a) const {
        return all(eq(*this, a));
    }

    bool operator!=(const LLVMArray &a) const {
        return any(neq(*this, a));
    }

    LLVMArray operator-() const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fneg <$w x $t0> $r1"
            : "$r0 = sub <$w x $t0> zeroinitializer, $r1";

        return from_index(
            jitc_trace_append_1(Type, op, 1, m_index));
    }

    LLVMArray operator~() const {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = xor <$w x $t1> $r1, $o0"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = xor <$w x $b0> $r0_0, $o0$n"
                               "$r0 = bitcast <$w x $b0> $r0_1 to <$w x $t0>";

        return from_index(
            jitc_trace_append_1(Type, op, 1, m_index));
    }

    LLVMArray operator|(const LLVMArray &a) const {
        // Simple constant propagation for masks
        if (std::is_same<Value, bool>::value) {
            if (is_all_true() || a.is_all_false())
                return *this;
            else if (a.is_all_true() || is_all_false())
                return a;
        }

        const char *op = std::is_integral<Value>::value
                             ? "$r0 = or <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = or <$w x $b0> $r0_0, $r0_1"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, a.index()));
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    LLVMArray operator|(const LLVMArray<bool> &m) const {
        // Simple constant propagation for masks
        if (m.is_all_false())
            return *this;
        else if (m.is_all_true())
            return LLVMArray(memcpy_cast<Value>(uint_with_size_t<Value>(-1)));

        using UInt = LLVMArray<uint_with_size_t<Value>>;
        UInt x = UInt::from_index(jitc_trace_append_1(
            UInt::Type, "$r0 = sext <$w x $t1> $r1 to <$w x $b0>", 1,
            m.index()));

        return *this | reinterpret_array<LLVMArray>(x);
    }

    LLVMArray operator&(const LLVMArray &a) const {
        // Simple constant propagation for masks
        if (std::is_same<Value, bool>::value) {
            if (is_all_true() || a.is_all_false())
                return a;
            else if (a.is_all_true() || is_all_false())
                return *this;
        }

        const char *op = std::is_integral<Value>::value
                             ? "$r0 = and <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = and <$w x $b0> $r0_0, $r0_1"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, a.index()));
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    LLVMArray operator&(const LLVMArray<bool> &m) const {
        // Simple constant propagation for masks
        if (m.is_all_true())
            return *this;
        else if (m.is_all_false())
            return LLVMArray(Value(0));

        using UInt = LLVMArray<uint_with_size_t<Value>>;
        UInt x = UInt::from_index(jitc_trace_append_1(
            UInt::Type, "$r0 = sext <$w x $t1> $r1 to <$w x $b0>", 1,
            m.index()));

        return *this & reinterpret_array<LLVMArray>(x);
    }

    LLVMArray operator^(const LLVMArray &a) const {
        // Simple constant propagation for masks
        if (std::is_same<Value, bool>::value) {
            if (is_all_false())
                return a;
            else if (a.is_all_false())
                return *this;
        }

        const char *op = std::is_integral<Value>::value
                             ? "$r0 = xor <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = xor <$w x $b0> $r0_0, $r0_1$n"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, a.index()));
    }

    LLVMArray operator<<(const LLVMArray<uint32_t> &v) const {
        return LLVMArray::from_index(jitc_trace_append_2(
            Type, "$r0 = shl <$w x $t0> $r1, $r2", 1, index(), v.index()));
    }

    LLVMArray operator>>(const LLVMArray<uint32_t> &v) const {
        if (std::is_integral<Value>::value && std::is_signed<Value>::value)
            return LLVMArray::from_index(jitc_trace_append_2(
                Type, "$r0 = ashr <$w x $t0> $r1, $r2", 1, index(), v.index()));
        else
            return LLVMArray::from_index(jitc_trace_append_2(
                Type, "$r0 = lshr <$w x $t0> $r1, $r2", 1, index(), v.index()));
    }

    LLVMArray& operator+=(const LLVMArray &v) {
        return operator=(*this + v);
    }

    LLVMArray& operator-=(const LLVMArray &v) {
        return operator=(*this - v);
    }

    LLVMArray& operator*=(const LLVMArray &v) {
        return operator=(*this * v);
    }

    LLVMArray& operator/=(const LLVMArray &v) {
        return operator=(*this / v);
    }

    LLVMArray& operator|=(const LLVMArray &v) {
        return operator=(*this | v);
    }

    LLVMArray& operator&=(const LLVMArray &v) {
        return operator=(*this & v);
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    LLVMArray& operator|=(const LLVMArray<bool> &v) {
        return operator=(*this | v);
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    LLVMArray& operator&=(const LLVMArray<bool> &v) {
        return operator=(*this & v);
    }

    LLVMArray& operator^=(const LLVMArray &v) {
        return operator=(*this ^ v);
    }

    LLVMArray& operator<<=(const LLVMArray<uint32_t> &v) {
        return operator=(*this << v);
    }

    LLVMArray& operator>>=(const LLVMArray<uint32_t> &v) {
        return operator=(*this >> v);
    }

    LLVMArray operator&&(const LLVMArray &a) const {
        if (!jitc_is_mask(Type))
            jitc_raise("Unsupported operand type");
        return operator&(a);
    }

    LLVMArray operator||(const LLVMArray &a) const {
        if (!jitc_is_mask(Type))
            jitc_raise("Unsupported operand type");
        return operator|(a);
    }

    LLVMArray operator!() const {
        if (!jitc_is_mask(Type))
            jitc_raise("Unsupported operand type");
        return operator~();
    }

    friend LLVMArray abs(const LLVMArray &a) {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        if (std::is_floating_point<Value>::value)
            return LLVMArray<Value>(sign_mask_neg<Value>()) & a;
        else
            return select(a > 0, a, -a);
    }

    friend LLVMArray sqrt(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_trace_append_1(Type,
            "$r0 = call <$w x $t0> @llvm.sqrt.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray round(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_trace_append_1(Type,
            "$r0 = call <$w x $t0> @llvm.nearbyint.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray floor(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_trace_append_1(Type,
            "$r0 = call <$w x $t0> @llvm.floor.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray ceil(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_trace_append_1(Type,
            "$r0 = call <$w x $t0> @llvm.ceil.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray trunc(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_trace_append_1(Type,
            "$r0 = call <$w x $t0> @llvm.trunc.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray fmadd(const LLVMArray &a, const LLVMArray &b,
                           const LLVMArray &c) {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        if (std::is_floating_point<Value>::value) {
            return LLVMArray::from_index(jitc_trace_append_3(
                Type,
                "$r0 = call <$w x $t0> @llvm.fma.v$w$a1(<$w x $t1> $r1, "
                "<$w x $t2> $r2, <$w x $t3> $r3)",
                1, a.index(), b.index(), c.index()));
        } else {
            return a*b + c;
        }
    }

    friend LLVMArray fmsub(const LLVMArray &a, const LLVMArray &b,
                           const LLVMArray &c) {
        return fmadd(a, b, -c);
    }

    friend LLVMArray fnmadd(const LLVMArray &a, const LLVMArray &b,
                            const LLVMArray &c) {
        return fmadd(-a, b, c);
    }

    friend LLVMArray fnmsub(const LLVMArray &a, const LLVMArray &b,
                            const LLVMArray &c) {
        return fmadd(-a, b, -c);
    }

    LLVMArray& eval() {
        jitc_var_eval(m_index);
        return *this;
    }

    const LLVMArray& eval() const {
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

    static LLVMArray map(void *ptr, size_t size, bool free = false) {
        return from_index(
            jitc_var_map(Type, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static LLVMArray copy(const void *ptr, size_t size) {
        return from_index(jitc_var_copy(Type, ptr, (uint32_t) size));
    }

    static LLVMArray from_index(uint32_t index) {
        LLVMArray result;
        result.m_index = index;
        return result;
    }

    static LLVMArray launch_index(size_t size) {
        return from_index(jitc_trace_append_0(
            Type,
            "$r0_0 = trunc i64 $i to $t0$n"
            "$r0_1 = insertelement <$w x $t0> undef, $t0 $r0_0, i32 0$n"
            "$r0_2 = shufflevector <$w x $t0> $r0_1, <$w x $t0> undef, "
                "<$w x i32> zeroinitializer$n"
            "$r0 = add <$w x $t0> $r0_2, $l0", 1, (uint32_t) size));
    }

protected:
    uint32_t m_index = 0;
};

template <typename Value>
void set_label(const LLVMArray<Value> &a, const char *label) {
    jitc_var_set_label(a.index(), label);
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array empty(size_t size) {
    size_t byte_size = size * sizeof(typename Array::Value);
    void *ptr = jitc_malloc(AllocType::Host, byte_size);
    return Array::from_index(jitc_var_map(Array::Type, ptr, (uint32_t) size, 1));
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array zero(size_t size) {
    if (size == 1) {
        return Array(0);
    } else {
        uint8_t value = 0;
        size_t byte_size = size * sizeof(typename Array::Value);
        void *ptr = jitc_malloc(AllocType::Host, byte_size);
        jitc_fill(VarType::UInt8, ptr, byte_size, &value);
        return Array::from_index(jitc_var_map(Array::Type, ptr, (uint32_t) size, 1));
    }
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array full(typename Array::Value value, size_t size) {
    if (size == 1) {
        return Array(value);
    } else {
        size_t byte_size = size * sizeof(typename Array::Value);
        void *ptr = jitc_malloc(AllocType::Host, byte_size);
        jitc_fill(Array::Type, ptr, size, &value);
        return Array::from_index(jitc_var_map(Array::Type, ptr, (uint32_t) size, 1));
    }
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array arange(ssize_t start, ssize_t stop, ssize_t step) {
    using UInt32 = LLVMArray<uint32_t>;
    using Value = typename Array::Value;

    size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
    UInt32 index = UInt32::launch_index(size);

    if (start == 0 && step == 1)
        return Array(index);
    else
        return fmadd(Array(index), Array((Value) step), Array((Value) start));
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array arange(size_t size) {
    return arange<Array>(0, (size_t) size, 1);
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array linspace(typename Array::Value min, typename Array::Value max, size_t size) {
    using UInt32 = LLVMArray<uint32_t>;
    using Value = typename Array::Value;

    using UInt32 = LLVMArray<uint32_t>;
    UInt32 index = UInt32::launch_index(size);

    Value step = (max - min) / Value(size - 1);
    return fmadd(Array(index), Array(step), Array(min));
}

template <typename Value>
LLVMArray<Value> select(const LLVMArray<bool> &m,
                        const LLVMArray<Value> &t,
                        const LLVMArray<Value> &f) {
    // Simple constant propagation for masks
    if (m.is_all_true()) {
        return t;
    } else if (m.is_all_false()) {
        return f;
    } else {
        return LLVMArray<Value>::from_index(jitc_trace_append_3(
            LLVMArray<Value>::Type,
            "$r0 = select <$w x $t1> $r1, <$w x $t2> $r2, <$w x $t3> $r3", 1,
            m.index(), t.index(), f.index()));
    }
}

template <typename OutArray, typename ValueIn>
OutArray reinterpret_array(const LLVMArray<ValueIn> &input) {
    using ValueOut = typename OutArray::Value;

    static_assert(
        sizeof(ValueIn) == sizeof(ValueOut),
        "reinterpret_array requires arrays with equal-sized element types!");

    if (std::is_integral<ValueIn>::value != std::is_integral<ValueOut>::value) {
        return OutArray::from_index(jitc_trace_append_1(
            OutArray::Type, "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>", 1,
            input.index()));
    } else {
        jitc_var_inc_ref_ext(input.index());
        return OutArray::from_index(input.index());
    }
}

template <typename Value>
LLVMArray<Value> min(const LLVMArray<Value> &a, const LLVMArray<Value> &b) {
    return LLVMArray<Value>::from_index(jitc_trace_append_2(
        LLVMArray<Value>::Type,
        "$r0 = call <$w x $t0> @llvm.minnum.v$w$a1(<$w x $t1> $r1, <$w x $t2> $r2)", 1,
        a.index(), b.index()));
}

template <typename Value>
LLVMArray<Value> max(const LLVMArray<Value> &a, const LLVMArray<Value> &b) {
    return LLVMArray<Value>::from_index(jitc_trace_append_2(
        LLVMArray<Value>::Type,
        "$r0 = call <$w x $t0> @llvm.maxnum.v$w$a1(<$w x $t1> $r1, <$w x $t2> $r2)", 1,
        a.index(), b.index()));
}

template <typename OutArray, typename Index,
          typename std::enable_if<OutArray::IsLLVM, int>::type = 0>
OutArray gather(const void *ptr, const LLVMArray<Index> &index,
                const LLVMArray<bool> &mask = true) {
    using UInt64 = LLVMArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr));
    using Value = typename OutArray::Value;
    constexpr size_t Size = sizeof(Value);

    uint32_t var;
    if (mask.is_all_false()) {
        return OutArray((Value) 0);
    } else if (Size != 1) {
        OutArray addr = OutArray::from_index(jitc_trace_append_2(
            OutArray::Type,
            "$r0_0 = inttoptr $t1 $r1 to $t0*$n"
            "$r0 = getelementptr $t0, $t0* $r0_0, <$w x $t2> $r2",
            1, base.index(), index.index()
        ));

        if (mask.is_all_true())
            var = jitc_trace_append_1(
                OutArray::Type,
                "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                "(<$w x $t0*> $r1, i32 $s0, <$w x i1> $O, <$w x $t0> $z0)",
                1, addr.index());
        else
            var = jitc_trace_append_2(
                OutArray::Type,
                "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                "(<$w x $t0*> $r1, i32 $s0, <$w x $t2> $r2, <$w x $t0> $z0)",
                1, addr.index(), mask.index());
    } else {
        using UInt32 = LLVMArray<uint32_t>;

        UInt32 addr = UInt32::from_index(jitc_trace_append_2(
            UInt32::Type,
            "$r0_0 = inttoptr $t1 $r1 to i1*$n"
            "$r0_1 = getelementptr i1, i1* $r0_0, <$w x $t2> $r2$n"
            "$r0   = bitcast <$w x i1*> $r0_1 to <$w x i32*>",
            1, base.index(), index.index()
        ));

        if (mask.is_all_true())
            var = jitc_trace_append_1(
                OutArray::Type,
                "$r0_0 = call <$w x i32> @llvm.masked.gather.v$wi32"
                "(<$w x i32*> $r1, i32 $s0, <$w x i1> $O, <$w x i32> $z1)$n"
                "$r0 = trunc <$w x i32> $r0_0 to <$w x $t0>",
                1, addr.index());
        else
            var = jitc_trace_append_2(
                OutArray::Type,
                "$r0_0 = call <$w x i32> @llvm.masked.gather.v$wi32"
                "(<$w x i32*> $r1, i32 $s0, <$w x $t2> $r2, <$w x i32> $z1)$n"
                "$r0 = trunc <$w x i32> $r0_0 to <$w x $t0>",
                1, addr.index(), mask.index());
    }

    return OutArray::from_index(var);
}

template <typename Value, typename Index>
LLVMArray<void_t> scatter(void *ptr,
                          const LLVMArray<Value> &value,
                          const LLVMArray<Index> &index,
                          const LLVMArray<bool> &mask = true) {
    using UInt64 = LLVMArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr));

    LLVMArray<Value> addr = LLVMArray<Value>::from_index(jitc_trace_append_2(
        LLVMArray<Value>::Type,
        "$r0_0 = inttoptr $t1 $r1 to $t0*$n"
        "$r0 = getelementptr $t0, $t0* $r0_0, <$w x $t2> $r2",
        1, base.index(), index.index()));

    uint32_t var;
    if (mask.is_all_false()) {
        return LLVMArray<void_t>();
    } else if (mask.is_all_true()) {
        var = jitc_trace_append_2(
            VarType::Invalid,
            "call void @llvm.masked.scatter.v$w$a1"
            "(<$w x $t1> $r1, <$w x $t1*> $r2, i32 $s1, <$w x i1> $O)",
            1, value.index(), addr.index());
    } else {
        var = jitc_trace_append_3(
            VarType::Invalid,
            "call void @llvm.masked.scatter.v$w$a1"
            "(<$w x $t1> $r1, <$w x $t1*> $r2, i32 $s1, <$w x $t3> $r3)",
            1, value.index(), addr.index(), mask.index());
    }

    jitc_var_mark_side_effect(var);
    jitc_var_inc_ref_ext(var);

    return LLVMArray<void_t>::from_index(var);
}

template <typename Array, typename Index,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array gather(const Array &src, const LLVMArray<Index> &index,
             const LLVMArray<bool> &mask = true) {
    if (mask.is_all_false())
        return Array(typename Array::Value(0));

    jitc_var_eval(src.index());
    Array result = gather<Array>(src.data(), index, mask);
    jitc_var_set_extra_dep(result.index(), src.index());
    return result;
}

template <typename Value, typename Index>
LLVMArray<void_t> scatter(LLVMArray<Value> &dst,
                          const LLVMArray<Value> &value,
                          const LLVMArray<Index> &index,
                          const LLVMArray<bool> &mask = true) {
    if (mask.is_all_false())
        return LLVMArray<void_t>();

    if (dst.data() == nullptr)
        jitc_var_eval(dst.index());

    LLVMArray<void_t> result = scatter(dst.data(), value, index, mask);
    jitc_var_set_extra_dep(result.index(), dst.index());
    jitc_var_mark_dirty(dst.index());
    return result;
}

template <typename Value>
LLVMArray<Value> copysign(const LLVMArray<Value> &v1,
                          const LLVMArray<Value> &v2) {
    if (!jitc_is_floating_point(LLVMArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return abs(v1) | (LLVMArray<Value>(sign_mask<Value>()) & v2);
}

template <typename Value>
LLVMArray<Value> copysign_neg(const LLVMArray<Value> &v1,
                              const LLVMArray<Value> &v2) {
    if (!jitc_is_floating_point(LLVMArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return abs(v1) | (LLVMArray<Value>(sign_mask<Value>()) & ~v2);
}

template <typename Value>
LLVMArray<Value> mulsign(const LLVMArray<Value> &v1,
                         const LLVMArray<Value> &v2) {
    if (!jitc_is_floating_point(LLVMArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return v1 ^ (LLVMArray<Value>(sign_mask<Value>()) & v2);
}

template <typename Value>
LLVMArray<Value> mulsign_neg(const LLVMArray<Value> &v1,
                             const LLVMArray<Value> &v2) {
    if (!jitc_is_floating_point(LLVMArray<Value>::Type))
        jitc_raise("Unsupported operand type");
    return v1 ^ (LLVMArray<Value>(sign_mask<Value>()) & ~v2);
}

inline bool all(const LLVMArray<bool> &v) {
    if (v.is_all_true()) {
        return true;
    } else if (v.is_all_false()) {
        return false;
    } else {
        v.eval();
        return (bool) jitc_all((uint8_t *) v.data(), (uint32_t) v.size());
    }
}

inline bool any(const LLVMArray<bool> &v) {
    if (v.is_all_true()) {
        return true;
    } else if (v.is_all_false()) {
        return false;
    } else {
        v.eval();
        return (bool) jitc_any((uint8_t *) v.data(), (uint32_t) v.size());
    }
}

inline bool none(const LLVMArray<bool> &v) {
    return !any(v);
}

template <typename Value> LLVMArray<Value> hsum(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Add, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hprod(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Mul, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hmax(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Max, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hmin(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Min, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}
