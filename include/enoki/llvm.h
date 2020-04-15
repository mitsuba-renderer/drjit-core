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
    static constexpr bool IsCUDA = false;

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
                         : (std::is_signed<T>::value
                                ? "$r0 = sext <$w x $t1> $r1 to <$w x $t0>"
                                : "$r0 = zext <$w x $t1> $r1 to <$w x $t0>");
            }
        } else {
            jitc_fail("Unsupported conversion!");
        }

        m_index = jitc_var_new_1(Type, op, 1, v.index());
    }

    LLVMArray(Value value) {
        m_index = mkfull_(value, 1);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    LLVMArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jitc_var_copy(AllocType::Host, Type, data,
                                (uint32_t) sizeof...(Args));
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

        // Simple constant propagation
        if (is_literal_zero())
            return v;
        else if (v.is_literal_zero())
            return *this;

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fadd <$w x $t0> $r1, $r2"
            : "$r0 = add <$w x $t0> $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator-(const LLVMArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fsub <$w x $t0> $r1, $r2"
            : "$r0 = sub <$w x $t0> $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator*(const LLVMArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        // Simple constant propagation
        if (is_literal_one())
            return v;
        else if (v.is_literal_one())
            return *this;

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fmul <$w x $t0> $r1, $r2"
            : "$r0 = mul <$w x $t0> $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator/(const LLVMArray &v) const {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        // Simple constant propagation
        if (v.is_literal_one())
            return *this;

        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fdiv <$w x $t0> $r1, $r2"
            : "$r0 = div <$w x $t0> $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator/(Value value) const {
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

        return operator/(LLVMArray(value));
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

        return LLVMArray<bool>::from_index(jitc_var_new_2(
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

        return LLVMArray<bool>::from_index(jitc_var_new_2(
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

        return LLVMArray<bool>::from_index(jitc_var_new_2(
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

        return LLVMArray<bool>::from_index(jitc_var_new_2(
            LLVMArray<bool>::Type, op, 1, m_index, a.index()));
    }

    friend LLVMArray<bool> eq(const LLVMArray &a, const LLVMArray &b) {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = icmp eq <$w x $t1> $r1, $r2"
                             : "$r0 = fcmp oeq <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_var_new_2(
            LLVMArray<bool>::Type, op, 1, a.index(), b.index()));
    }

    friend LLVMArray<bool> neq(const LLVMArray &a, const LLVMArray &b) {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = icmp ne <$w x $t1> $r1, $r2"
                             : "$r0 = fcmp one <$w x $t1> $r1, $r2";

        return LLVMArray<bool>::from_index(jitc_var_new_2(
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

        const char *op;
        if (std::is_floating_point<Value>::value) {
            if (jitc_llvm_version_major() > 7)
                op = "$r0 = fneg <$w x $t0> $r1";
            else
                op = "$r0 = fsub <$w x $t0> zeroinitializer, $r1";
        } else {
            op = "$r0 = sub <$w x $t0> $z, $r1";
        }

        return from_index(jitc_var_new_1(Type, op, 1, m_index));
    }

    LLVMArray operator~() const {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = xor <$w x $t1> $r1, $o0"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = xor <$w x $b0> $r0_0, $o0$n"
                               "$r0 = bitcast <$w x $b0> $r0_1 to <$w x $t0>";

        return from_index(jitc_var_new_1(Type, op, 1, m_index));
    }

    LLVMArray operator|(const LLVMArray &a) const {
        // Simple constant propagation
        if (std::is_same<Value, bool>::value) {
            if (is_literal_one() || a.is_literal_zero())
                return *this;
            else if (a.is_literal_one() || is_literal_zero())
                return a;
        }

        const char *op = std::is_integral<Value>::value
                             ? "$r0 = or <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = or <$w x $b0> $r0_0, $r0_1"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(jitc_var_new_2(Type, op, 1, m_index, a.index()));
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    LLVMArray operator|(const LLVMArray<bool> &m) const {
        // Simple constant propagation
        if (m.is_literal_zero())
            return *this;
        else if (m.is_literal_one())
            return LLVMArray(memcpy_cast<Value>(uint_with_size_t<Value>(-1)));

        using UInt = LLVMArray<uint_with_size_t<Value>>;
        UInt x = UInt::from_index(jitc_var_new_1(
            UInt::Type, "$r0 = sext <$w x $t1> $r1 to <$w x $b0>", 1,
            m.index()));

        return *this | reinterpret_array<LLVMArray>(x);
    }

    LLVMArray operator&(const LLVMArray &a) const {
        // Simple constant propagation
        if (std::is_same<Value, bool>::value) {
            if (is_literal_one() || a.is_literal_zero())
                return a;
            else if (a.is_literal_one() || is_literal_zero())
                return *this;
        }

        const char *op = std::is_integral<Value>::value
                             ? "$r0 = and <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = and <$w x $b0> $r0_0, $r0_1"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(jitc_var_new_2(Type, op, 1, m_index, a.index()));
    }

    template <typename T = Value, enable_if_t<!std::is_same<T, bool>::value> = 0>
    LLVMArray operator&(const LLVMArray<bool> &m) const {
        // Simple constant propagation
        if (m.is_literal_one())
            return *this;
        else if (m.is_literal_zero())
            return LLVMArray(Value(0));

        using UInt = LLVMArray<uint_with_size_t<Value>>;
        UInt x = UInt::from_index(jitc_var_new_1(
            UInt::Type, "$r0 = sext <$w x $t1> $r1 to <$w x $b0>", 1,
            m.index()));

        return *this & reinterpret_array<LLVMArray>(x);
    }

    LLVMArray operator^(const LLVMArray &a) const {
        // Simple constant propagation
        if (std::is_same<Value, bool>::value) {
            if (is_literal_zero())
                return a;
            else if (a.is_literal_zero())
                return *this;
        }

        const char *op = std::is_integral<Value>::value
                             ? "$r0 = xor <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = xor <$w x $b0> $r0_0, $r0_1$n"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(jitc_var_new_2(Type, op, 1, m_index, a.index()));
    }

    LLVMArray operator<<(const LLVMArray<uint32_t> &v) const {
        return LLVMArray::from_index(jitc_var_new_2(
            Type, "$r0 = shl <$w x $t0> $r1, $r2", 1, index(), v.index()));
    }

    LLVMArray operator>>(const LLVMArray<uint32_t> &v) const {
        if (std::is_integral<Value>::value && std::is_signed<Value>::value)
            return LLVMArray::from_index(jitc_var_new_2(
                Type, "$r0 = ashr <$w x $t0> $r1, $r2", 1, index(), v.index()));
        else
            return LLVMArray::from_index(jitc_var_new_2(
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

    LLVMArray& operator/=(Value v) {
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

        return LLVMArray::from_index(jitc_var_new_1(Type,
            "$r0 = call <$w x $t0> @llvm.sqrt.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray round(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_var_new_1(Type,
            "$r0 = call <$w x $t0> @llvm.nearbyint.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray floor(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_var_new_1(Type,
            "$r0 = call <$w x $t0> @llvm.floor.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray ceil(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_var_new_1(Type,
            "$r0 = call <$w x $t0> @llvm.ceil.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray trunc(const LLVMArray &a) {
        if (!jitc_is_floating_point(Type))
            jitc_raise("Unsupported operand type");

        return LLVMArray::from_index(jitc_var_new_1(Type,
            "$r0 = call <$w x $t0> @llvm.trunc.v$w$a1(<$w x $t1> $r1)", 1,
            a.index()));
    }

    friend LLVMArray fmadd(const LLVMArray &a,
                           const LLVMArray &b,
                           const LLVMArray &c) {
        if (!jitc_is_arithmetic(Type))
            jitc_raise("Unsupported operand type");

        // Simple constant propagation
        if (a.is_literal_one())
            return b + c;
        else if (b.is_literal_one())
            return a + c;
        else if (a.is_literal_zero() || b.is_literal_zero())
            return c;
        else if (c.is_literal_zero())
            return a * b;

        if (std::is_floating_point<Value>::value) {
            return LLVMArray::from_index(jitc_var_new_3(
                Type,
                "$r0 = call <$w x $t0> @llvm.fma.v$w$a1(<$w x $t1> $r1, "
                "<$w x $t2> $r2, <$w x $t3> $r3)",
                1, a.index(), b.index(), c.index()));
        } else {
            return a*b + c;
        }
    }

    LLVMArray& schedule() {
        jitc_var_schedule(m_index);
        return *this;
    }

    const LLVMArray& schedule() const {
        jitc_var_schedule(m_index);
        return *this;
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

    bool is_literal_one() const { return (bool) jitc_var_is_literal_one(m_index); }
    bool is_literal_zero() const { return (bool) jitc_var_is_literal_zero(m_index); }

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
        if (jitc_var_int_ref(m_index) > 0) {
            eval();
            *this = LLVMArray::from_index(
                jitc_var_copy(AllocType::HostAsync, LLVMArray<Value>::Type, data(),
                              (uint32_t) size()));
        }

        jitc_var_write(m_index, offset, &value);
    }

    static LLVMArray map(void *ptr, size_t size, bool free = false) {
        return from_index(
            jitc_var_map(Type, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static LLVMArray copy(const void *ptr, size_t size) {
        return from_index(
            jitc_var_copy(AllocType::Host, Type, ptr, (uint32_t) size));
    }

    static LLVMArray from_index(uint32_t index) {
        LLVMArray result;
        result.m_index = index;
        return result;
    }

    static LLVMArray launch_index(size_t size) {
        return from_index(jitc_var_new_0(
            Type,
            "$r0_0 = trunc i64 $i to $t0$n"
            "$r0_1 = insertelement <$w x $t0> undef, $t0 $r0_0, i32 0$n"
            "$r0_2 = shufflevector <$w x $t0> $r0_1, <$w x $t0> undef, "
                "<$w x i32> $z$n"
            "$r0 = add <$w x $t0> $r0_2, $l0", 1, (uint32_t) size));
    }

    static uint32_t mkfull_(Value value, uint32_t size) {
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
            "$r0 = shufflevector <$w x $t0> $r0_0, <$w x $t0> undef, <$w x i32> $z" :
            "$r0_0 = insertelement <$w x $t0> undef, $t0 %llu, i32 0$n"
            "$r0 = shufflevector <$w x $t0> $r0_0, <$w x $t0> undef, <$w x i32> $z",
            value_ull);

        return jitc_var_new_0(Type, value_str, 0, size);
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
    void *ptr = jitc_malloc(AllocType::HostAsync, byte_size);
    return Array::from_index(jitc_var_map(Array::Type, ptr, (uint32_t) size, 1));
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array zero(size_t size) {
    return Array::from_index(Array::mkfull_(typename Array::Value(0), (uint32_t) size));
}

template <typename Array,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array full(typename Array::Value value, size_t size) {
    return Array::from_index(Array::mkfull_(value, (uint32_t) size));
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
    // Simple constant propagation
    if (m.is_literal_one()) {
        return t;
    } else if (m.is_literal_zero()) {
        return f;
    } else {
        return LLVMArray<Value>::from_index(jitc_var_new_3(
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
        return OutArray::from_index(jitc_var_new_1(
            OutArray::Type, "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>", 1,
            input.index()));
    } else {
        jitc_var_inc_ref_ext(input.index());
        return OutArray::from_index(input.index());
    }
}

template <typename Value>
LLVMArray<Value> min(const LLVMArray<Value> &a, const LLVMArray<Value> &b) {
    // Portable intrinsic as a last resort
    const char *op = "$r0 = call <$w x $t0> @llvm.minnum.v$w$a1(<$w x $t1> "
                     "$r1, <$w x $t2> $r2)";

    // Prefer an X86-specific intrinsic (produces nicer machine code)
    if (std::is_same<Value, float>::value) {
        if (jitc_llvm_if_at_least(16, "+avx512f")) {
            op = "$2$r0 = call <$w x $t0> @llvm.x86.avx512.min.ps.512(<$w x $t1> "
                 "$r1, <$w x $t2> $r2, i32 4)";
        } else if (jitc_llvm_if_at_least(8, "+avx")) {
            op = "$3$r0 = call <$w x $t0> @llvm.x86.avx.min.ps.256(<$w x $t1> "
                 "$r1, <$w x $t2> $r2)";
        } else if (jitc_llvm_if_at_least(4, "+sse4.2")) {
            op = "$4$r0 = call <$w x $t0> @llvm.x86.sse.min.ps(<$w x $t1> $r1, "
                 "<$w x $t2> $r2)";
        }
    } else if (std::is_same<Value, double>::value) {
        if (jitc_llvm_if_at_least(8, "+avx512f")) {
            op = "$1$r0 = call <$w x $t0> @llvm.x86.avx512.min.pd.512(<$w x $t1> "
                 "$r1, <$w x $t2> $r2, i32 4)";
        } else if (jitc_llvm_if_at_least(4, "+avx")) {
            op = "$2$r0 = call <$w x $t0> @llvm.x86.avx.min.pd.256(<$w x $t1> "
                 "$r1, <$w x $t2> $r2)";
        } else if (jitc_llvm_if_at_least(2, "+sse4.2")) {
            op = "$3$r0 = call <$w x $t0> @llvm.x86.sse.min.pd(<$w x $t1> $r1, "
                 "<$w x $t2> $r2)";
        }
    }

    return LLVMArray<Value>::from_index(jitc_var_new_2(
        LLVMArray<Value>::Type, op, 1, a.index(), b.index()));
}

template <typename Value>
LLVMArray<Value> max(const LLVMArray<Value> &a, const LLVMArray<Value> &b) {
    // Portable intrinsic as a last resort
    const char *op = "$r0 = call <$w x $t0> @llvm.maxnum.v$w$a1(<$w x $t1> "
                     "$r1, <$w x $t2> $r2)";

    // Prefer an X86-specific intrinsic (produces nicer machine code)
    if (std::is_same<Value, float>::value) {
        if (jitc_llvm_if_at_least(16, "+avx512f")) {
            op = "$4$r0 = call <$w x $t0> @llvm.x86.avx512.max.ps.512(<$w x $t1> "
                 "$r1, <$w x $t2> $r2, i32$S 4)";
        } else if (jitc_llvm_if_at_least(8, "+avx")) {
            op = "$3$r0 = call <$w x $t0> @llvm.x86.avx.max.ps.256(<$w x $t1> "
                 "$r1, <$w x $t2> $r2)";
        } else if (jitc_llvm_if_at_least(4, "+sse4.2")) {
            op = "$2$r0 = call <$w x $t0> @llvm.x86.sse.max.ps(<$w x $t1> $r1, "
                 "<$w x $t2> $r2)";
        }
    } else if (std::is_same<Value, double>::value) {
        if (jitc_llvm_if_at_least(8, "+avx512f")) {
            op = "$3$r0 = call <$w x $t0> @llvm.x86.avx512.max.pd.512(<$w x $t1> "
                 "$r1, <$w x $t2> $r2, i32$S 4)";
        } else if (jitc_llvm_if_at_least(4, "+avx")) {
            op = "$2$r0 = call <$w x $t0> @llvm.x86.avx.max.pd.256(<$w x $t1> "
                 "$r1, <$w x $t2> $r2)";
        } else if (jitc_llvm_if_at_least(2, "+sse4.2")) {
            op = "$1$r0 = call <$w x $t0> @llvm.x86.sse.max.pd(<$w x $t1> $r1, "
                 "<$w x $t2> $r2)";
        }
    }

    return LLVMArray<Value>::from_index(jitc_var_new_2(
        LLVMArray<Value>::Type, op, 1, a.index(), b.index()));
}

template <typename OutArray, typename ValueIn>
OutArray jitc_llvm_f2i_cast(const LLVMArray<ValueIn> &a, int mode) {
    using ValueOut = typename OutArray::Value;
    constexpr bool Signed = std::is_signed<ValueOut>::value;
    constexpr size_t SizeIn = sizeof(ValueIn), SizeOut = sizeof(ValueOut);

    if (!jitc_is_floating_point(LLVMArray<ValueIn>::Type) ||
        !jitc_is_integral(LLVMArray<ValueOut>::Type))
        jitc_raise("Unsupported operand type");

    if (!((SizeIn == 4 && SizeOut == 4 &&
           jitc_llvm_if_at_least(16, "+avx512f")) ||
          ((SizeIn == 4 || SizeIn == 8) && (SizeOut == 4 || SizeOut == 8) &&
           jitc_llvm_if_at_least(8, "+avx512vl"))))
        return 0u;

    const char *in_t = SizeIn == 4 ? "ps" : "pd";
    const char *out_t =
        SizeOut == 4 ? (Signed ? "dq" : "udq") : (Signed ? "qq" : "uqq");

    char op[128];
    snprintf(op, sizeof(op),
             "$%i$r0 = call <$w x $t0> @llvm.x86.avx512.mask.cvt%s2%s.512(<$w "
             "x $t1> $r1, <$w x $t0> $z, i$w$S -1, i32$S %i)",
             (SizeIn == 4 && SizeOut == 4) ? 4 : 3, in_t, out_t, mode);

    return OutArray::from_index(jitc_var_new_1(OutArray::Type, op, 0, a.index()));
}

template <typename OutArray, typename ValueIn>
OutArray round2int(const LLVMArray<ValueIn> &a) {
    OutArray out = jitc_llvm_f2i_cast<OutArray>(a, 8);
    if (!out.valid())
        out = OutArray(round(a));
    return out;
}

template <typename OutArray, typename ValueIn>
OutArray floor2int(const LLVMArray<ValueIn> &a) {
    OutArray out = jitc_llvm_f2i_cast<OutArray>(a, 9);
    if (!out.valid())
        out = OutArray(floor(a));
    return out;
}

template <typename OutArray, typename ValueIn>
OutArray ceil2int(const LLVMArray<ValueIn> &a) {
    OutArray out = jitc_llvm_f2i_cast<OutArray>(a, 10);
    if (!out.valid())
        out = OutArray(ceil(a));
    return out;
}

template <typename OutArray, typename ValueIn>
OutArray trunc2int(const LLVMArray<ValueIn> &a) {
    OutArray out = jitc_llvm_f2i_cast<OutArray>(a, 11);
    if (!out.valid())
        out = OutArray(trunc(a));
    return out;
}

namespace detail {
template <typename Array, typename Index,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array gather_impl(const void *src_ptr,
                  uint32_t src_index,
                  const LLVMArray<Index> &index,
                  const LLVMArray<bool> &mask = true) {
    using Value = typename Array::Value;

    LLVMArray<void *> base = LLVMArray<void *>::from_index(
        jitc_var_copy_ptr(src_ptr, src_index));

    uint32_t var;
    if (sizeof(Value) != 1) {
        if (mask.is_literal_one())
            var = jitc_var_new_2(
                Array::Type,
                "$r0_0 = bitcast $t1 $r1 to $t0*$n"
                "$r0_1 = getelementptr $t0, $t0* $r0_0, <$w x $t2> $r2$n"
                "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                "(<$w x $t0*> $r0$S_1, i32 $s0, <$w x i1> $O, <$w x $t0> $z)",
                1, base.index(), index.index());
        else
            var = jitc_var_new_3(
                Array::Type,
                "$r0_0 = bitcast $t1 $r1 to $t0*$n"
                "$r0_1 = getelementptr $t0, $t0* $r0_0, <$w x $t2> $r2$n"
                "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                "(<$w x $t0*> $r0$S_1, i32 $s0, <$w x $t3> $r3, <$w x $t0> $z)",
                1, base.index(), index.index(), mask.index());
    } else {
        if (mask.is_literal_one())
            var = jitc_var_new_2(
                Array::Type,
                "$r0_0 = bitcast $t1 $r1 to i8*$n"
                "$r0_1 = getelementptr i8, i8* $r0_0, <$w x $t2> $r2$n"
                "$r0_2 = bitcast <$w x i8*> $r0_1 to <$w x i32*>$n"
                "$r0_3 = call <$w x i32> @llvm.masked.gather.v$wi32"
                "(<$w x i32*> $r0$S_2, i32 $s0, <$w x i1> $O, <$w x i32> $z)$n"
                "$r0 = trunc <$w x i32> $r0_3 to <$w x $t0>",
                1, base.index(), index.index());
        else
            var = jitc_var_new_3(
                Array::Type,
                "$r0_0 = bitcast $t1 $r1 to i8*$n"
                "$r0_1 = getelementptr i8, i8* $r0_0, <$w x $t2> $r2$n"
                "$r0_2 = bitcast <$w x i8*> $r0_1 to <$w x i32*>$n"
                "$r0_3 = call <$w x i32> @llvm.masked.gather.v$wi32"
                "(<$w x i32*> $r0$S_2, i32 $s0, <$w x $t3> $r3, <$w x i32> $z)$n"
                "$r0 = trunc <$w x i32> $r0_3 to <$w x $t0>",
                1, base.index(), index.index(), mask.index());
    }

    return Array::from_index(var);
}
}

template <typename Array, typename Index,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array gather(const void *src, const LLVMArray<Index> &index,
             const LLVMArray<bool> &mask = true) {
    if (mask.is_literal_zero())
        return Array(typename Array::Value(0));
    return detail::gather_impl<Array>(src, 0, index, mask);
}

template <typename Array, typename Index,
          typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array gather(const Array &src, const LLVMArray<Index> &index,
             const LLVMArray<bool> &mask = true) {
    if (mask.is_literal_zero())
        return Array(typename Array::Value(0));
    src.eval();
    return detail::gather_impl<Array>(src.data(), src.index(), index, mask);
}

template <typename Value, typename Index>
void scatter(LLVMArray<Value> &dst,
             const LLVMArray<Value> &value,
             const LLVMArray<Index> &index,
             const LLVMArray<bool> &mask = true) {
    if (mask.is_literal_zero())
        return;

    void *ptr = dst.data();

    if (!ptr) {
        dst.eval();
        ptr = dst.data();
    }

    if (jitc_var_int_ref(dst.index()) > 0) {
        dst = LLVMArray<Value>::from_index(jitc_var_copy(
            AllocType::HostAsync, LLVMArray<Value>::Type, ptr, (uint32_t) dst.size()));
        ptr = dst.data();
    }

    LLVMArray<void *> base = LLVMArray<void *>::from_index(
        jitc_var_copy_ptr(ptr, dst.index()));

    uint32_t var;
    if (mask.is_literal_one()) {
        var = jitc_var_new_3(
            VarType::Invalid,
            "$r0_0 = bitcast $t1 $r1 to $t2*$n"
            "$r0_1 = getelementptr $t2, $t2* $r0_0, <$w x $t3> $r3$n"
            "call void @llvm.masked.scatter.v$w$a2"
            "(<$w x $t2> $r2, <$w x $t2*> $r0$S_1, i32 $s1, <$w x i1> $O)",
            1, base.index(), value.index(), index.index());
    } else {
        var = jitc_var_new_4(
            VarType::Invalid,
            "$r0_0 = bitcast $t1 $r1 to $t2*$n"
            "$r0_1 = getelementptr $t2, $t2* $r0_0, <$w x $t3> $r3$n"
            "call void @llvm.masked.scatter.v$w$a2"
            "(<$w x $t2> $r2, <$w x $t2*> $r0$S_1, i32 $s1, <$w x $t4> $r4)",
            1, base.index(), value.index(), index.index(), mask.index());
    }

    jitc_var_mark_scatter(var, dst.index());
}

template <typename Value, typename Index>
void scatter_add(LLVMArray<Value> &dst,
                 const LLVMArray<Value> &value,
                 const LLVMArray<Index> &index,
                 const LLVMArray<bool> &mask = true) {
    if (mask.is_literal_zero()) {
        return;
    } else if (sizeof(Index) != sizeof(Value)) {
        using UIntSame = LLVMArray<uint_with_size_t<Value>>;
        return scatter_add(dst, value, UIntSame(index), mask);
    }

    void *ptr = dst.data();

    if (!ptr) {
        dst.eval();
        ptr = dst.data();
    }

    if (jitc_var_int_ref(dst.index()) > 0) {
        dst = LLVMArray<Value>::from_index(jitc_var_copy(
            AllocType::HostAsync, LLVMArray<Value>::Type, ptr, (uint32_t) dst.size()));
        ptr = dst.data();
    }

    LLVMArray<void *> base = LLVMArray<void *>::from_index(
        jitc_var_copy_ptr(ptr, dst.index()));

    uint32_t var;
    if (mask.is_literal_one()) {
        const char *op;
        if (sizeof(Value) == 4 && jitc_llvm_if_at_least(16, "+avx512dq") != 0)
            op = "$4call void @ek.scatter_add_v$w$a2($t1 $r1, <$w x $t2> $r2, <$w x $t3> $r3)";
        else if (sizeof(Value) == 8 && jitc_llvm_if_at_least(8, "+avx512dq") != 0)
            op = "$3call void @ek.scatter_add_v$w$a2($t1 $r1, <$w x $t2> $r2, <$w x $t3> $r3)";
        else
            op = "$0call void @ek.scatter_add_v$w$a2($t1 $r1, <$w x $t2> $r2, <$w x $t3> $r3)";

        var = jitc_var_new_3(VarType::Invalid, op, 1, base.index(),
                             value.index(), index.index());
    } else {
        const char *op;
        if (sizeof(Value) == 4 && jitc_llvm_if_at_least(16, "+avx512dq"))
            op = "$4call void @ek.masked_scatter_add_v$w$a2($t1 $r1, <$w x $t2> $r2, <$w x $t3> $r3, <$w x $t4> $r4)";
        else if (sizeof(Value) == 8 && jitc_llvm_if_at_least(8, "+avx512dq"))
            op = "$3call void @ek.masked_scatter_add_v$w$a2($t1 $r1, <$w x $t2> $r2, <$w x $t3> $r3, <$w x $t4> $r4)";
        else
            op = "$0call void @ek.masked_scatter_add_v$w$a2($t1 $r1, <$w x $t2> $r2, <$w x $t3> $r3, <$w x $t4> $r4)";

        var = jitc_var_new_4(VarType::Invalid, op, 1, base.index(),
                             value.index(), index.index(), mask.index());
    }

    jitc_var_mark_scatter(var, dst.index());
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
    if (v.is_literal_one()) {
        return true;
    } else if (v.is_literal_zero()) {
        return false;
    } else {
        v.eval();
        return (bool) jitc_all((uint8_t *) v.data(), (uint32_t) v.size());
    }
}

inline bool any(const LLVMArray<bool> &v) {
    if (v.is_literal_one()) {
        return true;
    } else if (v.is_literal_zero()) {
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
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Add, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hprod(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Mul, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hmax(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Max, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hmin(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = empty<Array>(1);
    jitc_reduce(Array::Type, ReductionType::Min, v.data(), (uint32_t) v.size(),
                result.data());
    return result;
}

template <typename Value, typename Array = LLVMArray<Value>>
Array popcnt(const LLVMArray<Value> &a) {
    if (!jitc_is_integral(Array::Type))
        jitc_raise("Unsupported operand type");

    return Array::from_index(jitc_var_new_1(
        Array::Type,
        "$r0 = call <$w x $t0> @llvm.ctpop.v$w$a1(<$w x $t1> $r1)",
        1, a.index()));
}

template <typename Value, typename Array = LLVMArray<Value>>
Array lzcnt(const LLVMArray<Value> &a) {
    if (!jitc_is_integral(Array::Type))
        jitc_raise("Unsupported operand type");

    return Array::from_index(jitc_var_new_1(
        Array::Type,
        "$r0 = call <$w x $t0> @llvm.ctlz.v$w$a1(<$w x $t1> $r1, i1$S 0)",
        1, a.index()));
}

template <typename Value, typename Array = LLVMArray<Value>>
Array tzcnt(const LLVMArray<Value> &a) {
    if (!jitc_is_integral(Array::Type))
        jitc_raise("Unsupported operand type");

    return Array::from_index(jitc_var_new_1(
        Array::Type,
        "$r0 = call <$w x $t0> @llvm.cttz.v$w$a1(<$w x $t1> $r1, i1$S 0)",
        1, a.index()));
}

template <typename T> void jitc_schedule(const LLVMArray<T> &a) {
    a.schedule();
}

inline std::vector<std::pair<void *, LLVMArray<uint32_t>>>
vcall(const char *domain, const LLVMArray<uint32_t> &a) {
    uint32_t bucket_count = 0;
    VCallBucket *buckets = jitc_vcall(domain, a.index(), &bucket_count);

    std::vector<std::pair<void *, LLVMArray<uint32_t>>> result;
    result.reserve(bucket_count);
    for (uint32_t i = 0; i < bucket_count; ++i) {
        jitc_var_inc_ref_ext(buckets[i].index);
        result.emplace_back(buckets[i].ptr,
                            LLVMArray<uint32_t>::from_index(buckets[i].index));
    }

    return result;
}

inline LLVMArray<uint32_t> compress(const LLVMArray<bool> &a) {
    uint32_t *perm = (uint32_t *) jitc_malloc(AllocType::Host, a.size() * sizeof(uint32_t));
    uint32_t size = 0;

    jitc_compress((const uint8_t *) a.data(), (uint32_t) a.size(), perm, &size);
    jitc_sync_stream();

    LLVMArray<uint32_t> result = LLVMArray<uint32_t>::from_index(
        jitc_var_map(VarType::UInt32, perm, size, 1));

    return result;
}

template <typename T>
inline LLVMArray<T> block_copy(const LLVMArray<T> &a, uint32_t block_size) {
    size_t size = a.eval().size();
    LLVMArray<T> output = empty<LLVMArray<T>>(size * block_size);
    jitc_block_copy(LLVMArray<T>::Type, a.data(), output.data(), (uint32_t) size, block_size);
    return output;
}

template <typename T>
inline LLVMArray<T> block_sum(const LLVMArray<T> &a, uint32_t block_size) {
    size_t size = a.eval().size();
    if (size % block_size != 0)
        jitc_raise("block_sum(): array size must be divisible by block size (%u)!", block_size);
    size /= block_size;
    LLVMArray<T> output = empty<LLVMArray<T>>(size);
    jitc_block_sum(LLVMArray<T>::Type, a.data(), output.data(), (uint32_t) size, block_size);
    return output;
}
