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
LLVMArray<Value> select(const LLVMArray<bool> &m, const LLVMArray<Value> &a,
                        const LLVMArray<Value> &b);

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

        m_index = jitc_trace_append_0(Type, value_str, 0);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    LLVMArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jitc_var_copy(Type, data, sizeof...(Args));
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
        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fadd <$w x $t0> $r1, $r2"
            : "$r0 = add <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator-(const LLVMArray &v) const {
        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fsub <$w x $t0> $r1, $r2"
            : "$r0 = sub <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator*(const LLVMArray &v) const {
        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fmul <$w x $t0> $r1, $r2"
            : "$r0 = mul <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray operator/(const LLVMArray &v) const {
        const char *op = std::is_floating_point<Value>::value
            ? "$r0 = fdiv <$w x $t0> $r1, $r2"
            : "$r0 = div <$w x $t0> $r1, $r2";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, v.m_index));
    }

    LLVMArray<bool> operator>(const LLVMArray &a) const {
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

    LLVMArray operator-() const {
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
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = or <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = or <$w x $b0> $r0_0, $r0_1"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, a.index()));
    }

    LLVMArray operator&(const LLVMArray &a) const {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = and <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = and <$w x $b0> $r0_0, $r0_1"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, a.index()));
    }

    LLVMArray operator^(const LLVMArray &a) const {
        const char *op = std::is_integral<Value>::value
                             ? "$r0 = xor <$w x $t1> $r1, $r2"
                             : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                               "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                               "$r0_2 = xor <$w x $b0> $r0_0, $r0_1"
                               "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";

        return from_index(
            jitc_trace_append_2(Type, op, 1, m_index, a.index()));
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

    LLVMArray& operator^=(const LLVMArray &v) {
        return operator=(*this ^ v);
    }

    friend LLVMArray sqrt(const LLVMArray &a) {
        return LLVMArray::from_index(
            "$r0 = call <$w x $t0> @llvm.sqrt.v$w$a1(<$w x $t1> $r1)", 1,
            a.index());
    }

    friend LLVMArray abs(const LLVMArray &a) {
        if (std::is_floating_point<Value>::value) {
            return LLVMArray::from_index(jitc_trace_append_1(
                Type, "$r0 = call <$w x $t0> @llvm.abs.v$w$a1(<$w x $t1> $r1)",
                1, a.index()));
        } else {
            return select(a > 0, a, -a);
        }
    }

    friend LLVMArray fmadd(const LLVMArray &a, const LLVMArray &b,
                           const LLVMArray &c) {
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

    static LLVMArray empty(size_t size) {
        size_t byte_size = size * sizeof(Value);
        void *ptr = jitc_malloc(AllocType::Host, byte_size);
        return from_index(jitc_var_map(Type, ptr, size, 1));
    }

    static LLVMArray zero(size_t size) {
        if (size == 1) {
            return LLVMArray(0);
        } else {
            uint8_t value = 0;
            size_t byte_size = size * sizeof(Value);
            void *ptr = jitc_malloc(AllocType::Host, byte_size);
            jitc_fill(VarType::UInt8, ptr, byte_size, &value);
            return from_index(jitc_var_map(Type, ptr, size, 1));
        }
    }

    static LLVMArray full(Value value, size_t size) {
        if (size == 1) {
            return LLVMArray(value);
        } else {
            size_t byte_size = size * sizeof(Value);
            void *ptr = jitc_malloc(AllocType::Host, byte_size);
            jitc_fill(Type, ptr, size, &value);
            return from_index(jitc_var_map(Type, ptr, size, 1));
        }
    }

    static LLVMArray arange(size_t size) {
        return arange(0, (size_t) size, 1);
    }

    static LLVMArray arange(ssize_t start, ssize_t stop, ssize_t step) {
        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);

        using UInt32 = LLVMArray<uint32_t>;
        UInt32 index = UInt32::launch_index(size);

        if (start == 0 && step == 1)
            return index;
        else
            return fmadd(LLVMArray(index), LLVMArray((Value) step), LLVMArray((Value) start));
    }

    LLVMArray eval() const {
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

    static LLVMArray from_index(uint32_t index) {
        LLVMArray result;
        result.m_index = index;
        return result;
    }

    static LLVMArray launch_index(size_t size) {
        uint32_t index = jitc_trace_append_0(
            Type,
            "$r0_0 = trunc i64 %index to $t0$n"
            "$r0_1 = insertelement <$w x $t0> undef, $t0 $r0_0, i32 0$n"
            "$r0_2 = shufflevector <$w x $t0> $r0_1, <$w x $t0> undef, "
                "<$w x i32> zeroinitializer$n"
            "$r0 = add <$w x $t0> $r0_2, $l0", 1);
        jitc_var_set_size(index, size, false);
        return from_index(index);
    }

    static LLVMArray launch_start() {
        return from_index(jitc_trace_append_0(
            Type,
            "$r0_0 = trunc i64 %start to $t0$n"
            "$r0_1 = insertelement <$w x $t0> undef, $t0 $r0_0, i32 0$n"
            "$r0 = shufflevector <$w x $t0> $r0_1, <$w x $t0> undef, "
                "<$w x i32> zeroinitializer", 1));
    }

    static LLVMArray launch_end() {
        return from_index(jitc_trace_append_0(
            Type,
            "$r0_0 = trunc i64 %end to $t0$n"
            "$r0_1 = insertelement <$w x $t0> undef, $t0 $r0_0, i32 0$n"
            "$r0 = shufflevector <$w x $t0> $r0_1, <$w x $t0> undef, "
                "<$w x i32> zeroinitializer", 1));
    }

    static LLVMArray<bool> launch_mask(size_t size) {
        using UInt32 = LLVMArray<uint32_t>;
        return UInt32::launch_index(size) < UInt32::launch_end();
    }

protected:
    uint32_t m_index = 0;
};

template <typename Value>
void set_label(const LLVMArray<Value> &a, const char *label) {
    jitc_var_set_label(a.index(), label);
}


template <typename Value>
LLVMArray<Value> select(const LLVMArray<bool> &m, const LLVMArray<Value> &t,
                        const LLVMArray<Value> &f) {
    return LLVMArray<Value>::from_index(jitc_trace_append_3(
        LLVMArray<Value>::Type,
        "$r0 = select <$w x $t1> $r1, <$w x $t2> $r2, <$w x $t3> $r3", 1,
        m.index(), t.index(), f.index()));
}

template <typename OutArray, size_t Stride = sizeof(typename OutArray::Value),
          typename Index, typename std::enable_if<OutArray::IsLLVM, int>::type = 0>
static OutArray gather(const void *ptr, const LLVMArray<Index> &index,
                       LLVMArray<bool> mask = true) {

    using UInt64 = LLVMArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr));
    size_t mask_size = mask.size(), index_size = index.size(),
           size = mask_size > index_size ? mask_size : index_size;
    mask &= LLVMArray<bool>::launch_mask(size);

    OutArray addr = OutArray::from_index(jitc_trace_append_2(
        OutArray::Type,
        "$r0_0 = inttoptr $t1 $r1 to $t0*$n"
        "$r0 = getelementptr $t0, $t0* $r0_0, <$w x $t2> $r2",
        1, base.index(), index.index()
    ));

    const char *op;
    switch (Stride) {
        case 1:
            op = "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                 "(<$w x $t0*> $r1, i32$s 1, <$w x $t2> $r2, <$w x $t0> $z0)";
            break;

        case 2:
            op = "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                 "(<$w x $t0*> $r1, i32$s 2, <$w x $t2> $r2, <$w x $t0> $z0)";
            break;

        case 4:
            op = "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                 "(<$w x $t0*> $r1, i32$s 4, <$w x $t2> $r2, <$w x $t0> $z0)";
            break;

        case 8:
            op = "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0"
                 "(<$w x $t0*> $r1, i32$s 8, <$w x $t2> $r2, <$w x $t0> $z0)";
            break;

        default: jitc_fail("Unsupported stride!");
    }

    return OutArray::from_index(
        jitc_trace_append_2(OutArray::Type, op, 1, addr.index(), mask.index()));
}

template <size_t Stride_ = 0, typename Value, typename Index>
static void scatter(void *ptr, const LLVMArray<Value> &value,
                    const LLVMArray<Index> &index,
                    LLVMArray<bool> mask = true) {

    constexpr size_t Stride = Stride_ != 0 ? Stride_ : sizeof(Value);

    using UInt64 = LLVMArray<uint64_t>;
    UInt64 base = UInt64::from_index(jitc_var_copy_ptr(ptr));

    size_t mask_size = mask.size(), index_size = index.size(),
           value_size = value.size();
    size_t size = mask_size > index_size ? mask_size : index_size;
    size = value_size > size ? value_size : size;

    mask &= LLVMArray<bool>::launch_mask(size);

    LLVMArray<Value> addr = LLVMArray<Value>::from_index(jitc_trace_append_2(
        LLVMArray<Value>::Type,
        "$r0_0 = inttoptr $t1 $r1 to $t0*$n"
        "$r0 = getelementptr $t0, $t0* $r0_0, <$w x $t2> $r2",
        1, base.index(), index.index()));

    const char *op;
    switch (Stride) {
        case 1:
            op = "call <$w x $t1> @llvm.masked.scatter.v$w$a1"
                 "(<$w x $t1> $r1, <$w x $t1*> $r2, i32$s 1, <$w x $t3> $r3)";
            break;

        case 2:
            op = "call <$w x $t1> @llvm.masked.scatter.v$w$a1"
                 "(<$w x $t1> $r1, <$w x $t1*> $r2, i32$s 2, <$w x $t3> $r3)";
            break;

        case 4:
            op = "call <$w x $t1> @llvm.masked.scatter.v$w$a1"
                 "(<$w x $t1> $r1, <$w x $t1*> $r2, i32$s 4, <$w x $t3> $r3)";
            break;

        case 8:
            op = "call <$w x $t1> @llvm.masked.scatter.v$w$a1"
                 "(<$w x $t1> $r1, <$w x $t1*> $r2, i32$s 8, <$w x $t3> $r3)";
            break;

        default: jitc_fail("Unsupported stride!");
    }

    uint32_t var = jitc_trace_append_3(VarType::Invalid, op, 1, value.index(),
                                       addr.index(), mask.index());

    jitc_var_mark_side_effect(var);
}

template <typename Array, size_t Stride = sizeof(typename Array::Value),
          typename Index, typename std::enable_if<Array::IsLLVM, int>::type = 0>
Array gather(const Array &src, const LLVMArray<Index> &index,
             const LLVMArray<bool> &mask = true) {

    jitc_set_scatter_gather_operand(src.index(), 1);
    Array result = gather<Array, Stride>(src.data(), index, mask);
    jitc_set_scatter_gather_operand(0, 0);
    return result;
}

template <size_t Stride = 0, typename Value, typename Index>
void scatter(LLVMArray<Value> &dst, const LLVMArray<Value> &value,
             const LLVMArray<Index> &index,
             const LLVMArray<bool> &mask = true) {

    jitc_set_scatter_gather_operand(dst.index(), 0);
    scatter<Stride>(dst.data(), value, index, mask);
    jitc_set_scatter_gather_operand(0, 0);
    jitc_var_mark_dirty(dst.index());
}

template <typename Value> LLVMArray<Value> hsum(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Add, v.data(), v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hprod(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Mul, v.data(), v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hmax(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Max, v.data(), v.size(),
                result.data());
    return result;
}

template <typename Value> LLVMArray<Value> hmin(const LLVMArray<Value> &v) {
    using Array = LLVMArray<Value>;
    if (v.size() == 1)
        return v;

    v.eval();
    Array result = Array::empty(1);
    jitc_reduce(Array::Type, ReductionType::Min, v.data(), v.size(),
                result.data());
    return result;
}