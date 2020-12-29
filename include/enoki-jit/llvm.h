/*
    enoki-jit/llvm.h -- Simple C++ array class with operator overloading (LLVM)

    This library implements convenient wrapper class around the C API in
    'enoki/jit.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/util.h>

NAMESPACE_BEGIN(enoki)

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

    ~LLVMArray() { jit_var_dec_ref_ext(m_index); }

    LLVMArray(const LLVMArray &a) : m_index(a.m_index) {
        jit_var_inc_ref_ext(m_index);
    }

    LLVMArray(LLVMArray &&a) : m_index(a.m_index) {
        a.m_index = 0;
    }

#if 0
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
            size_t size_1 = std::is_same<T,     bool>::value ? 0 : sizeof(T),
                   size_2 = std::is_same<Value, bool>::value ? 0 : sizeof(Value);

            if (size_1 == size_2) {
                m_index = v.index();
                jit_var_inc_ref_ext(m_index);
                return;
            } else {
                op = size_1 > size_2
                         ? "$r0 = trunc <$w x $t1> $r1 to <$w x $t0>"
                         : (std::is_signed<T>::value
                                ? "$r0 = sext <$w x $t1> $r1 to <$w x $t0>"
                                : "$r0 = zext <$w x $t1> $r1 to <$w x $t0>");
            }
        } else {
            jit_fail("Unsupported conversion!");
        }

        m_index = jit_var_new_1(0, Type, op, 1, v.index());
    }
#endif

    LLVMArray(Value value) {
        m_index = jit_var_new_literal(IsCUDA, Type, &value, 1, 0);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    LLVMArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jit_var_mem_copy(IsCUDA, AllocType::Host, Type, data,
                                    (uint32_t) sizeof...(Args));
    }

    LLVMArray &operator=(const LLVMArray &a) {
        jit_var_inc_ref_ext(a.m_index);
        jit_var_dec_ref_ext(m_index);
        m_index = a.m_index;
        return *this;
    }

    LLVMArray &operator=(LLVMArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    LLVMArray operator+(const LLVMArray &v) const {
        return steal(jitc_var_op_2(OpType::Add, m_index, v.m_index));
    }

    LLVMArray operator-(const LLVMArray &v) const {
        return steal(jitc_var_op_2(OpType::Sub, m_index, v.m_index));
    }

    LLVMArray operator*(const LLVMArray &v) const {
        return steal(jitc_var_op_2(OpType::Mul, m_index, v.m_index));
    }

    LLVMArray operator/(const LLVMArray &v) const {
        return steal(jitc_var_op_2(OpType::Div, m_index, v.m_index));
    }

    LLVMArray operator%(const LLVMArray &v) const {
        return steal(jitc_var_op_2(OpType::Mod, m_index, v.m_index));
    }

    LLVMArray& schedule() {
        jit_var_schedule(m_index);
        return *this;
    }

    const LLVMArray& schedule() const {
        jit_var_schedule(m_index);
        return *this;
    }

    LLVMArray& eval() {
        jit_var_eval(m_index);
        return *this;
    }

    const LLVMArray& eval() const {
        jit_var_eval(m_index);
        return *this;
    }

    bool valid() const { return m_index != 0; }

    size_t size() const {
        return jit_var_size(m_index);
    }

    uint32_t index() const {
        return m_index;
    }

    const char *str() {
        return jit_var_str(m_index);
    }

    const Value *data() const {
        return (const Value *) jit_var_ptr(m_index);
    }

    Value *data() {
        return (Value *) jit_var_ptr(m_index);
    }

    Value read(uint32_t offset) const {
        Value out;
        jit_var_read(m_index, offset, &out);
        return out;
    }

    void write(uint32_t offset, Value value) {
        if (jit_var_refs(m_index) > 1)
            *this = steal(jit_var_copy(m_index));

        jit_var_write(m_index, offset, &value);
    }

    static LLVMArray map(void *ptr, size_t size, bool free = false) {
        return steal(
            jit_var_mem_map(1, Type, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static LLVMArray copy(const void *ptr, size_t size) {
        return steal(jit_var_mem_copy(1, AllocType::Host, Type, ptr, (uint32_t) size));
    }

    static LLVMArray steal(uint32_t index) {
        LLVMArray result;
        result.m_index = index;
        return result;
    }

protected:
    uint32_t m_index = 0;
};

NAMESPACE_END(enoki)
