/*
    enoki-jit/cuda.h -- Simple C++ array class with operator overloading

    This library implements convenient wrapper class around the C API in
    'enoki/jit.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/traits.h>

NAMESPACE_BEGIN(enoki)

template <JitBackend Backend_, typename Value_> struct JitArray {
    using Value = Value_;
    using Mask = JitArray<Backend_, bool>;
    static constexpr VarType Type = var_type<Value>::value;
    static constexpr JitBackend Backend = Backend_;
	template <typename T> using ReplaceValue = JitArray<Backend_, T>;

    JitArray() = default;

    ~JitArray() { jit_var_dec_ref_ext(m_index); }

    JitArray(const JitArray &a) : m_index(a.m_index) {
        jit_var_inc_ref_ext(m_index);
    }

    template <typename T> JitArray(const JitArray<Backend_, T> &v) {
        m_index = jit_var_new_cast(v.index(), JitArray<Backend_, T>::Type, 0);
    }

    JitArray(JitArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    JitArray(Value value) {
        m_index = jit_var_new_literal(Backend, Type, &value);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    JitArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                   (uint32_t) sizeof...(Args));
    }

    JitArray &operator=(const JitArray &a) {
        jit_var_inc_ref_ext(a.m_index);
        jit_var_dec_ref_ext(m_index);
        m_index = a.m_index;
        return *this;
    }

    JitArray &operator=(JitArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    JitArray operator-() const {
        return steal(jit_var_new_op_1(OpType::Neg, m_index));
    }

    JitArray operator~() const {
        return steal(jit_var_new_op_1(OpType::Not, m_index));
    }

    JitArray operator+(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Add, m_index, v.m_index));
    }

    JitArray operator-(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Sub, m_index, v.m_index));
    }

    JitArray operator*(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Mul, m_index, v.m_index));
    }

    JitArray operator/(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Div, m_index, v.m_index));
    }

    JitArray operator%(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Mod, m_index, v.m_index));
    }

    Mask operator>(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(OpType::Gt, m_index, v.m_index));
    }

    Mask operator>=(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(OpType::Ge, m_index, v.m_index));
    }

    Mask operator<(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(OpType::Lt, m_index, v.m_index));
    }

    Mask operator<=(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(OpType::Le, m_index, v.m_index));
    }

    friend JitArray eq(const JitArray &v1, const JitArray &v2) {
        return Mask::steal(jit_var_new_op_2(OpType::Eq, v1.m_index, v2.m_index));
    }

    friend JitArray neq(const JitArray &v1, const JitArray &v2) {
        return Mask::steal(jit_var_new_op_2(OpType::Neq, v1.m_index, v2.m_index));
    }

    bool operator==(const JitArray &v) const {
        return all(eq(*this, v));
    }

    bool operator!=(const JitArray &v) const {
        return any(neq(*this, v));
    }

    JitArray operator|(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Or, m_index, v.m_index));
    }

    JitArray operator&(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::And, m_index, v.m_index));
    }

    JitArray operator^(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Xor, m_index, v.m_index));
    }

    JitArray operator<<(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Shl, m_index, v.m_index));
    }

    JitArray operator>>(const JitArray &v) const {
        return steal(jit_var_new_op_2(OpType::Shr, m_index, v.m_index));
    }

    JitArray &operator+=(const JitArray &v) { return operator=(*this + v); }
    JitArray &operator-=(const JitArray &v) { return operator=(*this - v); }
    JitArray &operator*=(const JitArray &v) { return operator=(*this * v); }
    JitArray &operator/=(const JitArray &v) { return operator=(*this / v); }
    JitArray &operator|=(const JitArray &v) { return operator=(*this | v); }
    JitArray &operator&=(const JitArray &v) { return operator=(*this & v); }
    JitArray &operator^=(const JitArray &v) { return operator=(*this ^ v); }
    JitArray& operator<<=(const JitArray &v) { return operator=(*this << v); }
    JitArray& operator>>=(const JitArray &v) { return operator=(*this >> v); }

    template <typename V = Value, enable_if_t<std::is_same<V, bool>::value> = 0>
    JitArray operator&&(const JitArray &v) const {
        return operator&(v);
    }

    template <typename V = Value, enable_if_t<std::is_same<V, bool>::value> = 0>
    JitArray operator||(const JitArray &v) const {
        return operator|(v);
    }

    template <typename V = Value, enable_if_t<std::is_same<V, bool>::value> = 0>
    JitArray operator!() const {
        return operator~();
    }

    JitArray& schedule() {
        jit_var_schedule(m_index);
        return *this;
    }

    const JitArray& schedule() const {
        jit_var_schedule(m_index);
        return *this;
    }

    JitArray& eval() {
        jit_var_eval(m_index);
        return *this;
    }

    const JitArray& eval() const {
        jit_var_eval(m_index);
        return *this;
    }

    bool valid() const { return m_index != 0; }

    size_t size() const {
        return jit_var_size(m_index);
    }

	void resize(size_t size) {
        uint32_t index = jit_var_resize(m_index, (uint32_t) size);
        jit_var_dec_ref_ext(m_index);
        m_index = index;
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

    static JitArray map(void *ptr, size_t size, bool free = false) {
        return steal(
            jit_var_mem_map(Backend, Type, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static JitArray copy(const void *ptr, size_t size) {
        return steal(jit_var_mem_copy(Backend, AllocType::Host, Type, ptr,
                                      (uint32_t) size));
    }

    static JitArray steal(uint32_t index) {
        JitArray result;
        result.m_index = index;
        return result;
    }

	// ------------------------------------------------------

    friend JitArray abs(const JitArray &v) {
        return Mask::steal(jit_var_new_op_2(OpType::Abs, v.m_index));
    }

    friend JitArray sqrt(const JitArray &v) {
        return Mask::steal(jit_var_new_op_2(OpType::Sqrt, v.m_index));
    }

    friend JitArray ceil(const JitArray &v) {
        return Mask::steal(jit_var_new_op_2(OpType::Ceil, v.m_index));
    }

    friend JitArray floor(const JitArray &v) {
        return Mask::steal(jit_var_new_op_2(OpType::Floor, v.m_index));
    }

    friend JitArray round(const JitArray &v) {
        return Mask::steal(jit_var_new_op_2(OpType::Round, v.m_index));
    }

    friend JitArray trunc(const JitArray &v) {
        return Mask::steal(jit_var_new_op_2(OpType::Trunc, v.m_index));
    }

    friend JitArray fmadd(const JitArray &a, const JitArray &b,
                          const JitArray &c) {
        return Mask::steal(
            jit_var_new_op_3(OpType::Fmadd, a.m_index, b.m_index.c.m_index));
    }

    friend JitArray select(const Mask &a, const JitArray &b,
                           const JitArray &c) {
        return Mask::steal(
            jit_var_new_op_3(OpType::Select, a.m_index, b.m_index.c.m_index));
    }

	friend const char *label(const JitArray &v) {
		return jit_var_label(v.m_index);
	}

	friend void set_label(const JitArray &v, const char *label) {
		jit_var_set_label(v.m_index, label);
	}
protected:
    uint32_t m_index = 0;
};

template <typename Array>
Array empty(size_t size) {
    size_t byte_size = size * sizeof(typename Array::Value);
    void *ptr =
        jit_malloc(Array::Backend == JitBackend::CUDA ? AllocType::Device
                                                      : AllocType::HostAsync,
                   byte_size);
    return Array::steal(
        jit_var_map_mem(Array::Backend, Array::Type, ptr, (uint32_t) size, 1));
}

template <typename Array>
Array zero(size_t size) {
    return Array::steal(
        jit_var_new_literal(Array::Backend, Array::Type, 0, (uint32_t) size));
}

template <typename Array>
Array full(const typename Array::Value &value, size_t size, bool eval = false) {
    return Array::steal(jit_var_new_literal(Array::Backend, Array::Type, &value,
                                            (uint32_t) size, eval));
}

template <typename Array>
Array arange(ssize_t start, ssize_t stop, ssize_t step) {
    using UInt32 = typename Array::template ReplaceValue<uint32_t>;
    using Value = typename Array::Value;

    size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);

    return fmadd(Array(UInt32::counter(size)), Array((Value) step),
                 Array((Value) start));
}

template <typename Array> Array arange(size_t size) {
    return arange<Array>(0, size, 1);
}

template <typename Array>
Array linspace(typename Array::Value min, typename Array::Value max, size_t size) {
    using UInt32 = typename Array::template ReplaceValue<uint32_t>;
    using Value = typename Array::Value;

    Value step = (max - min) / Value(size - 1);
    return fmadd(Array(UInt32::counter(size)), Array(step), Array(min));
}


#if 0
template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
void jit_var_schedule(Args&&... args) {
    bool unused[] = { (jit_var_schedule(args), false)..., false };
    (void) unused;
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 0)> = 0>
void jit_eval(Args&&... args) {
    jit_var_schedule(args...);
    if (sizeof...(Args) > 0)
        ::jit_eval();
}
#endif

template <typename T> using CUDAArray = JitArray<JitBackend::CUDA, T>;
template <typename T> using LLVMArray = JitArray<JitBackend::LLVM, T>;

NAMESPACE_END(enoki)
