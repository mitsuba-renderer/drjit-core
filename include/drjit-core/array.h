/*
    drjit-core/array.h -- Simple C++ array class with operator overloading

    This library implements convenient wrapper class around the C API in
    'drjit/jit.h'.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/traits.h>
#include <drjit-core/half.h>

NAMESPACE_BEGIN(drjit)

template <JitBackend Backend_, typename Value_> struct JitArray {
    using Value = Value_;
    using Mask = JitArray<Backend_, bool>;
    static constexpr VarType Type = var_type<Value>::value;
    static constexpr JitBackend Backend = Backend_;
    static constexpr bool IsArray = true;
    static constexpr bool IsClass =
        std::is_pointer<Value_>::value &&
        std::is_class<typename std::remove_pointer<Value_>::type>::value;
    template <typename T> using ReplaceValue = JitArray<Backend_, T>;
    using ActualValue = typename std::conditional<IsClass, uint32_t, Value>::type;

    JitArray() = default;

    ~JitArray() { jit_var_dec_ref(m_index); }

    JitArray(const JitArray &a) {
        m_index = jit_var_inc_ref(a.m_index);
    }

    template <typename T> JitArray(const JitArray<Backend_, T> &v) {
        m_index = jit_var_cast(v.index(), Type, 0);
    }

    JitArray(JitArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <bool B = IsClass, enable_if_t<B> = 0>
    JitArray(Value value) {
        m_index = jit_var_class(Backend, (void *) value);
    }

    template <bool B = IsClass, enable_if_t<!B> = 0>
    JitArray(Value value) {
        using half = drjit::half;
        switch (Type) {
            case VarType::Bool:    m_index = jit_var_bool(Backend, (bool) value); break;
            case VarType::Int32:   m_index = jit_var_i32 (Backend, (int32_t) value); break;
            case VarType::UInt32:  m_index = jit_var_u32 (Backend, (uint32_t) value); break;
            case VarType::Int64:   m_index = jit_var_i64 (Backend, (int64_t) value); break;
            case VarType::UInt64:  m_index = jit_var_u64 (Backend, (uint64_t) value); break;
            case VarType::Float16: m_index = jit_var_f16 (Backend, (half) value); break;
            case VarType::Float32: m_index = jit_var_f32 (Backend, (float) value); break;
            case VarType::Float64: m_index = jit_var_f64 (Backend, (double) value); break;
            default: jit_fail("JitArray(): tried to initialize scalar array with unsupported type!");
        }
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 && IsClass)> = 0>
    JitArray(Ts&&... ts) {
        uint32_t data[] = { jit_registry_get_id(Backend,  ts)... };
        m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                   sizeof...(Ts));
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 && !IsClass)> = 0>
    JitArray(Ts&&... ts) {
        Value data[] = { (Value) ts... };
        m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                   sizeof...(Ts));
    }

    template <typename T, enable_if_t<drjit::detail::is_arithmetic_v<T>> = 0>
    JitArray(T val) : JitArray((Value)val) {}

    JitArray &operator=(const JitArray &a) {
        uint32_t index = jit_var_inc_ref(a.m_index);
        jit_var_dec_ref(m_index);
        m_index = index;
        return *this;
    }

    JitArray &operator=(JitArray &&a) {
        uint32_t tmp = m_index;
        m_index = a.m_index;
        a.m_index = tmp;
        return *this;
    }

    JitArray operator-() const {
        return steal(jit_var_neg(m_index));
    }

    JitArray operator~() const {
        return steal(jit_var_not(m_index));
    }

    JitArray operator+(const JitArray &v) const {
        return steal(jit_var_add(m_index, v.m_index));
    }

    JitArray operator-(const JitArray &v) const {
        return steal(jit_var_sub(m_index, v.m_index));
    }

    JitArray operator*(const JitArray &v) const {
        return steal(jit_var_mul(m_index, v.m_index));
    }

    JitArray operator/(const JitArray &v) const {
        return steal(jit_var_div(m_index, v.m_index));
    }

    JitArray operator%(const JitArray &v) const {
        return steal(jit_var_mod(m_index, v.m_index));
    }

    JitArray operator&(const JitArray &v) const {
        return steal(jit_var_and(m_index, v.m_index));
    }

    JitArray operator|(const JitArray &v) const {
        return steal(jit_var_or(m_index, v.m_index));
    }

    Mask operator>(const JitArray &v) const {
        return Mask::steal(jit_var_gt(m_index, v.m_index));
    }

    Mask operator>=(const JitArray &v) const {
        return Mask::steal(jit_var_ge(m_index, v.m_index));
    }

    Mask operator<(const JitArray &v) const {
        return Mask::steal(jit_var_lt(m_index, v.m_index));
    }

    Mask operator<=(const JitArray &v) const {
        return Mask::steal(jit_var_le(m_index, v.m_index));
    }

    friend Mask eq(const JitArray &v1, const JitArray &v2) {
        return Mask::steal(jit_var_eq(v1.m_index, v2.m_index));
    }

    friend Mask neq(const JitArray &v1, const JitArray &v2) {
        return Mask::steal(jit_var_neq(v1.m_index, v2.m_index));
    }

    bool operator==(const JitArray &v) const {
        return all(eq(*this, v));
    }

    bool operator!=(const JitArray &v) const {
        return any(neq(*this, v));
    }

    template <typename T, enable_if_t<std::is_same<T, Value_>::value ||
                                      std::is_same<T, bool>::value> = 0>
    JitArray operator|(const JitArray<Backend_, T> &v) const {
        return steal(jit_var_or(m_index, v.index()));
    }

    template <typename T, enable_if_t<std::is_same<T, Value_>::value ||
                                      std::is_same<T, bool>::value> = 0>
    JitArray operator&(const JitArray<Backend_, T> &v) const {
        return steal(jit_var_and(m_index, v.index()));
    }

    JitArray operator^(const JitArray &v) const {
        return steal(jit_var_xor(m_index, v.m_index));
    }

    JitArray operator<<(const JitArray &v) const {
        return steal(jit_var_shl(m_index, v.m_index));
    }

    JitArray operator>>(const JitArray &v) const {
        return steal(jit_var_shr(m_index, v.m_index));
    }

    JitArray &operator+=(const JitArray &v) { return operator=(*this + v); }
    JitArray &operator-=(const JitArray &v) { return operator=(*this - v); }
    JitArray &operator*=(const JitArray &v) { return operator=(*this * v); }
    JitArray &operator/=(const JitArray &v) { return operator=(*this / v); }
    template <typename T = Value_, enable_if_t<!std::is_same<T, bool>::value> = 0>
    JitArray &operator|=(const JitArray &v) { return operator=(*this | v); }
    template <typename T = Value_, enable_if_t<!std::is_same<T, bool>::value> = 0>
    JitArray &operator&=(const JitArray &v) { return operator=(*this & v); }
    JitArray &operator|=(const JitArray<Backend, bool> &v) { return operator=(*this | v); }
    JitArray &operator&=(const JitArray<Backend, bool> &v) { return operator=(*this & v); }
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

    JitArray& make_opaque() {
        int unused;
        uint32_t rv = jit_var_schedule_force(m_index, &unused);
        jit_eval();
        jit_var_dec_ref(m_index);
        ((JitArray *) this)->m_index = rv;
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
        uint32_t index = jit_var_resize(m_index, size);
        jit_var_dec_ref(m_index);
        m_index = index;
    }

    uint32_t index() const {
        return m_index;
    }

    uint32_t* index_ptr() { return &m_index; }

    const char *str() const {
        return jit_var_str(m_index);
    }

    const Value *data() const {
        void* ptr_out = nullptr;
        uint32_t new_index = jit_var_data(m_index, &ptr_out);
        jit_var_dec_ref(m_index);
        ((JitArray &) *this).m_index = new_index;
        return (const Value *) ptr_out;
    }

    Value *data() {
        void* ptr_out = nullptr;
        uint32_t new_index = jit_var_data(m_index, &ptr_out);
        jit_var_dec_ref(m_index);
        m_index = new_index;
        return (Value *) ptr_out;
    }

    Value read(size_t offset) const {
        Value out;
        jit_var_read(m_index, offset, &out);
        return out;
    }

    void write(size_t offset, Value value) {
        uint32_t index = jit_var_write(m_index, offset, &value);
        jit_var_dec_ref(m_index);
        m_index = index;
    }

    static JitArray map(void *ptr, size_t size, bool free = false) {
        return steal(
            jit_var_mem_map(Backend, Type, ptr, size, free ? 1 : 0));
    }

    static JitArray copy(const void *ptr, size_t size) {
        return steal(
            jit_var_mem_copy(Backend, AllocType::Host, Type, ptr, size));
    }

    uint32_t release() {
        uint32_t result = m_index;
        m_index = 0;
        return result;
    }

    static JitArray steal(uint32_t index) {
        JitArray result;
        result.m_index = index;
        return result;
    }

    static JitArray borrow(uint32_t index) {
        JitArray result;
        result.m_index = jit_var_inc_ref(index);
        return result;
    }

    static JitArray<Backend_, uint32_t> counter(size_t size) {
        return JitArray<Backend_, uint32_t>::steal(
            jit_var_counter(Backend, size));
    }

    // ------------------------------------------------------

    friend JitArray abs(const JitArray &v) {
        return steal(jit_var_abs(v.m_index));
    }

    friend JitArray sqrt(const JitArray &v) {
        return steal(jit_var_sqrt(v.m_index));
    }

    friend JitArray ceil(const JitArray &v) {
        return steal(jit_var_ceil(v.m_index));
    }

    friend JitArray floor(const JitArray &v) {
        return steal(jit_var_floor(v.m_index));
    }

    friend JitArray round(const JitArray &v) {
        return steal(jit_var_round(v.m_index));
    }

    friend JitArray trunc(const JitArray &v) {
        return steal(jit_var_trunc(v.m_index));
    }

    friend JitArray fmadd(const JitArray &a, const JitArray &b,
                          const JitArray &c) {
        return steal(jit_var_fma(a.m_index, b.m_index, c.m_index));
    }

    friend JitArray select(const Mask &a, const JitArray &b,
                           const JitArray &c) {
        return steal(jit_var_select(a.index(), b.m_index, c.m_index));
    }

    friend JitArray min(const JitArray &a, const JitArray &b) {
        return steal(jit_var_min(a.m_index, b.m_index));
    }

    friend JitArray max(const JitArray &a, const JitArray &b) {
        return steal(jit_var_max(a.m_index, b.m_index));
    }

    friend JitArray hsum(const JitArray &v) {
        return steal(jit_var_reduce(Backend, Type, ReduceOp::Add, v.m_index));
    }

    friend JitArray hmul(const JitArray &v) {
        return steal(jit_var_reduce(Backend, Type, ReduceOp::Mul, v.m_index));
    }

    friend JitArray hmin(const JitArray &v) {
        return steal(jit_var_reduce(Backend, Type, ReduceOp::Min, v.m_index));
    }

    friend JitArray hmax(const JitArray &v) {
        return steal(jit_var_reduce(Backend, Type, ReduceOp::Max, v.m_index));
    }

    friend JitArray block_sum(const JitArray &v, uint32_t block_size) {
        return steal(jit_var_block_reduce(ReduceOp::Add, v.m_index, block_size, false));
    }

    friend JitArray block_prefix_sum(const JitArray &v, uint32_t block_size, bool exclusive = true, bool reverse = false) {
        return steal(jit_var_block_prefix_reduce(ReduceOp::Add, v.m_index, block_size, exclusive, reverse));
    }

    friend JitArray reverse(const JitArray &v) {
        return steal(jit_var_reverse(v.m_index));
    }

    friend bool all(const JitArray &a) { return jit_var_all(a.m_index); }
    friend bool any(const JitArray &a) { return jit_var_any(a.m_index); }
    friend bool none(const JitArray &a) { return !jit_var_any(a.m_index); }

	friend const char *label(const JitArray &v) {
		return jit_var_label(v.m_index);
	}

	friend void set_label(JitArray &v, const char *label) {
		uint32_t index = jit_var_set_label(v.m_index, 1, label);
		jit_var_dec_ref(v.m_index);
		v.m_index = index;
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
        jit_var_mem_map(Array::Backend, Array::Type, ptr, size, 1));
}

template <typename Array>
Array zeros(size_t size = 1) {
    typename Array::Value value = 0;
    return Array::steal(
        jit_var_literal(Array::Backend, Array::Type, &value, size));
}

template <typename Array>
Array full(const typename Array::Value &value, size_t size = 1) {
    return Array::steal(
        jit_var_literal(Array::Backend, Array::Type, &value, size, false));
}

template <typename Array>
Array opaque(const typename Array::Value &value, size_t size = 1) {
    return Array::steal(
        jit_var_literal(Array::Backend, Array::Type, &value, size, true));
}

template <typename Array, typename Index>
Array gather(const Array &source, const JitArray<Array::Backend, Index> index,
             const JitArray<Array::Backend, bool> &mask = true) {
    return Array::steal(
        jit_var_gather(source.index(), index.index(), mask.index()));
}

template <typename Array, typename Index>
void scatter(Array &target, const Array &value, const JitArray<Array::Backend, Index>& index,
             const JitArray<Array::Backend, bool> &mask = true) {
    target = Array::steal(jit_var_scatter(target.index(), value.index(),
                                          index.index(), mask.index(),
                                          ReduceOp::Identity));
}

template <typename Array, typename Index>
void scatter_reduce(ReduceOp op, Array &target, const Array &value,
                    const JitArray<Array::Backend, Index> &index,
                    const JitArray<Array::Backend, bool> &mask = true) {
    target = Array::steal(jit_var_scatter(target.index(), value.index(),
                                          index.index(), mask.index(), op));
}

template <typename Array>
Array scatter_inc(Array &target, const Array index, const JitArray<Array::Backend, bool> &mask = true) {
    return Array::steal(jit_var_scatter_inc(target.index_ptr(), index.index(), mask.index()));
}

template <typename Array, typename Index>
void scatter_add_kahan(Array &target_1, Array &target_2, const Array &value,
                          const JitArray<Array::Backend, Index> &index,
                          const JitArray<Array::Backend, bool> &mask = true) {
    jit_var_scatter_add_kahan(target_1.index_ptr(), target_2.index_ptr(),
                              value.index(), index.index(), mask.index());
}

template <typename Array>
Array arange(size_t start, size_t stop, size_t step) {
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

template <typename T> using CUDAArray = JitArray<JitBackend::CUDA, T>;
template <typename T> using LLVMArray = JitArray<JitBackend::LLVM, T>;

NAMESPACE_END(drjit)
