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

    ~CUDAArray() { jit_var_dec_ref_ext(m_index); }

    CUDAArray(const CUDAArray &a) : m_index(a.m_index) {
        jit_var_inc_ref_ext(m_index);
    }

    CUDAArray(CUDAArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

#if 0
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
                jit_var_inc_ref_ext(m_index);
                return;
            }
            op = "cvt.$t0.$t1 $r0, $r1";
        }

        m_index = jit_var_new_1(1, Type, op, 1, v.index());
    }
#endif

    CUDAArray(Value value) {
        m_index = jit_var_new_literal(IsCUDA, Type, &value);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
    CUDAArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jit_var_mem_copy(IsCUDA, AllocType::Host, Type, data,
                                    (uint32_t) sizeof...(Args));
    }

    CUDAArray &operator=(const CUDAArray &a) {
        jit_var_inc_ref_ext(a.m_index);
        jit_var_dec_ref_ext(m_index);
        m_index = a.m_index;
        return *this;
    }

    CUDAArray &operator=(CUDAArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    CUDAArray operator+(const CUDAArray &v) const {
        return steal(jitc_var_op_2(OpType::Add, m_index, v.m_index));
    }

    CUDAArray operator-(const CUDAArray &v) const {
        return steal(jitc_var_op_2(OpType::Sub, m_index, v.m_index));
    }

    CUDAArray operator*(const CUDAArray &v) const {
        return steal(jitc_var_op_2(OpType::Mul, m_index, v.m_index));
    }

    CUDAArray operator/(const CUDAArray &v) const {
        return steal(jitc_var_op_2(OpType::Div, m_index, v.m_index));
    }

    CUDAArray operator%(const CUDAArray &v) const {
        return steal(jitc_var_op_2(OpType::Mod, m_index, v.m_index));
    }

    CUDAArray& schedule() {
        jit_var_schedule(m_index);
        return *this;
    }

    const CUDAArray& schedule() const {
        jit_var_schedule(m_index);
        return *this;
    }

    CUDAArray& eval() {
        jit_var_eval(m_index);
        return *this;
    }

    const CUDAArray& eval() const {
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

    static CUDAArray map(void *ptr, size_t size, bool free = false) {
        return steal(
            jit_var_mem_map(1, Type, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static CUDAArray copy(const void *ptr, size_t size) {
        return steal(jit_var_mem_copy(1, AllocType::Host, Type, ptr, (uint32_t) size));
    }

    static CUDAArray steal(uint32_t index) {
        CUDAArray result;
        result.m_index = index;
        return result;
    }

protected:
    uint32_t m_index = 0;
};

NAMESPACE_END(enoki)
