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

template <typename Value_>
struct LLVMArray {
    using Value = Value_;
    static constexpr VarType Type = var_type<Value>::value;

    LLVMArray() = default;

    ~LLVMArray() { jitc_var_ext_ref_dec(m_index); }

    LLVMArray(const LLVMArray &a) : m_index(a.m_index) {
        jitc_var_ext_ref_inc(m_index);
    }

    LLVMArray(LLVMArray &&a) : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> LLVMArray(const LLVMArray<T> &v) {
        const char *op;

        if (std::is_floating_point<T>::value && std::is_integral<Value>::value)
            op = "cvt.rzi.$t0.$t1 $r0, $r1";
        else if (std::is_integral<T>::value && std::is_floating_point<Value>::value)
            op = "cvt.rn.$t0.$t1 $r0, $r1";
        else
            op = "cvt.$t0.$t1 $r0, $r1";

        m_index = jitc_trace_append_1(Type, op, 1, v.index_());
    }

    LLVMArray(Value value) {
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
    LLVMArray(Args&&... args) {
        Value data[] = { (Value) args... };
        m_index = jitc_var_copy(Type, sizeof...(Args), data);
    }

    LLVMArray &operator=(const LLVMArray &a) {
        jitc_var_ext_ref_inc(a.m_index);
        jitc_var_ext_ref_dec(m_index);
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

    LLVMArray& operator+=(const LLVMArray &v) {
        return operator=(*this + v);
    }

    static LLVMArray empty(size_t size) {
        size_t byte_size = size * sizeof(Value);
        void *ptr = jitc_malloc(AllocType::Device, byte_size);
        return from_index(jitc_var_map(Type, ptr, size, 1));
    }

    static LLVMArray zero(size_t size) {
        if (size == 1) {
            return LLVMArray(0);
        } else {
            uint8_t value = 0;
            size_t byte_size = size * sizeof(Value);
            void *ptr = jitc_malloc(AllocType::Device, byte_size);
            jitc_fill(VarType::UInt8, ptr, byte_size, &value);
            return from_index(jitc_var_map(Type, ptr, size, 1));
        }
    }

    static LLVMArray full(Value value, size_t size) {
        if (size == 1) {
            return LLVMArray(value);
        } else {
            size_t byte_size = size * sizeof(Value);
            void *ptr = jitc_malloc(AllocType::Device, byte_size);
            jitc_fill(Type, ptr, size, &value);
            return from_index(jitc_var_map(Type, ptr, size, 1));
        }
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
        return jitc_var_ptr(m_index);
    }

    Value *data() {
        return jitc_var_ptr(m_index);
    }

    static LLVMArray from_index(uint32_t index) {
        LLVMArray result;
        result.m_index = index;
        return result;
    }

protected:
    uint32_t m_index = 0;
};


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
