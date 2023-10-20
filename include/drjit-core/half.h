/*
    drjit-core/half.h -- minimal half precision number type

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <drjit-core/traits.h>
#include <drjit-core/array.h>

#include <cmath>
#include <cstring>

#if defined __aarch64__
#include <arm_fp16.h>
#elif defined(__F16C__)
#include <emmintrin.h>
#include <immintrin.h>
#endif


NAMESPACE_BEGIN(drjit)
struct dr_half;
NAMESPACE_END(drjit)

NAMESPACE_BEGIN(std)
template<> struct is_floating_point<drjit::dr_half> : true_type { };
template<> struct is_arithmetic<drjit::dr_half> : true_type { };
template<> struct is_signed<drjit::dr_half> : true_type { };
NAMESPACE_END(std)


NAMESPACE_BEGIN(drjit)
struct dr_half {

    uint16_t value;

    #define DRJIT_IF_SCALAR template <typename Value, enable_if_t<std::is_arithmetic<Value>::value> = 0>

    dr_half() = default;

    DRJIT_IF_SCALAR dr_half(Value val) : value(float32_to_float16(float(val))) { }

    dr_half operator+(dr_half h) const { return dr_half(float(*this) + float(h)); }
    dr_half operator-(dr_half h) const { return dr_half(float(*this) - float(h)); }
    dr_half operator*(dr_half h) const { return dr_half(float(*this) * float(h)); }
    dr_half operator/(dr_half h) const { return dr_half(float(*this) / float(h)); }

    dr_half operator-() const { return dr_half(-float(*this)); }

    DRJIT_IF_SCALAR friend dr_half operator+(Value val, dr_half h) { return dr_half(val) + h; }
    DRJIT_IF_SCALAR friend dr_half operator-(Value val, dr_half h) { return dr_half(val) - h; }
    DRJIT_IF_SCALAR friend dr_half operator*(Value val, dr_half h) { return dr_half(val) * h; }
    DRJIT_IF_SCALAR friend dr_half operator/(Value val, dr_half h) { return dr_half(val) / h; }

    dr_half& operator+=(dr_half h) { return operator=(*this + h); }
    dr_half& operator-=(dr_half h) { return operator=(*this - h); }
    dr_half& operator*=(dr_half h) { return operator=(*this * h); }
    dr_half& operator/=(dr_half h) { return operator=(*this / h); }

    bool operator==(dr_half h) const { return float(*this) == float(h); }
    bool operator!=(dr_half h) const { return float(*this) != float(h); }
    bool operator<(dr_half h) const  { return float(*this) < float(h); }
    bool operator>(dr_half h) const  { return float(*this) > float(h); }
    bool operator<=(dr_half h) const { return float(*this) <= float(h); }
    bool operator>=(dr_half h) const { return float(*this) >= float(h); }

    operator float() const { return float16_to_float32(value); }

    static dr_half from_binary(uint16_t value) { dr_half h; h.value = value; return h; }

    #undef DRJIT_IF_SCALAR
private:
    template <typename Dst, typename Src>
    static Dst memcpy_cast(const Src &src) {
        static_assert(sizeof(Src) == sizeof(Dst), "memcpy_cast: size mismatch!");
        Dst dst;
        std::memcpy(&dst, &src, sizeof(Dst));
        return dst;
    }

public:

    static uint16_t float32_to_float16(float value) {
        #if defined(__F16C__)
            return (uint16_t) _mm_cvtsi128_si32(
                _mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_CUR_DIRECTION));
        #elif defined(__aarch64__)
            return memcpy_cast<uint16_t>((__fp16) value);
        #else
            jitc_fail("Unsupported architecture");
        #endif
    }

    static float float16_to_float32(uint16_t value) {
        #if defined(__F16C__)
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int32_t) value)));
        #elif defined(__aarch64__)
            return (float)memcpy_cast<__fp16>(value);
        #else
            jitc_fail("Unsupported architecture");
        #endif
    }

    static dr_half sqrt(dr_half h) { 
        #if defined(__aarch64__)
            return (float)vsqrth_f16(memcpy_cast<__fp16>(h.value));
        #else
            return std::sqrt((float)h);
        #endif
    }
};

NAMESPACE_END(drjit)