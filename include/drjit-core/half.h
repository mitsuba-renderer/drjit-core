/*
    drjit-core/half.h -- minimal half precision number type

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <drjit-core/traits.h>

#include <cmath>
#include <cstring>

#if defined __aarch64__
#include <arm_fp16.h>
#elif defined(__F16C__)
#include <immintrin.h>
#endif


NAMESPACE_BEGIN(drjit)
struct half;

template<> struct is_signed<half>                   : std::true_type { };
template<> struct is_floating_point<drjit::half>    : std::true_type { };
template<> struct is_arithmetic<drjit::half>        : std::true_type { };
template<> struct is_scalar<drjit::half>            : std::true_type { };
NAMESPACE_END(drjit)


NAMESPACE_BEGIN(drjit)
struct half {

    uint16_t value = 0;

    #define DRJIT_IF_INT template <typename Value, enable_if_t<std::is_integral_v<Value>> = 0>

    half() = default;

    DRJIT_IF_INT half(Value val) : value(float32_to_float16((float)val)) { }

    explicit half(float val)    : value(float32_to_float16(val)) {}
    explicit half(double val)   : value(float32_to_float16((float)val)) {}

    half operator+(half h) const { return half(float(*this) + float(h)); }
    half operator-(half h) const { return half(float(*this) - float(h)); }
    half operator*(half h) const { return half(float(*this) * float(h)); }
    half operator/(half h) const { return half(float(*this) / float(h)); }

    half operator-() const { return half(-float(*this)); }

    half& operator+=(half h) { return operator=(*this + h); }
    half& operator-=(half h) { return operator=(*this - h); }
    half& operator*=(half h) { return operator=(*this * h); }
    half& operator/=(half h) { return operator=(*this / h); }

    bool operator==(half h) const { return float(*this) == float(h); }
    bool operator!=(half h) const { return float(*this) != float(h); }
    bool operator<(half h) const  { return float(*this) < float(h); }
    bool operator>(half h) const  { return float(*this) > float(h); }
    bool operator<=(half h) const { return float(*this) <= float(h); }
    bool operator>=(half h) const { return float(*this) >= float(h); }

    DRJIT_IF_INT bool operator==(Value val) const { return float(*this) == float(val); }
    DRJIT_IF_INT bool operator!=(Value val) const { return float(*this) != float(val); }
    DRJIT_IF_INT bool operator<(Value val) const  { return float(*this) < float(val); }
    DRJIT_IF_INT bool operator>(Value val) const  { return float(*this) > float(val); }
    DRJIT_IF_INT bool operator<=(Value val) const { return float(*this) <= float(val); }
    DRJIT_IF_INT bool operator>=(Value val) const { return float(*this) >= float(val); }

    operator float() const { return float16_to_float32(value); }

    static constexpr half from_binary(uint16_t value) { half h; h.value = value; return h; }

    #undef DRJIT_IF_INT
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
            jit_fail("Unsupported architecture");
        #endif
    }

    static float float16_to_float32(uint16_t value) {
        #if defined(__F16C__)
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int32_t) value)));
        #elif defined(__aarch64__)
            return (float) memcpy_cast<__fp16>(value);
        #else
            jit_fail("Unsupported architecture");
        #endif
    }

    static half sqrt(half h) { 
        #if defined(__aarch64__)
            return from_binary(memcpy_cast<uint16_t>(vsqrth_f16(memcpy_cast<__fp16>(h.value))));
        #else
            return half(std::sqrt((float) h));
        #endif
    }
};

NAMESPACE_END(drjit)