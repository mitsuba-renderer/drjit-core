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
#  include <arm_fp16.h>
#elif defined(__F16C__)
#  include <immintrin.h>
#endif

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)
template<> struct is_signed<drjit::half>            : std::true_type { };
template<> struct is_floating_point<drjit::half>    : std::true_type { };
template<> struct is_arithmetic<drjit::half>        : std::true_type { };
template<> struct is_scalar<drjit::half>            : std::true_type { };
NAMESPACE_END(detail)
NAMESPACE_END(drjit)

NAMESPACE_BEGIN(drjit)
struct half {
    uint16_t value;

    half() = default;
    half(const half &) = default;
    half(half &&) = default;
    half &operator=(const half &) = default;
    half &operator=(half &&) = default;

    template <typename Value,
              enable_if_t<drjit::detail::is_arithmetic_v<Value> &&
                          !std::is_same_v<Value, half>> = 0>
    half(Value val) : value(float32_to_float16((float) val)) {}

    half operator-() const { return from_binary(value ^ (uint16_t) 0x8000); }

    #define DRJIT_ARITH_T template <typename Value, enable_if_t<drjit::detail::is_arithmetic_v<Value>> = 0>
    DRJIT_ARITH_T half operator+(Value v) const { return half(operator float() + float(v)); }
    DRJIT_ARITH_T half operator-(Value v) const { return half(operator float() - float(v)); }
    DRJIT_ARITH_T half operator*(Value v) const { return half(operator float() * float(v)); }
    DRJIT_ARITH_T half operator/(Value v) const { return half(operator float() / float(v)); }

    DRJIT_ARITH_T half& operator+=(Value v) { return operator=(*this + v); }
    DRJIT_ARITH_T half& operator-=(Value v) { return operator=(*this - v); }
    DRJIT_ARITH_T half& operator*=(Value v) { return operator=(*this * v); }
    DRJIT_ARITH_T half& operator/=(Value v) { return operator=(*this / v); }

    DRJIT_ARITH_T bool operator==(Value v) const { return operator float() == (float) v; }
    DRJIT_ARITH_T bool operator!=(Value v) const { return operator float() != (float) v; }
    DRJIT_ARITH_T bool operator< (Value v) const { return operator float() <  (float) v; }
    DRJIT_ARITH_T bool operator> (Value v) const { return operator float() >  (float) v; }
    DRJIT_ARITH_T bool operator<=(Value v) const { return operator float() <= (float) v; }
    DRJIT_ARITH_T bool operator>=(Value v) const { return operator float() >= (float) v; }

    #undef DRJIT_ARITH_T

    operator float() const { return float16_to_float32(value); }

    static half from_binary(uint16_t value) { half h; h.value = value; return h; }
private:
    template <typename Out, typename In>
    static Out memcpy_cast(const In &src) {
        static_assert(sizeof(In) == sizeof(Out), "memcpy_cast: size mismatch!");
        Out dst;
        std::memcpy(&dst, &src, sizeof(Out));
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
            return float32_to_float16_fallback(value);
        #endif
    }

    static float float16_to_float32(uint16_t value) {
        #if defined(__F16C__)
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int32_t) value)));
        #elif defined(__aarch64__)
            return (float) memcpy_cast<__fp16>(value);
        #else
            return float16_to_float32_fallback(value);
        #endif
    }

    #if (!defined(__F16C__) && !defined(__aarch64__)) ||                           \
        defined(DRJIT_INCLUDE_FLOAT16_FALLBACK)
    /*
       The two functions below
           - ``float16_to_float32_fallback()``, and
           - ``float32_to_float16_fallback()``,
       are based on code by Paul A. Tessier (@Phernost). It is included with
       permission by the author, who released this code into the public domain.

       The Dr.Jit test suite compares the implementation against all possible
       half precision values (64K of them) and all single precision values (4B of
       them..). The implementation is equivalent to hardware rounding except for
       the conversion of NaN mantissa bits. This is arguably a quite minor point,
       and hopefully nobody relies on information there being preserved.
    */

    static float float16_to_float32_fallback(uint16_t value) {
        const uint32_t inf_h = 0x7c00, inf_f = 0x7f800000,
                       bit_diff = 16, sig_diff = 13;

        const float bias_mul = 0x1p+112f;

        uint32_t sign = value & (uint16_t) 0x8000,
                 bits = value ^ sign; // clear sign

        bool is_norm = bits < inf_h;
        bits = (sign << bit_diff) | (bits << sig_diff);
        bits = memcpy_cast<uint32_t>(memcpy_cast<float>(bits) * bias_mul);
        bits |= -!is_norm & inf_f;
        return memcpy_cast<float>(bits);
    }

    static uint16_t float32_to_float16_fallback(float value) {
      const uint32_t inf_h = 0x7c00, inf_f = 0x7f800000,
                     min_norm = 0x38800000,
                     sig_diff = 13, bit_diff = 16,
                     qnan_h = 0x7e00;

      const float sub_rnd = 0x1p-125f, sub_mul = 0x1p+13f,
                  bias_mul = 0x1p-112f;

      uint32_t bits = memcpy_cast<uint32_t>(value),
               sign = bits & (uint32_t) 0x80000000;

        bits ^= sign;
        bool is_nan = inf_f < bits, is_sub = bits < min_norm;

        float norm = memcpy_cast<float>(bits),
              subn = norm;

        subn *= sub_rnd; // round subnormals
        subn *= sub_mul; // correct subnormal exp
        norm *= bias_mul; // fix exp bias
        bits = memcpy_cast<uint32_t>(norm);
        bits += (bits >> sig_diff) & 1;              // add tie breaking bias
        bits += (uint32_t(1) << (sig_diff - 1)) - 1; // round up to half
        bits ^= -is_sub & (memcpy_cast<uint32_t>(subn) ^ bits);
        bits >>= sig_diff; // truncate
        bits ^= -(inf_h < bits) & (inf_h ^ bits); // fix overflow
        bits ^= -is_nan & (qnan_h ^ bits);
        bits |= sign >> bit_diff; // restore sign

        return (uint16_t) bits;
    }
#endif

    static half sqrt(half h) {
        #if defined(__aarch64__)
            return from_binary(
                memcpy_cast<uint16_t>(vsqrth_f16(memcpy_cast<__fp16>(h.value))));
        #else
            return half(std::sqrt((float) h));
        #endif
    }
};

NAMESPACE_END(drjit)
