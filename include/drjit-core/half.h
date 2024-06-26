/*
    drjit-core/half.h -- minimal half precision number type

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/nanostl.h>
#include <drjit-core/traits.h>
#include <drjit-core/intrin.h>
#include <cstdint>
#include <cstring>

NAMESPACE_BEGIN(drjit)

struct half {
    uint16_t value = 0;

    half() = default;
    half(const half &) = default;
    half(half &&) = default;
    half &operator=(const half &) = default;
    half &operator=(half &&) = default;

    template <typename Value, enable_if_t<std::is_arithmetic_v<Value> &&
                                          !std::is_same_v<Value, half>> = 0>
    half(Value val) : value(float32_to_float16((float) val)) {}

    half operator-() const { return from_binary(value ^ (uint16_t) 0x8000); }

    #define DRJIT_ARITH_T template <typename Value, enable_if_t<detail::is_arithmetic_v<Value>> = 0>
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

    static constexpr half from_binary(uint16_t value) { half h; h.value = value; return h; }
private:
    template <typename Dst, typename Src>
    static Dst bitcast(const Src &src) {
        static_assert(sizeof(Src) == sizeof(Dst),
                      "bitcast: size mismatch!");
        Dst dst;
        memcpy((void *) &dst, &src, sizeof(Dst));
        return dst;
    }

    static uint16_t float32_to_float16(float value) {
        #if defined(__F16C__)
            return (uint16_t) _mm_cvtsi128_si32(
                _mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_CUR_DIRECTION));
        #elif defined(__aarch64__)
            return bitcast<uint16_t>((__fp16) value);
        #else
            return float32_to_float16_fallback(value);
        #endif
    }

    static float float16_to_float32(uint16_t value) {
        #if defined(__F16C__)
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int32_t) value)));
        #elif defined(__aarch64__)
            return (float) bitcast<__fp16>(value);
        #else
            return float16_to_float32_fallback(value);
        #endif
    }

    static half sqrt(half h) {
        #if defined(__aarch64__)
            return from_binary(
                bitcast<uint16_t>(vsqrth_f16(bitcast<__fp16>(h.value))));
        #elif defined(__GNUC__)
            return half(__builtin_sqrtf((float) h));
        #else
            return half(sqrtf((float) h));
        #endif
    }

#if (!defined(__F16C__) && !defined(__aarch64__)) ||                           \
    defined(DRJIT_INCLUDE_FLOAT16_FALLBACK)
public:
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
        bits = bitcast<uint32_t>(bitcast<float>(bits) * bias_mul);
        bits |= -!is_norm & inf_f;
        return bitcast<float>(bits);
    }

    static uint16_t float32_to_float16_fallback(float value) {
      const uint32_t inf_h = 0x7c00, inf_f = 0x7f800000,
                     min_norm = 0x38800000,
                     sig_diff = 13, bit_diff = 16,
                     qnan_h = 0x7e00;

      const float sub_rnd = 0x1p-125f, sub_mul = 0x1p+13f,
                  bias_mul = 0x1p-112f;

      uint32_t bits = bitcast<uint32_t>(value),
               sign = bits & (uint32_t) 0x80000000;

        bits ^= sign;
        bool is_nan = inf_f < bits, is_sub = bits < min_norm;

        float norm = bitcast<float>(bits),
              subn = norm;

        subn *= sub_rnd; // round subnormals
        subn *= sub_mul; // correct subnormal exp
        norm *= bias_mul; // fix exp bias
        bits = bitcast<uint32_t>(norm);
        bits += (bits >> sig_diff) & 1;              // add tie breaking bias
        bits += (uint32_t(1) << (sig_diff - 1)) - 1; // round up to half
        bits ^= -(int32_t)is_sub & (bitcast<uint32_t>(subn) ^ bits);
        bits >>= sig_diff; // truncate
        bits ^= -(inf_h < bits) & (inf_h ^ bits); // fix overflow
        bits ^= -(int32_t)is_nan & (qnan_h ^ bits);
        bits |= sign >> bit_diff; // restore sign

        return (uint16_t) bits;
    }
#endif
};

NAMESPACE_BEGIN(detail)

template<> struct is_signed<drjit::half>         : std::true_type { };
template<> struct is_unsigned<drjit::half>       : std::false_type { };
template<> struct is_floating_point<drjit::half> : std::true_type { };
template<> struct is_arithmetic<drjit::half>     : std::true_type { };
template<> struct is_scalar<drjit::half>         : std::true_type { };

template <typename T> struct constants;
template <typename T> struct debug_init;

template <> struct constants<half> {
    static constexpr half E               = half::from_binary(0x4170);
    static constexpr half LogTwo          = half::from_binary(0x398c);
    static constexpr half InvLogTwo       = half::from_binary(0x3dc5);

    static constexpr half Pi              = half::from_binary(0x4248);
    static constexpr half InvPi           = half::from_binary(0x3518);
    static constexpr half SqrtPi          = half::from_binary(0x3f17);
    static constexpr half InvSqrtPi       = half::from_binary(0x3883);

    static constexpr half TwoPi           = half::from_binary(0x4648);
    static constexpr half InvTwoPi        = half::from_binary(0x3118);
    static constexpr half SqrtTwoPi       = half::from_binary(0x4103);
    static constexpr half InvSqrtTwoPi    = half::from_binary(0x3662);

    static constexpr half FourPi          = half::from_binary(0x4a48);
    static constexpr half InvFourPi       = half::from_binary(0x2d18);
    static constexpr half SqrtFourPi      = half::from_binary(0x4317);
    static constexpr half InvSqrtFourPi   = half::from_binary(0x3483);

    static constexpr half SqrtTwo         = half::from_binary(0x3da8);
    static constexpr half InvSqrtTwo      = half::from_binary(0x39a8);

    static constexpr half Infinity        = half::from_binary(0xfc00);
    static constexpr half NaN             = half::from_binary(0xffff);
};

template <> struct debug_init<half> {
    static constexpr half value = constants<half>::NaN;
};

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
