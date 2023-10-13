/*
    drjit-core/half.h -- minimal half precision number type

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

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

    dr_half()
    #if !defined(NDEBUG)
        : value(0x7FFF) /* Initialize with NaN */
    #endif
    { }

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
    /*
       Value float32<->float16 conversion code by Paul A. Tessier (@Phernost)
       Used with permission by the author, who released this code into the public domain
     */
    union Bits {
        float f;
        int32_t si;
        uint32_t ui;
    };

    static constexpr int const shift = 13;
    static constexpr int const shiftSign = 16;

    static constexpr int32_t const infN = 0x7F800000;  // flt32 infinity
    static constexpr int32_t const maxN = 0x477FE000;  // max flt16 normal as a flt32
    static constexpr int32_t const minN = 0x38800000;  // min flt16 normal as a flt32
    static constexpr int32_t const signN = (int32_t) 0x80000000; // flt32 sign bit

    static constexpr int32_t const infC = infN >> shift;
    static constexpr int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
    static constexpr int32_t const maxC = maxN >> shift;
    static constexpr int32_t const minC = minN >> shift;
    static constexpr int32_t const signC = signN >> shiftSign; // flt16 sign bit

    static constexpr int32_t const mulN = 0x52000000; // (1 << 23) / minN
    static constexpr int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

    static constexpr int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
    static constexpr int32_t const norC = 0x00400; // min flt32 normal down shifted

    static constexpr int32_t const maxD = infC - maxC - 1;
    static constexpr int32_t const minD = minC - subC - 1;

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
            Bits v, s;
            v.f = value;
            uint32_t sign = (uint32_t) (v.si & signN);
            v.si ^= sign;
            sign >>= shiftSign; // logical shift
            s.si = mulN;
            s.si = (int32_t) (s.f * v.f); // correct subnormals
            v.si ^= (s.si ^ v.si) & -(minN > v.si);
            v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
            v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
            v.ui >>= shift; // logical shift
            v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
            v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
            return (uint16_t) (v.ui | sign);
        #endif
    }

    static float float16_to_float32(uint16_t value) {
        #if defined(__F16C__)
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int32_t) value)));
        #elif defined(__aarch64__)
            return (float)memcpy_cast<__fp16>(value);
        #else
            Bits v;
            v.ui = value;
            int32_t sign = v.si & signC;
            v.si ^= sign;
            sign <<= shiftSign;
            v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
            v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
            Bits s;
            s.si = mulC;
            s.f *= float(v.si);
            int32_t mask = -(norC > v.si);
            v.si <<= shift;
            v.si ^= (s.si ^ v.si) & mask;
            v.si |= sign;
            return v.f;
        #endif
    }

    static dr_half sqrt(dr_half h) { 
        #if defined(__aarch64__)
            return (float)vsqrth_f16(memcpy_cast<__fp16>(h.value));
        #else
            return std::sqrt((float)h);
        #endif
    }

    static dr_half make_signed(dr_half h) {
        return h;
    }
};

NAMESPACE_END(drjit)