#pragma once

#include <stdint.h>
#include <type_traits>
#include <limits>
#include <cuda_fp16.h>

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define FINLINE __forceinline__
#define WarpSize 32
#define WarpMask 0xffffffff

template <typename T> struct SharedMemory {
    __device__ inline static T *get() {
        extern __shared__ int shared[];
        return (T *) shared;
    }
};

template <> struct SharedMemory<double> {
    __device__ inline static double *get() {
        extern __shared__ double shared_d[];
        return shared_d;
    }
};


DEVICE float fma_(float a, float b, float c) { return __fmaf_rn(a, b, c); }
DEVICE double fma_(double a, double b, double c) { return __fma_rn(a, b, c); }
DEVICE half fma_(half a, half b, half c) {
    #if __CUDA_ARCH__ >= 600
        return __hfma(a, b, c);
    #else
        return __fma_rn((float) a, (float) b, (float) c);
    #endif
}

template <typename T> DEVICE T add_(T a, T b) { return a + b; }
DEVICE half add_(half a, half b) {
    #if __CUDA_ARCH__ >= 600
        return __hadd(a, b);
    #else
        return (half) ((float) a + (float) b);
    #endif
}

template <typename T> DEVICE T mul_(T a, T b) { return a * b; }
DEVICE half mul_(half a, half b) {
    #if __CUDA_ARCH__ >= 600
        return __hmul(a, b);
    #else
        return (half) ((float) a * (float) b);
    #endif
}

template <typename T> DEVICE T sub_(T a, T b) { return a - b; }
DEVICE half sub_(half a, half b) {
    #if __CUDA_ARCH__ >= 600
        return __hsub(a, b);
    #else
        return (half) ((float) a - (float) b);
    #endif
}

DEVICE float max_(float a, float b) { return fmaxf(a, b); }
DEVICE double max_(double a, double b) { return fmax(a, b); }
DEVICE uint32_t max_(uint32_t a, uint32_t b) { return umax(a, b); }
DEVICE uint64_t max_(uint64_t a, uint64_t b) { return ullmax(a, b); }
DEVICE int32_t max_(int32_t a, int32_t b) { return max(a, b); }
DEVICE int64_t max_(int64_t a, int64_t b) { return llmax(a, b); }
DEVICE half max_(half a, half b) {
    #if __CUDA_ARCH__ >= 800
        return __hmax(a, b);
    #else
        return (half) fmaxf((float) a, (float) b);
    #endif
}
DEVICE float min_(float a, float b) { return fminf(a, b); }
DEVICE double min_(double a, double b) { return fmin(a, b); }
DEVICE uint32_t min_(uint32_t a, uint32_t b) { return umin(a, b); }
DEVICE uint64_t min_(uint64_t a, uint64_t b) { return ullmin(a, b); }
DEVICE int32_t min_(int32_t a, int32_t b) { return min(a, b); }
DEVICE int64_t min_(int64_t a, int64_t b) { return llmin(a, b); }
DEVICE half min_(half a, half b) {
    #if __CUDA_ARCH__ >= 800
        return __hmin(a, b);
    #else
        return (half) fminf((float) a, (float) b);
    #endif
}

template <typename T> struct reduction_add {
    using Value = std::conditional_t<std::is_same<half, T>::value, float, T>;
    __device__ Value init() { return (Value) 0; }
    __device__ Value operator()(Value a, Value b) const {
        return add_(a, b);
    }
};

template <typename T> struct reduction_mul {
    using Value = std::conditional_t<std::is_same<half, T>::value, float, T>;
    __device__ Value init() { return (Value) 1; }
    __device__ Value operator()(Value a, Value b) const {
        return mul_(a, b);
    }
};

template <typename T> struct reduction_max {
    using Value = T;
    __device__ Value init() {
        return std::is_integral<Value>::value
                   ?  std::numeric_limits<Value>::min()
                   : -std::numeric_limits<Value>::infinity();
    }
    __device__ Value operator()(Value a, Value b) const {
        return max_(a, b);
    }
};

template <> struct reduction_max<half> {
    using Value = half;
    __device__ half init() { return __ushort_as_half((unsigned short) 0xFC00U); }
    __device__ half operator()(half a, half b) const {
        return max_(a, b);
    }
};

template <typename T> struct reduction_min {
    using Value = T;
    __device__ Value init() {
        return std::is_integral<Value>::value
                   ? std::numeric_limits<Value>::max()
                   : std::numeric_limits<Value>::infinity();
    }
    __device__ Value operator()(Value a, Value b) const {
        return min_(a, b);
    }
};

template <> struct reduction_min<half> {
    using Value = half;
    __device__ half init() { return __ushort_as_half((unsigned short) 0x7BFFU); }
    __device__ half operator()(half a, half b) const {
        return min_(a, b);
    }
};

template <typename T> struct reduction_or {
    using Value = T;
    __device__ Value init() { return (Value) 0; }
    __device__ Value operator()(Value a, Value b) const {
        return a | b;
    }
};

template <typename T> struct reduction_and {
    using Value = T;
    __device__ Value init() { return (Value) -1; }
    __device__ Value operator()(Value a, Value b) const {
        return a & b;
    }
};

template <size_t> struct uint_with_size;
template <> struct uint_with_size<2> { using type = uint16_t; };
template <> struct uint_with_size<4> { using type = uint32_t; };
template <> struct uint_with_size<8> { using type = uint64_t; };
template <size_t Size> using uint_with_size_t = typename uint_with_size<Size>::type;

template <typename Target, typename Source>
__device__ Target memcpy_cast(Source source) {
    static_assert(sizeof(Source) == sizeof(Target), "memcpy_cast: sizes must be identical!");
    Target target;
    memcpy(&target, &source, sizeof(Source));
    return target;
}

/// Helper routines to write tagged values while bypassing the L1 cache
__device__ void store_with_status(uint16_t *p, uint16_t value, uint32_t status) {
    uint32_t v = ((uint32_t) value) | (((uint32_t) status) << 16);

    asm("st.global.cg.u32 [%0], %1;"
        :
        : "l"(p)
          "r"(v)
        : "memory");
}

__device__ void store_with_status(uint32_t *p, uint32_t value, uint32_t status) {
    uint64_t v = ((uint64_t) value) | (((uint64_t) status) << 32);

    asm("st.global.cg.u64 [%0], %1;"
        :
        : "l"(p)
          "l"(v)
        : "memory");
}

__device__ void store_with_status(uint64_t *p, uint64_t value, uint32_t status) {
    uint64_t status_shift = ((uint64_t) status) << 32,
             v0 = uint32_t(value)  | status_shift,
             v1 = (value >> 32)    | status_shift;

    asm("st.global.cg.u64 [%0], %1;\n"
        "st.global.cg.u64 [%0 + 8], %2;"
        :
        : "l"(p)
          "l"(v0)
          "l"(v1)
        : "memory");
}

__device__ void load_with_status(uint16_t *p, uint16_t &value, uint32_t &status) {
    uint32_t v;

    asm("ld.global.cg.u32 %0, [%1];"
        : "=r"(v)
        : "l"(p)
        : "memory");

    value = (uint16_t) v;
    status = (uint32_t) (v >> 16);
}

__device__ void load_with_status(uint32_t *p, uint32_t &value, uint32_t &status) {
    uint64_t v;

    asm("ld.global.cg.u64 %0, [%1];"
        : "=l"(v)
        : "l"(p)
        : "memory");

    value = (uint32_t) v;
    status = (uint32_t) (v >> 32);
}

__device__ void load_with_status(uint64_t *p, uint64_t &value, uint32_t &status) {
    uint64_t v0, v1;

    asm("ld.global.cg.u64 %0, [%2];\n"
        "ld.global.cg.u64 %1, [%2 + 8];"
        : "=l"(v0)
          "=l"(v1)
        : "l"(p)
        : "memory");

    uint32_t v0_lo = (uint32_t) v0,
             v1_lo = (uint32_t) v1,
             v0_hi = (uint32_t) (v0 >> 32),
             v1_hi = (uint32_t) (v1 >> 32);

    value = (uint64_t) v0_lo + (((uint64_t) v1_lo) << 32);
    status = v0_hi == v1_hi ? v0_hi : 0;
}


// Vectorized types for 128 bit loads
template <typename T> struct Vec2 {
    T data[2];

    template <typename Func> __device__ T reduce(Func f) {
        return f(data[0], data[1]);
    }
};

template <typename T> struct Vec4 {
    T data[4];

    template <typename Func> __device__ T reduce(Func f) {
        return f(f(data[0], data[1]), f(data[2], data[3]));
    }
};

template <typename T> struct Vec8 {
    T data[8];

    template <typename Func> __device__ T reduce(Func f) {
        return f(f(f(data[0], data[1]),
                   f(data[2], data[3])),
                 f(f(data[4], data[5]),
                   f(data[6], data[7])));
    }
};

// Precomputed integer division helper
// Based on libidivide (https://github.com/ridiculousfish/libdivide)
struct divisor {
    uint32_t magic;
    uint32_t shift;
    uint32_t value;

    static __device__ uint32_t mulhi(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ void div_rem(uint32_t input, uint32_t *out_div, uint32_t *out_rem) const {
        uint32_t div = 0, rem = 0;

        if (!magic) {
            div = input >> shift;
            rem = input - (div << shift);
        } else {
            uint32_t hi = mulhi(input, magic);
            div = (((input - hi) >> 1) + hi) >> shift;
            rem = input - div * value;
        }

        *out_div = div;
        *out_rem = rem;
    }
};
