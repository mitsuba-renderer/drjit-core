#pragma once

#include <stdint.h>
#include <type_traits>
#include <limits>
#include <cuda_fp16.h>

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define FINLINE __forceinline__
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

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
