#pragma once

#include <stdint.h>
#include <type_traits>
#include <limits>

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
