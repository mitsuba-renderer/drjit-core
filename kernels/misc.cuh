#include "common.h"

KERNEL void fill_64(uint64_t *out, uint32_t size, uint64_t value) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        out[i] = value;
}

#define BLOCK_KERNEL(Suffix, Type)                                             \
    KERNEL void block_copy_##Suffix(const Type *in, uint32_t size,             \
                                    uint32_t block_size, Type *out) {          \
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;                  \
        if (idx >= size)                                                       \
            return;                                                            \
                                                                               \
        Type value = in[idx];                                                  \
        for (uint32_t i = 0; i < block_size; ++i)                              \
            out[idx * block_size + i] = value;                                 \
    }                                                                          \
                                                                               \
    KERNEL void block_sum_##Suffix(const Type *in, uint32_t size,              \
                                   uint32_t block_size, Type *out) {           \
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;                  \
        if (idx >= size)                                                       \
            return;                                                            \
                                                                               \
        Type value = 0;                                                        \
        for (uint32_t i = 0; i < block_size; ++i)                              \
            value += out[idx * block_size + i];                                \
                                                                               \
        out[idx] = value;                                                      \
    }

BLOCK_KERNEL(u32, uint32_t);
BLOCK_KERNEL(f32, float);
BLOCK_KERNEL(f64, double);
