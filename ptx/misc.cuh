#include "common.h"

KERNEL void fill_64(uint64_t *ptr, size_t size, uint64_t value) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        ptr[i] = value;
}

