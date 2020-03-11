#include <stdint.h>

#define KERNEL_FUNC extern "C" __global__

KERNEL_FUNC void fill_64(uint64_t *ptr, size_t size, uint64_t value) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        ptr[i] = value;
}

KERNEL_FUNC void cuda_hsum(const float *ptr, size_t size, float *out) {
    float value = 0.f;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        value += ptr[i];

    atomicAdd(out, value);
}
