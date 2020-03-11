#include <stdint.h>

__global__ void fill_64(uint64_t *out, uint64_t value, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        out[i] = value;
}
