#include <stdint.h>
#include "cuda.h"

template <typename T> __global__ void fill(T *out, T value, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        out[i] = value;
}

void jit_cuda_fill_8(uint8_t *ptr, size_t size, uint8_t value) {
    cudaStream_t stream;
    int num_sm;
    jit_cuda_get_config(&stream, &num_sm);
    (void) num_sm;
    cuda_check(cudaMemsetAsync(ptr, value, size, stream));
}

void jit_cuda_fill_16(uint16_t *ptr, size_t size, uint16_t value) {
    cudaStream_t stream;
    int num_sm;
    jit_cuda_get_config(&stream, &num_sm);
    fill<<<num_sm, 1024, 0, stream>>>(ptr, value, size);
}

void jit_cuda_fill_32(uint32_t *ptr, size_t size, uint32_t value) {
    cudaStream_t stream;
    int num_sm;
    jit_cuda_get_config(&stream, &num_sm);
    fill<<<num_sm, 1024, 0, stream>>>(ptr, value, size);
}

void jit_cuda_fill_64(uint64_t *ptr, size_t size, uint64_t value) {
    cudaStream_t stream;
    int num_sm;
    jit_cuda_get_config(&stream, &num_sm);
    fill<<<num_sm, 1024, 0, stream>>>(ptr, value, size);
}
