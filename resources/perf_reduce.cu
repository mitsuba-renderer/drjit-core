#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include "block_reduce.cuh"

#define __global___FUNC extern "C" __global__

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

__global__ void fill(uint32_t *target, uint32_t size) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        target[i] = i/100;
}

__global__ void fill2(uint32_t *target, uint32_t size) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        target[i] = 1;
}

void cuda_check_impl(cudaError_t errval, const char *file, const int line) {
    if (errval != cudaSuccess && errval != cudaErrorCudartUnloading)
        fprintf(stderr, "cuda_check(): runtime API error = %04d \"%s\" in "
                 "%s:%i.\n", (int) errval, cudaGetErrorName(errval), file, line);
}

struct alignas(16) Vec4 {
    uint32_t data[4];

    template <typename Func> __device__ uint32_t reduce(Func f) {
        return f(f(data[0], data[1]), f(data[2], data[3]));
    }
};

__global__ void reduce_1(const uint32_t *in, uint32_t *out, uint32_t size, uint32_t block_size) {
    block_reduce<reduction_add<uint32_t>, uint32_t, uint32_t, 1024>(in, out, size, block_size);
}

__global__ void reduce_4(const uint32_t *in, uint32_t *out, uint32_t size, uint32_t block_size) {
    block_reduce<reduction_add<uint32_t>, uint32_t, Vec4, 256>(in, out, size, block_size);
}

void do_reduce(const uint32_t *in, uint32_t *out, uint32_t size, uint32_t block_size) {
    uint32_t threads = 1024, vec_size = 4;

    threads /= vec_size;
    size /= vec_size;
    block_size /= vec_size;

    int smem   = threads / 32 * sizeof(uint32_t),
        blocks = size / threads;

    printf("Launching %u blocks, %u threads, %u smem bytes\n", blocks, threads, smem);

    if (vec_size == 4)
        reduce_4<<<blocks, threads, smem>>>(in, out, size, block_size);
    else
        reduce_1<<<blocks, threads, smem>>>(in, out, size, block_size);
}


int main(int argc, char **argv) {
    size_t size = 1024*1024*1024;
    uint32_t *d_in, *d_out, *d_out_2, *d_out_3, *d_out_4, *d_tmp = nullptr;

    cuda_check(cudaMalloc(&d_in, size * sizeof(uint32_t)));
    cuda_check(cudaMalloc(&d_out, sizeof(uint32_t)));
    cuda_check(cudaMalloc(&d_out_2, sizeof(uint32_t)*(1024*1024)));
    cuda_check(cudaMalloc(&d_out_3, sizeof(uint32_t)*512));
    cuda_check(cudaMalloc(&d_out_4, sizeof(uint32_t)));
    fill<<<128,512>>>(d_in, size);

    size_t temp_size = 0;
    cub::DeviceReduce::Sum(d_tmp, temp_size, d_in, d_out, size);

    printf("Launching CUB\n");
    cuda_check(cudaMalloc(&d_tmp, temp_size));
    cub::DeviceReduce::Sum(d_tmp, temp_size, d_in, d_out, size);
    uint32_t out;
    cuda_check(cudaMemcpy(&out, d_out, 4, cudaMemcpyDefault));
    printf("%x\n", out);

    printf("Launching ours\n");
    do_reduce(d_in, d_out_2, 1024*1024*1024, 1024);
    do_reduce(d_out_2, d_out_3, 1024*1024, 1024);
    do_reduce(d_out_3, d_out_4, 1024, 1024);

    cuda_check(cudaMemcpy(&out, d_out_4, 4, cudaMemcpyDefault));
    printf("%x\n", out);

    return 0;
}
