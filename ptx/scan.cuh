#include "common.h"

inline __device__ uint4 scan_4(uint4 value, uint32_t size, uint32_t *sum_out) {
    uint32_t *shared = SharedMemory<uint32_t>::get();
    uint32_t si = threadIdx.x;

    uint4 sum = value;

    // Unrolled inclusive scan over 4 elements
    sum.y += sum.x;
    sum.z += sum.y;
    sum.w += sum.z;

    // Reduce using shared memory
    shared[si] = 0;
    si += size;
    shared[si] = sum.w;

    uint32_t sum4_inclusive = sum.w;
    for (uint32_t offset = 1; offset < size; offset <<= 1) {
        __syncthreads();
        sum4_inclusive = shared[si] + shared[si - offset];
        __syncthreads();
        shared[si] = sum4_inclusive;
    }

    uint32_t sum4_exclusive = sum4_inclusive - sum.w;

    // Convert back to exclusive scan
    sum.x  = sum4_exclusive;
    sum.y += sum4_exclusive - value.y;
    sum.z += sum4_exclusive - value.z;
    sum.w += sum4_exclusive - value.w;

    if (sum_out)
        *sum_out = sum4_inclusive;

    return sum;
}

KERNEL void scan_small(const uint32_t *in, uint32_t *out, uint32_t size) {
    uint32_t i = threadIdx.x * 4;

    uint4 value = make_uint4(
        i     < size ? in[i]     : 0u,
        i + 1 < size ? in[i + 1] : 0u,
        i + 2 < size ? in[i + 2] : 0u,
        i + 3 < size ? in[i + 3] : 0u
    );

    value = scan_4(value, blockDim.x, nullptr);

    if (i < size)
        out[i] = value.x;
    if (i + 1 < size)
        out[i + 1] = value.y;
    if (i + 2 < size)
        out[i + 2] = value.z;
    if (i + 3 < size)
        out[i + 3] = value.w;
}

KERNEL void scan_large(const uint32_t *in, uint32_t *out, uint32_t *block_sums) {
    uint32_t thread_count = 1024,
             i            = blockIdx.x * thread_count + threadIdx.x;

    uint4 value = ((const uint4 *) in)[i];

    uint32_t sum;
    value = scan_4(value, thread_count, &sum);

    ((uint4 *) out)[i] = value;

    if (threadIdx.x == thread_count - 1)
        block_sums[blockIdx.x] = sum;
}

KERNEL void scan_offset(uint32_t *data, const uint32_t *block_sums) {
    const uint32_t thread_count = 1024;

    uint32_t i      = blockIdx.x * thread_count + threadIdx.x,
             offset = block_sums[blockIdx.x];

    uint4 value = ((const uint4 *) data)[i];

    value.x += offset; value.y += offset;
    value.z += offset; value.w += offset;

    ((uint4 *) data)[i] = value;
}
