#include "internal.h"
#include "log.h"

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (unlikely(errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED)) {
        const char *msg = nullptr;
        cuGetErrorString(errval, &msg);
        jit_fail("cuda_check(): driver API error = %04d \"%s\" in "
                 "%s:%i.\n", (int) errval, msg, file, line);
    }
}

void cuda_check_impl(cudaError_t errval, const char *file, const int line) {
    if (unlikely(errval != cudaSuccess && errval != cudaErrorCudartUnloading))
        jit_fail("cuda_check(): runtime API error = %04d \"%s\" in "
                 "%s:%i.\n", (int) errval, cudaGetErrorName(errval), file, line);
}

/// Fill a device memory region with 'size' 8-bit values.
void jit_cuda_fill_8(void *ptr, size_t size, uint8_t value) {
    Stream *stream = jit_get_stream("jit_cuda_fill_8");
    cuda_check(cuMemsetD8Async(ptr, value, size, stream->handle));
}

/// Fill a device memory region with 'size' 16-bit values.
void jit_cuda_fill_16(void *ptr, size_t size, uint16_t value) {
    Stream *stream = jit_get_stream("jit_cuda_fill_16");
    cuda_check(cuMemsetD16Async(ptr, value, size, stream->handle));
}

/// Fill a device memory region with 'size' 32-bit values.
void jit_cuda_fill_32(void *ptr, size_t size, uint32_t value) {
    Stream *stream = jit_get_stream("jit_cuda_fill_32");
    cuda_check(cuMemsetD32Async(ptr, value, size, stream->handle));
}

/// Fill a device memory region with 'size' 64-bit values.
void jit_cuda_fill_64(void *ptr, size_t size, uint64_t value) {
    Stream *stream = jit_get_stream("jit_cuda_fill_64");
    int num_sm = state.devices[stream->device].num_sm;
    void *args[] = { &ptr, &size, &value };
    cuda_check(cuLaunchKernel(kernel_fill_64, num_sm, 1, 1, 1024,
                              1, 1, 0, stream->handle, args, nullptr));
}
