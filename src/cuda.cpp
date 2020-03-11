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
void jit_cuda_fill_8(uint8_t *ptr, size_t size, uint8_t value) {
    (void) ptr; (void) size; (void) value;
}

/// Fill a device memory region with 'size' 16-bit values.
void jit_cuda_fill_16(uint16_t *ptr, size_t size, uint16_t value) {
    (void) ptr; (void) size; (void) value;
}

/// Fill a device memory region with 'size' 32-bit values.
void jit_cuda_fill_32(uint32_t *ptr, size_t size, uint32_t value) {
    (void) ptr; (void) size; (void) value;
}

/// Fill a device memory region with 'size' 64-bit values.
void jit_cuda_fill_64(uint64_t *ptr, size_t size, uint64_t value) {
    (void) ptr; (void) size; (void) value;
}

