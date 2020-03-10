#pragma once
#if defined(ENOKI_CUDA)

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_runtime_api.h>

#if !defined(likely)
#  define likely(x)   __builtin_expect(!!(x), 1)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/// Fill a device memory region with 'size' 8-bit values.
extern void jit_cuda_fill_8(uint8_t *ptr, size_t size, uint8_t value);
/// Fill a device memory region with 'size' 16-bit values.
extern void jit_cuda_fill_16(uint16_t *ptr, size_t size, uint16_t value);
/// Fill a device memory region with 'size' 32-bit values.
extern void jit_cuda_fill_32(uint32_t *ptr, size_t size, uint32_t value);
/// Fill a device memory region with 'size' 64-bit values.
extern void jit_cuda_fill_64(uint64_t *ptr, size_t size, uint64_t value);

/// Query the launch context (CUDA stream and number of SMs of the target device)
extern void jit_cuda_get_config(cudaStream_t *stream_out, int *num_sm_out);

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
extern void cuda_check_impl(CUresult errval, const char *file, const int line);
extern void cuda_check_impl(cudaError_t errval, const char *file, const int line);

#endif
