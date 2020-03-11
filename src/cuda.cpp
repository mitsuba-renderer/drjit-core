#include "internal.h"
#include "log.h"

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (unlikely(errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED)) {
        const char *msg = nullptr;
        cuGetErrorString(errval, &msg);
        jit_fail("cuda_check(): API error = %04d (\"%s\") in "
                 "%s:%i.", (int) errval, msg, file, line);
    }
}

/// Fill a device memory region with 'size' 8-bit values.
void jit_fill_8(void *ptr, size_t size, uint8_t value) {
    Stream *stream = active_stream;
    if (stream) {
        cuda_check(cuMemsetD8Async(ptr, value, size, stream->handle));
    } else {
        uint8_t *p = (uint8_t *) ptr;
        for (size_t i = 0; i < size; ++i)
            p[i] = value;
    }
}

/// Fill a device memory region with 'size' 16-bit values.
void jit_fill_16(void *ptr, size_t size, uint16_t value) {
    Stream *stream = active_stream;
    if (stream) {
        cuda_check(cuMemsetD16Async(ptr, value, size, stream->handle));
    } else {
        uint16_t *p = (uint16_t *) ptr;
        for (size_t i = 0; i < size; ++i)
            p[i] = value;
    }
}

/// Fill a device memory region with 'size' 32-bit values.
void jit_fill_32(void *ptr, size_t size, uint32_t value) {
    Stream *stream = active_stream;
    if (stream) {
        cuda_check(cuMemsetD32Async(ptr, value, size, stream->handle));
    } else {
        uint32_t *p = (uint32_t *) ptr;
        for (size_t i = 0; i < size; ++i)
            p[i] = value;
    }
}

/// Fill a device memory region with 'size' 64-bit values.
void jit_fill_64(void *ptr, size_t size, uint64_t value) {
    Stream *stream = active_stream;
    if (stream) {
        int num_sm = state.devices[stream->device].num_sm;
        void *args[] = { &ptr, &size, &value };
        cuda_check(cuLaunchKernel(kernel_fill_64, num_sm, 1, 1, 1024,
                                  1, 1, 0, stream->handle, args, nullptr));
    } else {
        uint64_t *p = (uint64_t *) ptr;
        for (size_t i = 0; i < size; ++i)
            p[i] = value;
    }
}
