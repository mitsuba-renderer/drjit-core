#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <ctime>
#include "internal.h"
#include "log.h"

static Buffer log_buffer;

void jit_log(LogLevel log_level, const char* fmt, ...) {
    if (log_level > state.log_level)
        return;

    va_list args;
    va_start(args, fmt);
    if (likely(!state.log_to_buffer)) {
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
    } else {
        log_buffer.vfmt(fmt, args);
        log_buffer.put("\n");
    }
    va_end(args);
}

char *jit_log_buffer() {
    char *value = strdup(log_buffer.get());
    log_buffer.clear();
    return value;
}

void jit_raise(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_buffer.vfmt(fmt, args);
    va_end(args);
    throw std::runtime_error(log_buffer.get());
}

void jit_fail(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "Critical failure in Enoki JIT compiler: ");
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
    exit(EXIT_FAILURE);
}

static char jit_string_buf[64];

const char *jit_mem_string(size_t size) {
    const char *orders[] = {
        "B", "KiB", "MiB", "GiB",
        "TiB", "PiB", "EiB"
    };
    float value = (float) size;

    int i = 0;
    for (i = 0; i < 6 && value > 1024.f; ++i)
        value /= 1024.f;

    snprintf(jit_string_buf, 64,
             i > 0 ? "%.3g %s" : "%.0f %s", value,
             orders[i]);

    return jit_string_buf;
}

const char *jit_time_string(float value) {
    struct Order { float factor; const char* suffix; };
    const Order orders[] = { { 0, "us" },   { 1000, "ms" },
                             { 1000, "s" }, { 60, "m" },
                             { 60, "h" },   { 24, "d" },
                             { 7, "w" },    { (float) 52.1429, "y" } };

    int i = 0;
    for (i = 0; i < 7 && value > orders[i+1].factor; ++i)
        value /= orders[i+1].factor;

    snprintf(jit_string_buf, 64, "%.5g %s", value, orders[i].suffix);

    return jit_string_buf;
}

Buffer::Buffer() : m_start(nullptr), m_cur(nullptr), m_end(nullptr) {
    const size_t size = 1024;
    m_start = (char *) malloc(size);
    if (unlikely(m_start == nullptr))
        jit_fail("Buffer(): out of memory!");
    m_end = m_start + size;
    clear();
}

size_t Buffer::fmt(const char *format, ...) {
    size_t written;
    do {
        size_t size = m_end - m_cur;
        va_list args;
        va_start(args, format);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);

    return written;
}

size_t Buffer::vfmt(const char *format, va_list args_) {
    size_t written;
    va_list args;
    do {
        size_t size = m_end - m_cur;
        va_copy(args, args_);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);
    return written;
}

void Buffer::expand() {
    size_t old_alloc_size = m_end - m_start,
           new_alloc_size = 2 * old_alloc_size,
           used_size      = m_cur - m_start,
           copy_size      = std::min(used_size + 1, old_alloc_size);

    char *tmp = (char *) malloc(new_alloc_size);
    if (unlikely(m_start == nullptr))
        jit_fail("Buffer::expand() out of memory!");
    memcpy(tmp, m_start, copy_size);
    free(m_start);

    m_start = tmp;
    m_end = m_start + new_alloc_size;
    m_cur = m_start + used_size;
}

#if defined(ENOKI_CUDA)
struct CUDAErrorList {
    CUresult id;
    const char *value;
};

static CUDAErrorList __cuda_error_list[] = {
    { CUDA_SUCCESS,
     "CUDA_SUCCESS"},
    { CUDA_ERROR_INVALID_VALUE,
     "CUDA_ERROR_INVALID_VALUE"},
    { CUDA_ERROR_OUT_OF_MEMORY,
     "CUDA_ERROR_OUT_OF_MEMORY"},
    { CUDA_ERROR_NOT_INITIALIZED,
     "CUDA_ERROR_NOT_INITIALIZED"},
    { CUDA_ERROR_DEINITIALIZED,
     "CUDA_ERROR_DEINITIALIZED"},
    { CUDA_ERROR_PROFILER_DISABLED,
     "CUDA_ERROR_PROFILER_DISABLED"},
    { CUDA_ERROR_PROFILER_NOT_INITIALIZED,
     "CUDA_ERROR_PROFILER_NOT_INITIALIZED"},
    { CUDA_ERROR_PROFILER_ALREADY_STARTED,
     "CUDA_ERROR_PROFILER_ALREADY_STARTED"},
    { CUDA_ERROR_PROFILER_ALREADY_STOPPED,
     "CUDA_ERROR_PROFILER_ALREADY_STOPPED"},
    { CUDA_ERROR_NO_DEVICE,
     "CUDA_ERROR_NO_DEVICE"},
    { CUDA_ERROR_INVALID_DEVICE,
     "CUDA_ERROR_INVALID_DEVICE"},
    { CUDA_ERROR_INVALID_IMAGE,
     "CUDA_ERROR_INVALID_IMAGE"},
    { CUDA_ERROR_INVALID_CONTEXT,
     "CUDA_ERROR_INVALID_CONTEXT"},
    { CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
     "CUDA_ERROR_CONTEXT_ALREADY_CURRENT"},
    { CUDA_ERROR_MAP_FAILED,
     "CUDA_ERROR_MAP_FAILED"},
    { CUDA_ERROR_UNMAP_FAILED,
     "CUDA_ERROR_UNMAP_FAILED"},
    { CUDA_ERROR_ARRAY_IS_MAPPED,
     "CUDA_ERROR_ARRAY_IS_MAPPED"},
    { CUDA_ERROR_ALREADY_MAPPED,
     "CUDA_ERROR_ALREADY_MAPPED"},
    { CUDA_ERROR_NO_BINARY_FOR_GPU,
     "CUDA_ERROR_NO_BINARY_FOR_GPU"},
    { CUDA_ERROR_ALREADY_ACQUIRED,
     "CUDA_ERROR_ALREADY_ACQUIRED"},
    { CUDA_ERROR_NOT_MAPPED,
     "CUDA_ERROR_NOT_MAPPED"},
    { CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
     "CUDA_ERROR_NOT_MAPPED_AS_ARRAY"},
    { CUDA_ERROR_NOT_MAPPED_AS_POINTER,
     "CUDA_ERROR_NOT_MAPPED_AS_POINTER"},
    { CUDA_ERROR_ECC_UNCORRECTABLE,
     "CUDA_ERROR_ECC_UNCORRECTABLE"},
    { CUDA_ERROR_UNSUPPORTED_LIMIT,
     "CUDA_ERROR_UNSUPPORTED_LIMIT"},
    { CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
     "CUDA_ERROR_CONTEXT_ALREADY_IN_USE"},
    { CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
     "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"},
    { CUDA_ERROR_INVALID_PTX,
     "CUDA_ERROR_INVALID_PTX"},
    { CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
     "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT"},
    { CUDA_ERROR_NVLINK_UNCORRECTABLE,
     "CUDA_ERROR_NVLINK_UNCORRECTABLE"},
    { CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
     "CUDA_ERROR_JIT_COMPILER_NOT_FOUND"},
    { CUDA_ERROR_INVALID_SOURCE,
     "CUDA_ERROR_INVALID_SOURCE"},
    { CUDA_ERROR_FILE_NOT_FOUND,
     "CUDA_ERROR_FILE_NOT_FOUND"},
    { CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
     "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"},
    { CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
     "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"},
    { CUDA_ERROR_OPERATING_SYSTEM,
     "CUDA_ERROR_OPERATING_SYSTEM"},
    { CUDA_ERROR_INVALID_HANDLE,
     "CUDA_ERROR_INVALID_HANDLE"},
    { CUDA_ERROR_NOT_FOUND,
     "CUDA_ERROR_NOT_FOUND"},
    { CUDA_ERROR_NOT_READY,
     "CUDA_ERROR_NOT_READY"},
    { CUDA_ERROR_ILLEGAL_ADDRESS,
     "CUDA_ERROR_ILLEGAL_ADDRESS"},
    { CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
     "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"},
    { CUDA_ERROR_LAUNCH_TIMEOUT,
     "CUDA_ERROR_LAUNCH_TIMEOUT"},
    { CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"},
    { CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
     "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"},
    { CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
     "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"},
    { CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
     "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"},
    { CUDA_ERROR_CONTEXT_IS_DESTROYED,
     "CUDA_ERROR_CONTEXT_IS_DESTROYED"},
    { CUDA_ERROR_ASSERT,
     "CUDA_ERROR_ASSERT"},
    { CUDA_ERROR_TOO_MANY_PEERS,
     "CUDA_ERROR_TOO_MANY_PEERS"},
    { CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
     "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"},
    { CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
     "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"},
    { CUDA_ERROR_HARDWARE_STACK_ERROR,
     "CUDA_ERROR_HARDWARE_STACK_ERROR"},
    { CUDA_ERROR_ILLEGAL_INSTRUCTION,
     "CUDA_ERROR_ILLEGAL_INSTRUCTION"},
    { CUDA_ERROR_MISALIGNED_ADDRESS,
     "CUDA_ERROR_MISALIGNED_ADDRESS"},
    { CUDA_ERROR_INVALID_ADDRESS_SPACE,
     "CUDA_ERROR_INVALID_ADDRESS_SPACE"},
    { CUDA_ERROR_INVALID_PC,
     "CUDA_ERROR_INVALID_PC"},
    { CUDA_ERROR_LAUNCH_FAILED,
     "CUDA_ERROR_LAUNCH_FAILED"},
    { CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
     "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE"},
    { CUDA_ERROR_NOT_PERMITTED,
     "CUDA_ERROR_NOT_PERMITTED"},
    { CUDA_ERROR_NOT_SUPPORTED,
     "CUDA_ERROR_NOT_SUPPORTED"},
    { CUDA_ERROR_UNKNOWN,
     "CUDA_ERROR_UNKNOWN"},
    { (CUresult) -1, nullptr }
};

static const char *cuda_error_string(CUresult id) {
    int index = 0;

    while (__cuda_error_list[index].id != id &&
           __cuda_error_list[index].id != (CUresult) -1)
        index++;

    if (__cuda_error_list[index].id == id)
        return __cuda_error_list[index].value;
    else
        return "Invalid CUDA error status!";
}

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (unlikely(errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED))
        jit_fail("cuda_check(): driver API error = %04d \"%s\" in "
                 "%s:%i.\n", (int) errval, cuda_error_string(errval), file, line);
}

void cuda_check_impl(cudaError_t errval, const char *file, const int line) {
    if (unlikely(errval != cudaSuccess && errval != cudaErrorCudartUnloading))
        jit_fail("cuda_check(): runtime API error = %04d \"%s\" in "
                 "%s:%i.\n", (int) errval, cudaGetErrorName(errval), file, line);
}

static timespec timer_value { 0, 0 };

float timer() {
    timespec timer_value_2;
    clock_gettime(CLOCK_REALTIME, &timer_value_2);
    float result = (timer_value_2.tv_sec - timer_value.tv_sec) * 1e6f +
                   (timer_value_2.tv_nsec - timer_value.tv_nsec) * 1e-3f;
    timer_value = timer_value_2;
    return result;
}

#endif
