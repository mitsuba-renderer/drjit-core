#pragma once

#include "api.h"
#include "cuda.h"

/// Log levels for jit_log()
enum LogLevel : uint32_t {
    Error = 0,
    Warn  = 1,
    Info  = 2,
    Debug = 3,
    Trace = 4
};

/// Print a log message with the specified log level and message
extern void jit_log(LogLevel level, const char* fmt, ...);

/// Raise a std::runtime_error with the given message
[[noreturn]] extern void jit_raise(const char* fmt, ...);

/// Immediately terminate the application due to a fatal internal error
[[noreturn]] extern void jit_fail(const char* fmt, ...);

/// Convert a number of bytes into a human-readable string (returns static buffer!)
extern const char *jit_mem_string(size_t size);

#if defined(ENOKI_CUDA)
    #define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
    ENOKI_EXPORT extern void cuda_check_impl(CUresult errval, const char *file, const int line);
    ENOKI_EXPORT extern void cuda_check_impl(cudaError_t errval, const char *file, const int line);
#endif
