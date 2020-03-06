#pragma once

#include "api.h"
#include "cuda.h"

/// Log levels for jit_log()
enum LogLevel {
    Error = 0,
    Warn  = 1,
    Info  = 2,
    Debug = 3,
    Trace = 4
};

/// Print a log message with the specified log level and message
extern ENOKI_EXPORT void jit_log(LogLevel level, const char* format, ...);

/// Raise a std::runtime_error with the given message
extern ENOKI_EXPORT void jit_raise(const char* format, ...);

/// Immediately terminate the application due to a fatal internal error
[[noreturn]] extern ENOKI_EXPORT void jit_fail(const char* format, ...);

#if defined(ENOKI_CUDA)
    #define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
    ENOKI_EXPORT extern void cuda_check_impl(CUresult errval, const char *file, const int line);
    ENOKI_EXPORT extern void cuda_check_impl(cudaError_t errval, const char *file, const int line);
#endif
