#pragma once

#include <enoki/jit.h>
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

/// Return and clear the log buffer
extern char *jit_log_buffer();

/// Convert a number of bytes into a human-readable string (returns static buffer!)
extern const char *jit_mem_string(size_t size);

/// Convert a time in microseconds into a human-readable string (returns static buffer!)
extern const char *jit_time_string(float us);

#if defined(ENOKI_CUDA)
    #define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
    extern void cuda_check_impl(CUresult errval, const char *file, const int line);
    extern void cuda_check_impl(cudaError_t errval, const char *file, const int line);
#endif

/// Return the number of microseconds since the previous timer() call
extern float timer();
