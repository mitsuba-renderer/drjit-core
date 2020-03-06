#pragma once

#include "api.h"
#include "cuda.h"

// ====================================================================
//                              Logging
// ====================================================================

enum LogLevel { Trace, Debug, Info, Warn, Error };
extern ENOKI_EXPORT void jit_log(LogLevel level, const char* format, ...);
extern ENOKI_EXPORT void jit_raise(const char* format, ...);
[[noreturn]] extern ENOKI_EXPORT void jit_fail(const char* format, ...);

#if defined(ENOKI_CUDA)
    #define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
    ENOKI_EXPORT extern void cuda_check_impl(CUresult errval, const char *file, const int line);
    ENOKI_EXPORT extern void cuda_check_impl(cudaError_t errval, const char *file, const int line);
#endif
