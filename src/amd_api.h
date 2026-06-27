/*
    src/amd_api.h -- Low-level helpers for the AMD/HIP backend.
*/

#pragma once

#include <drjit-core/jit.h>

#if defined(DRJIT_ENABLE_AMD)

#  include <hip/hip_runtime.h>
#  include <hip/hiprtc.h>

extern int jitc_amd_runtime_version_v;

extern void jitc_amd_check(hipError_t result, const char *expr,
                           const char *file, int line);

extern void jitc_amd_rtc_check(hiprtcResult result, const char *expr,
                               const char *file, int line);

extern void jitc_amd_sync_stream(uintptr_t stream);

#  define hip_check(expr)                                                    \
    do {                                                                     \
        jitc_amd_check((expr), #expr, __FILE__, __LINE__);                   \
    } while (0)

#  define hiprtc_check(expr)                                                 \
    do {                                                                     \
        jitc_amd_rtc_check((expr), #expr, __FILE__, __LINE__);               \
    } while (0)

#endif
