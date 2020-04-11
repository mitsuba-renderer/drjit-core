/*
    src/cuda_api.cpp -- Low-level interface to CUDA driver API

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "util.h"
#include "io.h"
#include "../kernels/kernels.h"
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif
#include <lz4.h>

#if defined(ENOKI_DYNAMIC_CUDA)
// Driver API
CUresult (*cuCtxEnablePeerAccess)(CUcontext, unsigned int) = nullptr;
CUresult (*cuCtxSynchronize)() = nullptr;
CUresult (*cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice) = nullptr;
CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
CUresult (*cuDeviceGetCount)(int *) = nullptr;
CUresult (*cuDeviceGetName)(char *, int, CUdevice) = nullptr;
CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
CUresult (*cuDeviceTotalMem)(size_t *, CUdevice) = nullptr;
CUresult (*cuDriverGetVersion)(int *) = nullptr;
CUresult (*cuEventCreate)(CUevent *, unsigned int) = nullptr;
CUresult (*cuEventDestroy)(CUevent) = nullptr;
CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
CUresult (*cuEventSynchronize)(CUevent) = nullptr;
CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;
CUresult (*cuGetErrorName)(CUresult, const char **) = nullptr;
CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;
CUresult (*cuInit)(unsigned int) = nullptr;
CUresult (*cuLaunchHostFunc)(CUstream, void (*)(void *), void *) = nullptr;
CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, CUstream, void **,
                           void **) = nullptr;
CUresult (*cuLinkAddData)(CUlinkState, int, void *, size_t,
                          const char *, unsigned int, int *,
                          void **) = nullptr;
CUresult (*cuLinkComplete)(CUlinkState, void **, size_t *) = nullptr;
CUresult (*cuLinkCreate)(unsigned int, int *, void **,
                         CUlinkState *) = nullptr;
CUresult (*cuLinkDestroy)(CUlinkState) = nullptr;
CUresult (*cuMemAdvise)(void *, size_t, int, CUdevice) = nullptr;
CUresult (*cuMemAlloc)(void **, size_t) = nullptr;
CUresult (*cuMemAllocHost)(void **, size_t) = nullptr;
CUresult (*cuMemAllocManaged)(void **, size_t, unsigned int) = nullptr;
CUresult (*cuMemHostRegister)(void*, size_t, unsigned int) = nullptr;
CUresult (*cuMemHostUnregister)(void*) = nullptr;
CUresult (*cuMemFree)(void *) = nullptr;
CUresult (*cuMemFreeHost)(void *) = nullptr;
CUresult (*cuMemPrefetchAsync)(const void *, size_t, CUdevice, CUstream) = nullptr;
CUresult (*cuMemcpy)(void *, const void *, size_t) = nullptr;
CUresult (*cuMemcpyAsync)(void *, const void *, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD16Async)(void *, unsigned short, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream) = nullptr;
CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
CUresult (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
CUresult (*cuModuleUnload)(CUmodule) = nullptr;
CUresult (*cuOccupancyMaxPotentialBlockSize)(int *, int *, CUfunction, void *,
                                             size_t, int) = nullptr;
CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
CUresult (*cuStreamCreate)(CUstream *, unsigned int) = nullptr;
CUresult (*cuStreamDestroy)(CUstream) = nullptr;
CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int) = nullptr;

static void *jit_cuda_handle = nullptr;
#endif

// Enoki API
static CUmodule *jit_cuda_module = nullptr;

CUfunction *jit_cuda_fill_64 = nullptr;
CUfunction *jit_cuda_mkperm_phase_1_tiny = nullptr;
CUfunction *jit_cuda_mkperm_phase_1_small = nullptr;
CUfunction *jit_cuda_mkperm_phase_1_large = nullptr;
CUfunction *jit_cuda_mkperm_phase_3 = nullptr;
CUfunction *jit_cuda_mkperm_phase_4_tiny = nullptr;
CUfunction *jit_cuda_mkperm_phase_4_small = nullptr;
CUfunction *jit_cuda_mkperm_phase_4_large = nullptr;
CUfunction *jit_cuda_transpose = nullptr;
CUfunction *jit_cuda_scan_small_u32 = nullptr;
CUfunction *jit_cuda_scan_large_u32 = nullptr;
CUfunction *jit_cuda_scan_large_u32_init = nullptr;
CUfunction *jit_cuda_compress_small = nullptr;
CUfunction *jit_cuda_compress_large = nullptr;
CUfunction *jit_cuda_block_copy[(int)VarType::Count] { };
CUfunction *jit_cuda_block_sum [(int)VarType::Count] { };
CUfunction *jit_cuda_reductions[(int) ReductionType::Count]
                               [(int) VarType::Count] = { };
int jit_cuda_devices = 0;

static bool jit_cuda_init_attempted = false;
static bool jit_cuda_init_success = false;

bool jit_cuda_init() {
    if (jit_cuda_init_attempted)
        return jit_cuda_init_success;
    jit_cuda_init_attempted = true;

    // We have our own caching scheme, disable CUDA's JIT cache
#if !defined(_WIN32)
    putenv((char*)"CUDA_CACHE_DISABLE=1");
#else
    (void) _wputenv(L"CUDA_CACHE_DISABLE=1");
#endif

#if defined(ENOKI_DYNAMIC_CUDA)
    jit_cuda_handle = nullptr;
#  if defined(_WIN32)
#    define dlsym(ptr, name) GetProcAddress((HMODULE) ptr, name)
    const char* cuda_fname = "nvcuda.dll",
              * cuda_glob = nullptr;
#  elif defined(__linux__)
    const char *cuda_fname  = "libcuda.so",
               *cuda_glob   = "/usr/lib/x86_64-linux-gnu/libcuda.so.*";
#  else
    const char *cuda_fname  = "libcuda.dylib",
               *cuda_glob   = cuda_fname;
#  endif

#  if !defined(_WIN32)
    // Don't dlopen libcuda.so if it was loaded by another library
    if (dlsym(RTLD_NEXT, "cuInit"))
        jit_cuda_handle = RTLD_NEXT;
#  endif

    if (!jit_cuda_handle) {
        jit_cuda_handle = jit_find_library(cuda_fname, cuda_glob, "ENOKI_LIBCUDA_PATH");

        if (!jit_cuda_handle) {
            jit_log(Warn, "jit_cuda_init(): %s could not be loaded -- "
                          "disabling CUDA backend! Set the 'ENOKI_LIBCUDA_PATH' "
                          "environment variable to specify its path.", cuda_fname);
            return false;
        }
    }

    const char *symbol = nullptr;

    do {
        #define LOAD(name, ...)                                      \
            symbol = strlen(__VA_ARGS__ "") > 0                      \
                ? (#name "_" __VA_ARGS__) : #name;                   \
            name = decltype(name)(dlsym(jit_cuda_handle, symbol));   \
            if (!name)                                               \
                break;                                               \
            symbol = nullptr

        LOAD(cuCtxEnablePeerAccess);
        LOAD(cuCtxSynchronize);
        LOAD(cuDeviceCanAccessPeer);
        LOAD(cuDeviceGet);
        LOAD(cuDeviceGetAttribute);
        LOAD(cuDeviceGetCount);
        LOAD(cuDeviceGetName);
        LOAD(cuDevicePrimaryCtxRelease);
        LOAD(cuDevicePrimaryCtxRetain);
        LOAD(cuDeviceTotalMem, "v2");
        LOAD(cuDriverGetVersion);
        LOAD(cuEventCreate);
        LOAD(cuEventDestroy, "v2");
        LOAD(cuEventRecord, "ptsz");
        LOAD(cuEventSynchronize);
        LOAD(cuFuncSetAttribute);
        LOAD(cuGetErrorName);
        LOAD(cuGetErrorString);
        LOAD(cuInit);
        LOAD(cuLaunchHostFunc, "ptsz");
        LOAD(cuLaunchKernel, "ptsz");
        LOAD(cuLinkAddData, "v2");
        LOAD(cuLinkComplete);
        LOAD(cuLinkCreate, "v2");
        LOAD(cuLinkDestroy);
        LOAD(cuMemAdvise);
        LOAD(cuMemAlloc, "v2");
        LOAD(cuMemAllocHost, "v2");
        LOAD(cuMemAllocManaged);
        LOAD(cuMemFree, "v2");
        LOAD(cuMemFreeHost);
        LOAD(cuMemHostRegister, "v2");
        LOAD(cuMemHostUnregister);
        LOAD(cuMemPrefetchAsync, "ptsz");
        LOAD(cuMemcpy, "ptds");
        LOAD(cuMemcpyAsync, "ptsz");
        LOAD(cuMemsetD16Async, "ptsz");
        LOAD(cuMemsetD32Async, "ptsz");
        LOAD(cuMemsetD8Async, "ptsz");
        LOAD(cuModuleGetFunction);
        LOAD(cuModuleLoadData);
        LOAD(cuModuleUnload);
        LOAD(cuOccupancyMaxPotentialBlockSize);
        LOAD(cuCtxSetCurrent);
        LOAD(cuStreamCreate);
        LOAD(cuStreamDestroy, "v2");
        LOAD(cuStreamSynchronize, "ptsz");
        LOAD(cuStreamWaitEvent, "ptsz");

        #undef LOAD
    } while (false);

    if (symbol) {
        jit_log(LogLevel::Warn,
                "jit_cuda_init(): could not find symbol \"%s\" -- disabling "
                "CUDA backend!", symbol);
        return false;
    }
#endif

    CUresult rv = cuInit(0);
    if (rv != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(rv, &msg);
        jit_log(LogLevel::Warn,
                "jit_cuda_init(): cuInit failed (%s) -- disabling CUDA backend", msg);
        return false;
    }

    cuda_check(cuDeviceGetCount(&jit_cuda_devices));

    if (jit_cuda_devices == 0) {
        jit_log(
            LogLevel::Warn,
            "jit_cuda_init(): No devices found -- disabling CUDA backend!");
        return false;
    }

    int cuda_version, cuda_version_major, cuda_version_minor;
    cuda_check(cuDriverGetVersion(&cuda_version));

    cuda_version_major = cuda_version / 1000;
    cuda_version_minor = (cuda_version % 1000) / 10;

    jit_log(LogLevel::Info,
            "jit_cuda_init(): enabling CUDA backend (version %i.%i)",
            cuda_version_major, cuda_version_minor);


    for (uint32_t j = 0; j < (uint32_t) ReductionType::Count; j++)
        for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++)
            jit_cuda_reductions[j][k] =
                (CUfunction *) malloc_check(sizeof(CUfunction) * jit_cuda_devices);

    jit_cuda_module = (CUmodule *) malloc_check(sizeof(CUmodule) * jit_cuda_devices);

    jit_lz4_init();

    for (int i = 0; i < jit_cuda_devices; ++i) {
        CUcontext context = nullptr;
        cuda_check(cuDevicePrimaryCtxRetain(&context, i));
        cuda_check(cuCtxSetCurrent(context));
        int cc_minor, cc_major, shared_memory_bytes;

        cuda_check(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
        cuda_check(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
        cuda_check(cuDeviceGetAttribute(
            &shared_memory_bytes,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, i));

        const char *kernels = cc_major >= 7 ? kernels_70 : kernels_50;
        int kernels_size_uncompressed = cc_major >= 7
                                            ? kernels_70_size_uncompressed
                                            : kernels_50_size_uncompressed;
        int kernels_size_compressed   = cc_major >= 7
                                            ? kernels_70_size_compressed
                                            : kernels_50_size_compressed;
        size_t kernels_hash           = cc_major >= 7
                                            ? kernels_70_hash
                                            : kernels_50_hash;

        // Decompress supplemental PTX content
        char *uncompressed =
            (char *) malloc_check(size_t(kernels_size_uncompressed) + jit_lz4_dict_size + 1);
        memcpy(uncompressed, jit_lz4_dict, jit_lz4_dict_size);
        char *uncompressed_ptx = uncompressed + jit_lz4_dict_size;

        if (LZ4_decompress_safe_usingDict(
                kernels, uncompressed_ptx,
                kernels_size_compressed,
                kernels_size_uncompressed,
                uncompressed,
                jit_lz4_dict_size) != kernels_size_uncompressed)
            jit_fail("jit_cuda_init(): decompression of builtin kernels failed!");

        uncompressed_ptx[kernels_size_uncompressed] = '\0';

        hash_combine(kernels_hash, cc_minor + size_t(cc_major) * 10);

        Kernel kernel;
        if (!jit_kernel_load(uncompressed_ptx, kernels_size_uncompressed, true, kernels_hash, kernel)) {
            jit_cuda_compile(uncompressed_ptx, kernels_size_uncompressed, kernel);
            jit_kernel_write(uncompressed_ptx, kernels_size_uncompressed, true, kernels_hash, kernel);
        }

        free(uncompressed);

        // .. and register it with CUDA
        CUmodule m;
        cuda_check(cuModuleLoadData(&m, kernel.data));
        free(kernel.data);
        jit_cuda_module[i] = m;

        #define LOAD(name)                                                       \
            if (i == 0)                                                          \
                jit_cuda_##name = (CUfunction *) malloc_check(                   \
                    sizeof(CUfunction) * jit_cuda_devices);                      \
            cuda_check(cuModuleGetFunction(&jit_cuda_##name[i], m, #name))

        LOAD(fill_64);
        LOAD(mkperm_phase_1_tiny);
        LOAD(mkperm_phase_1_small);
        LOAD(mkperm_phase_1_large);
        LOAD(mkperm_phase_3);
        LOAD(mkperm_phase_4_tiny);
        LOAD(mkperm_phase_4_small);
        LOAD(mkperm_phase_4_large);
        LOAD(transpose);
        LOAD(scan_small_u32);
        LOAD(scan_large_u32);
        LOAD(scan_large_u32_init);
        LOAD(compress_small);
        LOAD(compress_large);

        #undef LOAD

        #define MAXIMIZE_SHARED(name)                                            \
            cuda_check(cuFuncSetAttribute(                                       \
                jit_cuda_##name[i],                                              \
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,                 \
                shared_memory_bytes))

        // Max out the amount of shared memory available to the following kernels
        MAXIMIZE_SHARED(mkperm_phase_1_tiny);
        MAXIMIZE_SHARED(mkperm_phase_1_small);
        MAXIMIZE_SHARED(mkperm_phase_4_tiny);
        MAXIMIZE_SHARED(mkperm_phase_4_small);

        #undef MAXIMIZE_SHARED

        char name[16];
        CUfunction func;
        for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
            snprintf(name, sizeof(name), "block_copy_%s", var_type_name_short[k]);
            CUresult rv = cuModuleGetFunction(&func, m, name);
            if (rv != CUDA_ERROR_NOT_FOUND) {
                cuda_check(rv);
                if (i == 0)
                    jit_cuda_block_copy[k] = (CUfunction *) malloc_check(
                        sizeof(CUfunction) * jit_cuda_devices);
                jit_cuda_block_copy[k][i] = func;
            }

            snprintf(name, sizeof(name), "block_sum_%s", var_type_name_short[k]);
            rv = cuModuleGetFunction(&func, m, name);
            if (rv != CUDA_ERROR_NOT_FOUND) {
                cuda_check(rv);
                if (i == 0)
                    jit_cuda_block_sum[k] = (CUfunction *) malloc_check(
                        sizeof(CUfunction) * jit_cuda_devices);
                jit_cuda_block_sum[k][i] = func;
            }

            for (uint32_t j = 0; j < (uint32_t) ReductionType::Count; j++) {
                snprintf(name, sizeof(name), "reduce_%s_%s", reduction_name[j],
                         var_type_name_short[k]);
                rv = cuModuleGetFunction(&func, m, name);
                if (rv != CUDA_ERROR_NOT_FOUND) {
                    cuda_check(rv);
                    if (i == 0)
                        jit_cuda_reductions[j][k] = (CUfunction *) malloc_check(
                            sizeof(CUfunction) * jit_cuda_devices);
                    jit_cuda_reductions[j][k][i] = func;
                }
            }
        }
    }
    cuda_check(cuCtxSetCurrent(nullptr));

    jit_cuda_init_success = true;
    return true;
}

void jit_cuda_compile(const char *buffer, size_t buffer_size, Kernel &kernel) {
    const uintptr_t log_size = 4096;
    char error_log[log_size], info_log[log_size];

    CUjit_option arg[5] = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE
    };

    void *argv[5] = {
        (void *) info_log,
        (void *) log_size,
        (void *) error_log,
        (void *) log_size,
        (void *) 1
    };

    CUlinkState link_state;
    cuda_check(cuLinkCreate(5, arg, argv, &link_state));

    int rt = cuLinkAddData(link_state, CU_JIT_INPUT_PTX, (void *) buffer,
                           buffer_size, nullptr, 0, nullptr, nullptr);
    if (rt != CUDA_SUCCESS)
        jit_fail("jit_llvm_compile(): compilation failed. Please see the PTX "
                 "assembly listing and error message below:\n\n%s\n\n%s",
                 buffer, error_log);

    void *link_output = nullptr;
    size_t link_output_size = 0;
    cuda_check(cuLinkComplete(link_state, &link_output, &link_output_size));
    if (rt != CUDA_SUCCESS)
        jit_fail("jit_llvm_compile(): compilation failed. Please see the PTX "
                 "assembly listing and error message below:\n\n%s\n\n%s",
                 buffer, error_log);

    jit_trace("Detailed linker output:\n%s", info_log);

    kernel.data = malloc_check(link_output_size);
    kernel.size = (uint32_t) link_output_size;
    memcpy(kernel.data, link_output, link_output_size);

    // Destroy the linker invocation
    cuda_check(cuLinkDestroy(link_state));
}

void jit_cuda_shutdown() {
    if (!jit_cuda_init_success)
        return;

    jit_log(Info, "jit_cuda_shutdown()");


    for (int i = 0; i < jit_cuda_devices; ++i) {
        CUcontext context = nullptr;
        cuda_check(cuDevicePrimaryCtxRetain(&context, i));
        cuda_check(cuCtxSetCurrent(nullptr));
        cuda_check(cuModuleUnload(jit_cuda_module[i]));
        cuda_check(cuDevicePrimaryCtxRelease(i));
        cuda_check(cuDevicePrimaryCtxRelease(i));
    }

    cuda_check(cuCtxSetCurrent(nullptr));

    jit_cuda_devices = 0;

    #define Z(x) do { free(x); x = nullptr; } while (0)

    Z(jit_cuda_fill_64);
    Z(jit_cuda_mkperm_phase_1_tiny);
    Z(jit_cuda_mkperm_phase_1_small);
    Z(jit_cuda_mkperm_phase_1_large);
    Z(jit_cuda_mkperm_phase_3);
    Z(jit_cuda_mkperm_phase_4_tiny);
    Z(jit_cuda_mkperm_phase_4_small);
    Z(jit_cuda_mkperm_phase_4_large);
    Z(jit_cuda_transpose);
    Z(jit_cuda_scan_small_u32);
    Z(jit_cuda_scan_large_u32);
    Z(jit_cuda_scan_large_u32_init);
    Z(jit_cuda_compress_small);
    Z(jit_cuda_compress_large);
    Z(jit_cuda_module);

    for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
        Z(jit_cuda_block_copy[k]);
        Z(jit_cuda_block_sum[k]);
        for (uint32_t j = 0; j < (uint32_t) ReductionType::Count; j++)
            Z(jit_cuda_reductions[j][k]);
    }

    #undef Z

#if defined(ENOKI_DYNAMIC_CUDA)
    #define Z(x) x = nullptr

    Z(cuCtxEnablePeerAccess); Z(cuCtxSynchronize); Z(cuDeviceCanAccessPeer);
    Z(cuDeviceGet); Z(cuDeviceGetAttribute); Z(cuDeviceGetCount);
    Z(cuDeviceGetName); Z(cuDevicePrimaryCtxRelease);
    Z(cuDevicePrimaryCtxRetain); Z(cuDeviceTotalMem); Z(cuDriverGetVersion);
    Z(cuEventCreate); Z(cuEventDestroy); Z(cuEventRecord);
    Z(cuEventSynchronize); Z(cuFuncSetAttribute); Z(cuGetErrorName);
    Z(cuGetErrorString); Z(cuInit); Z(cuLaunchHostFunc); Z(cuLaunchKernel);
    Z(cuLinkAddData); Z(cuLinkComplete); Z(cuLinkCreate); Z(cuLinkDestroy);
    Z(cuMemAdvise); Z(cuMemAlloc); Z(cuMemAllocHost); Z(cuMemAllocManaged);
    Z(cuMemHostRegister); Z(cuMemHostUnregister); Z(cuMemFree);
    Z(cuMemFreeHost); Z(cuMemPrefetchAsync); Z(cuMemcpy); Z(cuMemcpyAsync);
    Z(cuMemsetD16Async); Z(cuMemsetD32Async); Z(cuMemsetD8Async);
    Z(cuModuleGetFunction); Z(cuModuleLoadData); Z(cuModuleUnload);
    Z(cuOccupancyMaxPotentialBlockSize); Z(cuCtxSetCurrent); Z(cuStreamCreate);
    Z(cuStreamDestroy); Z(cuStreamSynchronize); Z(cuStreamWaitEvent);

#if !defined(_WIN32)
    if (jit_cuda_handle != RTLD_NEXT)
        dlclose(jit_cuda_handle);
#else
    FreeLibrary((HMODULE) jit_cuda_handle);
#endif

    jit_cuda_handle = nullptr;

    #undef Z
#endif

    jit_cuda_init_success = false;
    jit_cuda_init_attempted = false;
}

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (unlikely(errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED)) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        jit_fail("cuda_check(): API error %04i (%s): \"%s\" in "
                 "%s:%i.", (int) errval, name, msg, file, line);
    }
}
