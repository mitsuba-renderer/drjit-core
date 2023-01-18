/*
    src/cuda_api.cpp -- Low-level interface to CUDA driver API

    Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#define DR_CUDA_SYM(...) __VA_ARGS__ = nullptr;

#include "cuda_api.h"
#include "log.h"
#include "internal.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#if !defined(DRJIT_DYNAMIC_CUDA)

bool jitc_cuda_api_init() { return true; }
void jitc_cuda_api_shutdown() { }

void *jitc_cuda_lookup(const char *name) {
    #if defined(_WIN32)
        jitc_raise("jit_cuda_lookup(): currently unsupported on Windows when the "
                   "DRJIT_DYNAMIC_CUDA flag is disabled.");
    #else
        void *ptr = dlsym(RTLD_DEFAULT, name);
        if (!ptr)
            jitc_raise("jit_cuda_lookup(): function \"%s\" not found!", name);
        return ptr;
    #endif
}

#else // DRJIT_DYNAMIC_CUDA

static void *jitc_cuda_handle = nullptr;

bool jitc_cuda_api_init() {
    if (jitc_cuda_handle)
        return true;

#  if defined(_WIN32)
    const char* cuda_fname = "nvcuda.dll",
              * cuda_glob = nullptr;
#  elif defined(__linux__)
    const char *cuda_fname  = "libcuda.so",
               *cuda_glob   = "/usr/lib/{x86_64-linux-gnu,aarch64-linux-gnu}/libcuda.so.*";
#  else
    const char *cuda_fname  = "libcuda.dylib",
               *cuda_glob   = cuda_fname;
#  endif

#  if !defined(_WIN32)
    // Don't dlopen libcuda.so if it was loaded by another library
    if (dlsym(RTLD_NEXT, "cuInit"))
        jitc_cuda_handle = RTLD_NEXT;
#  endif

    if (!jitc_cuda_handle) {
        jitc_cuda_handle = jitc_find_library(cuda_fname, cuda_glob, "DRJIT_LIBCUDA_PATH");

        if (!jitc_cuda_handle) // CUDA library cannot be loaded, give up
            return false;
    }

    const char *symbol = nullptr;

    do {
        #define LOAD(name, ...)                                      \
            symbol = strlen(__VA_ARGS__ "") > 0                      \
                ? (#name "_" __VA_ARGS__) : #name;                   \
            name = decltype(name)(dlsym(jitc_cuda_handle, symbol));  \
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
        LOAD(cuDevicePrimaryCtxRelease, "v2");
        LOAD(cuDevicePrimaryCtxRetain);
        LOAD(cuDeviceTotalMem, "v2");
        LOAD(cuDriverGetVersion);
        LOAD(cuEventCreate);
        LOAD(cuEventDestroy, "v2");
        LOAD(cuEventRecord);
        LOAD(cuEventSynchronize);
        LOAD(cuEventElapsedTime);
        LOAD(cuFuncSetAttribute);
        LOAD(cuGetErrorName);
        LOAD(cuGetErrorString);
        LOAD(cuInit);
        LOAD(cuLaunchHostFunc);
        LOAD(cuLaunchKernel);
        LOAD(cuLinkAddData, "v2");
        LOAD(cuLinkComplete);
        LOAD(cuLinkCreate, "v2");
        LOAD(cuLinkDestroy);
        LOAD(cuMemAdvise);
        LOAD(cuMemAlloc, "v2");
        LOAD(cuMemAllocHost, "v2");
        LOAD(cuMemFree, "v2");
        LOAD(cuMemFreeHost);

        LOAD(cuMemcpy);
        LOAD(cuMemcpyAsync);
        LOAD(cuMemsetD16Async);
        LOAD(cuMemsetD32Async);
        LOAD(cuMemsetD8Async);
        LOAD(cuModuleGetFunction);
        LOAD(cuModuleLoadData);
        LOAD(cuModuleUnload);
        LOAD(cuOccupancyMaxPotentialBlockSize);
        LOAD(cuCtxPushCurrent, "v2");
        LOAD(cuCtxPopCurrent, "v2");
        LOAD(cuStreamCreate);
        LOAD(cuStreamDestroy, "v2");
        LOAD(cuStreamSynchronize);
        LOAD(cuStreamWaitEvent);
        LOAD(cuPointerGetAttribute);
        LOAD(cuArrayCreate, "v2");
        LOAD(cuArray3DCreate, "v2");
        LOAD(cuArray3DGetDescriptor, "v2");
        LOAD(cuArrayDestroy);
        LOAD(cuTexObjectCreate);
        LOAD(cuTexObjectGetResourceDesc);
        LOAD(cuTexObjectDestroy);
        LOAD(cuMemcpy2DAsync, "v2");
        LOAD(cuMemcpy3DAsync, "v2");
        #undef LOAD
    } while (false);

    if (symbol) {
        jitc_cuda_api_shutdown();
        jitc_log(LogLevel::Warn,
                "jit_cuda_api_init(): could not find symbol \"%s\" -- disabling "
                "CUDA backend!", symbol);
        return false;
    }

    // These two functions are optional
    cuMemAllocAsync =
        decltype(cuMemAllocAsync)(dlsym(jitc_cuda_handle, "cuMemAllocAsync"));
    cuMemFreeAsync =
        decltype(cuMemFreeAsync)(dlsym(jitc_cuda_handle, "cuMemFreeAsync"));

    return true;
}

void jitc_cuda_api_shutdown() {
    if (!jitc_cuda_handle)
        return;

    #define Z(x) x = nullptr
    Z(cuCtxEnablePeerAccess); Z(cuCtxSynchronize); Z(cuDeviceCanAccessPeer);
    Z(cuDeviceGet); Z(cuDeviceGetAttribute); Z(cuDeviceGetCount);
    Z(cuDeviceGetName); Z(cuDevicePrimaryCtxRelease);
    Z(cuDevicePrimaryCtxRetain); Z(cuDeviceTotalMem); Z(cuDriverGetVersion);
    Z(cuEventCreate); Z(cuEventDestroy); Z(cuEventRecord);
    Z(cuEventSynchronize); Z(cuEventElapsedTime); Z(cuFuncSetAttribute);
    Z(cuGetErrorName); Z(cuGetErrorString); Z(cuInit); Z(cuLaunchHostFunc);
    Z(cuLaunchKernel); Z(cuLinkAddData); Z(cuLinkComplete); Z(cuLinkCreate);
    Z(cuLinkDestroy); Z(cuMemAdvise); Z(cuMemAlloc); Z(cuMemAllocHost);
    Z(cuMemFree); Z(cuMemFreeHost); Z(cuMemcpy); Z(cuMemcpyAsync);
    Z(cuMemsetD16Async); Z(cuMemsetD32Async); Z(cuMemsetD8Async);
    Z(cuModuleGetFunction); Z(cuModuleLoadData); Z(cuModuleUnload);
    Z(cuOccupancyMaxPotentialBlockSize); Z(cuCtxPushCurrent);
    Z(cuCtxPopCurrent); Z(cuStreamCreate); Z(cuStreamDestroy);
    Z(cuStreamSynchronize); Z(cuStreamWaitEvent); Z(cuPointerGetAttribute);
    Z(cuArrayCreate); Z(cuArray3DCreate); Z(cuArray3DGetDescriptor);
    Z(cuArrayDestroy); Z(cuTexObjectCreate); Z(cuTexObjectGetResourceDesc);
    Z(cuTexObjectDestroy); Z(cuMemcpy2DAsync); Z(cuMemcpy3DAsync);
    #undef Z

#if !defined(_WIN32)
    if (jitc_cuda_handle != RTLD_NEXT)
        dlclose(jitc_cuda_handle);
#else
    FreeLibrary((HMODULE) jitc_cuda_handle);
#endif

    jitc_cuda_handle = nullptr;
}

void *jitc_cuda_lookup(const char *name) {
    void *ptr = dlsym(jitc_cuda_handle, name);
    if (!ptr)
        jitc_raise("jit_cuda_lookup(): function \"%s\" not found!", name);
    return ptr;
}
#endif
