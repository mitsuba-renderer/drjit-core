/*
    src/cuda_api.h -- Low-level interface to CUDA driver API

    Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

/// Try to resolve the CUDA API functions
extern bool jitc_cuda_api_init();

/// Free any resources allocated by jitc_cuda_api_init()
extern void jitc_cuda_api_shutdown();

/// Look up a device driver function
extern void *jitc_cuda_lookup(const char *name);

#if !defined(DRJIT_DYNAMIC_CUDA)
#  include <cuda.h>
#else
#  define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#  define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#  define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN 97
#  define CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED 115
#  define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#  define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#  define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#  define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50
#  define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING 41
#  define CU_DEVICE_ATTRIBUTE_TCC_DRIVER 35
#  define CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED 90

#  define CU_DEVICE_CPU -1

#  define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#  define CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT 9
#  define CU_FUNC_CACHE_PREFER_L1 2

#  define CU_JIT_INPUT_PTX 1
#  define CU_JIT_INFO_LOG_BUFFER 3
#  define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4
#  define CU_JIT_ERROR_LOG_BUFFER 5
#  define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6
#  define CU_JIT_OPTIMIZATION_LEVEL 7
#  define CU_JIT_GENERATE_DEBUG_INFO 11
#  define CU_JIT_LOG_VERBOSE 12
#  define CU_JIT_GENERATE_LINE_INFO 13

#  define CU_LAUNCH_PARAM_BUFFER_POINTER (void *) 1
#  define CU_LAUNCH_PARAM_BUFFER_SIZE (void *) 2
#  define CU_LAUNCH_PARAM_END (void *) 0

#  define CU_MEM_ATTACH_GLOBAL 1
#  define CU_MEM_ADVISE_SET_READ_MOSTLY 1
#  define CU_SHAREDMEM_CARVEOUT_MAX_L1 0

#  define CU_STREAM_DEFAULT 0
#  define CU_STREAM_NON_BLOCKING 1
#  define CU_EVENT_DEFAULT 0
#  define CU_EVENT_DISABLE_TIMING 2
#  define CU_MEMORYTYPE_HOST 1
#  define CU_POINTER_ATTRIBUTE_MEMORY_TYPE 2

#  define CUDA_ERROR_INVALID_VALUE 1
#  define CUDA_ERROR_NOT_INITIALIZED 3
#  define CUDA_ERROR_DEINITIALIZED 4
#  define CUDA_ERROR_NOT_FOUND 500
#  define CUDA_ERROR_OUT_OF_MEMORY 2
#  define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED 704
#  define CUDA_SUCCESS 0

#define CU_RESOURCE_TYPE_ARRAY 0
#define CU_TR_FILTER_MODE_POINT 0
#define CU_TR_FILTER_MODE_LINEAR 1
#define CU_TRSF_NORMALIZED_COORDINATES 2
#define CU_TR_ADDRESS_MODE_WRAP 0
#define CU_TR_ADDRESS_MODE_CLAMP 1
#define CU_TR_ADDRESS_MODE_MIRROR 2
#define CU_MEMORYTYPE_DEVICE 2
#define CU_MEMORYTYPE_ARRAY 3

#define CU_AD_FORMAT_FLOAT 0x20
#define CU_RES_VIEW_FORMAT_FLOAT_1X32 0x16
#define CU_RES_VIEW_FORMAT_FLOAT_2X32 0x17
#define CU_RES_VIEW_FORMAT_FLOAT_4X32 0x18

using CUcontext    = struct CUctx_st *;
using CUmodule     = struct CUmod_st *;
using CUfunction   = struct CUfunc_st *;
using CUlinkState  = struct CUlinkState_st *;
using CUstream     = struct CUstream_st *;
using CUevent      = struct CUevent_st *;
using CUarray      = struct CUarray_st *;
using CUtexObject  = struct CUtexObject_st *;
using CUresult     = int;
using CUdevice     = int;
using CUdeviceptr  = void *;
using CUjit_option = int;

struct CUDA_ARRAY_DESCRIPTOR {
    size_t Width;
    size_t Height;
    int Format;
    unsigned int NumChannels;
};

struct CUDA_ARRAY3D_DESCRIPTOR {
    size_t Width;
    size_t Height;
    size_t Depth;
    int Format;
    unsigned int NumChannels;
    unsigned int Flags;
};

struct CUDA_RESOURCE_DESC {
    int resType;
    union {
        struct { CUarray hArray; } array;
        struct { int reserved[32]; } reserved;
    } res;
    unsigned int flags;
};

struct CUDA_TEXTURE_DESC {
    int addressMode[3];
    int filterMode;
    unsigned int flags;
    unsigned int maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};

struct CUDA_RESOURCE_VIEW_DESC {
    int format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
    unsigned int reserved[16];
};

struct CUDA_MEMCPY2D {
    size_t srcXInBytes;
    size_t srcY;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};

struct CUDA_MEMCPY3D {
    size_t srcXInBytes;
    size_t srcY;
    size_t srcZ;
    size_t srcLOD;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    void *reserved0;
    size_t srcPitch;
    size_t srcHeight;
    size_t dstXInBytes;
    size_t dstY;
    size_t dstZ;
    size_t dstLOD;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    void *reserved1;
    size_t dstPitch;
    size_t dstHeight;
    size_t WidthInBytes;
    size_t Height;
    size_t Depth;
};

#if !defined(DR_CUDA_SYM)
#  define DR_CUDA_SYM(x) extern x;
#endif

// Driver API
DR_CUDA_SYM(CUresult (*cuCtxEnablePeerAccess)(CUcontext, unsigned int));
DR_CUDA_SYM(CUresult (*cuCtxSynchronize)());
DR_CUDA_SYM(CUresult (*cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice));
DR_CUDA_SYM(CUresult (*cuDeviceGet)(CUdevice *, int));
DR_CUDA_SYM(CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice));
DR_CUDA_SYM(CUresult (*cuDeviceGetCount)(int *));
DR_CUDA_SYM(CUresult (*cuDeviceGetName)(char *, int, CUdevice));
DR_CUDA_SYM(CUresult (*cuDevicePrimaryCtxRelease)(CUdevice));
DR_CUDA_SYM(CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice));
DR_CUDA_SYM(CUresult (*cuDeviceTotalMem)(size_t *, CUdevice));
DR_CUDA_SYM(CUresult (*cuDriverGetVersion)(int *));
DR_CUDA_SYM(CUresult (*cuEventCreate)(CUevent *, unsigned int));
DR_CUDA_SYM(CUresult (*cuEventDestroy)(CUevent));
DR_CUDA_SYM(CUresult (*cuEventRecord)(CUevent, CUstream));
DR_CUDA_SYM(CUresult (*cuEventSynchronize)(CUevent));
DR_CUDA_SYM(CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent));
DR_CUDA_SYM(CUresult (*cuFuncSetAttribute)(CUfunction, int, int));
DR_CUDA_SYM(CUresult (*cuGetErrorName)(CUresult, const char **));
DR_CUDA_SYM(CUresult (*cuGetErrorString)(CUresult, const char **));
DR_CUDA_SYM(CUresult (*cuInit)(unsigned int));
DR_CUDA_SYM(CUresult (*cuLaunchHostFunc)(CUstream, void (*)(void *), void *));
DR_CUDA_SYM(CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, CUstream, void **,
                                       void **));
DR_CUDA_SYM(CUresult (*cuLinkAddData)(CUlinkState, int, void *, size_t, const char *,
                                      unsigned int, int *, void **));
DR_CUDA_SYM(CUresult (*cuLinkComplete)(CUlinkState, void **, size_t *));
DR_CUDA_SYM(CUresult (*cuLinkCreate)(unsigned int, int *, void **, CUlinkState *));
DR_CUDA_SYM(CUresult (*cuLinkDestroy)(CUlinkState));
DR_CUDA_SYM(CUresult (*cuPointerGetAttribute)(void* data, int, void*));
DR_CUDA_SYM(CUresult (*cuMemAdvise)(void *, size_t, int, CUdevice));
DR_CUDA_SYM(CUresult (*cuMemAlloc)(void **, size_t));
DR_CUDA_SYM(CUresult (*cuMemAllocHost)(void **, size_t));
DR_CUDA_SYM(CUresult (*cuMemFree)(void *));
DR_CUDA_SYM(CUresult (*cuMemFreeHost)(void *));
DR_CUDA_SYM(CUresult (*cuMemcpy)(void *, const void *, size_t));
DR_CUDA_SYM(CUresult (*cuMemcpyAsync)(void *, const void *, size_t, CUstream));
DR_CUDA_SYM(CUresult (*cuMemsetD16Async)(void *, unsigned short, size_t, CUstream));
DR_CUDA_SYM(CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream));
DR_CUDA_SYM(CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream));
DR_CUDA_SYM(CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *));
DR_CUDA_SYM(CUresult (*cuModuleLoadData)(CUmodule *, const void *));
DR_CUDA_SYM(CUresult (*cuModuleUnload)(CUmodule));
DR_CUDA_SYM(CUresult (*cuOccupancyMaxPotentialBlockSize)(int *, int *, CUfunction,
                                                         void *, size_t, int));
DR_CUDA_SYM(CUresult (*cuCtxPushCurrent)(CUcontext));
DR_CUDA_SYM(CUresult (*cuCtxPopCurrent)(CUcontext*));
DR_CUDA_SYM(CUresult (*cuStreamCreate)(CUstream *, unsigned int));
DR_CUDA_SYM(CUresult (*cuStreamDestroy)(CUstream));
DR_CUDA_SYM(CUresult (*cuStreamSynchronize)(CUstream));
DR_CUDA_SYM(CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int));
DR_CUDA_SYM(CUresult (*cuMemAllocAsync)(CUdeviceptr *, size_t, CUstream));
DR_CUDA_SYM(CUresult (*cuMemFreeAsync)(CUdeviceptr, CUstream));

DR_CUDA_SYM(CUresult (*cuArrayCreate)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *));
DR_CUDA_SYM(CUresult (*cuArray3DCreate)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *));
DR_CUDA_SYM(CUresult (*cuArray3DGetDescriptor)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray));
DR_CUDA_SYM(CUresult (*cuArrayDestroy)(CUarray));
DR_CUDA_SYM(CUresult (*cuTexObjectCreate)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                                          const CUDA_TEXTURE_DESC *,
                                          const CUDA_RESOURCE_VIEW_DESC *));
DR_CUDA_SYM(CUresult (*cuTexObjectDestroy)(CUtexObject));
DR_CUDA_SYM(CUresult (*cuTexObjectGetResourceDesc)(CUDA_RESOURCE_DESC *,
                                                   CUtexObject));
DR_CUDA_SYM(CUresult (*cuMemcpy3DAsync)(const CUDA_MEMCPY3D *, CUstream));
DR_CUDA_SYM(CUresult (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D *, CUstream));
#endif
