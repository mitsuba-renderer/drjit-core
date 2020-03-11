#pragma once

#include <stdlib.h>

enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_DEINITIALIZED = 4
};

enum CUjit_option {
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_LOG_VERBOSE = 12
};

enum CUjitInputType {
    CU_JIT_INPUT_PTX = 1
};

typedef enum CUfunction_attribute {
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
} CUfunction_attribute;


enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorCudartUnloading = 4,
    cudaErrorPeerAccessAlreadyEnabled = 704
};

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

enum cudaDeviceAttr {
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrConcurrentManagedAccess = 89
};

using CUdevice    = int;
using CUmodule    = void*;
using CUfunction  = void*;
using CUlinkState = void*;

using cudaStream_t = void*;
using cudaEvent_t  = void*;

#define cudaCpuDeviceId -1
#define cudaStreamNonBlocking 1
#define cudaEventDisableTiming 2
#define cudaMemAttachGlobal 1
#define cudaMemAdviseSetReadMostly 1
#define CU_SHAREDMEM_CARVEOUT_MAX_L1  0
#define CU_LAUNCH_PARAM_END (void *) 0
#define CU_LAUNCH_PARAM_BUFFER_POINTER (void *) 1
#define CU_LAUNCH_PARAM_BUFFER_SIZE (void *) 2

// Driver API
extern CUresult (*cuDeviceGetName)(char *, int, CUdevice);
extern CUresult (*cuDeviceTotalMem)(size_t *, CUdevice);
extern CUresult (*cuFuncGetAttribute)(int *, CUfunction_attribute, CUfunction);
extern CUresult (*cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int);
extern CUresult (*cuGetErrorString)(CUresult, const char **);
extern CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                                  unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, cudaStream_t,
                                  void **, void **);
extern CUresult (*cuLinkComplete)(CUlinkState, void **, size_t *);
extern CUresult (*cuLinkAddData)(CUlinkState, CUjitInputType, void *, size_t,
                                 const char *, unsigned int, CUjit_option *, void **);
extern CUresult (*cuLinkCreate)(unsigned int, CUjit_option *, void **, CUlinkState *);
extern CUresult (*cuLinkDestroy)(CUlinkState);
extern CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
extern CUresult (*cuModuleLoadData)(CUmodule *, const void *);
extern CUresult (*cuModuleUnload)(CUmodule);
extern CUresult (*cuOccupancyMaxPotentialBlockSize)(int *, int *, CUfunction,
                                                    void *, size_t, int);

// Runtime API
extern const char *(*cudaGetErrorName)(cudaError_t);
extern cudaError_t (*cudaDeviceCanAccessPeer)(int *, int, int);
extern cudaError_t (*cudaDeviceEnablePeerAccess)(int, unsigned int);
extern cudaError_t (*cudaDeviceGetAttribute)(int *, cudaDeviceAttr, int);
extern cudaError_t (*cudaDeviceSynchronize)();
extern cudaError_t (*cudaEventCreateWithFlags)(cudaEvent_t *, unsigned int);
extern cudaError_t (*cudaEventDestroy)(cudaEvent_t);
extern cudaError_t (*cudaEventRecord)(cudaEvent_t, cudaStream_t);
extern cudaError_t (*cudaFree)(void *devPtr);
extern cudaError_t (*cudaFreeHost)(void *ptr);
extern cudaError_t (*cudaGetDeviceCount)(int *);
extern cudaError_t (*cudaLaunchHostFunc)(cudaStream_t, void (*)(void*), void *);
extern cudaError_t (*cudaMalloc)(void **devPtr, size_t size);
extern cudaError_t (*cudaMallocHost)(void **ptr, size_t size);
extern cudaError_t (*cudaMallocManaged)(void **, size_t, unsigned int);
extern cudaError_t (*cudaMemAdvise)(const void *, size_t, int, int);
extern cudaError_t (*cudaMemPrefetchAsync)(const void *, size_t, int, cudaStream_t);
extern cudaError_t (*cudaMemcpyAsync)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
extern cudaError_t (*cudaSetDevice)(int);
extern cudaError_t (*cudaStreamCreateWithFlags)(cudaStream_t *, unsigned int);
extern cudaError_t (*cudaStreamDestroy)(cudaStream_t);
extern cudaError_t (*cudaStreamSynchronize)(cudaStream_t);
extern cudaError_t (*cudaStreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int);

/// Try to load CUDA
extern bool jit_cuda_init();
