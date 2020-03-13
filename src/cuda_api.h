#pragma once

#include <enoki/jit.h>

#define CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS 89
#define CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY 83
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING 41

#define CU_DEVICE_CPU -1

#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#define CU_FUNC_ATTRIBUTE_NUM_REGS 4
#define CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT 9

#define CU_JIT_ERROR_LOG_BUFFER 5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6
#define CU_JIT_INFO_LOG_BUFFER 3
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4
#define CU_JIT_INPUT_PTX 1
#define CU_JIT_LOG_VERBOSE 12

#define CU_LAUNCH_PARAM_BUFFER_POINTER (void *) 1
#define CU_LAUNCH_PARAM_BUFFER_SIZE (void *) 2
#define CU_LAUNCH_PARAM_END (void *) 0

#define CU_MEM_ATTACH_GLOBAL 1
#define CU_MEM_ADVISE_SET_READ_MOSTLY 1
#define CU_SHAREDMEM_CARVEOUT_MAX_L1 0

#define CU_STREAM_NON_BLOCKING 1
#define CU_EVENT_DISABLE_TIMING 2

#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED 704
#define CUDA_SUCCESS 0


using CUresult    = int;
using CUdevice    = int;
using CUcontext   = void *;
using CUmodule    = void *;
using CUfunction  = void *;
using CUlinkState = void *;
using CUstream    = void *;
using CUevent     = void *;

// Driver API
extern CUresult (*cuCtxEnablePeerAccess)(CUcontext, unsigned int);
extern CUresult (*cuCtxSynchronize)();
extern CUresult (*cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice);
extern CUresult (*cuDeviceGet)(CUdevice *, int);
extern CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
extern CUresult (*cuDeviceGetCount)(int *);
extern CUresult (*cuDeviceGetName)(char *, int, CUdevice);
extern CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
extern CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
extern CUresult (*cuDeviceTotalMem)(size_t *, CUdevice);
extern CUresult (*cuDriverGetVersion)(int *);
extern CUresult (*cuEventCreate)(CUevent *, unsigned int);
extern CUresult (*cuEventDestroy)(CUevent);
extern CUresult (*cuEventRecord)(CUevent, CUstream);
extern CUresult (*cuFuncGetAttribute)(int *, int, CUfunction);
extern CUresult (*cuFuncSetAttribute)(CUfunction, int, int);
extern CUresult (*cuGetErrorString)(CUresult, const char **);
extern CUresult (*cuInit)(unsigned int);
extern CUresult (*cuLaunchHostFunc)(CUstream, void (*)(void *), void *);
extern CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                                  unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, CUstream, void **,
                                  void **);
extern CUresult (*cuLinkAddData)(CUlinkState, int, void *, size_t, const char *,
                                 unsigned int, int *, void **);
extern CUresult (*cuLinkComplete)(CUlinkState, void **, size_t *);
extern CUresult (*cuLinkCreate)(unsigned int, int *, void **, CUlinkState *);
extern CUresult (*cuLinkDestroy)(CUlinkState);
extern CUresult (*cuMemAdvise)(void *, size_t, int, CUdevice);
extern CUresult (*cuMemAlloc)(void **, size_t);
extern CUresult (*cuMemAllocHost)(void **, size_t);
extern CUresult (*cuMemAllocManaged)(void **, size_t, unsigned int);
extern CUresult (*cuMemFree)(void *);
extern CUresult (*cuMemFreeHost)(void *);
extern CUresult (*cuMemPrefetchAsync)(const void *, size_t, CUdevice, CUstream);
extern CUresult (*cuMemcpyAsync)(void *, const void *, size_t, CUstream);
extern CUresult (*cuMemsetD16Async)(void *, unsigned short, size_t, CUstream);
extern CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream);
extern CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream);
extern CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
extern CUresult (*cuModuleLoadData)(CUmodule *, const void *);
extern CUresult (*cuModuleUnload)(CUmodule);
extern CUresult (*cuOccupancyMaxPotentialBlockSize)(int *, int *, CUfunction,
                                                    void *, size_t, int);
extern CUresult (*cuCtxSetCurrent)(CUcontext);
extern CUresult (*cuStreamCreate)(CUstream *, unsigned int);
extern CUresult (*cuStreamDestroy)(CUstream);
extern CUresult (*cuStreamSynchronize)(CUstream);
extern CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int);

// Enoki API
extern CUfunction kernel_fill_64;
extern CUfunction kernel_reductions[(int) ReductionType::Count][(int) VarType::Count];

/// Try to load CUDA
extern bool jit_cuda_init();

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
extern void cuda_check_impl(CUresult errval, const char *file, const int line);
