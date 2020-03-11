#include "internal.h"
#include "log.h"
#include <dlfcn.h>

// Driver API
CUresult (*cuDeviceGetName)(char *, int, CUdevice) = nullptr;
CUresult (*cuDeviceTotalMem)(size_t *, CUdevice) = nullptr;
CUresult (*cuFuncGetAttribute)(int *, CUfunction_attribute, CUfunction) = nullptr;
CUresult (*cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int) = nullptr;
CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;
CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, cudaStream_t,
                           void **, void **) = nullptr;
CUresult (*cuLinkAddData)(CUlinkState, CUjitInputType, void *, size_t,
                          const char *, unsigned int, CUjit_option *, void **) = nullptr;
CUresult (*cuLinkComplete)(CUlinkState, void **, size_t *) = nullptr;
CUresult (*cuLinkCreate)(unsigned int, CUjit_option *, void **, CUlinkState *) = nullptr;
CUresult (*cuLinkDestroy)(CUlinkState) = nullptr;
CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
CUresult (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
CUresult (*cuModuleUnload)(CUmodule) = nullptr;
CUresult (*cuOccupancyMaxPotentialBlockSize)(int *, int *, CUfunction,
                                             void *, size_t, int) = nullptr;

// Runtime API
const char *(*cudaGetErrorName)(cudaError_t) = nullptr;
cudaError_t (*cudaDeviceCanAccessPeer)(int *, int, int) = nullptr;
cudaError_t (*cudaDeviceEnablePeerAccess)(int, unsigned int) = nullptr;
cudaError_t (*cudaDeviceGetAttribute)(int *, cudaDeviceAttr, int) = nullptr;
cudaError_t (*cudaDeviceSynchronize)() = nullptr;
cudaError_t (*cudaEventCreateWithFlags)(cudaEvent_t *, unsigned int) = nullptr;
cudaError_t (*cudaEventDestroy)(cudaEvent_t) = nullptr;
cudaError_t (*cudaEventRecord)(cudaEvent_t, cudaStream_t) = nullptr;
cudaError_t (*cudaFree)(void *devPtr) = nullptr;
cudaError_t (*cudaFreeHost)(void *ptr) = nullptr;
cudaError_t (*cudaGetDeviceCount)(int *) = nullptr;
cudaError_t (*cudaLaunchHostFunc)(cudaStream_t, void (*)(void*), void *) = nullptr;
cudaError_t (*cudaMalloc)(void **devPtr, size_t size) = nullptr;
cudaError_t (*cudaMallocHost)(void **ptr, size_t size) = nullptr;
cudaError_t (*cudaMallocManaged)(void **, size_t, unsigned int) = nullptr;
cudaError_t (*cudaMemAdvise)(const void *, size_t, int, int) = nullptr;
cudaError_t (*cudaMemPrefetchAsync)(const void *, size_t, int, cudaStream_t) = nullptr;
cudaError_t (*cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = nullptr;
cudaError_t (*cudaMemcpyAsync)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) = nullptr;
cudaError_t (*cudaSetDevice)(int) = nullptr;
cudaError_t (*cudaStreamCreateWithFlags)(cudaStream_t *, unsigned int) = nullptr;
cudaError_t (*cudaStreamDestroy)(cudaStream_t) = nullptr;
cudaError_t (*cudaStreamSynchronize)(cudaStream_t) = nullptr;
cudaError_t (*cudaStreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int) = nullptr;

#define LOAD(name)                                                             \
    name = decltype(name)(dlsym(lib, #name));                                  \
    if (!name)                                                                 \
        return false;

#define LOAD_SUFFIX(name, suffix)                                              \
    name = decltype(name)(dlsym(lib, #name "_" suffix));                       \
    if (!name)                                                                 \
        return false;

bool jit_cuda_init() {
    if (cuDeviceGetName)
        return true;

    void *lib = dlopen("libcuda.so", RTLD_NOW);
    if (!lib) {
        jit_log(LogLevel::Warn, "libcuda.so could not be found -- disabling CUDA backend!");
        return false;
    }

    LOAD(cuDeviceGetName);
    LOAD(cuFuncGetAttribute);
    LOAD(cuFuncSetAttribute);
    LOAD(cuGetErrorString);
    LOAD(cuLinkComplete);
    LOAD(cuLinkDestroy);
    LOAD(cuModuleGetFunction);
    LOAD(cuModuleLoadData);
    LOAD(cuModuleUnload);
    LOAD(cuOccupancyMaxPotentialBlockSize);
    LOAD_SUFFIX(cuDeviceTotalMem, "v2");
    LOAD_SUFFIX(cuLaunchKernel, "ptsz");
    LOAD_SUFFIX(cuLinkAddData, "v2");
    LOAD_SUFFIX(cuLinkCreate, "v2");

    lib = dlopen("libcudart.so", RTLD_NOW);
    if (!lib) {
        jit_log(LogLevel::Warn, "libcudart.so could not be found -- disabling CUDA backend!");
        return false;
    }

    LOAD(cudaDeviceCanAccessPeer);
    LOAD(cudaDeviceEnablePeerAccess);
    LOAD(cudaDeviceGetAttribute);
    LOAD(cudaDeviceSynchronize);
    LOAD(cudaEventCreateWithFlags);
    LOAD(cudaEventDestroy);
    LOAD(cudaFree);
    LOAD(cudaFreeHost);
    LOAD(cudaGetDeviceCount);
    LOAD(cudaGetErrorName);
    LOAD(cudaMalloc);
    LOAD(cudaMallocHost);
    LOAD(cudaMallocManaged);
    LOAD(cudaMemAdvise);
    LOAD(cudaSetDevice);
    LOAD(cudaStreamCreateWithFlags);
    LOAD(cudaStreamDestroy);
    LOAD_SUFFIX(cudaStreamSynchronize, "ptsz");
    LOAD_SUFFIX(cudaStreamWaitEvent, "ptsz");
    LOAD_SUFFIX(cudaEventRecord, "ptsz");
    LOAD_SUFFIX(cudaLaunchHostFunc, "ptsz");
    LOAD_SUFFIX(cudaMemPrefetchAsync, "ptsz");
    LOAD_SUFFIX(cudaMemcpyAsync, "ptsz");

    jit_log(LogLevel::Info, "jit_cuda_init(): enabled CUDA backend.");
    return true;
}
