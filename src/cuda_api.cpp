#include "internal.h"
#include "log.h"
#include <dlfcn.h>
#include <zlib.h>
#include "../ptx/kernels.h"

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
CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD16Async)(void *, unsigned short, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream) = nullptr;
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
cudaError_t (*cudaRuntimeGetVersion)(int*) = nullptr;
cudaError_t (*cudaSetDevice)(int) = nullptr;
cudaError_t (*cudaStreamCreateWithFlags)(cudaStream_t *, unsigned int) = nullptr;
cudaError_t (*cudaStreamDestroy)(cudaStream_t) = nullptr;
cudaError_t (*cudaStreamSynchronize)(cudaStream_t) = nullptr;
cudaError_t (*cudaStreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int) = nullptr;

// Enoki API
CUfunction kernel_fill_64 = nullptr;

#define LOAD(name)                                                             \
    name = decltype(name)(dlsym(lib, #name));                                  \
    if (!name)                                                                 \
        return false;

#define LOAD_SUFFIX(name, suffix)                                              \
    name = decltype(name)(dlsym(lib, #name "_" suffix));                       \
    if (!name)                                                                 \
        return false;

static bool jit_cuda_init_attempted = false;
static bool jit_cuda_init_success = false;
static CUmodule jit_cuda_module = nullptr;

extern "C" {
    extern unsigned char kernels_ptx_compressed[];
};

int inflate(const void *src, uint32_t src_size, void *dst, uint32_t dst_size) {
    z_stream strm;
    memset(&strm, 0, sizeof(z_stream));
    strm.total_in = strm.avail_in = src_size;
    strm.total_out = strm.avail_out = dst_size;
    strm.next_in   = (unsigned char *) src;
    strm.next_out  = (unsigned char *) dst;

    int rv = inflateInit2(&strm, (15 + 32));
    if (rv == Z_OK) {
        rv = inflate(&strm, Z_FINISH);
        if (rv == Z_STREAM_END)
            rv = strm.total_out;
    }
    inflateEnd(&strm);
    return rv;
}

bool jit_cuda_init() {
    if (jit_cuda_init_attempted)
        return jit_cuda_init_success;
    jit_cuda_init_attempted = true;

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
    LOAD_SUFFIX(cuMemsetD16Async, "ptsz");
    LOAD_SUFFIX(cuMemsetD32Async, "ptsz");
    LOAD_SUFFIX(cuMemsetD8Async, "ptsz");

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
    LOAD(cudaRuntimeGetVersion);
    LOAD(cudaSetDevice);
    LOAD(cudaStreamCreateWithFlags);
    LOAD(cudaStreamDestroy);
    LOAD_SUFFIX(cudaEventRecord, "ptsz");
    LOAD_SUFFIX(cudaLaunchHostFunc, "ptsz");
    LOAD_SUFFIX(cudaMemPrefetchAsync, "ptsz");
    LOAD_SUFFIX(cudaMemcpyAsync, "ptsz");
    LOAD_SUFFIX(cudaStreamSynchronize, "ptsz");
    LOAD_SUFFIX(cudaStreamWaitEvent, "ptsz");

    int cuda_version, cuda_version_major, cuda_version_minor;
    cuda_check(cudaRuntimeGetVersion(&cuda_version));
    cuda_version_major = cuda_version / 1000;
    cuda_version_minor = (cuda_version % 1000) / 10;

    jit_log(LogLevel::Info, "jit_cuda_init(): enabled CUDA backend (version %i.%i)",
            cuda_version_major, cuda_version_minor);

    // Dummy operations to create a context
    cuda_check(cudaSetDevice(0));
    cuda_check(cudaFree(0));

    // Decompress supplemental PTX content
    std::unique_ptr<char[]> uncompressed(
        new char[kernels_ptx_uncompressed_size + 1]);

    int rv = inflate(kernels_ptx_compressed,
                     (uint32_t) sizeof(kernels_ptx_compressed),
                     uncompressed.get(),
                     (uint32_t) kernels_ptx_uncompressed_size);

    if (rv != (int) kernels_ptx_uncompressed_size)
        jit_fail("jit_cuda_init(): decompression of precompiled kernels failed "
                 "(%i)!", rv);
    uncompressed[kernels_ptx_uncompressed_size] = '\0';

    // .. and register it with CUDA
    cuda_check(cuModuleLoadData(&jit_cuda_module, uncompressed.get()));

    cuda_check(cuModuleGetFunction(&kernel_fill_64, jit_cuda_module, "fill_64"));

    jit_cuda_init_success = true;
    return true;
}
