#include "internal.h"
#include "log.h"
#include "var.h"
#include "util.h"
#include "../ptx/kernels.h"
#include <dlfcn.h>
#include <zlib.h>

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
CUresult (*cuFuncGetAttribute)(int *, int, CUfunction) = nullptr;
CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;
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
CUresult (*cuMemFree)(void *) = nullptr;
CUresult (*cuMemFreeHost)(void *) = nullptr;
CUresult (*cuMemPrefetchAsync)(const void *, size_t, CUdevice, CUstream) = nullptr;
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
CUfunction jit_cuda_fill_64 = nullptr;
CUfunction jit_cuda_mkperm_phase_1_shared = nullptr;
CUfunction jit_cuda_mkperm_phase_1_global = nullptr;
CUfunction jit_cuda_mkperm_phase_2_shared = nullptr;
CUfunction jit_cuda_mkperm_phase_2_global = nullptr;
CUfunction jit_cuda_transpose = nullptr;
CUfunction jit_cuda_scan_small = nullptr;
CUfunction jit_cuda_scan_large = nullptr;
CUfunction jit_cuda_scan_offset = nullptr;

CUfunction jit_cuda_reductions[(int) ReductionType::Count]
                              [(int) VarType::Count] = {};
int jit_cuda_devices = 0;

#define LOAD(name, ...)                                                        \
    symbol = strlen(__VA_ARGS__ "") > 0 ? (#name "_" __VA_ARGS__) : #name;     \
    name = decltype(name)(dlsym(jit_cuda_handle, symbol));                     \
    if (!name)                                                                 \
        break;                                                                 \
    symbol = nullptr
#define Z(x) x = nullptr

static bool jit_cuda_init_attempted = false;
static bool jit_cuda_init_success = false;
static CUmodule jit_cuda_module = nullptr;

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

#if defined(ENOKI_DYNAMIC_CUDA)
#if defined(__linux__)
    const char *cuda_fname  = "libcuda.so",
               *cuda_glob   = "/usr/lib/x86_64-linux-gnu/libcuda.so.*";
#else
    const char *cuda_fname  = "libcuda.dylib",
               *cuda_glob   = cuda_fname;
#endif

    jit_cuda_handle = jit_find_library(cuda_fname, cuda_glob, "ENOKI_LIBCUDA_PATH");

    if (!jit_cuda_handle) {
        jit_log(Warn, "jit_cuda_init(): %s could not be loaded -- "
                      "disabling CUDA backend! Set the 'ENOKI_LIBCUDA_PATH' "
                      "environment variable to specify its path.", cuda_fname);
        return false;
    }

    const char *symbol = nullptr;

    do {
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
        LOAD(cuFuncGetAttribute);
        LOAD(cuFuncSetAttribute);
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
        LOAD(cuMemPrefetchAsync, "ptsz");
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


    CUcontext context_0 = nullptr;
    for (int i = 0; i < jit_cuda_devices; ++i) {
        CUcontext context = nullptr;
        cuda_check(cuDevicePrimaryCtxRetain(&context, i));
        if (i == 0)
            context_0 = context;
    }
    cuda_check(cuCtxSetCurrent(context_0));

    // Decompress supplemental PTX content
    std::unique_ptr<char[]> uncompressed(
        new char[kernels_ptx_uncompressed_size + 1]);

    int zrv = inflate(kernels_ptx_compressed,
                     (uint32_t) sizeof(kernels_ptx_compressed),
                     uncompressed.get(),
                     (uint32_t) kernels_ptx_uncompressed_size);

    if (zrv != (int) kernels_ptx_uncompressed_size)
        jit_fail("jit_cuda_init(): decompression of precompiled kernels failed "
                 "(%i)!", zrv);
    uncompressed[kernels_ptx_uncompressed_size] = '\0';

    // .. and register it with CUDA
    cuda_check(cuModuleLoadData(&jit_cuda_module, uncompressed.get()));
    cuda_check(cuModuleGetFunction(&jit_cuda_fill_64, jit_cuda_module, "fill_64"));
    cuda_check(cuModuleGetFunction(&jit_cuda_mkperm_phase_1_shared, jit_cuda_module, "mkperm_phase_1_shared"));
    cuda_check(cuModuleGetFunction(&jit_cuda_mkperm_phase_2_shared, jit_cuda_module, "mkperm_phase_2_shared"));
    cuda_check(cuModuleGetFunction(&jit_cuda_mkperm_phase_1_global, jit_cuda_module, "mkperm_phase_1_global"));
    cuda_check(cuModuleGetFunction(&jit_cuda_mkperm_phase_2_global, jit_cuda_module, "mkperm_phase_2_global"));
    cuda_check(cuModuleGetFunction(&jit_cuda_transpose, jit_cuda_module, "transpose"));
    cuda_check(cuModuleGetFunction(&jit_cuda_scan_small, jit_cuda_module, "scan_small"));
    cuda_check(cuModuleGetFunction(&jit_cuda_scan_large, jit_cuda_module, "scan_large"));
    cuda_check(cuModuleGetFunction(&jit_cuda_scan_offset, jit_cuda_module, "scan_offset"));

    for (uint32_t i = 0; i < (uint32_t) ReductionType::Count; i++) {
        for (uint32_t j = 0; j < (uint32_t) VarType::Count; j++) {
            char name[16];
            CUfunction func;
            snprintf(name, sizeof(name), "reduce_%s_%s", reduction_name[i],
                     var_type_name_short[j]);
            CUresult rv = cuModuleGetFunction(&func, jit_cuda_module, name);
            if (rv == CUDA_ERROR_NOT_FOUND)
                continue;
            cuda_check(rv);
            jit_cuda_reductions[i][j] = func;
        }
    }

    jit_cuda_init_success = true;
    return true;
}

void jit_cuda_shutdown() {
    if (!jit_cuda_init_success)
        return;

    jit_log(Info, "jit_cuda_shutdown()");

    cuda_check(cuModuleUnload(jit_cuda_module));
    cuda_check(cuCtxSetCurrent(nullptr));

    for (int i = 0; i < jit_cuda_devices; ++i)
        cuda_check(cuDevicePrimaryCtxRelease(i));

    jit_cuda_module = nullptr;
    jit_cuda_devices = 0;

    jit_cuda_fill_64 = nullptr;
    jit_cuda_mkperm_phase_1_shared = nullptr;
    jit_cuda_mkperm_phase_2_shared = nullptr;
    jit_cuda_mkperm_phase_1_global = nullptr;
    jit_cuda_mkperm_phase_2_global = nullptr;
    jit_cuda_transpose = nullptr;
    jit_cuda_scan_small = nullptr;
    jit_cuda_scan_large = nullptr;
    jit_cuda_scan_offset = nullptr;

    memset(jit_cuda_reductions, 0, sizeof(jit_cuda_reductions));

#if defined(ENOKI_DYNAMIC_CUDA)
    Z(cuCtxEnablePeerAccess); Z(cuCtxSynchronize); Z(cuDeviceCanAccessPeer);
    Z(cuDeviceGet); Z(cuDeviceGetAttribute); Z(cuDeviceGetCount);
    Z(cuDeviceGetName); Z(cuDevicePrimaryCtxRelease);
    Z(cuDevicePrimaryCtxRetain); Z(cuDeviceTotalMem); Z(cuDriverGetVersion);
    Z(cuEventCreate); Z(cuEventDestroy); Z(cuEventRecord);
    Z(cuEventSynchronize); Z(cuFuncGetAttribute); Z(cuFuncSetAttribute);
    Z(cuGetErrorString); Z(cuInit); Z(cuLaunchHostFunc); Z(cuLaunchKernel);
    Z(cuLinkAddData); Z(cuLinkComplete); Z(cuLinkCreate); Z(cuLinkDestroy);
    Z(cuMemAdvise); Z(cuMemAlloc); Z(cuMemAllocHost); Z(cuMemAllocManaged);
    Z(cuMemFree); Z(cuMemFreeHost); Z(cuMemPrefetchAsync);
    Z(cuMemcpyAsync); Z(cuMemsetD16Async); Z(cuMemsetD32Async);
    Z(cuMemsetD8Async); Z(cuModuleGetFunction); Z(cuModuleLoadData);
    Z(cuModuleUnload); Z(cuOccupancyMaxPotentialBlockSize); Z(cuCtxSetCurrent);
    Z(cuStreamCreate); Z(cuStreamDestroy); Z(cuStreamSynchronize);
    Z(cuStreamWaitEvent);

    dlclose(jit_cuda_handle);
    jit_cuda_handle = nullptr;
#endif

    jit_cuda_init_success = false;
    jit_cuda_init_attempted = false;
}

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (unlikely(errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED)) {
        const char *msg = nullptr;
        cuGetErrorString(errval, &msg);
        jit_fail("cuda_check(): API error = %04d (\"%s\") in "
                 "%s:%i.", (int) errval, msg, file, line);
    }
}
