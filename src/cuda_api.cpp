#include "internal.h"
#include "log.h"
#include "var.h"
#include "util.h"
#include "io.h"
#include "../ptx/kernels.h"
#include <dlfcn.h>
#include <zlib.h>
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
CUresult (*cuFuncSetCacheConfig)(CUfunction, int) = nullptr;
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
CUfunction *jit_cuda_mkperm_phase_1_shared = nullptr;
CUfunction *jit_cuda_mkperm_phase_1_global = nullptr;
CUfunction *jit_cuda_mkperm_phase_3 = nullptr;
CUfunction *jit_cuda_mkperm_phase_4_shared = nullptr;
CUfunction *jit_cuda_mkperm_phase_4_global = nullptr;
CUfunction *jit_cuda_transpose = nullptr;
CUfunction *jit_cuda_scan_small_u8 = nullptr;
CUfunction *jit_cuda_scan_small_u32 = nullptr;
CUfunction *jit_cuda_scan_large_u8 = nullptr;
CUfunction *jit_cuda_scan_large_u32 = nullptr;
CUfunction *jit_cuda_scan_offset = nullptr;

CUfunction *jit_cuda_reductions[(int) ReductionType::Count]
                               [(int) VarType::Count] = {};
int jit_cuda_devices = 0;

static bool jit_cuda_init_attempted = false;
static bool jit_cuda_init_success = false;

bool jit_cuda_init() {
    if (jit_cuda_init_attempted)
        return jit_cuda_init_success;
    jit_cuda_init_attempted = true;

    // We have our own caching scheme, disable CUDA's JIT cache
#ifdef _MSC_VER
    _putenv("CUDA_CACHE_DISABLE=1");
#else
    putenv((char *) "CUDA_CACHE_DISABLE=1");
#endif

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
        LOAD(cuFuncSetCacheConfig);
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

    // Decompress supplemental PTX content
    char *uncompressed =
        (char *) malloc_check(kernels_ptx_size_uncompressed + kernels_dict_size + 1);
    memcpy(uncompressed, kernels_dict, kernels_dict_size);

    if (LZ4_decompress_safe_usingDict((const char *) kernels_ptx,
                            uncompressed + kernels_dict_size,
                            (int) kernels_ptx_size_compressed,
                            (int) kernels_ptx_size_uncompressed,
                            uncompressed,
                            (int) kernels_dict_size) !=
        (int) kernels_ptx_size_uncompressed)
        jit_fail("jit_cuda_init(): decompression of precompiled kernels failed!");

    char *uncompressed_ptx = uncompressed + kernels_dict_size;
    uncompressed_ptx[kernels_ptx_size_uncompressed] = '\0';

    for (uint32_t j = 0; j < (uint32_t) ReductionType::Count; j++)
        for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++)
            jit_cuda_reductions[j][k] =
                (CUfunction *) malloc_check(sizeof(CUfunction) * jit_cuda_devices);

    jit_cuda_module = (CUmodule *) malloc_check(sizeof(CUmodule) * jit_cuda_devices);

    for (int i = 0; i < jit_cuda_devices; ++i) {
        CUcontext context = nullptr;
        cuda_check(cuDevicePrimaryCtxRetain(&context, i));
        cuda_check(cuCtxSetCurrent(context));
        int cc_minor, cc_major;

        cuda_check(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
        cuda_check(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));

        size_t hash = kernels_ptx_hash;
        hash_combine(hash, cc_minor + cc_major * 10);

        Kernel kernel;
        if (!jit_kernel_load(uncompressed_ptx, kernels_ptx_size_uncompressed, false, hash, kernel)) {
            jit_cuda_compile(uncompressed_ptx, kernels_ptx_size_uncompressed, kernel);
            jit_kernel_write(uncompressed_ptx, kernels_ptx_size_uncompressed, false, hash, kernel);
        }

        // .. and register it with CUDA
        CUmodule m;
        cuda_check(cuModuleLoadData(&m, kernel.data));
        free(kernel.data);
        jit_cuda_module[i] = m;

        #define LOAD(name)                                                       \
            if (i == 0)                                                          \
                jit_cuda_##name = (CUfunction *) malloc_check(                   \
                    sizeof(CUfunction) * jit_cuda_devices);                      \
            cuda_check(cuModuleGetFunction(&jit_cuda_##name[i], m, #name));

        LOAD(fill_64);
        LOAD(mkperm_phase_1_shared);
        LOAD(mkperm_phase_1_global);
        LOAD(mkperm_phase_3);
        LOAD(mkperm_phase_4_shared);
        LOAD(mkperm_phase_4_global);
        LOAD(transpose);
        LOAD(scan_small_u8);
        LOAD(scan_small_u32);
        LOAD(scan_large_u8);
        LOAD(scan_large_u32);
        LOAD(scan_offset);

        #undef LOAD

        for (uint32_t j = 0; j < (uint32_t) ReductionType::Count; j++) {
            for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
                char name[16];
                CUfunction func;
                snprintf(name, sizeof(name), "reduce_%s_%s", reduction_name[j],
                         var_type_name_short[k]);
                CUresult rv = cuModuleGetFunction(&func, m, name);
                if (rv == CUDA_ERROR_NOT_FOUND)
                    continue;
                cuda_check(rv);
                if (i == 0)
                    jit_cuda_reductions[j][k] = (CUfunction *) malloc_check(
                        sizeof(CUfunction) * jit_cuda_devices);
                jit_cuda_reductions[j][k][i] = func;
            }
        }
    }
    cuda_check(cuCtxSetCurrent(nullptr));
    free(uncompressed);

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
    kernel.size = link_output_size;
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
    Z(jit_cuda_mkperm_phase_1_shared);
    Z(jit_cuda_mkperm_phase_1_global);
    Z(jit_cuda_mkperm_phase_3);
    Z(jit_cuda_mkperm_phase_4_shared);
    Z(jit_cuda_mkperm_phase_4_global);
    Z(jit_cuda_transpose);
    Z(jit_cuda_scan_small_u8);
    Z(jit_cuda_scan_small_u32);
    Z(jit_cuda_scan_large_u8);
    Z(jit_cuda_scan_large_u32);
    Z(jit_cuda_scan_offset);
    Z(jit_cuda_module);

    for (uint32_t j = 0; j < (uint32_t) ReductionType::Count; j++)
        for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++)
            Z(jit_cuda_reductions[j][k]);

    #undef Z

#if defined(ENOKI_DYNAMIC_CUDA)
    #define Z(x) x = nullptr

    Z(cuCtxEnablePeerAccess); Z(cuCtxSynchronize); Z(cuDeviceCanAccessPeer);
    Z(cuDeviceGet); Z(cuDeviceGetAttribute); Z(cuDeviceGetCount);
    Z(cuDeviceGetName); Z(cuDevicePrimaryCtxRelease);
    Z(cuDevicePrimaryCtxRetain); Z(cuDeviceTotalMem); Z(cuDriverGetVersion);
    Z(cuEventCreate); Z(cuEventDestroy); Z(cuEventRecord);
    Z(cuEventSynchronize); Z(cuFuncSetAttribute); Z(cuFuncSetCacheConfig);
    Z(cuGetErrorString); Z(cuInit); Z(cuLaunchHostFunc); Z(cuLaunchKernel);
    Z(cuLinkAddData); Z(cuLinkComplete); Z(cuLinkCreate); Z(cuLinkDestroy);
    Z(cuMemAdvise); Z(cuMemAlloc); Z(cuMemAllocHost); Z(cuMemAllocManaged);
    Z(cuMemHostRegister); Z(cuMemHostUnregister); Z(cuMemFree);
    Z(cuMemFreeHost); Z(cuMemPrefetchAsync); Z(cuMemcpy); Z(cuMemcpyAsync);
    Z(cuMemsetD16Async); Z(cuMemsetD32Async); Z(cuMemsetD8Async);
    Z(cuModuleGetFunction); Z(cuModuleLoadData); Z(cuModuleUnload);
    Z(cuOccupancyMaxPotentialBlockSize); Z(cuCtxSetCurrent); Z(cuStreamCreate);
    Z(cuStreamDestroy); Z(cuStreamSynchronize); Z(cuStreamWaitEvent);

    dlclose(jit_cuda_handle);
    jit_cuda_handle = nullptr;

    #undef Z
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
