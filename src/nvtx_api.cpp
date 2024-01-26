/*
    src/nvtx_api.cpp-- Low-level interface to the NVTX profiling API

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "nvtx_api.h"
#include <stdlib.h>
#include <stdio.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#if defined(_MSC_VER)
#  pragma warning(disable : 4996) // '_wgetenv': This function or variable has been superseded
#  define NVTX_API __stdcall
#  define NVTX_STR(x) L##x
#  define NVTX_GETENV _wgetenv
#  define NVTX_CHAR wchar_t
#  define NVTX_DLOPEN(x) LoadLibraryW(x)
#  define NVTX_DLCLOSE FreeLibrary
#  define NVTX_DLSYM GetProcAddress
#else
#  define NVTX_STR(x) x
#  define NVTX_GETENV getenv
#  define NVTX_CHAR char
#  define NVTX_DLOPEN(x) dlopen(x, RTLD_LAZY)
#  define NVTX_DLCLOSE dlclose
#  define NVTX_DLSYM dlsym
#endif

#define NVTX_ETID_CALLBACKS  1
#define NVTX_ETID_VERSIONINFO 3
#define NVTX_CBID_CORE_SIZE 16
#define NVTX_CBID_CUDA_SIZE 9
#define NVTX_CBID_CORE2_SIZE 16
#define NVTX_CBID_CUDART_SIZE 7
#define NVTX_CBID_OPENCL_SIZE 15
#define NVTX_CBID_SYNC_SIZE 7

#define NVTX_CBID_ALL_SIZE                                                     \
    (NVTX_CBID_CORE_SIZE + NVTX_CBID_CUDA_SIZE + NVTX_CBID_OPENCL_SIZE +       \
     NVTX_CBID_CUDART_SIZE + NVTX_CBID_CORE2_SIZE + NVTX_CBID_SYNC_SIZE)

#define NVTX_CORE_2_OFFSET                                                     \
    (NVTX_CBID_CORE_SIZE + NVTX_CBID_CUDA_SIZE + NVTX_CBID_OPENCL_SIZE +       \
     NVTX_CBID_CUDART_SIZE)

#define NVTX_DOMAIN_MARK_EX           (NVTX_CORE_2_OFFSET + 1)
#define NVTX_DOMAIN_RANGE_PUSH_EX     (NVTX_CORE_2_OFFSET + 4)
#define NVTX_DOMAIN_RANGE_POP         (NVTX_CORE_2_OFFSET + 5)
#define NVTX_DOMAIN_REGISTER_STRING_A (NVTX_CORE_2_OFFSET + 10)
#define NVTX_DOMAIN_CREATE_A          (NVTX_CORE_2_OFFSET + 12)
#define NVTX_DOMAIN_DESTROY           (NVTX_CORE_2_OFFSET + 14)
#define NVTX_INITIALIZE               (NVTX_CORE_2_OFFSET + 15)

struct NvtxExportTableCallbacks {
    size_t struct_size;
    int(NVTX_API *GetModuleFunctionTable)(int, void ***, unsigned int *);
};

struct NvtxExportTableVersionInfo {
    size_t struct_size;
    uint32_t version;
    uint32_t reserved0;
    void(NVTX_API *SetInjectionNvtxVersion)(uint32_t);
};

static NvtxExportTableCallbacks nvtxCallbacks;
static NvtxExportTableVersionInfo nvtxVersionInfo;

static uint32_t nvtxSizes[]{ NVTX_CBID_CORE_SIZE,   NVTX_CBID_CUDA_SIZE,
                             NVTX_CBID_OPENCL_SIZE, NVTX_CBID_CUDART_SIZE,
                             NVTX_CBID_CORE2_SIZE,  NVTX_CBID_SYNC_SIZE };

static void *nvtxPointers[NVTX_CBID_ALL_SIZE];
static void **nvtxPointers_p[NVTX_CBID_ALL_SIZE];
nvtxDomainHandle_t nvtxDomain = nullptr;

void NVTX_API (*nvtxDomainMarkEx)(nvtxDomainHandle_t,
                                  const nvtxEventAttributes_t *) = nullptr;
int NVTX_API (*nvtxDomainRangePushEx)(nvtxDomainHandle_t,
                                      const nvtxEventAttributes_t *);
int NVTX_API (*nvtxDomainRangePop)(nvtxDomainHandle_t) = nullptr;
nvtxStringHandle_t
    NVTX_API (*nvtxDomainRegisterStringA)(nvtxDomainHandle_t,
                                          const char *) = nullptr;
nvtxDomainHandle_t NVTX_API (*nvtxDomainCreateA)(const char *) = nullptr;
void NVTX_API (*nvtxDomainDestroy)(nvtxDomainHandle_t) = nullptr;
void NVTX_API (*nvtxInitialize)(const void *) = nullptr;

static const void *NVTX_API nvtxGetExportTable(uint32_t etid) {
    switch (etid) {
        case NVTX_ETID_CALLBACKS:
            return &nvtxCallbacks;
        case NVTX_ETID_VERSIONINFO:
            return &nvtxVersionInfo;
        default:
            return 0;
    }
}

static void NVTX_API nvtxEtiSetInjectionNvtxVersion(uint32_t) {}

static int NVTX_API nvtxEtiGetModuleFunctionTable(
    int module_id, void ***out_table, unsigned int *out_size) {
    int module_count = (int) (sizeof(nvtxSizes) / sizeof(uint32_t));
    module_id -= 1;

    if (module_id >= 0 && module_id < module_count) {
        void **p = (void **) nvtxPointers_p;

        for (int i = 0; i < module_id; ++i)
            p += nvtxSizes[i];

        if (out_table)
            *out_table = (void **) p;
        if (out_size)
            *out_size = nvtxSizes[module_id];

        return 1;
    } else {
        return 0;
    }
}

static void *nvtxHandle = nullptr;

void jitc_nvtx_shutdown() {
    if (nvtxDomain && nvtxDomainDestroy) {
        nvtxDomainDestroy(nvtxDomain);
        nvtxDomain = nullptr;
    }

    if (nvtxHandle) {
        NVTX_DLCLOSE(nvtxHandle);
        nvtxHandle = nullptr;
    }

    nvtxDomainMarkEx = nullptr;
    nvtxDomainRangePushEx = nullptr;
    nvtxDomainRangePop = nullptr;
    nvtxDomainRegisterStringA = nullptr;
    nvtxDomainCreateA = nullptr;
    nvtxDomainDestroy = nullptr;
    nvtxInitialize = nullptr;
}

bool jitc_nvtx_init() {
    // Skip if already initialized
    if (nvtxDomain)
        return true;

    typedef int(NVTX_API * nvtxInitializeInjection)(void *exportTable);

    nvtxInitializeInjection init = nullptr;

    #if defined(_WIN32)
        init = (nvtxInitializeInjection) NVTX_DLSYM(
            nvtxHandle, "InitializeInjectionNvtx2"
        );
    #endif

    if (!init) {
        const NVTX_CHAR *libpath =
            NVTX_GETENV(NVTX_STR("NVTX_INJECTION64_PATH"));
        if (libpath) {
            nvtxHandle = NVTX_DLOPEN(libpath);
            if (nvtxHandle) {
                init = (nvtxInitializeInjection) NVTX_DLSYM(
                    nvtxHandle, "InitializeInjectionNvtx2");
            }
        }
    }

    if (init) {
        nvtxVersionInfo.struct_size = sizeof(NvtxExportTableVersionInfo);
        nvtxVersionInfo.version = NVTX_VERSION;
        nvtxVersionInfo.reserved0 = 0;
        nvtxVersionInfo.SetInjectionNvtxVersion =
            nvtxEtiSetInjectionNvtxVersion;
        nvtxCallbacks.struct_size = sizeof(NvtxExportTableCallbacks);
        nvtxCallbacks.GetModuleFunctionTable = nvtxEtiGetModuleFunctionTable;

        for (size_t i = 0; i < NVTX_CBID_ALL_SIZE; ++i) {
            nvtxPointers[i] = nullptr;
            nvtxPointers_p[i] = &nvtxPointers[i];
        }

        int rv = init((void *) nvtxGetExportTable);
        if (rv == 1) {
            nvtxDomainMarkEx          = (decltype(nvtxDomainMarkEx)) nvtxPointers[NVTX_DOMAIN_MARK_EX];
            nvtxDomainRangePushEx     = (decltype(nvtxDomainRangePushEx)) nvtxPointers[NVTX_DOMAIN_RANGE_PUSH_EX];
            nvtxDomainRangePop        = (decltype(nvtxDomainRangePop)) nvtxPointers[NVTX_DOMAIN_RANGE_POP];
            nvtxDomainRegisterStringA = (decltype(nvtxDomainRegisterStringA)) nvtxPointers[NVTX_DOMAIN_REGISTER_STRING_A];
            nvtxDomainCreateA         = (decltype(nvtxDomainCreateA)) nvtxPointers[NVTX_DOMAIN_CREATE_A];
            nvtxDomainDestroy         = (decltype(nvtxDomainDestroy)) nvtxPointers[NVTX_DOMAIN_DESTROY];
            nvtxInitialize            = (decltype(nvtxInitialize)) nvtxPointers[NVTX_INITIALIZE];

            if (nvtxDomainMarkEx && nvtxDomainRangePushEx && nvtxDomainRangePop &&
               nvtxDomainRegisterStringA && nvtxDomainCreateA && nvtxDomainDestroy) {
                if (nvtxInitialize)
                    nvtxInitialize(0);
                nvtxDomain = nvtxDomainCreateA("Dr.Jit");
                if (nvtxDomain)
                    return true;
            }
        }
    }

    jitc_nvtx_shutdown();
    return false;
}
