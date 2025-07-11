/*
    src/optix_api.cpp -- Low-level interface to OptiX API

    Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#define DR_OPTIX_SYM(...) __VA_ARGS__ = nullptr;

// We target ABI 87 and upgrade to 105 if it is available
#define DR_OPTIX_ABI_VERSION_87 87
#define DR_OPTIX_ABI_VERSION_105 105
#define DR_OPTIX_FUNCTION_TABLE_SIZE_87 48
#define DR_OPTIX_FUNCTION_TABLE_SIZE_105 52

#include "optix.h"
#include "optix_api.h"
#include "internal.h"
#include "log.h"

#if defined(_WIN32)
#  include <windows.h>
#  include <cfgmgr32.h>
#else
#  include <dlfcn.h>
#endif

static void *jitc_optix_handle = nullptr;

#if defined(_WIN32)
void *jitc_optix_win32_load_alternative();
#endif

static void *jitc_optix_table_87[DR_OPTIX_FUNCTION_TABLE_SIZE_87] { };
static void *jitc_optix_table_105[DR_OPTIX_FUNCTION_TABLE_SIZE_105] { };

static const char *jitc_optix_table_names_87[DR_OPTIX_FUNCTION_TABLE_SIZE_87] = {
    "optixGetErrorName",
    "optixGetErrorString",
    "optixDeviceContextCreate",
    "optixDeviceContextDestroy",
    "optixDeviceContextGetProperty",
    "optixDeviceContextSetLogCallback",
    "optixDeviceContextSetCacheEnabled",
    "optixDeviceContextSetCacheLocation",
    "optixDeviceContextSetCacheDatabaseSizes",
    "optixDeviceContextGetCacheEnabled",
    "optixDeviceContextGetCacheLocation",
    "optixDeviceContextGetCacheDatabaseSizes",
    "optixModuleCreate",
    "optixModuleCreateWithTasks",
    "optixModuleGetCompilationState",
    "optixModuleDestroy",
    "optixBuiltinISModuleGet",
    "optixTaskExecute",
    "optixProgramGroupCreate",
    "optixProgramGroupDestroy",
    "optixProgramGroupGetStackSize",
    "optixPipelineCreate",
    "optixPipelineDestroy",
    "optixPipelineSetStackSize",
    "optixAccelComputeMemoryUsage",
    "optixAccelBuild",
    "optixAccelGetRelocationInfo",
    "optixAccelCheckRelocationCompatibility",
    "optixAccelRelocate",
    "optixAccelCompact",
    "optixAccelEmitProperty",
    "optixConvertPointerToTraversableHandle",
    "optixOpacityMicromapArrayComputeMemoryUsage",
    "optixOpacityMicromapArrayBuild",
    "optixOpacityMicromapArrayGetRelocationInfo",
    "optixOpacityMicromapArrayRelocate",
    "optixDisplacementMicromapArrayComputeMemoryUsage",
    "optixDisplacementMicromapArrayBuild",
    "optixSbtRecordPackHeader",
    "optixLaunch",
    "optixDenoiserCreate",
    "optixDenoiserDestroy",
    "optixDenoiserComputeMemoryResources",
    "optixDenoiserSetup",
    "optixDenoiserInvoke",
    "optixDenoiserComputeIntensity",
    "optixDenoiserComputeAverageColor",
    "optixDenoiserCreateWithUserModel"
};

static const char *jitc_optix_table_names_105[DR_OPTIX_FUNCTION_TABLE_SIZE_105] = {
    "optixGetErrorName",
    "optixGetErrorString",
    "optixDeviceContextCreate",
    "optixDeviceContextDestroy",
    "optixDeviceContextGetProperty",
    "optixDeviceContextSetLogCallback",
    "optixDeviceContextSetCacheEnabled",
    "optixDeviceContextSetCacheLocation",
    "optixDeviceContextSetCacheDatabaseSizes",
    "optixDeviceContextGetCacheEnabled",
    "optixDeviceContextGetCacheLocation",
    "optixDeviceContextGetCacheDatabaseSizes",
    "optixModuleCreate",
    "optixModuleCreateWithTasks",
    "optixModuleGetCompilationState",
    "optixModuleDestroy",
    "optixBuiltinISModuleGet",
    "optixTaskExecute",
    "optixProgramGroupCreate",
    "optixProgramGroupDestroy",
    "optixProgramGroupGetStackSize",
    "optixPipelineCreate",
    "optixPipelineDestroy",
    "optixPipelineSetStackSize",
    "optixAccelComputeMemoryUsage",
    "optixAccelBuild",
    "optixAccelGetRelocationInfo",
    "optixCheckRelocationCompatibility",
    "optixAccelRelocate",
    "optixAccelCompact",
    "optixAccelEmitProperty",
    "optixConvertPointerToTraversableHandle",
    "optixOpacityMicromapArrayComputeMemoryUsage",
    "optixOpacityMicromapArrayBuild",
    "optixOpacityMicromapArrayGetRelocationInfo",
    "optixOpacityMicromapArrayRelocate",
    "optixDisplacementMicromapArrayComputeMemoryUsage",
    "optixDisplacementMicromapArrayBuild",
    "optixClusterAccelComputeMemoryUsage",
    "optixClusterAccelBuild",
    "optixSbtRecordPackHeader",
    "optixLaunch",
    "optixCoopVecMatrixConvert",
    "optixCoopVecMatrixComputeSize",
    "optixDenoiserCreate",
    "optixDenoiserDestroy",
    "optixDenoiserComputeMemoryResources",
    "optixDenoiserSetup",
    "optixDenoiserInvoke",
    "optixDenoiserComputeIntensity",
    "optixDenoiserComputeAverageColor",
    "optixDenoiserCreateWithUserModel"
};

bool jitc_optix_has_abi_105 = false;

bool jitc_optix_api_init() {
    if (jitc_optix_handle)
        return true;

    if (jitc_cuda_version_major == 12 && jitc_cuda_version_minor == 7) {
        jitc_log(
            Warn,
            "jit_optix_api_init(): DrJit considers the driver of your graphics "
            "card buggy and prone to miscompilation (we explicitly do not "
            "support OptiX with CUDA version 12.7, which corresponds to driver "
            "versions 565 to 559). Please install an older (< 565) or newer driver (>= 570).");
        return false;
    }

#if defined(_WIN32)
    const char* optix_fname = "nvoptix.dll";
#elif defined(__linux__)
    const char *optix_fname  = "libnvoptix.so.1";
#else
    const char *optix_fname  = "libnvoptix.dylib";
#endif

#if !defined(_WIN32)
    // Don't dlopen OptiX if it was loaded by another library
    if (dlsym(RTLD_NEXT, "optixLaunch"))
        jitc_optix_handle = RTLD_NEXT;
#endif

    if (!jitc_optix_handle) {
        jitc_optix_handle = jitc_find_library(optix_fname, optix_fname, "DRJIT_LIBOPTIX_PATH");

#if defined(_WIN32)
        if (!jitc_optix_handle)
            jitc_optix_handle = jitc_optix_win32_load_alternative();
#endif

        if (!jitc_optix_handle) {
            jitc_log(Error, "jit_optix_api_init(): %s could not be loaded -- "
                            "disabling OptiX backend! Set the DRJIT_LIBOPTIX_PATH "
                            "environment variable to specify its path. If you "
                            "are using the Windows Subsystem for Linux 2 (\"WSL 2\"), "
                            "please follow the setup instructions at "
                            "https://github.com/mitsuba-renderer/mitsuba3/blob/master/docs/src/optix_setup.rst "
                            "to make OptiX availablet to WSL.",
                            optix_fname);
            return false;
        }
    }

    // Load optixQueryFunctionTable from library
    optixQueryFunctionTable = decltype(optixQueryFunctionTable)(
        dlsym(jitc_optix_handle, "optixQueryFunctionTable"));

    if (!optixQueryFunctionTable) {
        jitc_log(Warn, "jit_optix_api_init(): could not find symbol optixQueryFunctionTable");
        jitc_optix_api_shutdown();
        return false;
    }

    int rv = optixQueryFunctionTable(DR_OPTIX_ABI_VERSION_105, 0, 0, 0,
                                     &jitc_optix_table_105,
                                     sizeof(jitc_optix_table_105));
    if (rv) {
        jitc_optix_has_abi_105 = false;
        memset(jitc_optix_table_105, 0, sizeof(jitc_optix_table_105));

        // Next, try ABI 87
        rv = optixQueryFunctionTable(DR_OPTIX_ABI_VERSION_87, 0, 0, 0,
                                     &jitc_optix_table_87,
                                     sizeof(jitc_optix_table_87));
        if (rv) {
            jitc_log(Warn,
                    "jit_optix_api_init(): Failed to load OptiX library! Very likely, "
                    "your NVIDIA graphics driver is too old and not compatible "
                    "with the version of OptiX that is being used. In particular, "
                    "OptiX 8.0 requires driver revision R535 or newer.");
            jitc_optix_api_shutdown();
            return false;
        }
    } else {
        jitc_optix_has_abi_105 = true;
    }

    #define LOAD(name) name = (decltype(name)) jitc_optix_lookup(#name)

    LOAD(optixGetErrorName);
    LOAD(optixGetErrorString);
    LOAD(optixDeviceContextCreate);
    LOAD(optixDeviceContextDestroy);
    LOAD(optixDeviceContextSetCacheEnabled);
    LOAD(optixDeviceContextSetCacheLocation);
    LOAD(optixModuleCreate);
    LOAD(optixModuleCreateWithTasks);
    LOAD(optixModuleGetCompilationState);
    LOAD(optixModuleDestroy);
    LOAD(optixTaskExecute);
    LOAD(optixProgramGroupCreate);
    LOAD(optixProgramGroupDestroy);
    LOAD(optixPipelineCreate);
    LOAD(optixPipelineDestroy);
    LOAD(optixLaunch);
    LOAD(optixSbtRecordPackHeader);
    LOAD(optixPipelineSetStackSize);
    LOAD(optixProgramGroupGetStackSize);

    if (jitc_optix_has_abi_105) {
        LOAD(optixCoopVecMatrixConvert);
        LOAD(optixCoopVecMatrixComputeSize);
    }

    #undef LOAD

    jitc_log(Info, "jit_optix_api_init(): loaded OptiX (via %s ABI).",
             jitc_optix_has_abi_105 ? "9.0" : "8.0");

    return true;
}

void jitc_optix_api_shutdown() {
    if (!jitc_optix_handle)
        return;

    jitc_log(Info, "jit_optix_api_shutdown()");

    #if !defined(_WIN32)
        if (jitc_optix_handle != RTLD_NEXT)
            dlclose(jitc_optix_handle);
    #else
        FreeLibrary((HMODULE) jitc_optix_handle);
    #endif
    jitc_optix_handle = nullptr;

    memset(jitc_optix_table_87, 0, sizeof(jitc_optix_table_87));
    memset(jitc_optix_table_105, 0, sizeof(jitc_optix_table_105));

    #define Z(x) x = nullptr
    Z(optixGetErrorName); Z(optixGetErrorString); Z(optixDeviceContextCreate);
    Z(optixDeviceContextDestroy); Z(optixDeviceContextSetCacheEnabled);
    Z(optixDeviceContextSetCacheLocation); Z(optixModuleCreate);
    Z(optixModuleCreateWithTasks); Z(optixModuleGetCompilationState);
    Z(optixModuleDestroy); Z(optixTaskExecute); Z(optixProgramGroupCreate);
    Z(optixProgramGroupDestroy); Z(optixPipelineCreate);
    Z(optixPipelineDestroy); Z(optixLaunch); Z(optixSbtRecordPackHeader);
    Z(optixPipelineSetStackSize); Z(optixProgramGroupGetStackSize);
    Z(optixCoopVecMatrixConvert); Z(optixCoopVecMatrixComputeSize);
    #undef Z
}

void *jitc_optix_lookup(const char *name) {
    if (jitc_optix_has_abi_105) {
        for (size_t i = 0; i < DR_OPTIX_FUNCTION_TABLE_SIZE_105; ++i) {
            if (strcmp(name, jitc_optix_table_names_105[i]) == 0)
                return jitc_optix_table_105[i];
        }
    } else {
        for (size_t i = 0; i < DR_OPTIX_FUNCTION_TABLE_SIZE_87; ++i) {
            if (strcmp(name, jitc_optix_table_names_87[i]) == 0)
                return jitc_optix_table_87[i];
        }
    }
    jitc_raise("jit_optix_lookup(): function \"%s\" not found!", name);
}

#if defined(_WIN32)
/**
 * Alternative way of finding OptiX based on the official API: nvoptix.dll
 * may not be on the path. Since it is co-located with the OpenGL drivers,
 * we should also enumerate all of them and double-check there.
 */
void *jitc_optix_win32_load_alternative() {
    const char *guid        = "{4d36e968-e325-11ce-bfc1-08002be10318}",
               *suffix      = "nvoptix.dll",
               *driver_name = "OpenGLDriverName";

    unsigned long size  = 0,
                  flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT,
                  suffix_len = (unsigned long) strlen(suffix);

    if (CM_Get_Device_ID_List_SizeA(&size, guid, flags))
        return nullptr;

    std::unique_ptr<char[]> dev_names(new char[size]);
    if (CM_Get_Device_ID_ListA(guid, dev_names.get(), size, flags))
        return nullptr;

    for (char *p = dev_names.get(); *p != '\0'; p += strlen(p) + 1) {
        unsigned long node_handle = 0;
        if (CM_Locate_DevNodeA(&node_handle, p, CM_LOCATE_DEVNODE_NORMAL))
            continue;

        HKEY reg_key = 0;
        if (CM_Open_DevNode_Key(node_handle, KEY_QUERY_VALUE, 0,
                                RegDisposition_OpenExisting, &reg_key,
                                CM_REGISTRY_SOFTWARE))
            continue;

        auto guard = scope_guard([reg_key]{ RegCloseKey(reg_key); });

        if (RegQueryValueExA(reg_key, driver_name, 0, 0, 0, &size))
            continue;

        std::unique_ptr<char[]> path(new char[size + suffix_len]);
        if (RegQueryValueExA(reg_key, driver_name, 0, 0, (LPBYTE) path.get(), &size))
            continue;

        for (int i = (int) size - 1; i >= 0 && path[i] != '\\'; --i)
            path[i] = '\0';

        strncat(path.get(), suffix, suffix_len);
        void* handle = (void *) LoadLibraryA(path.get());

        if (handle)
            return handle;
    }
    return nullptr;
}
#endif
