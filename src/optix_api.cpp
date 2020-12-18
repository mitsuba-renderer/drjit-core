#include <enoki-jit/optix.h>
#include "optix_api.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "internal.h"
#include <unistd.h>

#define OPTIX_ABI_VERSION 41

#if defined(_WIN32)
#  include <windows.h>
#  include <cfgmgr32.h>
#else
#  include <dlfcn.h>
#endif

static void *jit_optix_table[38] { };
static const char *jit_optix_table_names[38] = {
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
    "optixModuleCreateFromPTX",
    "optixModuleDestroy",
    "optixBuiltinISModuleGet",
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
    "optixConvertPointerToTraversableHandle",
    "optixSbtRecordPackHeader",
    "optixLaunch",
    "optixDenoiserCreate",
    "optixDenoiserDestroy",
    "optixDenoiserComputeMemoryResources",
    "optixDenoiserSetup",
    "optixDenoiserInvoke",
    "optixDenoiserSetModel",
    "optixDenoiserComputeIntensity",
    "optixDenoiserComputeAverageColor"
};

using OptixResult = int;
using OptixProgramGroupKind = int;
using OptixCompileDebugLevel = int;
using OptixDeviceContext = void*;
using OptixLogCallback = void (*)(unsigned int, const char *, const char *, void *);

#define OPTIX_COMPILE_DEBUG_LEVEL_NONE           0x2350
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF 0
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL ((int) 0xFFFFFFFF)
#define OPTIX_PROGRAM_GROUP_KIND_RAYGEN          0x2421
#define OPTIX_SBT_RECORD_HEADER_SIZE             32

struct OptixDeviceContextOptions {
    OptixLogCallback logCallbackFunction;
    void *logCallbackData;
    int logCallbackLevel;
    int validationMode;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
};

struct OptixPipelineLinkOptions {
    unsigned int maxTraceDepth;
    OptixCompileDebugLevel debugLevel;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char* entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char* entryFunctionNameCH;
    OptixModule moduleAH;
    const char* entryFunctionNameAH;
    OptixModule moduleIS;
    const char* entryFunctionNameIS;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixProgramGroupOptions {
    int placeholder;
};

struct OptixShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr  missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    CUdeviceptr  hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    CUdeviceptr  callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

OptixResult (*optixQueryFunctionTable)(int, unsigned int, void *, const void **,
                                       void *, size_t) = nullptr;
const char *(*optixGetErrorName)(OptixResult r) = nullptr;
const char *(*optixGetErrorString)(OptixResult ) = nullptr;
OptixResult (*optixDeviceContextCreate)(
    CUcontext fromContext, const OptixDeviceContextOptions *,
    OptixDeviceContext *context) = nullptr;
OptixResult (*optixDeviceContextDestroy)(OptixDeviceContext) = nullptr;
OptixResult (*optixDeviceContextSetCacheEnabled)(OptixDeviceContext, int) = nullptr;
OptixResult (*optixDeviceContextSetCacheLocation)(OptixDeviceContext, const char *) = nullptr;
OptixResult (*optixModuleCreateFromPTX)(OptixDeviceContext,
                                        const OptixModuleCompileOptions *,
                                        const OptixPipelineCompileOptions *,
                                        const char *, size_t, char *, size_t *,
                                        OptixModule *) = nullptr;
OptixResult (*optixModuleDestroy)(OptixModule) = nullptr;
OptixResult (*optixProgramGroupCreate)(OptixDeviceContext,
                                       const OptixProgramGroupDesc *,
                                       unsigned int,
                                       const OptixProgramGroupOptions *, char *,
                                       size_t *, OptixProgramGroup *) = nullptr;
OptixResult (*optixProgramGroupDestroy)(OptixProgramGroup) = nullptr;

OptixResult (*optixPipelineCreate)(OptixDeviceContext,
                                   const OptixPipelineCompileOptions *,
                                   const OptixPipelineLinkOptions *,
                                   const OptixProgramGroup *, unsigned int,
                                   char *, size_t *, OptixPipeline *) = nullptr;
OptixResult (*optixPipelineDestroy)(OptixPipeline) = nullptr;
OptixResult (*optixLaunch)(OptixPipeline, CUstream, CUdeviceptr, size_t,
                           const OptixShaderBindingTable *, unsigned int,
                           unsigned int, unsigned int) = nullptr;
OptixResult (*optixSbtRecordPackHeader)(OptixProgramGroup, void*) = nullptr;

#define jit_optix_check(err) jit_optix_check_impl((err), __FILE__, __LINE__)
extern void jit_optix_check_impl(OptixResult errval, const char *file, const int line);

#if !defined(_WIN32)
void *jit_optix_win32_load_alternative();
#endif

static bool jit_optix_init_attempted = false;
static bool jit_optix_init_success = false;
static void *jit_optix_handle = nullptr;

bool jit_optix_init() {
    if (jit_optix_init_attempted)
        return jit_optix_init_success;

    jit_optix_init_attempted = true;
    jit_optix_handle = nullptr;

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
        jit_optix_handle = RTLD_NEXT;
#endif

    if (!jit_optix_handle) {
        jit_optix_handle = jit_find_library(optix_fname, optix_fname, "ENOKI_LIBOPTIX_PATH");

#if defined(_WIN32)
        if (!jit_optix_handle)
            jit_optix_handle = jit_optix_win32_load_alternative();
#endif

        if (!jit_optix_handle) {
            jit_log(Warn, "jit_optix_init(): %s could not be loaded -- "
                          "disabling OptiX backend! Set the ENOKI_LIBOPTIX_PATH "
                          "environment variable to specify its path.", optix_fname);
            return false;
        }
    }

    // Load optixQueryFunctionTable from library
    optixQueryFunctionTable = decltype(optixQueryFunctionTable)(
        dlsym(jit_optix_handle, "optixQueryFunctionTable"));

    if (!optixQueryFunctionTable) {
        jit_log(Warn, "jit_optix_init(): could not find symbol optixQueryFunctionTable");
        return false;
    }

    int rv = optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0,
                                     &jit_optix_table, sizeof(jit_optix_table));
    if (rv) {
        jit_log(Warn,
                "jit_optix_init(): Failed to load OptiX library! Very likely, "
                "your NVIDIA graphics driver is too old and not compatible "
                "with the version of OptiX that is being used. In particular, "
                "OptiX 7.2 requires driver revision R455.28 or newer on Linux "
                "and R456.71 or newer on Windows.");

        return false;
    }

    #define LOOKUP(name) name = (decltype(name)) jit_optix_lookup(#name)

    LOOKUP(optixGetErrorName);
    LOOKUP(optixGetErrorString);
    LOOKUP(optixDeviceContextCreate);
    LOOKUP(optixDeviceContextDestroy);
    LOOKUP(optixDeviceContextSetCacheEnabled);
    LOOKUP(optixDeviceContextSetCacheLocation);
    LOOKUP(optixModuleCreateFromPTX);
    LOOKUP(optixModuleDestroy);
    LOOKUP(optixProgramGroupCreate);
    LOOKUP(optixProgramGroupDestroy);
    LOOKUP(optixPipelineCreate);
    LOOKUP(optixPipelineDestroy);
    LOOKUP(optixLaunch);
    LOOKUP(optixSbtRecordPackHeader);

    #undef LOOKUP

    jit_log(LogLevel::Info,
            "jit_optix_init(): loaded OptiX (via 7.2 ABI).");

    jit_optix_init_success = true;
    return true;
}

void jit_optix_log(unsigned int /* level */, const char *tag, const char *message, void *) {
    size_t len = strlen(message);
    fprintf(stderr, "jitc_optix_log(): [%s] %s%s", tag, message,
            (len > 0 && message[len - 1] == '\n') ? "" : "\n");
}

OptixDeviceContext jit_optix_context() {
    ThreadState *ts = thread_state(true);

    if (ts->optix_context)
        return ts->optix_context;

    if (!jit_optix_init())
        jit_raise("Could not create OptiX context!");

    int log_level = std::max(0, std::max((int) state.log_level_stderr,
                                         (int) state.log_level_callback) - 1);

    OptixDeviceContextOptions ctx_opts {
        jit_optix_log, nullptr, log_level,

#if defined(NDEBUG)
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
#else
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
#endif
    };

    OptixDeviceContext ctx;
    jit_optix_check(optixDeviceContextCreate(ts->context, &ctx_opts, &ctx));

#if !defined(_WIN32)
    jit_optix_check(optixDeviceContextSetCacheLocation(ctx, jit_temp_path));
#else
    char path[PATH_MAX];
    wcstombs(path, jit_temp_path, PATH_MAX);
    jit_optix_check(optixDeviceContextSetCacheLocation(ctx, path));
#endif
    jit_optix_check(optixDeviceContextSetCacheEnabled(ctx, 1));

    ts->optix_context = ctx;

    return ctx;
}

void jit_optix_context_destroy(ThreadState *ts) {
    jit_optix_check(optixDeviceContextDestroy(ts->optix_context));
}

void *jit_optix_lookup(const char *name) {
    for (int i = 0; i < 38; ++i) {
        if (strcmp(name, jit_optix_table_names[i]) == 0)
            return jit_optix_table[i];
    }
    jit_raise("jit_optix_lookup(): function \"%s\" not found!", name);
}

void jit_optix_configure(const OptixPipelineCompileOptions *pco,
                         const OptixShaderBindingTable *sbt,
                         const OptixProgramGroup *pg,
                         uint32_t pg_count) {
    ThreadState *ts = thread_state(true);
    jit_log(Info, "jit_optix_configure()");
    ts->optix_pipeline_compile_options = pco;
    ts->optix_shader_binding_table = sbt;

    ts->optix_program_groups.clear();
    ts->optix_program_groups.push_back(nullptr);
    for (uint32_t i = 0; i < pg_count; ++i)
        ts->optix_program_groups.push_back(pg[i]);
}

void jit_optix_compile(ThreadState *ts, const char *buffer,
                       size_t buffer_size, Kernel &kernel,
                       uint64_t kernel_hash) {
    char error_log[16384];

    // =====================================================
    // 1. Compile an OptiX module
    // =====================================================

    OptixModuleCompileOptions module_opts { };
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    size_t log_size = sizeof(error_log);
    int rv = optixModuleCreateFromPTX(
        ts->optix_context, &module_opts, ts->optix_pipeline_compile_options,
        buffer, buffer_size, error_log, &log_size, &kernel.optix.mod);
    if (rv) {
        jit_log(Warn, "jit_optix_compile(): optixModuleCreateFromPTX(): %s",
                error_log);
        jit_optix_check(rv);
    }

    // =====================================================
    // 2. Create an OptiX program group
    // =====================================================

    char kernel_name[33];
    snprintf(kernel_name, sizeof(kernel_name), "__raygen__enoki_%016llx",
             (unsigned long long) kernel_hash);

    OptixProgramGroupOptions pgo { };
    OptixProgramGroupDesc pgd { };
    pgd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgd.raygen.module = kernel.optix.mod;
    pgd.raygen.entryFunctionName = kernel_name;

    log_size = sizeof(error_log);
    rv = optixProgramGroupCreate(ts->optix_context, &pgd, 1, &pgo, error_log,
                                 &log_size, &kernel.optix.group);
    if (rv) {
        jit_log(Warn, "jit_optix_compile(): optixProgramGroupCreate(): %s",
                error_log);
        jit_optix_check(rv);
    }

    ts->optix_program_groups[0] = kernel.optix.group;

    void *sbt_record =
        jit_malloc(AllocType::Host, OPTIX_SBT_RECORD_HEADER_SIZE);
    jit_optix_check(optixSbtRecordPackHeader(kernel.optix.group, sbt_record));

    kernel.optix.sbt_record =
        jit_malloc_migrate(sbt_record, AllocType::Device, 1);

    // =====================================================
    // 3. Create an OptiX pipeline
    // =====================================================

    OptixPipelineLinkOptions link_options {};
    link_options.maxTraceDepth = 1;
    link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    log_size = sizeof(error_log);
    rv = optixPipelineCreate(
        ts->optix_context, ts->optix_pipeline_compile_options, &link_options,
        ts->optix_program_groups.data(), ts->optix_program_groups.size(),
        error_log, &log_size, &kernel.optix.pipeline);
    if (rv) {
        jit_log(Warn, "jit_optix_compile(): optixProgramGroupCreate(): %s",
                error_log);
        jit_optix_check(rv);
    }

    kernel.size = 0;
    kernel.data = nullptr;
}

void jit_optix_free(const Kernel &kernel) {
    jit_optix_check(optixPipelineDestroy(kernel.optix.pipeline));
    jit_optix_check(optixProgramGroupDestroy(kernel.optix.group));
    jit_optix_check(optixModuleDestroy(kernel.optix.mod));
    jit_free(kernel.optix.sbt_record);
}

void jit_optix_launch(ThreadState *ts, const Kernel &kernel, uint32_t size,
                      const void *args, uint32_t args_size) {
    OptixShaderBindingTable sbt;
    memcpy(&sbt, ts->optix_shader_binding_table,
           sizeof(OptixShaderBindingTable));
    sbt.raygenRecord = (CUdeviceptr) kernel.optix.sbt_record;

    jit_optix_check(optixLaunch(kernel.optix.pipeline, ts->stream,
                                (CUdeviceptr) args, args_size, &sbt, size, 1, 1));
}

void jit_optix_trace(uint32_t nargs, uint32_t *args) {
    VarType types[] { VarType::UInt64,  VarType::Float32, VarType::Float32,
                      VarType::Float32, VarType::Float32, VarType::Float32,
                      VarType::Float32, VarType::Float32, VarType::Float32,
                      VarType::Float32, VarType::UInt32,  VarType::UInt32,
                      VarType::UInt32,  VarType::UInt32,  VarType::UInt32 };

    if (nargs < 15)
        jit_raise("jit_optix_trace(): too few arguments (got %u < 15)", nargs);
    else if (nargs > 23)
        jit_raise("jit_optix_trace(): too many arguments (got %u > 23)", nargs);

    for (uint32_t i = 0; i < nargs; ++i) {
        VarType vt = jit_var_type(args[i]);
        if (i < 15) {
            if (vt != types[i])
                jit_raise("jit_optix_trace(): type mismatch for arg. %u", i);
        } else {
            if (var_type_size[(int) vt] != 4)
                jit_raise("jit_optix_trace(): size mismatch for arg. %u", i);
        }
    }

    uint32_t decl = jit_var_new_0(1, VarType::Invalid, "", 1, 1), index = decl;
    jit_var_inc_ref_ext(index);

    Buffer buf(100);
    for (uint32_t i = 0; i < nargs; ++i) {
        VarType vt = jit_var_type(args[i]);
        const char *tname = var_type_name_ptx[(int) vt],
                   *tname_bin = tname;

        if (i >= 25) {
            tname = "u32";
            tname_bin = "b32";
        }

        buf.clear();
        buf.fmt(".reg.%s $r1_%u$n"
                "mov.%s $r1_%u, $r2", tname, i, tname_bin, i);
        uint32_t prev = index;
        index = jit_var_new_3(1, VarType::Invalid, buf.get(), 0, decl, args[i], index);
        jit_var_dec_ref_ext(prev);
    }

    buf.clear();
    buf.put("call (");
    for (uint32_t i = 15; i < nargs; ++i)
        buf.fmt("$r1_%u%s", i, i + 1 < nargs ? ", " : "");
    buf.fmt("), _optix_trace_%u, (", nargs - 15);
    for (uint32_t i = 0; i < nargs; ++i)
        buf.fmt("$r1_%u%s", i, i + 1 < nargs ? ", " : "");
    buf.put(")");

    uint32_t prev = index;
    index = jit_var_new_2(1, VarType::Invalid, buf.get(), 0, decl, index);
    jit_var_dec_ref_ext(prev);

    jit_var(index)->optix = 1;

    for (uint32_t i = 15; i < nargs; ++i) {
        buf.clear();
        buf.fmt("mov.b32 $r0, $r1_%u", i);
        args[i] = jit_var_new_2(1, (VarType) jit_var_type(args[i]), buf.get(), 0, decl, index);
    }

    jit_var_dec_ref_ext(index);
    jit_var_dec_ref_ext(decl);
}

void jit_optix_shutdown() {
    if (!jit_optix_init_success)
        return;

    jit_log(Info, "jit_optix_shutdown()");

#if defined(ENOKI_JIT_DYNAMIC_OPTIX)
    #define Z(x) x = nullptr

    #if !defined(_WIN32)
        if (jit_optix_handle != RTLD_NEXT)
            dlclose(jit_optix_handle);
    #else
        FreeLibrary((HMODULE) jit_optix_handle);
    #endif

    memset(jit_optix_table, 0, sizeof(jit_optix_table));

    Z(optixGetErrorName);
    Z(optixGetErrorString);
    Z(optixDeviceContextCreate);
    Z(optixDeviceContextDestroy);
    Z(optixDeviceContextSetCacheEnabled);
    Z(optixDeviceContextSetCacheLocation);
    Z(optixModuleCreateFromPTX);
    Z(optixModuleDestroy);
    Z(optixProgramGroupCreate);
    Z(optixProgramGroupDestroy);
    Z(optixPipelineCreate);
    Z(optixPipelineDestroy);
    Z(optixLaunch);
    Z(optixSbtRecordPackHeader);

    jit_optix_handle = nullptr;

    #undef Z
#endif

    jit_optix_init_success = false;
    jit_optix_init_attempted = false;
}

void jit_optix_check_impl(OptixResult errval, const char *file,
                          const int line) {
    if (unlikely(errval != 0)) {
        const char *name = optixGetErrorName(errval),
                   *msg  = optixGetErrorString(errval);
        jit_fail("jit_optix_check(): API error %04i (%s): \"%s\" in "
                 "%s:%i.", (int) errval, name, msg, file, line);
    }
}

void jitc_optix_check_impl(int errval, const char *file, const int line) {
    if (errval) {
        lock_guard guard(state.mutex);
        jit_optix_check_impl(errval, file, line);
    }
}

#if defined(_WIN32)
/**
 * Alternative way of finding OptiX based on the official API: nvoptix.dll
 * may not be on the path. Since it is co-located with the OpenGL drivers,
 * we should also enumerate all of them and double-check there.
 */
void *jit_optix_win32_load_alternative() {
    const wchar_t *guid        = L"{4d36e968-e325-11ce-bfc1-08002be10318}",
          wchar_t *suffix      = L"nvoptix.dll";
          wchar_t *driver_name = L"OpenGLDriverName";

    uint32_t size = 0,
             flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;

    if (CM_Get_Device_ID_List_SizeW(&size, guid, flags))
        return nullptr;

    std::unique_ptr<wchar_t[]> dev_names(new wchar_t[size]);
    if (CM_Get_Device_ID_ListW(guid, dev_names.get(), size, flags))
        return nullptr;

    for (wchar_t *p = dev_names.get(); *p != '\0'; p += wcslen(p) + 1) {
        uint32_t node_handle = 0;
        if (CM_Locate_DevNodeW(&handle, p, CM_LOCATE_DEVNODE_NORMAL))
            continue;

        HKEY reg_key = 0;
        if (CM_Open_DevNode_Key(node_handle, KEY_QUERY_VALUE, 0,
                                RegDisposition_OpenExisting, &reg_key,
                                CM_REGISTRY_SOFTWARE))
            continue;

        scoped_guard guard([reg_key]{ RegCloseKey(reg_key); });

        if (RegQueryValueExW(reg_key, driver_name, 0, 0, 0, &size))
            continue;

        std::unique_ptr<wchar_t[]> path = new wchar_t[size + wcslen(suffix)];
        if (RegQueryValueExW(reg_key, driver_name, 0, 0, path.get(), &size))
            continue;

        for (int i = (int) size - 1; i >= 0 && path[i] != L'\\'; --i)
            path[i] = L'\0';

        wcsncat(path.get(), suffix, wcslen(suffix));
        void* handle = (void *) LoadLibraryW(path.get());

        if (handle)
            return handle;
    }
}

#endif
