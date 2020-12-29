#include <enoki-jit/optix.h>
#include <tsl/robin_map.h>
#include "optix_api.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "internal.h"

#define OPTIX_ABI_VERSION 41

#if defined(_WIN32)
#  include <windows.h>
#  include <cfgmgr32.h>
#else
#  include <dlfcn.h>
#endif

static void *jitc_optix_table[38] { };
static const char *jitc_optix_table_names[38] = {
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
#define OPTIX_PROGRAM_GROUP_KIND_CALLABLES       0x2425

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

struct OptixProgramGroupCallables {
    OptixModule moduleDC;
    const char* entryFunctionNameDC;
    OptixModule moduleCC;
    const char* entryFunctionNameCC;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupCallables callables;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixProgramGroupOptions {
    int opaque;
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

#define jitc_optix_check(err) jitc_optix_check_impl((err), __FILE__, __LINE__)
extern void jitc_optix_check_impl(OptixResult errval, const char *file, const int line);

#if defined(_WIN32)
void *jitc_optix_win32_load_alternative();
#endif

static bool jitc_optix_init_attempted = false;
static bool jitc_optix_init_success = false;
static void *jitc_optix_handle = nullptr;

bool jitc_optix_init() {
    if (jitc_optix_init_attempted)
        return jitc_optix_init_success;

    jitc_optix_init_attempted = true;
    jitc_optix_handle = nullptr;

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
        jitc_optix_handle = jitc_find_library(optix_fname, optix_fname, "ENOKI_LIBOPTIX_PATH");

#if defined(_WIN32)
        if (!jitc_optix_handle)
            jitc_optix_handle = jitc_optix_win32_load_alternative();
#endif

        if (!jitc_optix_handle) {
            jitc_log(Warn, "jit_optix_init(): %s could not be loaded -- "
                          "disabling OptiX backend! Set the ENOKI_LIBOPTIX_PATH "
                          "environment variable to specify its path.", optix_fname);
            return false;
        }
    }

    // Load optixQueryFunctionTable from library
    optixQueryFunctionTable = decltype(optixQueryFunctionTable)(
        dlsym(jitc_optix_handle, "optixQueryFunctionTable"));

    if (!optixQueryFunctionTable) {
        jitc_log(Warn, "jit_optix_init(): could not find symbol optixQueryFunctionTable");
        return false;
    }

    int rv = optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0,
                                     &jitc_optix_table, sizeof(jitc_optix_table));
    if (rv) {
        jitc_log(Warn,
                "jit_optix_init(): Failed to load OptiX library! Very likely, "
                "your NVIDIA graphics driver is too old and not compatible "
                "with the version of OptiX that is being used. In particular, "
                "OptiX 7.2 requires driver revision R455.28 or newer on Linux "
                "and R456.71 or newer on Windows.");

        return false;
    }

    #define LOOKUP(name) name = (decltype(name)) jitc_optix_lookup(#name)

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

    jitc_log(LogLevel::Info,
            "jit_optix_init(): loaded OptiX (via 7.2 ABI).");

    jitc_optix_init_success = true;
    return true;
}

void jitc_optix_log(unsigned int /* level */, const char *tag, const char *message, void *) {
    size_t len = strlen(message);
    fprintf(stderr, "jit_optix_log(): [%s] %s%s", tag, message,
            (len > 0 && message[len - 1] == '\n') ? "" : "\n");
}

OptixDeviceContext jitc_optix_context() {
    ThreadState *ts = thread_state(true);

    if (ts->optix_context)
        return ts->optix_context;

    if (!jitc_optix_init())
        jitc_raise("Could not create OptiX context!");

    int log_level = std::max((int) state.log_level_stderr,
                             (int) state.log_level_callback) + 1;

    OptixDeviceContextOptions ctx_opts {
        jitc_optix_log, nullptr,
        std::max(0, std::min(4, log_level)),
#if defined(NDEBUG)
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
#else
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
#endif
    };

    OptixDeviceContext ctx;
    jitc_optix_check(optixDeviceContextCreate(ts->context, &ctx_opts, &ctx));

#if !defined(_WIN32)
    jitc_optix_check(optixDeviceContextSetCacheLocation(ctx, jitc_temp_path));
#else
    size_t len = wcstombs(nullptr, jitc_temp_path, 0) + 1;
    std::unique_ptr<char[]> temp(new char[len]);
    wcstombs(temp.get(), jitc_temp_path, len);
    jitc_optix_check(optixDeviceContextSetCacheLocation(ctx, temp.get()));
#endif
    jitc_optix_check(optixDeviceContextSetCacheEnabled(ctx, 1));

    ts->optix_context = ctx;

    return ctx;
}

void jitc_optix_context_destroy(ThreadState *ts) {
    if (ts->optix_context) {
        jitc_optix_check(optixDeviceContextDestroy(ts->optix_context));
        ts->optix_context = nullptr;
    }
}

void *jitc_optix_lookup(const char *name) {
    for (int i = 0; i < 38; ++i) {
        if (strcmp(name, jitc_optix_table_names[i]) == 0)
            return jitc_optix_table[i];
    }
    jitc_raise("jit_optix_lookup(): function \"%s\" not found!", name);
}

void jitc_optix_configure(const OptixPipelineCompileOptions *pco,
                         const OptixShaderBindingTable *sbt,
                         const OptixProgramGroup *pg,
                         uint32_t pg_count) {
    ThreadState *ts = thread_state(true);
    jitc_log(Info, "jit_optix_configure(pg_count=%u)", pg_count);
    ts->optix_pipeline_compile_options = pco;
    ts->optix_shader_binding_table = sbt;

    ts->optix_program_groups.clear();
    for (uint32_t i = 0; i < pg_count; ++i)
        ts->optix_program_groups.push_back(pg[i]);
}

void jitc_optix_compile(ThreadState *ts, const char *buffer,
                       size_t buffer_size, Kernel &kernel,
                       uint64_t kernel_hash) {
    char error_log[16384];

    // =====================================================
    // 2. Compile an OptiX module
    // =====================================================

    OptixModuleCompileOptions module_opts { };
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    size_t log_size = sizeof(error_log);
    int rv = optixModuleCreateFromPTX(
        ts->optix_context, &module_opts, ts->optix_pipeline_compile_options,
        buffer, buffer_size, error_log, &log_size, &kernel.optix.mod);
    if (rv) {
        jitc_fail("jit_optix_compile(): optixModuleCreateFromPTX() failed. Please see the PTX "
                 "assembly listing and error message below:\n\n%s\n\n%s", buffer, error_log);
        jitc_optix_check(rv);
    }

    // =====================================================
    // 3. Create an OptiX program group
    // =====================================================

    OptixProgramGroupOptions pgo { };
    std::unique_ptr<OptixProgramGroupDesc[]> pgd(
        new OptixProgramGroupDesc[pg_map.size()]);
    memset(pgd.get(), 0, pg_map.size() * sizeof(OptixProgramGroupDesc));

    for (auto &kv : pg_map) {
        char kernel_name[36];
        uint32_t i = kv.second;
        snprintf(kernel_name, sizeof(kernel_name),
                 i == 0 ? "__raygen__enoki_%016llx"
                        : "__direct_callable__%016llx",
                 (unsigned long long) kv.first);

        if (i == 0) {
            pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            pgd[0].raygen.module = kernel.optix.mod;
            pgd[0].raygen.entryFunctionName = strdup(kernel_name);
        } else {
            pgd[i].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            pgd[i].callables.moduleDC = kernel.optix.mod;
            pgd[i].callables.entryFunctionNameDC = strdup(kernel_name);
        }
    }

    kernel.optix.pg = new OptixProgramGroup[pg_map.size()];
    kernel.optix.pg_count = pg_map.size();
    kernel.optix.sbt_count = sbt_refs.size();

    log_size = sizeof(error_log);
    rv = optixProgramGroupCreate(ts->optix_context, pgd.get(),
                                 pg_map.size(), &pgo, error_log,
                                 &log_size, kernel.optix.pg);
    if (rv) {
        jitc_fail("jit_optix_compile(): optixProgramGroupCreate() failed. Please see the PTX "
                 "assembly listing and error message below:\n\n%s\n\n%s", buffer, error_log);
        jitc_optix_check(rv);
    }

    size_t stride = OPTIX_SBT_RECORD_HEADER_SIZE;
    uint8_t *sbt_record = (uint8_t *)
        jitc_malloc(AllocType::HostPinned, sbt_refs.size() * stride);

    for (size_t i = 0; i < sbt_refs.size(); ++i) {
        auto it = pg_map.find(sbt_refs[i]);
        if (it == pg_map.end())
            jitc_fail("Could not find raygen/callable!");
        jitc_optix_check(optixSbtRecordPackHeader(
            kernel.optix.pg[it->second], sbt_record + stride * i));
    }

    kernel.optix.sbt_record = (uint8_t *)
        jitc_malloc_migrate(sbt_record, AllocType::Device, 1);

    // =====================================================
    // 4. Create an OptiX pipeline
    // =====================================================

    OptixPipelineLinkOptions link_options {};
    link_options.maxTraceDepth = 1;
    link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    size_t size_before = ts->optix_program_groups.size();
    for (uint32_t i = 0; i < pg_map.size(); ++i) {
        if (i == 0)
            free((char *) pgd[0].raygen.entryFunctionName);
        else
            free((char *) pgd[i].callables.entryFunctionNameDC);
        ts->optix_program_groups.push_back(kernel.optix.pg[i]);
    }

    log_size = sizeof(error_log);
    rv = optixPipelineCreate(
        ts->optix_context, ts->optix_pipeline_compile_options, &link_options,
        ts->optix_program_groups.data(), ts->optix_program_groups.size(),
        error_log, &log_size, &kernel.optix.pipeline);
    if (rv) {
        jitc_fail("jit_optix_compile(): optixPipelineCreate() failed. Please see the PTX "
                 "assembly listing and error message below:\n\n%s\n\n%s", buffer, error_log);
        jitc_optix_check(rv);
    }

    kernel.size = 0;
    kernel.data = nullptr;
    ts->optix_program_groups.resize(size_before);
}

void jitc_optix_free(const Kernel &kernel) {
    jitc_optix_check(optixPipelineDestroy(kernel.optix.pipeline));
    for (uint32_t i = 0; i < kernel.optix.pg_count; ++i)
        jitc_optix_check(optixProgramGroupDestroy(kernel.optix.pg[i]));
    delete[] kernel.optix.pg;
    jitc_optix_check(optixModuleDestroy(kernel.optix.mod));
    jitc_free(kernel.optix.sbt_record);
}

void jitc_optix_launch(ThreadState *ts, const Kernel &kernel, uint32_t size,
                      const void *args, uint32_t args_size) {
    OptixShaderBindingTable sbt;
    memcpy(&sbt, ts->optix_shader_binding_table, sizeof(OptixShaderBindingTable));
    sbt.raygenRecord = (CUdeviceptr) kernel.optix.sbt_record;

    if (kernel.optix.sbt_count > 1) {
        sbt.callablesRecordBase = (CUdeviceptr) (kernel.optix.sbt_record + OPTIX_SBT_RECORD_HEADER_SIZE);
        sbt.callablesRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.callablesRecordCount = kernel.optix.sbt_count - 1;
    }

    jitc_optix_check(optixLaunch(kernel.optix.pipeline, ts->stream,
                                (CUdeviceptr) args, args_size, &sbt, size, 1, 1));
}

void jitc_optix_trace(uint32_t nargs, uint32_t *args, uint32_t mask) {
    VarType types[] { VarType::UInt64,  VarType::Float32, VarType::Float32,
                      VarType::Float32, VarType::Float32, VarType::Float32,
                      VarType::Float32, VarType::Float32, VarType::Float32,
                      VarType::Float32, VarType::UInt32,  VarType::UInt32,
                      VarType::UInt32,  VarType::UInt32,  VarType::UInt32 };

    if (nargs < 15)
        jitc_raise("jit_optix_trace(): too few arguments (got %u < 15)", nargs);

    uint32_t np = nargs - 15;
    if (np > 8)
        jitc_raise("jit_optix_trace(): too many payloads (got %u > 8)", np);

    jitc_log(Info, "jit_optix_trace(): %u payload value%s.", np, np == 1 ? "" : "s");

    for (uint32_t i = 0; i < nargs; ++i) {
        VarType vt = jitc_var_type(args[i]);
        if (i < 15) {
            if (vt != types[i])
                jitc_raise("jit_optix_trace(): type mismatch for arg. %u", i);
        } else {
            if (var_type_size[(int) vt] != 4)
                jitc_raise("jit_optix_trace(): size mismatch for arg. %u", i);
        }
    }

    uint32_t decl = jitc_var_new_0(1, VarType::Void, "", 1, 1), index = decl;
    jitc_var_inc_ref_ext(index);

    Buffer buf(100);
    for (uint32_t i = 0; i < nargs; ++i) {
        VarType vt = jitc_var_type(args[i]);
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
        index = jitc_var_new_3(1, VarType::Void, buf.get(), 0, decl, args[i], index);
        jitc_var_dec_ref_ext(prev);
    }

    buf.clear();
    for (uint32_t i = 15; i < nargs; ++i) {
        buf.fmt(".reg.u32 $r1_%u$n", i + np);
        buf.fmt("mov.u32 $r1_%u, 0$n", i + np);
    }
    buf.put("@$r2 call (");
    for (uint32_t i = 15; i < nargs; ++i)
        buf.fmt("$r1_%u%s", i + np, i + 1 < nargs ? ", " : "");
    buf.fmt("), _optix_trace_%u, (", np);
    for (uint32_t i = 0; i < nargs; ++i)
        buf.fmt("$r1_%u%s", i, i + 1 < nargs ? ", " : "");
    buf.put(")");

    uint32_t prev = index;
    index = jitc_var_new_3(1, VarType::Void, buf.get(), 0, decl, mask, index);
    jitc_var_dec_ref_ext(prev);

    jitc_var(index)->optix = 1;

    for (uint32_t i = 15; i < nargs; ++i) {
        buf.clear();
        buf.fmt("mov.b32 $r0, $r1_%u", i + np);
        args[i] = jitc_var_new_2(1, (VarType) jitc_var_type(args[i]), buf.get(), 0, decl, index);
    }

    jitc_var_dec_ref_ext(index);
    jitc_var_dec_ref_ext(decl);
}

void jitc_optix_mark(uint32_t index) {
    jitc_var(index)->optix = true;
}

void jitc_optix_shutdown() {
    if (!jitc_optix_init_success)
        return;

    jitc_log(Info, "jit_optix_shutdown()");

#if defined(ENOKI_JIT_DYNAMIC_OPTIX)
    #define Z(x) x = nullptr

    #if !defined(_WIN32)
        if (jitc_optix_handle != RTLD_NEXT)
            dlclose(jitc_optix_handle);
    #else
        FreeLibrary((HMODULE) jitc_optix_handle);
    #endif

    memset(jitc_optix_table, 0, sizeof(jitc_optix_table));

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

    jitc_optix_handle = nullptr;

    #undef Z
#endif

    jitc_optix_init_success = false;
    jitc_optix_init_attempted = false;
}

void jitc_optix_check_impl(OptixResult errval, const char *file,
                          const int line) {
    if (unlikely(errval != 0)) {
        const char *name = optixGetErrorName(errval),
                   *msg  = optixGetErrorString(errval);
        jitc_fail("jit_optix_check(): API error %04i (%s): \"%s\" in "
                 "%s:%i.", (int) errval, name, msg, file, line);
    }
}

void jit_optix_check_impl(int errval, const char *file, const int line) {
    if (errval) {
        lock_guard guard(state.mutex);
        jitc_optix_check_impl(errval, file, line);
    }
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
                  suffix_len = strlen(suffix);

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
