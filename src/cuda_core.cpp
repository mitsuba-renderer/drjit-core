#include "cuda.h"
#include "log.h"
#include "var.h"
#include "util.h"
#include "internal.h"
#include "io.h"
#include "optix.h"
#include "resources/kernels.h"
#include <lz4.h>

CUresult jitc_cuda_cuinit_result = CUDA_ERROR_NOT_INITIALIZED;

int jitc_cuda_version_major = 0;
int jitc_cuda_version_minor = 0;
uint32_t jitc_cuda_arg_limit = 0;

// Dr.Jit kernel functions
static CUmodule *jitc_cuda_module = nullptr;

CUfunction *jitc_cuda_fill_64 = nullptr;
CUfunction *jitc_cuda_mkperm_phase_1_tiny = nullptr;
CUfunction *jitc_cuda_mkperm_phase_1_small = nullptr;
CUfunction *jitc_cuda_mkperm_phase_1_large = nullptr;
CUfunction *jitc_cuda_mkperm_phase_3 = nullptr;
CUfunction *jitc_cuda_mkperm_phase_4_tiny = nullptr;
CUfunction *jitc_cuda_mkperm_phase_4_small = nullptr;
CUfunction *jitc_cuda_mkperm_phase_4_large = nullptr;
CUfunction *jitc_cuda_transpose = nullptr;
CUfunction *jitc_cuda_compress_small = nullptr;
CUfunction *jitc_cuda_compress_large = nullptr;
CUfunction *jitc_cuda_compress_large_init = nullptr;
CUfunction *jitc_cuda_poke[(int)VarType::Count] { };
CUfunction *jitc_cuda_block_reduce[(int) ReduceOp::Count]
                                  [(int) VarType::Count][10] = { };
CUfunction *jitc_cuda_block_reduce_vec[(int) ReduceOp::Count]
                                      [(int) VarType::Count] = { };
CUfunction *jitc_cuda_block_prefix_reduce[(int) ReduceOp::Count]
                                         [(int) VarType::Count][10] = { };
CUfunction *jitc_cuda_reduce_dot[(int) VarType::Count] = { };
CUfunction *jitc_cuda_aggregate = nullptr;

std::pair<CUmodule, bool> jitc_cuda_compile(const char *buf, bool release_state_lock) {
    const uintptr_t log_size = 16384;
    char error_log[log_size], info_log[log_size];
    info_log[0] = '\0';
    error_log[0] = '\0';

    CUjit_option arg[] = {
        CU_JIT_OPTIMIZATION_LEVEL,
        CU_JIT_LOG_VERBOSE,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_GENERATE_LINE_INFO,
        CU_JIT_GENERATE_DEBUG_INFO
    };

    /// Generate debug ine information when this code is run within NSight Compute
    const bool have_line_info =
        getenv("NV_NSIGHT_INJECTION_TRANSPORT_TYPE") != nullptr;

    void *argv[] = {
        (void *) 4,
        (void *) 1,
        (void *) info_log,
        (void *) log_size,
        (void *) error_log,
        (void *) log_size,
        (void *) (uintptr_t) have_line_info,
        (void *) 0
    };

    uint32_t nargs = (uint32_t) (sizeof(arg) / sizeof(CUjit_option));

    CUmodule mod = nullptr;
    CUresult rv = (CUresult) 0;

    for (int i = 0; i < 2; ++i) {
        if (release_state_lock) {
            unlock_guard guard(state.lock);
            rv = cuModuleLoadDataEx(&mod, buf, nargs, arg, argv);
        } else {
            rv = cuModuleLoadDataEx(&mod, buf, nargs, arg, argv);
        }

        if (rv == CUDA_ERROR_OUT_OF_MEMORY) {
            if (i == 0)
                jitc_flush_malloc_cache(true);
            else
                cuda_check(rv);
        } else {
            break;
        }
    }

    if (rv != CUDA_SUCCESS)
        jitc_fail("jit_cuda_compile(): compilation failed. Please see the PTX "
                  "assembly listing and error message below:\n\n%s\n\n%s",
                  buf, error_log);

    bool cache_hit = info_log[0] == '\0';
    if (!cache_hit)
        jitc_log(Trace, "Detailed linker output:\n%s", info_log);

    return { mod, cache_hit };
}

void jitc_cuda_sync_stream(uintptr_t stream) {
    ThreadState* ts = thread_state(JitBackend::CUDA);
    CUevent sync_event = ts->sync_stream_event;
    scoped_set_context guard(ts->context);
    cuda_check(cuEventRecord(sync_event, (CUstream) ts->stream));
    if (stream != 2)
        cuda_check(cuStreamWaitEvent((CUstream)stream, sync_event, CU_EVENT_DEFAULT));
    else
        cuda_check(cuStreamWaitEvent_ptsz(nullptr, sync_event, CU_EVENT_DEFAULT));
}

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (unlikely(errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED)) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        jitc_fail("cuda_check(): API error %04i (%s): \"%s\" in "
                  "%s:%i.", (int) errval, name, msg, file, line);
    }
}

bool jitc_cuda_init() {
    /// Was the CUDA backend already initialized?
    if (jitc_cuda_module)
        return true;

    // First, dynamically load CUDA into the process
    if (!jitc_cuda_api_init())
        return false;

    // The following call may fail if there aren't any CUDA-capable GPUs
    jitc_cuda_cuinit_result = cuInit(0);
    if (jitc_cuda_cuinit_result != CUDA_SUCCESS)
        return false;

    int device_count = 0;
    cuda_check(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        jitc_log(LogLevel::Warn,
                 "jit_cuda_init(): No devices found -- disabling CUDA backend!");
        return false;
    }

    int cuda_version;
    cuda_check(cuDriverGetVersion(&cuda_version));

    jitc_cuda_version_major = cuda_version / 1000;
    jitc_cuda_version_minor = (cuda_version % 1000) / 10;

    if (jitc_cuda_version_major < 10) {
        jitc_cuda_api_shutdown();
        jitc_log(Warn,
                "jit_cuda_init(): your version of CUDA is too old (found %i.%i, "
                "at least 10.x is required) -- disabling CUDA backend!",
                jitc_cuda_version_major, jitc_cuda_version_minor);
        return false;
    }

    // Maximal amount of data (measured in # of 8-byte pointers) that can be
    // passed to a CUDA kernel depends on the CUDA version
    bool cuda_12_1_or_newer =
        (jitc_cuda_version_major > 12 ||
         (jitc_cuda_version_major == 12 && jitc_cuda_version_minor >= 1));

    jitc_cuda_arg_limit = cuda_12_1_or_newer ? 4096 : 512;

    jitc_log(Info, "jit_cuda_init(): enabling CUDA backend (version %i.%i)",
             jitc_cuda_version_major, jitc_cuda_version_minor);

    size_t asize = sizeof(CUfunction) * device_count;
    for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
        jitc_cuda_poke[k] = (CUfunction *) malloc_check_zero(asize);
        for (uint32_t j = 0; j < (uint32_t) ReduceOp::Count; j++) {
            for (int l = 0; l < 10; ++l) {
                jitc_cuda_block_reduce[j][k][l] = (CUfunction *) malloc_check_zero(asize);
                jitc_cuda_block_prefix_reduce[j][k][l] = (CUfunction *) malloc_check_zero(asize);
            }
            jitc_cuda_block_reduce_vec[j][k] = (CUfunction *) malloc_check_zero(asize);
        }
        jitc_cuda_reduce_dot[k] = (CUfunction *) malloc_check_zero(asize);
    }

    jitc_cuda_module =
        (CUmodule *) malloc_check_zero(sizeof(CUmodule) * device_count);

    jitc_lz4_init();

    for (int i = 0; i < device_count; ++i) {
        int pci_bus_id = 0, pci_dom_id = 0, pci_dev_id = 0, sm_count = 0,
            unified_addr = 0, shared_memory_bytes = 0, cc_minor = 0,
            cc_major = 0, memory_pool = 0;
        bool preemptable = true;
        size_t mem_total = 0;
        char name[256];

        CUcontext context = nullptr;
        cuda_check(cuDevicePrimaryCtxRetain(&context, i));
        scoped_set_context guard(context);

        cuda_check(cuDeviceGetName(name, sizeof(name), i));
        cuda_check(cuDeviceGetAttribute(&pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, i));
        cuda_check(cuDeviceGetAttribute(&pci_dev_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, i));
        cuda_check(cuDeviceGetAttribute(&pci_dom_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, i));
        cuda_check(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, i));
        cuda_check(cuDeviceGetAttribute(&unified_addr, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, i));
        cuda_check(cuDeviceGetAttribute(&shared_memory_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, i));
        cuda_check(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
        cuda_check(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
        cuda_check(cuDeviceTotalMem(&mem_total, i));

        if (jitc_cuda_version_major > 11 || (jitc_cuda_version_major == 11 && jitc_cuda_version_minor >= 2))
            cuda_check(cuDeviceGetAttribute(&memory_pool, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, i));

        // Determine the device compute capability
        int cc = cc_major * 10 + cc_minor;

        // Determine if we need special workarounds for long-running kernels on Windows
        #if defined(_WIN32)
            int tcc_driver = 0, compute_preemption = 0;
            cuda_check(cuDeviceGetAttribute(&tcc_driver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, i));
            cuda_check(cuDeviceGetAttribute(&compute_preemption, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, i));
            preemptable = compute_preemption || tcc_driver;
        #endif

        jitc_log(Info,
                " - Found CUDA device %i: \"%s\" "
                "(PCI ID %02x:%02x.%i, compute cap. %i.%i, %i SMs w/%s shared mem., %s global mem.%s)",
                i, name, pci_bus_id, pci_dev_id, pci_dom_id, cc_major, cc_minor, sm_count,
                std::string(jitc_mem_string(shared_memory_bytes)).c_str(),
                std::string(jitc_mem_string(mem_total)).c_str(),
                preemptable ? "" : ", non-preemptable");

        if (unified_addr == 0) {
            jitc_log(Warn, " - Warning: device does *not* support unified addressing, skipping ..");
            cuda_check(cuDevicePrimaryCtxRelease(i));
            continue;
        }

        if (cc < 50) {
            jitc_log(Warn, " - Warning: compute capability of device too low, skipping ..");
            cuda_check(cuDevicePrimaryCtxRelease(i));
            continue;
        }

        // Choose an appropriate set of builtin kernels
        const char *kernels           = cc >= 70 ? kernels_70 : kernels_50;
        int kernels_size_uncompressed = cc >= 70 ? kernels_70_size_uncompressed
                                                 : kernels_50_size_uncompressed;
        int kernels_size_compressed   = cc >= 70 ? kernels_70_size_compressed
                                                 : kernels_50_size_compressed;

        // Decompress the supplemental PTX content
        char *uncompressed =
            (char *) malloc_check(size_t(kernels_size_uncompressed) + jitc_lz4_dict_size + 1);
        memcpy(uncompressed, jitc_lz4_dict, jitc_lz4_dict_size);
        char *uncompressed_ptx = uncompressed + jitc_lz4_dict_size;

        int uncompressed_size_actual = LZ4_decompress_safe_usingDict(
            kernels, uncompressed_ptx, kernels_size_compressed,
            kernels_size_uncompressed, uncompressed, jitc_lz4_dict_size);
        if (uncompressed_size_actual != kernels_size_uncompressed)
            jitc_fail("jit_cuda_init(): decompression of builtin kernels failed!"
                      " Expected %d bytes (negative value indicates an error), got %d.",
                      kernels_size_uncompressed, uncompressed_size_actual);

        uncompressed_ptx[kernels_size_uncompressed] = '\0';

        CUmodule m = jitc_cuda_compile(uncompressed_ptx, /* release_state_lock */ false).first;
        jitc_cuda_module[i] = m;
        free(uncompressed);

        #define LOAD(name)                                                       \
            if (i == 0)                                                          \
                jitc_cuda_##name = (CUfunction *) malloc_check(                  \
                    sizeof(CUfunction) * device_count);                          \
            cuda_check(cuModuleGetFunction(&jitc_cuda_##name[i], m, #name))

        LOAD(fill_64);
        LOAD(mkperm_phase_1_tiny);
        LOAD(mkperm_phase_1_small);
        LOAD(mkperm_phase_1_large);
        LOAD(mkperm_phase_3);
        LOAD(mkperm_phase_4_tiny);
        LOAD(mkperm_phase_4_small);
        LOAD(mkperm_phase_4_large);
        LOAD(transpose);
        LOAD(compress_small);
        LOAD(compress_large);
        LOAD(compress_large_init);
        LOAD(aggregate);

        #undef LOAD

        #define MAXIMIZE_SHARED(name)                                            \
            cuda_check(cuFuncSetAttribute(                                       \
                jitc_cuda_##name[i],                                             \
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,                 \
                shared_memory_bytes))

        // Max out the amount of shared memory available to the following kernels
        MAXIMIZE_SHARED(mkperm_phase_1_tiny);
        MAXIMIZE_SHARED(mkperm_phase_1_small);
        MAXIMIZE_SHARED(mkperm_phase_4_tiny);
        MAXIMIZE_SHARED(mkperm_phase_4_small);

        #undef MAXIMIZE_SHARED

        CUfunction func;
        for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
            snprintf(name, sizeof(name), "poke_%s", type_name_short[k]);
            if (strstr(kernels_list, name)) {
                cuda_check(cuModuleGetFunction(&func, m, name));
                jitc_cuda_poke[k][i] = func;
            }

            for (uint32_t j = 0; j < (uint32_t) ReduceOp::Count; j++) {
                snprintf(name, sizeof(name), "block_reduce_%s_%s_vec_1024", red_name[j],
                         type_name_short[k]);
                if (strstr(kernels_list, name)) {
                    cuda_check(cuModuleGetFunction(&func, m, name));
                    jitc_cuda_block_reduce_vec[j][k][i] = func;
                }
                for (uint32_t l = 0; l < 10; ++l) {
                    snprintf(name, sizeof(name), "block_reduce_%s_%s_%u", red_name[j],
                             type_name_short[k], 1 << (l + 1));
                    if (strstr(kernels_list, name)) {
                        cuda_check(cuModuleGetFunction(&func, m, name));
                        jitc_cuda_block_reduce[j][k][l][i] = func;
                    }
                    snprintf(name, sizeof(name), "block_prefix_reduce_%s_%s_%u", red_name[j],
                             type_name_short[k], 1 << (l + 1));
                    if (strstr(kernels_list, name)) {
                        cuda_check(cuModuleGetFunction(&func, m, name));
                        jitc_cuda_block_prefix_reduce[j][k][l][i] = func;
                    }
                }
            }

            snprintf(name, sizeof(name), "reduce_dot_%s", type_name_short[k]);
            if (strstr(kernels_list, name)) {
                cuda_check(cuModuleGetFunction(&func, m, name));
                jitc_cuda_reduce_dot[k][i] = func;
            }
        }

        Device device;
        device.id = i;
        device.compute_capability = cc_major * 10 + cc_minor;
        device.shared_memory_bytes = (uint32_t) shared_memory_bytes;
        device.sm_count = (uint32_t) sm_count;
        device.memory_pool = memory_pool != 0;
        device.preemptable = preemptable;
        device.context = context;

        cuda_check(cuStreamCreate(&device.stream, CU_STREAM_DEFAULT));
        cuda_check(cuEventCreate(&device.event, CU_EVENT_DISABLE_TIMING));
        cuda_check(cuEventCreate(&device.sync_stream_event, CU_EVENT_DISABLE_TIMING));

        uint32_t driver_to_ptx_isa_mappling[][2] = {
            { 10, 10 },  { 11, 11 },  { 20, 12 },  { 21, 13 },  { 22, 14 },
            { 23, 14 },  { 30, 20 },  { 31, 21 },  { 32, 22 },  { 40, 23 },
            { 41, 23 },  { 42, 30 },  { 50, 31 },  { 55, 32 },  { 60, 40 },
            { 65, 42 },  { 70, 43 },  { 75, 50 },  { 80, 51 },  { 90, 60 },
            { 91, 61 },  { 92, 62 },  { 100, 63 }, { 101, 64 }, { 102, 65 },
            { 110, 70 }, { 111, 71 }, { 112, 72 }, { 113, 73 }, { 114, 74 },
            { 115, 75 }, { 116, 76 }, { 117, 77 }, { 118, 78 }, { 120, 80 },
            { 121, 81 }, { 122, 82 }, { 123, 83 }, { 124, 84 }, { 125, 85 },
            { 126, 85 }, { 127, 86 }, { 128, 87 }
        };

        const uint32_t table_size = sizeof(driver_to_ptx_isa_mappling) /
                                    (uint32_t) (sizeof(uint32_t) * 2);

        uint32_t driver_version = jitc_cuda_version_major*10+jitc_cuda_version_minor;
        uint32_t ptx_version = 0;

        for (uint32_t j = 0; j < table_size; ++j) {
            uint32_t driver_version_j = driver_to_ptx_isa_mappling[j][0],
                     ptx_version_j    = driver_to_ptx_isa_mappling[j][1];

            if (driver_version >= driver_version_j)
                ptx_version = ptx_version_j;
            else
                break;
        }

        device.ptx_version = ptx_version;
        state.devices.push_back(device);
    }

    // Enable P2P communication if possible
    for (const Device &a : state.devices) {
        for (const Device &b : state.devices) {
            if (a.id == b.id)
                continue;

            int peer_ok = 0;
            scoped_set_context guard(a.context);
            cuda_check(cuDeviceCanAccessPeer(&peer_ok, a.id, b.id));
            if (peer_ok) {
                jitc_log(Debug, " - Enabling peer access from device %i -> %i",
                        a.id, b.id);
                CUresult rv_2 = cuCtxEnablePeerAccess(b.context, 0);
                if (rv_2 == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
                    continue;
                cuda_check(rv_2);
            }
        }
    }

    return true;
}

void jitc_cuda_shutdown() {
    jitc_log(Info, "jit_cuda_shutdown()");

    for (auto &dev : state.devices) {
        {
            scoped_set_context guard(dev.context);
#if defined(DRJIT_ENABLE_OPTIX)
            jitc_optix_context_destroy(dev);
#endif
            cuda_check(cuModuleUnload(jitc_cuda_module[dev.id]));
            cuda_check(cuStreamDestroy(dev.stream));
            cuda_check(cuEventDestroy(dev.event));
            cuda_check(cuEventDestroy(dev.sync_stream_event));
        }
        cuda_check(cuDevicePrimaryCtxRelease(dev.id));
    }
    state.devices.clear();

    #define Z(x) do { free(x); x = nullptr; } while (0)

    Z(jitc_cuda_fill_64);
    Z(jitc_cuda_mkperm_phase_1_tiny);
    Z(jitc_cuda_mkperm_phase_1_small);
    Z(jitc_cuda_mkperm_phase_1_large);
    Z(jitc_cuda_mkperm_phase_3);
    Z(jitc_cuda_mkperm_phase_4_tiny);
    Z(jitc_cuda_mkperm_phase_4_small);
    Z(jitc_cuda_mkperm_phase_4_large);
    Z(jitc_cuda_transpose);
    Z(jitc_cuda_compress_small);
    Z(jitc_cuda_compress_large);
    Z(jitc_cuda_compress_large_init);
    Z(jitc_cuda_module);

    for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
        Z(jitc_cuda_poke[k]);
        for (uint32_t j = 0; j < (uint32_t) ReduceOp::Count; j++) {
            Z(jitc_cuda_block_reduce_vec[j][k]);
            for (uint32_t l = 0; l < 10; ++l) {
                Z(jitc_cuda_block_reduce[j][k][l]);
                Z(jitc_cuda_block_prefix_reduce[j][k][l]);
            }
        }
        Z(jitc_cuda_reduce_dot[k]);
    }

    jitc_cuda_api_shutdown();
}
