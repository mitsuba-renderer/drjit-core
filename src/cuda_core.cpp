#include "cuda.h"
#include "log.h"
#include "var.h"
#include "util.h"
#include "internal.h"
#include "io.h"
#include "optix.h"
#include "strbuf.h"
#include "unit.h"
#include "profile.h"
#include "resources/kernels.h"
#include <string>

/// One named PTX input of a driver JIT compilation
struct CUDALinkInput {
    const char *name;
    const char *ptx;
    size_t size;
};

/// Compile and link one or more PTX inputs into a module. The optional
/// 'cache_hit_out' receives whether the driver's cache served the result.
static CUmodule jitc_cuda_compile(const CUDALinkInput *in, size_t n,
                                  bool release_state_lock = false,
                                  bool *cache_hit_out = nullptr);

CUresult jitc_cuda_cuinit_result = CUDA_ERROR_NOT_INITIALIZED;

int jitc_cuda_version_major = 0;
int jitc_cuda_version_minor = 0;
uint32_t jitc_cuda_arg_limit = 0;

bool jitc_cuda_supports_256bit(const ThreadState *ts, bool uses_optix) {
    return ts->compute_capability >= 120 &&
           (!uses_optix ||
            jitc_cuda_version_major > 13 ||
            (jitc_cuda_version_major == 13 && jitc_cuda_version_minor >= 2));
}

// Dr.Jit kernel functions
CUfunction *jitc_cuda_fill_64 = nullptr;
CUfunction *jitc_cuda_block_mkperm_phase_1_tiny = nullptr;
CUfunction *jitc_cuda_block_mkperm_phase_1_small = nullptr;
CUfunction *jitc_cuda_block_mkperm_phase_1_large = nullptr;
CUfunction *jitc_cuda_block_mkperm_phase_3 = nullptr;
CUfunction *jitc_cuda_block_mkperm_phase_4_tiny = nullptr;
CUfunction *jitc_cuda_block_mkperm_phase_4_small = nullptr;
CUfunction *jitc_cuda_block_mkperm_phase_4_large = nullptr;
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
CUfunction *jitc_cuda_gemm[(int) VarType::Count][4][3] = { };

// ====================================================================
//              Builtin kernel storage & lazy JIT compilation
// ====================================================================

// Decompressed builtin-kernel PTX blob: preamble '\0' entry_0 '\0' entry_1 ...
static char *jitc_cuda_kernels = nullptr;

// Scratch buffer (reused, guarded by state.lock) for assembling preamble + entry
static StringBuffer jitc_cuda_ptx_buf;

/// Return a pointer to builtin kernel ``name``'s PTX, or ``NULL`` if absent.
static const char *jitc_cuda_find_kernel(const char *name) {
    size_t name_len = strlen(name);
    const char *p = kernels_list;
    uint32_t idx = 0;
    while (const char *comma = strchr(p, ',')) {
        if ((size_t) (comma - p) == name_len && memcmp(p, name, name_len) == 0)
            return jitc_cuda_kernels + kernels_75_offsets[idx];
        p = comma + 1;
        ++idx;
    }
    return nullptr;
}

/// Compile the builtin kernel ``name`` into its own module on the given device
static CUfunction jitc_cuda_compile_kernel(int device, const char *name) {
    const char *entry = jitc_cuda_find_kernel(name);
    if (!entry)
        return nullptr;

    CUDADevice &dev = state.devices[device];

    jitc_cuda_ptx_buf.clear();
    jitc_cuda_ptx_buf.put(jitc_cuda_kernels, kernels_75_preamble_size);
    jitc_cuda_ptx_buf.put(entry, strlen(entry));

    scoped_set_context guard(dev.context);
    CUDALinkInput input { name, jitc_cuda_ptx_buf.get(),
                          jitc_cuda_ptx_buf.size() };
    CUmodule m = jitc_cuda_compile(&input, 1);
    dev.modules.push_back(m);

    CUfunction func = nullptr;
    cuda_check(cuModuleGetFunction(&func, m, name));
    return func;
}

CUfunction jitc_cuda_poke_function(int device, VarType vt) {
    CUfunction &slot = jitc_cuda_poke[(int) vt][state.devices[device].id];
    if (!slot) {
        char name[128];
        snprintf(name, sizeof(name), "poke_%s", type_name_short[(int) vt]);
        slot = jitc_cuda_compile_kernel(device, name);
    }
    return slot;
}

CUfunction jitc_cuda_block_reduce_function(int device, ReduceOp op, VarType vt,
                                           int kernel_id) {
    CUfunction &slot = jitc_cuda_block_reduce[(int) op][(int) vt][kernel_id]
                                             [state.devices[device].id];
    if (!slot) {
        char name[128];
        snprintf(name, sizeof(name), "block_reduce_%s_%s_%u", red_name[(int) op],
                 type_name_short[(int) vt], 1u << (kernel_id + 1));
        slot = jitc_cuda_compile_kernel(device, name);
    }
    return slot;
}

CUfunction jitc_cuda_block_reduce_vec_function(int device, ReduceOp op,
                                               VarType vt) {
    CUfunction &slot =
        jitc_cuda_block_reduce_vec[(int) op][(int) vt][state.devices[device].id];
    if (!slot) {
        char name[128];
        snprintf(name, sizeof(name), "block_reduce_%s_%s_vec_1024",
                 red_name[(int) op], type_name_short[(int) vt]);
        slot = jitc_cuda_compile_kernel(device, name);
    }
    return slot;
}

CUfunction jitc_cuda_block_prefix_reduce_function(int device, ReduceOp op,
                                                  VarType vt, int kernel_id) {
    CUfunction &slot = jitc_cuda_block_prefix_reduce[(int) op][(int) vt][kernel_id]
                                                    [state.devices[device].id];
    if (!slot) {
        char name[128];
        snprintf(name, sizeof(name), "block_prefix_reduce_%s_%s_%u",
                 red_name[(int) op], type_name_short[(int) vt],
                 1u << (kernel_id + 1));
        slot = jitc_cuda_compile_kernel(device, name);
    }
    return slot;
}

CUfunction jitc_cuda_reduce_dot_function(int device, VarType vt) {
    CUfunction &slot = jitc_cuda_reduce_dot[(int) vt][state.devices[device].id];
    if (!slot) {
        char name[128];
        snprintf(name, sizeof(name), "reduce_dot_%s", type_name_short[(int) vt]);
        slot = jitc_cuda_compile_kernel(device, name);
    }
    return slot;
}

CUfunction jitc_cuda_gemm_function(int device, VarType vt, int tile,
                                   int transpose) {
    // tile 0->BM=8..3->BM=64, transpose 0->nn, 1->nt, 2->tn.
    static const char *gemm_suffix[3] = { "nn", "nt", "tn" };
    CUfunction &slot =
        jitc_cuda_gemm[(int) vt][tile][transpose][state.devices[device].id];
    if (!slot) {
        char name[128];
        snprintf(name, sizeof(name), "gemm_%s_%u_%s", type_name_short[(int) vt],
                 8u << tile, gemm_suffix[transpose]);
        slot = jitc_cuda_compile_kernel(device, name);
    }
    return slot;
}

/// Option list and log buffers of one driver JIT compilation
struct CUDAJitOptions {
    static constexpr size_t log_size = 16384;

    static constexpr uint32_t nargs = 8;

    char info_log[log_size];
    char error_log[log_size];
    CUjit_option arg[nargs];
    void *argv[nargs];

    CUDAJitOptions() {
        // Potentially generate line-info. The presence of this environment
        // variable indicates that Nsight Compute is attached.
        const bool have_line_info =
            getenv("CUDA_INJECTION64_PATH") != nullptr;

        info_log[0] = '\0';
        error_log[0] = '\0';

        arg[0] = CU_JIT_OPTIMIZATION_LEVEL;          argv[0] = (void *) 4;
        arg[1] = CU_JIT_LOG_VERBOSE;                 argv[1] = (void *) 1;
        arg[2] = CU_JIT_INFO_LOG_BUFFER;             argv[2] = info_log;
        arg[3] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;  argv[3] = (void *) log_size;
        arg[4] = CU_JIT_ERROR_LOG_BUFFER;            argv[4] = error_log;
        arg[5] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; argv[5] = (void *) log_size;
        arg[6] = CU_JIT_GENERATE_LINE_INFO;          argv[6] = (void *) (uintptr_t) have_line_info;
        arg[7] = CU_JIT_GENERATE_DEBUG_INFO;         argv[7] = (void *) 0;
    }

    /// Did the driver's cache serve every input? Only an actual JIT compile
    /// logs 'ptxas'-prefixed lines, while the link stage logs plain 'info'.
    bool cache_hit() const { return strstr(info_log, "ptxas") == nullptr; }
};

/// Load a linked cubin, optionally without the central Dr.Jit lock
static CUresult jitc_cuda_load_module(CUmodule *mod, const void *cubin,
                                      bool release_state_lock) {
    CUresult rv = CUDA_SUCCESS;

    for (int i = 0; i < 2; ++i) {
        if (release_state_lock) {
            unlock_guard guard(state.lock);
            rv = cuModuleLoadData(mod, cubin);
        } else {
            rv = cuModuleLoadData(mod, cubin);
        }

        if (rv != CUDA_ERROR_OUT_OF_MEMORY)
            break;
        if (i == 0)
            jitc_flush_malloc_cache(true);
        else
            cuda_check(rv);
    }

    return rv;
}

static CUmodule jitc_cuda_compile(const CUDALinkInput *in, size_t n,
                                  bool release_state_lock,
                                  bool *cache_hit_out) {
    CUDAJitOptions opt;
    CUlinkState ls = nullptr;
    void *cubin = nullptr;
    size_t cubin_size = 0;
    CUmodule mod = nullptr;
    CUresult rv = CUDA_SUCCESS;

    auto link = [&] {
        rv = cuLinkCreate(opt.nargs, opt.arg, opt.argv, &ls);
        for (size_t i = 0; i < n && rv == CUDA_SUCCESS; ++i)
            rv = cuLinkAddData(ls, CU_JIT_INPUT_PTX, (void *) in[i].ptx,
                               in[i].size + 1, in[i].name, 0, nullptr,
                               nullptr);
        if (rv == CUDA_SUCCESS)
            rv = cuLinkComplete(ls, &cubin, &cubin_size);
    };

    if (release_state_lock) {
        unlock_guard guard(state.lock);
        link();
    } else {
        link();
    }

    if (cache_hit_out)
        *cache_hit_out = opt.cache_hit();

    if (rv == CUDA_SUCCESS)
        rv = jitc_cuda_load_module(&mod, cubin, release_state_lock);

    if (rv == CUDA_ERROR_INVALID_PTX ||
        rv == CUDA_ERROR_UNSUPPORTED_PTX_VERSION ||
        rv == CUDA_ERROR_INVALID_VALUE) {
        if (n == 1)
            jitc_fail(
                "jit_cuda_compile(): compilation failed. Please see the PTX "
                "assembly listing and error message below:\n\n%s\n\n%s",
                in[0].ptx, opt.error_log);
        jitc_fail("jit_cuda_compile(): linking failed. Error message:\n\n%s",
                  opt.error_log);
    }
    cuda_check(rv);

    cuda_check(cuLinkDestroy(ls));

    if (opt.info_log[0] != '\0')
        jitc_log(Trace, "Detailed linker output:\n%s", opt.info_log);

    return mod;
}

// ============================================================================
//  Per-unit kernel compilation
// ============================================================================

/// Dr.Jit parallelizes the compilation of kernels with callable subroutines.
/// There isn't a great API for this. Dr.Jit does it by first compiling all the
/// callables individually in parallel (to warm the cache) and then performing
/// one combined compile+link step to produce the final kernel.

struct CUDACompileJob : UnitCompileJob {
    std::string error;     // error log text when the compilation failed
    CUresult rv = CUDA_SUCCESS;
    bool cache_hit = false; // served by the driver's cache (its JIT did not run)
};

/// Warm the driver cache with one unit. The 'resolver' slot satisfies any extern
/// references to the callable table.
static void jitc_cuda_compile_unit(CUcontext ctx, CUDACompileJob &job,
                                   const StringBuffer &resolver) {
    scoped_set_context guard(ctx);

    CUDAJitOptions opt;
    CUlinkState ls = nullptr;
    void *cubin = nullptr;
    size_t cubin_size = 0;

    CUresult rv = cuLinkCreate(opt.nargs, opt.arg, opt.argv, &ls);
    if (rv == CUDA_SUCCESS)
        rv = cuLinkAddData(ls, CU_JIT_INPUT_PTX, (void *) job.source,
                           job.source_size + 1, job.symbol, 0, nullptr,
                           nullptr);
    if (rv == CUDA_SUCCESS)
        rv = cuLinkAddData(ls, CU_JIT_INPUT_PTX, (void *) resolver.get(),
                           resolver.size() + 1, "callables", 0, nullptr,
                           nullptr);
    if (rv == CUDA_SUCCESS)
        rv = cuLinkComplete(ls, &cubin, &cubin_size);

    job.cache_hit = opt.cache_hit();
    if (rv != CUDA_SUCCESS) {
        job.rv = rv;
        job.error = opt.error_log;
    }

    if (ls)
        cuLinkDestroy(ls);
}

/// Render the callable table declaration
static void jitc_cuda_append_table_def(StringBuffer &out, size_t slots) {
    out.fmt(".visible .global .align 8 .u64 callables[%zu];\n", slots);
}

/// Fill the callable table with resolved addresses for each callable
static void jitc_cuda_fill_table(CUmodule mod) {
    if (callable_units.empty())
        return;

    CUdeviceptr table = 0;
    size_t size = 0;
    cuda_check(cuModuleGetGlobal(&table, &size, mod, "callables"));

    for (size_t i = 0; i < callable_units.size(); ++i) {
        char name[40];
        snprintf(name, sizeof(name), "addr_%016llx%016llx",
                 (unsigned long long) callable_units[i].hash.high64,
                 (unsigned long long) callable_units[i].hash.low64);
        CUdeviceptr ptr = 0;
        size_t unused = 0;
        cuda_check(cuModuleGetGlobal(&ptr, &unused, mod, name));

        // The null slot (0) is left as-is. Module-globals are zero-initialized
        cuda_check(cuMemcpy((uint8_t *) table + (1 + i) * sizeof(uint64_t),
                            ptr, sizeof(uint64_t)));
    }
}

static ProfilerRegion profiler_region_cuda_compile("jit_cuda_compile");

bool jitc_cuda_kernel_compile(ThreadState *ts, Kernel &kernel) {
    ProfilerPhase phase(profiler_region_cuda_compile);

    size_t n_units = 1 + callable_units.size();
    int device = ts->device;

    std::vector<CUDACompileJob> jobs(n_units);
    std::vector<uint32_t> misses;
    StringBuffer table_src, resolver_src;

    for (size_t i = 0; i < n_units; ++i)
        jitc_unit_job_init(i, jobs[i]);

    // Precompile each unit in parallel to warm the CUDA driver cache so that
    // the code below only needs to link. Skip for single-unit kernels.
    if (n_units > 1) {
        for (size_t i = 0; i < n_units; ++i) {
            UnitArtifact unused;
            if (!jitc_unit_cache_lookup(JitBackend::CUDA, jobs[i].unit_hash,
                                        (uint64_t) device, unused))
                misses.push_back((uint32_t) i);
        }

        // The per-kernel definition of the callable table, and its single-slot
        // variant that resolves the units' extern references during prewarming
        table_src.put(unit_prologue.get(), unit_prologue.size());
        resolver_src.put(unit_prologue.get(), unit_prologue.size());
        jitc_cuda_append_table_def(table_src, n_units);
        jitc_cuda_append_table_def(resolver_src, 1);
    }

    // Release the lock while prewarming
    if (!misses.empty()) {
        unlock_guard guard(state.lock);
        CUcontext ctx = ts->context;
        jitc_unit_compile_parallel(
            misses, [&](uint32_t i) { return jobs[i].source_size; },
            [&](uint32_t i) {
                jitc_cuda_compile_unit(ctx, jobs[i], resolver_src);
            });
    }

    for (uint32_t i : misses) {
        const CUDACompileJob &job = jobs[i];
        if (job.rv != CUDA_SUCCESS)
            jitc_fail("jit_cuda_kernel_compile(): compilation of unit "
                      "\"%s\" failed. Please see the PTX assembly "
                      "listing and error message below:\n\n%s\n\n%s",
                      job.symbol, job.source, job.error.c_str());
    }

    // Link the kernel from its units, resolving the per-unit compilations
    // above through the driver's cache
    std::vector<CUDALinkInput> inputs;
    inputs.reserve(n_units + 1);
    for (size_t i = 0; i < n_units; ++i)
        inputs.push_back({ jobs[i].symbol, jobs[i].source,
                           jobs[i].source_size });
    if (table_src.size() > 0)
        inputs.push_back({ "callables", table_src.get(), table_src.size() });

    bool link_cached = false;
    kernel.cuda.mod = jitc_cuda_compile(inputs.data(), inputs.size(),
                                        /* release_state_lock = */ true,
                                        &link_cached);
    jitc_cuda_fill_table(kernel.cuda.mod);

    for (uint32_t i : misses) {
        UnitArtifact empty { };
        jitc_unit_cache_insert(JitBackend::CUDA, jobs[i].unit_hash,
                               (uint64_t) device, empty, nullptr);
    }

    // Soft hit: the driver's cache served every prewarmed unit, and the final
    // link did not need to compile anything either
    bool all_cached = link_cached;
    for (uint32_t i : misses)
        all_cached &= jobs[i].cache_hit;
    return all_cached;
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
    if (jitc_cuda_kernels)
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

    // The precompiled PTX shipped with Dr.Jit-core targets PTX ISA 8.2
    // (built with CUDA 12.2), which requires driver R535 or newer.
    if (jitc_cuda_version_major < 12 ||
        (jitc_cuda_version_major == 12 && jitc_cuda_version_minor < 2)) {
        jitc_cuda_api_shutdown();
        jitc_log(Warn,
                "jit_cuda_init(): your version of CUDA is too old (found %i.%i, "
                "at least 12.2 / driver R535 is required) -- disabling CUDA backend!",
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
        for (int l = 0; l < 4; ++l)
            for (int t = 0; t < 3; ++t)
                jitc_cuda_gemm[k][l][t] = (CUfunction *) malloc_check_zero(asize);
    }

    // Decompress the builtin-kernel PTX blob once.
    jitc_cuda_kernels =
        jitc_lz4_inflate(kernels_75, kernels_75_size_compressed,
                         kernels_75_size_uncompressed, "builtin kernels");

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
                jitc_mem_string(shared_memory_bytes).c_str(),
                jitc_mem_string(mem_total).c_str(),
                preemptable ? "" : ", non-preemptable");

        if (unified_addr == 0) {
            jitc_log(Warn, " - Warning: device does *not* support unified addressing, skipping ..");
            cuda_check(cuDevicePrimaryCtxRelease(i));
            continue;
        }

        if (cc < 75) {
            jitc_log(Warn, " - Warning: compute capability of device too low (need >= 7.5), skipping ..");
            cuda_check(cuDevicePrimaryCtxRelease(i));
            continue;
        }

        // Eagerly build a single module for the small set of non-parameterized kernels
        static const char *core_kernels[] = {
            "fill_64",
            "block_mkperm_phase_1_tiny",  "block_mkperm_phase_1_small",
            "block_mkperm_phase_1_large", "block_mkperm_phase_3",
            "block_mkperm_phase_4_tiny",  "block_mkperm_phase_4_small",
            "block_mkperm_phase_4_large", "transpose",
            "compress_small",             "compress_large",
            "compress_large_init",        "aggregate"
        };

        jitc_cuda_ptx_buf.clear();
        jitc_cuda_ptx_buf.put(jitc_cuda_kernels, kernels_75_preamble_size);
        for (const char *kname : core_kernels) {
            const char *entry = jitc_cuda_find_kernel(kname);
            if (!entry)
                jitc_fail("jit_cuda_init(): core kernel \"%s\" is missing from "
                          "the builtin PTX!", kname);
            jitc_cuda_ptx_buf.put(entry, strlen(entry));
        }

        CUDALinkInput input { "drjit", jitc_cuda_ptx_buf.get(),
                              jitc_cuda_ptx_buf.size() };
        CUmodule m = jitc_cuda_compile(&input, 1);

        #define LOAD(name)                                                       \
            if (i == 0)                                                          \
                jitc_cuda_##name = (CUfunction *) malloc_check(                  \
                    sizeof(CUfunction) * device_count);                          \
            cuda_check(cuModuleGetFunction(&jitc_cuda_##name[i], m, #name))

        LOAD(fill_64);
        LOAD(block_mkperm_phase_1_tiny);
        LOAD(block_mkperm_phase_1_small);
        LOAD(block_mkperm_phase_1_large);
        LOAD(block_mkperm_phase_3);
        LOAD(block_mkperm_phase_4_tiny);
        LOAD(block_mkperm_phase_4_small);
        LOAD(block_mkperm_phase_4_large);
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
        MAXIMIZE_SHARED(block_mkperm_phase_1_tiny);
        MAXIMIZE_SHARED(block_mkperm_phase_1_small);
        MAXIMIZE_SHARED(block_mkperm_phase_4_tiny);
        MAXIMIZE_SHARED(block_mkperm_phase_4_small);

        #undef MAXIMIZE_SHARED

        CUDADevice device;
        device.id = i;
        device.compute_capability = cc_major * 10 + cc_minor;
        device.shared_memory_bytes = (uint32_t) shared_memory_bytes;
        device.sm_count = (uint32_t) sm_count;
        device.memory_pool = memory_pool != 0;
        device.preemptable = preemptable;
        device.context = context;
        device.modules.push_back(m);

        cuda_check(cuStreamCreate(&device.stream, CU_STREAM_DEFAULT));
        cuda_check(cuEventCreate(&device.event, CU_EVENT_DISABLE_TIMING));
        cuda_check(cuEventCreate(&device.sync_stream_event, CU_EVENT_DISABLE_TIMING));

        // This table maps from CUDA version to the PTX ISA version based on the table available here:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes
        uint32_t driver_to_ptx_isa_mapping[][2] = {
            { 10, 10 },  { 11, 11 },  { 20, 12 },  { 21, 13 },  { 22, 14 },
            { 23, 14 },  { 30, 20 },  { 31, 21 },  { 32, 22 },  { 40, 23 },
            { 41, 23 },  { 42, 30 },  { 50, 31 },  { 55, 32 },  { 60, 40 },
            { 65, 42 },  { 70, 43 },  { 75, 50 },  { 80, 51 },  { 90, 60 },
            { 91, 61 },  { 92, 62 },  { 100, 63 }, { 101, 64 }, { 102, 65 },
            { 110, 70 }, { 111, 71 }, { 112, 72 }, { 113, 73 }, { 114, 74 },
            { 115, 75 }, { 116, 76 }, { 117, 77 }, { 118, 78 }, { 120, 80 },
            { 121, 81 }, { 122, 82 }, { 123, 83 }, { 124, 84 }, { 125, 85 },
            { 126, 85 }, { 127, 86 }, { 128, 87 }, { 129, 88 }, { 130, 90 }
        };

        const uint32_t table_size = sizeof(driver_to_ptx_isa_mapping) /
                                    (uint32_t) (sizeof(uint32_t) * 2);

        uint32_t driver_version = jitc_cuda_version_major*10+jitc_cuda_version_minor;
        uint32_t ptx_version = 0;

        for (uint32_t j = 0; j < table_size; ++j) {
            uint32_t driver_version_j = driver_to_ptx_isa_mapping[j][0],
                     ptx_version_j    = driver_to_ptx_isa_mapping[j][1];

            if (driver_version >= driver_version_j)
                ptx_version = ptx_version_j;
            else
                break;
        }

        device.ptx_version = ptx_version;
        state.devices.push_back(device);
    }

    // Enable P2P communication if possible
    for (const CUDADevice &a : state.devices) {
        for (const CUDADevice &b : state.devices) {
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

    jitc_unit_cache_flush((int) JitBackend::CUDA);

    for (auto &dev : state.devices) {
        {
            scoped_set_context guard(dev.context);
#if defined(DRJIT_ENABLE_OPTIX)
            jitc_optix_context_destroy(dev);
#endif
            for (CUmodule m : dev.modules)
                cuda_check(cuModuleUnload(m));
            dev.modules.clear();
            cuda_check(cuStreamDestroy(dev.stream));
            cuda_check(cuEventDestroy(dev.event));
            cuda_check(cuEventDestroy(dev.sync_stream_event));
        }
        cuda_check(cuDevicePrimaryCtxRelease(dev.id));
    }
    state.devices.clear();

    #define Z(x) do { free(x); x = nullptr; } while (0)

    Z(jitc_cuda_fill_64);
    Z(jitc_cuda_block_mkperm_phase_1_tiny);
    Z(jitc_cuda_block_mkperm_phase_1_small);
    Z(jitc_cuda_block_mkperm_phase_1_large);
    Z(jitc_cuda_block_mkperm_phase_3);
    Z(jitc_cuda_block_mkperm_phase_4_tiny);
    Z(jitc_cuda_block_mkperm_phase_4_small);
    Z(jitc_cuda_block_mkperm_phase_4_large);
    Z(jitc_cuda_transpose);
    Z(jitc_cuda_compress_small);
    Z(jitc_cuda_compress_large);
    Z(jitc_cuda_compress_large_init);

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
        for (int l = 0; l < 4; ++l)
            for (int t = 0; t < 3; ++t)
                Z(jitc_cuda_gemm[k][l][t]);
    }

    free(jitc_cuda_kernels);
    jitc_cuda_kernels = nullptr;

    jitc_cuda_api_shutdown();
}

// ====================================================================
//                       Event API implementation
// ====================================================================

JitEvent jitc_cuda_event_create(bool enable_timing) {
    ThreadState* ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    EventData* event = new EventData(JitBackend::CUDA, enable_timing);
    event->ts = ts;

    unsigned int flags = enable_timing ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING;
    cuda_check(cuEventCreate(&event->cuda_event, flags));

    return (JitEvent)event;
}

void jitc_cuda_event_destroy(JitEvent event) {
    EventData* e = (EventData*)event;
    scoped_set_context guard(e->ts->context);
    cuda_check(cuEventDestroy(e->cuda_event));
    delete e;
}

void jitc_cuda_event_record(JitEvent event) {
    EventData* e = (EventData*)event;
    ThreadState* ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);
    cuda_check(cuEventRecord(e->cuda_event, ts->stream));
}

int jitc_cuda_event_query(JitEvent event) {
    EventData* e = (EventData*)event;
    scoped_set_context guard(e->ts->context);
    CUresult result = cuEventQuery(e->cuda_event);

    if (result == CUDA_SUCCESS)
        return 1;
    else if (result == CUDA_ERROR_NOT_READY)
        return 0;
    else
        cuda_check(result);
    return 0;
}

void jitc_cuda_event_wait(JitEvent event) {
    EventData* e = (EventData*)event;
    scoped_set_context guard(e->ts->context);
    CUevent cuda_event = e->cuda_event;
    // Release the lock while waiting
    unlock_guard guard_2(state.lock);
    cuda_check(cuEventSynchronize(cuda_event));
}

float jitc_cuda_event_elapsed_time(JitEvent start, JitEvent end) {
    EventData* s = (EventData*)start;
    EventData* e = (EventData*)end;

    if (!s->enable_timing || !e->enable_timing)
        jitc_raise("jit_event_elapsed_time(): both events must have timing enabled");

    scoped_set_context guard(s->ts->context);
    float ms;
    cuda_check(cuEventElapsedTime(&ms, s->cuda_event, e->cuda_event));
    return ms;
}
