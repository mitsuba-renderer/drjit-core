#include <drjit-core/optix.h>
#include <tsl/robin_map.h>
#include "optix.h"
#include "optix_api.h"
#include "internal.h"
#include "log.h"
#include "eval.h"
#include "var.h"
#include "op.h"
#include "util.h"
#include "trace.h"
#include "unit.h"

static bool jitc_optix_cache_hit = false;
static bool jitc_optix_cache_global_disable = false;
uint32_t jitc_optix_max_coopvec_size = 0;

void jitc_optix_log(unsigned int level, const char *tag, const char *message, void *) {
    // Note: cannot use jitc_var_log here. Parallel OptiX compilation may enter this
    // region from another thread, causing deadlocks (with the Dr.Jit-Core lock + Python GIL)
    if (level <= (uint32_t) state.log_level_combined)
        fprintf(stderr, "jit_optix_log(): [%s] %s", tag, message);

    if (strcmp(tag, "DISKCACHE") == 0 &&
        strncmp(message, "Cache miss for key", 18) == 0)
        jitc_optix_cache_hit = false;

    if (strcmp(tag, "DISK CACHE") == 0 &&
        strncmp(message, "OPTIX_CACHE_MAXSIZE is set to 0", 31) == 0)
        jitc_optix_cache_global_disable = true;
}

static OptixPipelineCompileOptions jitc_optix_default_compile_options() {
    OptixPipelineCompileOptions pco { };
    pco.numAttributeValues = 2;
    pco.pipelineLaunchParamsVariableName = "params";

    // The kernels generated via the default options actually don't do any ray
    // tracing, so the following declarations may seem unnecessary. However,
    // this combination produces the leanest kernels.
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pco.usesPrimitiveTypeFlags = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    if (jit_flag(JitFlag::Debug))
        pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                             OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    else
        pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    return pco;
}

OptixDeviceContext jitc_optix_context() {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    OptixDeviceContext &ctx = state.devices[ts->device].optix_context;

    if (!ctx) {
        if (!jitc_optix_api_init())
            jitc_raise("Could not initialize OptiX!");

        OptixDeviceContextOptions ctx_opts {
            jitc_optix_log, nullptr, 4,
            jit_flag(JitFlag::Debug) ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                     : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
        };

        jitc_optix_check(optixDeviceContextCreate(ts->context, &ctx_opts, &ctx));

        /* OptiX keeps its own cache database alongside ours. When Dr.Jit has no
           usable directory, leave it off rather than letting OptiX pick one. */
        const char *cache_dir = jitc_cache_dir();
        if (cache_dir)
            jitc_optix_check(optixDeviceContextSetCacheLocation(ctx, cache_dir));
        jitc_optix_check(optixDeviceContextSetCacheEnabled(ctx, cache_dir != nullptr));
    }

    // =====================================================
    // Create default OptiX pipeline for testcases, etc.
    // =====================================================

    if (!state.optix_default_sbt_index) {
        OptixPipelineCompileOptions pco = jitc_optix_default_compile_options();
        OptixModuleCompileOptions mco { };
        if (jit_flag(JitFlag::Debug)) {
            mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        } else {
            mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        }

        const char *minimal = ".version 6.0 .target sm_50 .address_size 64 "
                              ".entry __miss__dr() { ret; }";

        char log[128];
        size_t log_size = sizeof(log);

        OptixModule mod;
        jitc_optix_check(optixModuleCreate(
            ctx, &mco, &pco, minimal, strlen(minimal), log, &log_size, &mod));

        OptixProgramGroupDesc pgd { };
        pgd.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgd.miss.module = mod;
        pgd.miss.entryFunctionName = "__miss__dr";

        OptixProgramGroupOptions pgo { };
        OptixProgramGroup pg;
        log_size = sizeof(log);
        jitc_optix_check(optixProgramGroupCreate(ctx, &pgd, 1, &pgo, log, &log_size, &pg));

        OptixShaderBindingTable sbt {};
        sbt.missRecordBase = jitc_malloc(JitBackend::CUDA,
                                         OPTIX_SBT_RECORD_HEADER_SIZE,
                                         /*shared=*/true);
        jitc_optix_check(optixSbtRecordPackHeader(pg, sbt.missRecordBase));
        sbt.missRecordBase = jitc_malloc_migrate(sbt.missRecordBase,
                                                 JitBackend::CUDA);
        sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.missRecordCount = 1;

        uint32_t pipeline_index = jitc_optix_configure_pipeline(&pco, mod, &pg, 1),
                 sbt_index = jitc_optix_configure_sbt(&sbt, pipeline_index);

        state.optix_default_pipeline =
            (OptixPipelineData *) jitc_var_extra(jitc_var(pipeline_index))
                ->callback_data;
        state.optix_default_sbt_index = sbt_index;
        state.optix_default_sbt =
            (OptixShaderBindingTable *) jitc_var_extra(jitc_var(sbt_index))
                ->callback_data;

        if (!state.optix_default_pipeline || !state.optix_default_sbt)
            jitc_fail("jitc_optix_context(): could not find default pipeline/SBT entries!");

        jitc_var_dec_ref(pipeline_index);
    }

    return ctx;
}

void jitc_optix_context_destroy(CUDADevice &d) {
    if (d.optix_context) {
        jitc_optix_check(optixDeviceContextDestroy(d.optix_context));
        d.optix_context = nullptr;
    }
}

uint32_t jitc_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                                       OptixModule module,
                                       const OptixProgramGroup *pg,
                                       uint32_t pg_count) {
    jitc_log(InfoSym, "jitc_optix_configure_pipeline(pg_count=%u)", pg_count);

    if (!pco || !pg || pg_count == 0)
        jitc_raise("jitc_optix_configure_pipeline(): invalid input arguments!");

    OptixPipelineData *p = new OptixPipelineData();
    p->module = module;
    p->program_groups = std::vector<OptixProgramGroup>();
    memcpy(&p->compile_options, pco, sizeof(OptixPipelineCompileOptions));
    for (uint32_t i = 0; i < pg_count; ++i)
        p->program_groups.push_back(pg[i]);

    uint32_t index =
        jitc_var_new_node_0(JitBackend::CUDA, VarKind::Nop,
                            VarType::Void, 1, 0, (uintptr_t) p);

    // Free pipeline resources when this variable is destroyed
    auto callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (!free)
            return;
        jitc_log(InfoSym, "jit_optix_configure_pipeline(): free optix pipeline");
        OptixPipelineData *p = (OptixPipelineData*) ptr;
        for (size_t i = 0; i < p->program_groups.size(); i++)
            jitc_optix_check(optixProgramGroupDestroy(p->program_groups[i]));
        if (p->module)
            jitc_optix_check(optixModuleDestroy(p->module));
        delete p;
    };

    jitc_var_set_callback(index, callback, p, true);

    return index;
}

uint32_t jitc_optix_configure_sbt(const OptixShaderBindingTable *sbt,
                                  uint32_t pipeline) {
    jitc_log(InfoSym, "jitc_optix_configure_sbt()");

    if (!sbt || !pipeline)
        jitc_raise("jitc_optix_configure_sbt(): invalid input arguments!");

    if (jitc_var_type(pipeline) != VarType::Void)
        jitc_raise("jitc_optix_configure_sbt(): type mismatch for pipeline argument!");

    OptixShaderBindingTable *p = new OptixShaderBindingTable();
    memcpy(p, sbt, sizeof(OptixShaderBindingTable));

    uint32_t index = jitc_var_new_node_1(
        JitBackend::CUDA, VarKind::Nop, VarType::Void, 1, 0, pipeline,
        jitc_var(pipeline), (uintptr_t) p);

    // Free SBT resources when this variable is destroyed
    auto callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (!free)
            return;
        jitc_log(InfoSym, "jit_optix_configure_sbt(): free optix shader binding table");
        OptixShaderBindingTable *sbt = (OptixShaderBindingTable*) ptr;
        jitc_free(sbt->hitgroupRecordBase);
        jitc_free(sbt->missRecordBase);
        delete sbt;
    };

    jitc_var_set_callback(index, callback, p, true);

    return index;
}

void jitc_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt) {
    memcpy(jitc_var_extra(jitc_var(index))->callback_data, sbt,
           sizeof(OptixShaderBindingTable));
}

uint32_t jitc_optix_sbt_owner_handle(uint32_t sbt_index) {
    void *sbt_ptr = (void *) jitc_var(sbt_index)->literal;
    return jitc_var_mem_map(JitBackend::CUDA, VarType::UInt64, sbt_ptr, 1,
                            /* free = */ 0);
}

// ============================================================================
//  Per-unit kernel compilation
// ============================================================================

/// Compiled per-unit OptiX modules are held by the unit cache as
/// { ptr[0] = module }, filed under JitBackend::CUDA so that
/// jitc_cuda_shutdown() releases them ahead of the OptiX context.
static void jitc_optix_unit_release(UnitArtifact &a) {
    jitc_optix_check(optixModuleDestroy((OptixModule) a.ptr[0]));
}

/// Every module of a pipeline must be compiled with matching pipeline/module
/// compile options, so they enter the unit cache key alongside the device.
static uint64_t jitc_optix_config_salt(int device,
                                       const OptixPipelineCompileOptions &pco,
                                       const OptixModuleCompileOptions &mco) {
    uint32_t data[] = { (uint32_t) device,
                        (uint32_t) pco.numAttributeValues,
                        (uint32_t) pco.numPayloadValues,
                        (uint32_t) pco.usesMotionBlur,
                        (uint32_t) pco.traversableGraphFlags,
                        (uint32_t) pco.usesPrimitiveTypeFlags,
                        (uint32_t) pco.exceptionFlags,
                        (uint32_t) mco.debugLevel,
                        (uint32_t) mco.optLevel,
                        (uint32_t) mco.maxRegisterCount };
    return XXH3_64bits(data, sizeof(data));
}

struct OptixCompileJob : UnitCompileJob {
    OptixModule mod = nullptr;
    OptixTask task = nullptr;
    char error_log[2048];
    size_t log_size = sizeof(error_log);
};

bool jitc_optix_compile(ThreadState *ts, Kernel &kernel) {
    if (!optixModuleCreateWithTasks)
        jitc_fail("jit_optix_compile(): OptiX not initialized, make sure "
                  "evaluation happens before Optix shutdown!");

    // =====================================================
    // 1. Compile the kernel's units into OptiX modules
    // =====================================================

    OptixModuleCompileOptions mco { };
    if (jit_flag(JitFlag::Debug)) {
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    } else {
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    }

    if (jitc_optix_max_coopvec_size > 64) {
        // In large neural networks, spilling-related costs dominate.
        // In this case, prefer a larger register count over occupancy.
        mco.maxRegisterCount = 255;
    }

    jitc_optix_cache_hit = !jitc_optix_cache_global_disable;
    OptixDeviceContext &optix_context = state.devices[ts->device].optix_context;
    OptixPipelineData &pipeline = *ts->optix_pipeline;
    uint64_t salt =
        jitc_optix_config_salt(ts->device, pipeline.compile_options, mco);

    size_t n_units = 1 + callable_units.size();
    std::vector<OptixCompileJob> jobs(n_units);
    std::vector<uint32_t> misses;

    for (size_t i = 0; i < n_units; ++i) {
        OptixCompileJob &job = jobs[i];
        jitc_unit_job_init(i, job);
        job.error_log[0] = '\0';

        UnitArtifact artifact;
        if (jitc_unit_cache_lookup(JitBackend::CUDA, job.unit_hash, salt,
                                   artifact)) {
            job.mod = (OptixModule) artifact.ptr[0];
            continue;
        }

        misses.push_back((uint32_t) i);
    }

    // Release the lock while compiling. The job sources stay valid
    // throughout (see UnitCompileJob in unit.h)
    if (!misses.empty()) {
        unlock_guard guard(state.lock);

        // Issue one deferred module build per miss (cheap), then execute the
        // resulting task graphs of all units together through nanothread,
        // which balances the heavy work across the pool.
        for (uint32_t i : misses) {
            OptixCompileJob &job = jobs[i];
            int rv = optixModuleCreateWithTasks(
                optix_context, &mco, &pipeline.compile_options,
                job.source, job.source_size, job.error_log,
                &job.log_size, &job.mod, &job.task);
            if (rv) {
                jitc_log(Error,
                         "jit_optix_compile(): optixModuleCreateWithTasks() "
                         "failed for unit \"%s\". Please see the PTX assembly "
                         "listing and error message below:\n\n%s\n\n%s",
                         job.symbol, job.source, job.error_log);
                jitc_optix_check(rv);
            }
        }

        std::function<void(OptixTask)> execute_task = [&](OptixTask task) {
            unsigned int max_new_tasks = std::max(pool_size(), 1u);

            std::unique_ptr<OptixTask[]> new_tasks =
                std::make_unique<OptixTask[]>(max_new_tasks);
            unsigned int new_task_count = 0;
            optixTaskExecute(task, new_tasks.get(), max_new_tasks,
                             &new_task_count);

            parallel_for(
                drjit::blocked_range<size_t>(0, new_task_count, 1),
                [&](const drjit::blocked_range<size_t> &range) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        OptixTask new_task = new_tasks[i];
                        execute_task(new_task);
                    }
                }
            );
        };

        parallel_for(
            drjit::blocked_range<size_t>(0, misses.size(), 1),
            [&](const drjit::blocked_range<size_t> &range) {
                for (auto i = range.begin(); i != range.end(); ++i)
                    execute_task(jobs[misses[i]].task);
            }
        );

        for (uint32_t i : misses) {
            OptixCompileJob &job = jobs[i];
            int compilation_state = 0;
            jitc_optix_check(
                optixModuleGetCompilationState(job.mod, &compilation_state));
            if (compilation_state != OPTIX_MODULE_COMPILE_STATE_COMPLETED)
                jitc_fail("jit_optix_compile(): compilation of unit \"%s\" "
                          "did not complete succesfully (state: %#06x).\n"
                          "Please see the PTX assembly listing and error "
                          "message below:\n\n%s\n\n%s",
                          job.symbol, compilation_state, job.source,
                          job.error_log);
            else if (job.error_log[0])
                jitc_log(Trace, "Detailed compile output of unit \"%s\":\n%s",
                         job.symbol, job.error_log);
        }

        // Publish the new modules; the cache pins them
        for (uint32_t i : misses) {
            OptixCompileJob &job = jobs[i];
            UnitArtifact artifact { };
            artifact.ptr[0] = job.mod;
            jitc_unit_cache_insert(JitBackend::CUDA, job.unit_hash, salt,
                                   artifact, jitc_optix_unit_release);
            job.mod = (OptixModule) artifact.ptr[0];
        }
    }

    // =====================================================
    // 2. Create an OptiX program group
    // =====================================================

    size_t n_programs = n_units;

    OptixProgramGroupOptions pgo { };
    std::unique_ptr<OptixProgramGroupDesc[]> pgd(
        new OptixProgramGroupDesc[n_programs]);
    memset(pgd.get(), 0, n_programs * sizeof(OptixProgramGroupDesc));

    pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgd[0].raygen.module = jobs[0].mod;
    pgd[0].raygen.entryFunctionName = strdup(kernel_name);

    bool continuation_callables = jitc_optix_use_continuation_callables();

    for (uint32_t i = 0; i < (uint32_t) callable_units.size(); ++i) {
        XXH128_hash_t ch = callable_units[i].hash;

        char *name = (char *) malloc_check(58);
        snprintf(name, 58, "__%s_callable__%016llx%016llx",
                 continuation_callables ? "continuation" : "direct",
                 (unsigned long long) ch.high64,
                 (unsigned long long) ch.low64);

        uint32_t index = 1 + i;
        pgd[index].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

        if (continuation_callables) {
            pgd[index].callables.moduleCC = jobs[index].mod;
            pgd[index].callables.entryFunctionNameCC = name;
        } else {
            pgd[index].callables.moduleDC = jobs[index].mod;
            pgd[index].callables.entryFunctionNameDC = name;
        }
    }

    char error_log[16384];
    size_t log_size;

    kernel.optix.pg = new OptixProgramGroup[n_programs];
    kernel.optix.pg_count = (uint32_t) n_programs;

    log_size = sizeof(error_log);
    error_log[0] = '\0';
    int rv = optixProgramGroupCreate(optix_context, pgd.get(),
                                     (unsigned int) n_programs, &pgo, error_log,
                                     &log_size, kernel.optix.pg);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixProgramGroupCreate() "
                 "failed. Please see the error message below:\n\n%s",
                 error_log);
        jitc_optix_check(rv);
    } else if (error_log[0]) {
        jitc_log(Trace, "Detailed program group creation output:\n%s", error_log);
    }

    const size_t stride = OPTIX_SBT_RECORD_HEADER_SIZE;
    uint8_t *sbt_record = (uint8_t *)
        jitc_malloc(JitBackend::CUDA, n_programs * stride, /*shared=*/true);

    for (size_t i = 0; i < n_programs; ++i)
        jitc_optix_check(optixSbtRecordPackHeader(
            kernel.optix.pg[i], sbt_record + stride * i));

    kernel.optix.sbt_record = (uint8_t *)
        jitc_malloc_migrate(sbt_record, JitBackend::CUDA);

    // =====================================================
    // 3. Create an OptiX pipeline
    // =====================================================

    OptixPipelineLinkOptions link_options {};
    link_options.maxTraceDepth = 1;

    size_t size_before = pipeline.program_groups.size();

    for (uint32_t i = 0; i < n_programs; ++i) {
        if (i == 0)
            free((char *) pgd[0].raygen.entryFunctionName);
        else if (continuation_callables)
            free((char *) pgd[i].callables.entryFunctionNameCC);
        else
            free((char *) pgd[i].callables.entryFunctionNameDC);
        pipeline.program_groups.push_back(kernel.optix.pg[i]);
    }

    log_size = sizeof(error_log);
    error_log[0] = '\0';
    rv = optixPipelineCreate(optix_context, &pipeline.compile_options,
                             &link_options, pipeline.program_groups.data(),
                             (unsigned int) pipeline.program_groups.size(),
                             error_log, &log_size, &kernel.optix.pipeline);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixPipelineCreate() failed. "
                 "Please see the error message below:\n\n%s", error_log);
        jitc_optix_check(rv);
    } else if (error_log[0]) {
        jitc_log(Trace, "Detailed pipeline link output:\n%s", error_log);
    }

    // Setup the direct stack and continuation stack size.
    // See OptiX documentation for more detail:
    // https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#7235

    OptixStackSizes ssp = {};
    for (size_t i = 0; i < pipeline.program_groups.size(); ++i) {
        OptixStackSizes ss;
        rv = optixProgramGroupGetStackSize(pipeline.program_groups[i], &ss,
                                           kernel.optix.pipeline);
        if (rv) {
            jitc_log(Error, "jit_optix_compile(): optixProgramGroupGetStackSize() "
                            "failed:\n\n%s", error_log);
            jitc_optix_check(rv);
        }
        ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
        ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
        ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
        ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
        ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
        ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
        ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
    }

    unsigned int max_dc_depth = 2; // Support nested calls
    unsigned int dc_stack_size_from_traversal = 0; // DC is not invoked from IS or AH.
    unsigned int dc_stack_size_from_state = max_dc_depth * ssp.dssDC; // DC is invoked from RG, MS, or CH.
    unsigned int continuation_stack_size = ssp.cssRG + std::max(std::max(ssp.cssCH, ssp.cssMS), ssp.cssAH + ssp.cssIS);

    unsigned int max_traversable_graph_depth = 2; // Support instancing
    if (pipeline.compile_options.traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
        max_traversable_graph_depth = 1;

    rv = optixPipelineSetStackSize(kernel.optix.pipeline,
                                   dc_stack_size_from_traversal,
                                   dc_stack_size_from_state,
                                   continuation_stack_size,
                                   max_traversable_graph_depth);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixPipelineSetStackSize() "
                        "failed:\n\n%s", error_log);
        jitc_optix_check(rv);
    }

    kernel.size = 0;
    pipeline.program_groups.resize(size_before);
    return jitc_optix_cache_hit;
}

void jitc_optix_free(const Kernel &kernel) {
    jitc_optix_check(optixPipelineDestroy(kernel.optix.pipeline));
    for (uint32_t i = 0; i < kernel.optix.pg_count; ++i)
        jitc_optix_check(optixProgramGroupDestroy(kernel.optix.pg[i]));
    delete[] kernel.optix.pg;
    // The modules referenced by the program groups are owned by the unit
    // cache and released in jitc_unit_cache_flush()
    jitc_free(kernel.optix.sbt_record);
}

void jitc_optix_launch(ThreadState *ts, const Kernel &kernel,
                       uint32_t launch_size, const void *args,
                       uint32_t n_args) {
    OptixShaderBindingTable &sbt = *ts->optix_sbt;
    sbt.raygenRecord = kernel.optix.sbt_record;

    if (kernel.optix.pg_count > 1) {
        sbt.callablesRecordBase = kernel.optix.sbt_record + OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.callablesRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.callablesRecordCount = kernel.optix.pg_count - 1;
    }

    uint32_t offset = 0;

    /* We accept kernel launches up to the maximum value of 2^30 threads by
       default. When using an older non-preemptable WDDM driver setup on
       Windows, a long-running kernel may freeze the OS graphics and eventually
       cause a device reset. That's not good, so we submit smaller batches that
       correspond roughly to 1 sample/pixel for a full HD resolution frame. */
    uint32_t limit = state.devices[ts->device].preemptable ? (1 << 30) : (1 << 21);

    while (launch_size > 0) {
        uint32_t sub_launch_size = launch_size < limit ? launch_size : limit;

        // Bytes 4..8 used to store optional offset parameter
        if (offset != 0)
            cuMemsetD32Async(
                (CUdeviceptr) ((uint8_t *) args + sizeof(uint32_t)),
                offset, 1, ts->stream);

        jitc_optix_check(
            optixLaunch(kernel.optix.pipeline, ts->stream, (CUdeviceptr) args,
                        n_args * sizeof(void *), &sbt,
                        sub_launch_size, 1, 1));

        launch_size -= sub_launch_size;
        offset += sub_launch_size;
    }
}

void jitc_optix_ray_trace(uint32_t n_args, uint32_t *args,
                          uint32_t n_hit_object_field,
                          OptixHitObjectField *hit_object_fields,
                          uint32_t *hit_object_out, int reorder,
                          uint32_t reorder_hint, uint32_t reorder_hint_num_bits,
                          int invoke, uint32_t mask, uint32_t pipeline,
                          uint32_t sbt) {
    VarType types[]{ VarType::UInt64,  VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32 };
    if (n_args < 15)
        jitc_raise("jit_optix_ray_trace(): too few arguments (got %u < 15)", n_args);

    uint32_t np = n_args - 15, size = 0;
    if (np > 32)
        jitc_raise("jit_optix_ray_trace(): too many payloads (got %u > 32)", np);

    if (jitc_var_type(pipeline) != VarType::Void ||
        jitc_var_type(sbt) != VarType::Void)
        jitc_raise("jit_optix_ray_trace(): type mismatch for pipeline argument!");

    // Validate input types, determine size of the operation
    bool symbolic = false, dirty = false;
    for (uint32_t i = 0; i <= n_args; ++i) {
        uint32_t index = i < n_args ? args[i] : mask;
        VarType ref = VarType::Void;
        if (i < 15)
            ref = types[i];
        else if (i - 15 < np)
            ref = VarType::UInt32; // payloads are all UInt32
        else
            ref = VarType::Bool;
        const Variable *v = jitc_var(index);
        if ((VarType) v->type != ref)
            jitc_raise("jit_optix_ray_trace(): type mismatch for arg. %u (got %s, "
                       "expected %s)",
                       i, type_name[v->type], type_name[(int) ref]);
        size = std::max(size, v->size);
        symbolic |= (bool) v->symbolic;
        dirty |= v->is_dirty();
    }

    for (uint32_t i = 0; i <= n_args; ++i) {
        uint32_t index = (i < n_args) ? args[i] : mask;
        const Variable *v = jitc_var(index);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_optix_ray_trace(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::CUDA));

        for (uint32_t i = 0; i <= n_args; ++i) {
            uint32_t index = (i < n_args) ? args[i] : mask;
            if (jitc_var(index)->is_dirty())
                jitc_raise_dirty_error(index);
        }
    }

    for (uint32_t i = 0; i < n_hit_object_field; ++i)
        if (hit_object_fields[i] >= OptixHitObjectField::Count)
            jitc_raise("jit_optix_ray_trace(): unknown hit object field!");


    if (reorder) {
        if (reorder_hint_num_bits > 16)
            jitc_fail("jit_optix_ray_trace(): a maximum of 16 bits can be used for "
                      "the reordering key!");
        if ((VarType) jitc_var(reorder_hint)->type != VarType::UInt32)
            jitc_raise("jit_optix_ray_trace(): 'reorder_hint' must be an "
                       "unsigned 32-bit array.");
    }

    // Potentially apply any masks on the mask stack
    Ref valid = steal(jitc_var_mask_apply(mask, size));

    jitc_log(InfoSym,
             "jit_optix_ray_trace(): "
             "tracing %u ray%s, %u payload value%s, %u hitobject field%s%s.",
             size, size != 1 ? "s" : "", np, np == 1 ? "" : "s",
             n_hit_object_field, n_hit_object_field == 1 ? "" : "s",
             symbolic ? " ([symbolic])" : "");

    // Fill payload information for node
    TraceData *td = new TraceData();
    td->invoke = invoke;
    td->reorder = reorder;
    td->reorder_hint = reorder_hint;
    td->reorder_hint_num_bits = reorder_hint_num_bits;
    td->indices.reserve(n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        uint32_t id = args[i];
        td->indices.push_back(id);
        jitc_var_inc_ref(id);
    }
    td->hit_object_fields.reserve(n_hit_object_field);
    for (uint32_t i = 0; i < n_hit_object_field; ++i)
        td->hit_object_fields.push_back((uint32_t) hit_object_fields[i]);

    Ref index = steal(jitc_var_new_node_3(
        JitBackend::CUDA, VarKind::TraceRay, VarType::Void, size,
        symbolic, valid, jitc_var(valid), pipeline, jitc_var(pipeline), sbt,
        jitc_var(sbt), (uintptr_t) td));

    Variable *v = jitc_var(index);
    v->optix = 1;

    if (reorder && reorder_hint_num_bits > 0) {
        v->dep[3] = reorder_hint;
        jitc_var_inc_ref(reorder_hint);
    }

    // Extract payload values
    if (invoke)
        for (uint32_t i = 0; i < np; ++i)
            args[15 + i] = jitc_var_new_node_1(
                JitBackend::CUDA, VarKind::Extract, VarType::UInt32,
                size, symbolic, index, jitc_var(index), (uint64_t) i);

    // Extract hit object queries
    for (uint32_t i = 0; i < n_hit_object_field; ++i) {
        VarType field_type = VarType::Void;
        switch (hit_object_fields[i]) {
            case OptixHitObjectField::IsHit:
            case OptixHitObjectField::InstanceId:
            case OptixHitObjectField::InstanceIndex:
            case OptixHitObjectField::PrimitiveIndex:
                field_type = VarType::UInt32;
                break;
            case OptixHitObjectField::SBTDataPointer:
                field_type = VarType::Pointer;
                break;
            case OptixHitObjectField::RayTMax:
                field_type = VarType::Float32;
                break;
            case OptixHitObjectField::Attribute0:
            case OptixHitObjectField::Attribute1:
            case OptixHitObjectField::Attribute2:
            case OptixHitObjectField::Attribute3:
            case OptixHitObjectField::Attribute4:
            case OptixHitObjectField::Attribute5:
            case OptixHitObjectField::Attribute6:
            case OptixHitObjectField::Attribute7:
                field_type = VarType::UInt32;
                break;
            default:
                jitc_fail("jit_optix_ray_trace(): unhandled hit object "
                          "field type (value %u)!", (uint32_t) hit_object_fields[i]);
        }
        hit_object_out[i] = jitc_var_new_node_1(
            JitBackend::CUDA, VarKind::Extract, field_type, size, symbolic,
            index, jitc_var(index), (uint64_t) 32 + i);
    }

    // Free resources when this variable is destroyed
    auto callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (free)
            delete (TraceData *) ptr;
    };

    jitc_var_set_callback(index, callback, td, true);
}

uint32_t jitc_optix_sbt_data_load(uint32_t sbt_data_ptr, VarType type,
                                  uint32_t offset, uint32_t mask_) {

    Variable *v_sbt_data_ptr = jitc_var(sbt_data_ptr);
    uint32_t size = v_sbt_data_ptr->size;
    bool symbolic = v_sbt_data_ptr->symbolic;

    Ref mask = steal(jitc_var_mask_apply(mask_, size));

    Variable *v_mask = jitc_var(mask);
    if (v_mask->is_literal() && v_mask->literal == 0) {
        uint64_t value = 0;
        return jitc_var_literal(JitBackend::CUDA, type, &value, v_mask->size, 0);
    }

    return jitc_var_new_node_2(JitBackend::CUDA, VarKind::VectorLoad, type,
                               size, symbolic, sbt_data_ptr, jitc_var(sbt_data_ptr),
                               mask, jitc_var(mask), offset);
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
        lock_guard guard(state.lock);
        jitc_optix_check_impl(errval, file, line);
    }
}
