#include "cuda_green.h"
#include "cuda.h"
#include "cuda_api.h"
#include "log.h"
#include "common.h"

struct CUDAGreenContext {
    uint32_t sm_count = 0;
    CUgreenCtx green_1 = nullptr;
    CUgreenCtx green_2 = nullptr;
    Device device = {};
};

// Snapshot of the CUDA thread-local state before entering a green context
struct ThreadStateBackup {
    ThreadState *ts = nullptr;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUevent event = nullptr;
    CUevent sync_stream_event = nullptr;
};

CUDAGreenContext *jitc_cuda_green_context_make(uint32_t sm_count_requested,
                                               uint32_t *sm_count_actual,
                                               void **other_context_out) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    if (!ts)
        jitc_raise("jit_cuda_green_context_make(): CUDA backend is inactive.");

    if (sm_count_requested == 0)
        jitc_raise("jit_cuda_green_context_make(): sm_count must be > 0.");

    if (!cuCtxGetDevResource || !cuDevSmResourceSplitByCount ||
        !cuDevResourceGenerateDesc || !cuGreenCtxCreate || !cuCtxFromGreenCtx)
        jitc_raise("jit_cuda_green_context_make(): required driver symbols are missing. "
                   "Ensure a recent CUDA driver is installed and that green contexts are supported.");

    // Step 1: fetch SM resources for this CUDA context
    Device &parent = state.devices[ts->device];
    CUdevResource sm_resource = {};
    CUresult rc = cuCtxGetDevResource(parent.context, &sm_resource, CU_DEV_RESOURCE_TYPE_SM);
    cuda_check(rc);
    if (sm_resource.sm.smCount == 0)
        jitc_raise("jit_cuda_green_context_make(): cuCtxGetDevResource() failed. "
                   "This context/device does not expose SM partitioning.");

    if (sm_resource.sm.minSmPartitionSize == 0 ||
        sm_resource.sm.minSmPartitionSize > sm_resource.sm.smCount)
        sm_resource.sm.minSmPartitionSize = 1;

    if (sm_resource.sm.smCoscheduledAlignment == 0 ||
        sm_resource.sm.smCoscheduledAlignment > sm_resource.sm.minSmPartitionSize)
        sm_resource.sm.smCoscheduledAlignment = 1;

    // Step 2: compute how many SMs we can allocate (respect min/alignment)
    uint32_t total_sm  = sm_resource.sm.smCount;
    uint32_t rounded   = std::max(sm_count_requested, sm_resource.sm.minSmPartitionSize);
    rounded            = ceil_div(rounded, sm_resource.sm.smCoscheduledAlignment) *
                         sm_resource.sm.smCoscheduledAlignment;


    if (rounded > total_sm)
        jitc_raise("jit_cuda_green_context_make(): requested %u SMs, but device only has %u.",
                   rounded, total_sm);

    CUdevResource primary_res = {}, extra_res = {};
    unsigned int nb_groups = 1;
    // Step 3: split the SM resource into (primary, extra)
    cuda_check(cuDevSmResourceSplitByCount(&primary_res, &nb_groups,
                                           &sm_resource, &extra_res,
                                           CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
                                           rounded));
    if (nb_groups == 0)
        jitc_raise("jit_cuda_green_context_make(): split operation returned no groups.");

    CUdevResourceDesc desc_primary = nullptr, desc_other = nullptr;
    cuda_check(cuDevResourceGenerateDesc(&desc_primary, &primary_res, 1));
    bool have_other = extra_res.sm.smCount > 0;
    if (have_other)
        cuda_check(cuDevResourceGenerateDesc(&desc_other, &extra_res, 1));

    // Step 4: create green contexts and convert them into CUcontext handles
    CUgreenCtx green_1 = nullptr, green_2 = nullptr;
    CUcontext  primary_ctx  = nullptr,  other_ctx  = nullptr;
    cuda_check(cuGreenCtxCreate(&green_1, desc_primary, parent.id,
                                CU_GREEN_CTX_DEFAULT_STREAM));
    cuda_check(cuCtxFromGreenCtx(&primary_ctx, green_1));
    if (have_other) {
        cuda_check(cuGreenCtxCreate(&green_2, desc_other, parent.id,
                                    CU_GREEN_CTX_DEFAULT_STREAM));
        cuda_check(cuCtxFromGreenCtx(&other_ctx, green_2));
    }

    // Assemble context and device (copy parent, then patch changed fields)
    CUDAGreenContext *ctx = new CUDAGreenContext();
    ctx->sm_count = rounded;
    ctx->green_1 = green_1;
    ctx->green_2 = green_2;

    ctx->device = parent;
    ctx->device.context = primary_ctx;
    ctx->device.sm_count = rounded;
    ctx->device.stream = nullptr;
    ctx->device.event = nullptr;
    ctx->device.sync_stream_event = nullptr;
    {
        Device &d = ctx->device;
        scoped_set_context guard(d.context);
        cuda_check(cuStreamCreate(&d.stream, CU_STREAM_DEFAULT));
        cuda_check(cuEventCreate(&d.event, CU_EVENT_DISABLE_TIMING));
        cuda_check(cuEventCreate(&d.sync_stream_event, CU_EVENT_DISABLE_TIMING));
    }

    if (sm_count_actual)
        *sm_count_actual = rounded;

    if (other_context_out)
        *other_context_out = other_ctx;

    jitc_log(Debug,
             "jit_cuda_green_context_make(): requested %u SMs, driver provided (%u, %u) SM split and SM alignment %u. Contexts (%p, %p).",
             sm_count_requested,
             primary_res.sm.smCount,
             extra_res.sm.smCount,
             sm_resource.sm.smCoscheduledAlignment,
             (void *) primary_ctx,
             (void *) other_ctx);

    return ctx;
}

void jitc_cuda_green_context_release(CUDAGreenContext *ctx) {
    if (!ctx)
        return;

    jitc_log(Debug, "jit_cuda_green_context_release(): primary=%p, other=%p, actual_sm=%u",
             (void *) ctx->green_1, (void *) ctx->green_2,
             ctx->sm_count);

    // Destroy stream/events under the green context's CUcontext
    {
        const Device &d = ctx->device;
        scoped_set_context guard(d.context);
        cuda_check(cuStreamDestroy(d.stream));
        cuda_check(cuEventDestroy(d.event));
        cuda_check(cuEventDestroy(d.sync_stream_event));
    }

    // Destroy green contexts
    cuda_check(cuGreenCtxDestroy(ctx->green_2));
    cuda_check(cuGreenCtxDestroy(ctx->green_1));
    delete ctx;
}

void *jitc_cuda_green_context_enter(CUDAGreenContext *ctx) {
    if (!ctx)
        return nullptr;

    ThreadState *ts = thread_state(JitBackend::CUDA);
    if (!ts)
        jitc_raise("jit_cuda_green_context_enter(): CUDA backend is inactive.");

    ThreadStateBackup *backup = new ThreadStateBackup();
    backup->ts = ts;
    backup->context = ts->context;
    backup->stream = ts->stream;
    backup->event = ts->event;
    backup->sync_stream_event = ts->sync_stream_event;

    ts->context = ctx->device.context;
    ts->stream = ctx->device.stream;
    ts->event = ctx->device.event;
    ts->sync_stream_event = ctx->device.sync_stream_event;

    jitc_log(Debug, "jit_cuda_green_context_enter(): context=%p, stream=%p, sm=%u",
             (void *) ctx->device.context, (void *) ctx->device.stream,
             ctx->sm_count);
    return backup;
}

void jitc_cuda_green_context_leave(void *token) {
    if (!token)
        return;

    ThreadStateBackup *backup = (ThreadStateBackup *) token;
    ThreadState *ts = backup->ts;

    ts->context = backup->context;
    ts->stream = backup->stream;
    ts->event = backup->event;
    ts->sync_stream_event = backup->sync_stream_event;

    jitc_log(Debug, "jit_cuda_green_context_leave(): restored context=%p, stream=%p",
             (void *) backup->context, (void *) backup->stream);
    delete backup;
}
