#include "amd_api.h"

#if defined(DRJIT_ENABLE_AMD)

#include "amd_ts.h"
#include "internal.h"
#include "io.h"
#include "log.h"
#include "malloc.h"
#include "strbuf.h"

#include <algorithm>
#include <cstring>

#if defined(DRJIT_ENABLE_HIPRT)
#  include <hiprt/hiprt.h>
#endif

int jitc_amd_runtime_version_v = 0;

static std::string jitc_amd_normalize_arch(const char *gcn_arch_name) {
    std::string result = gcn_arch_name ? gcn_arch_name : "";
    size_t suffix = result.find(':');
    if (suffix != std::string::npos)
        result.resize(suffix);
    return result;
}

void jitc_amd_check(hipError_t result, const char *expr,
                    const char *file, int line) {
    if (likely(result == hipSuccess))
        return;

    jitc_raise("%s:%i: HIP call \"%s\" failed: %s", file, line, expr,
               hipGetErrorString(result));
}

void jitc_amd_rtc_check(hiprtcResult result, const char *expr,
                        const char *file, int line) {
    if (likely(result == HIPRTC_SUCCESS))
        return;

    jitc_raise("%s:%i: HIPRTC call \"%s\" failed: %s", file, line, expr,
               hiprtcGetErrorString(result));
}

bool jitc_amd_init() {
    int device_count = 0;
    hipError_t result = hipGetDeviceCount(&device_count);
    if (result != hipSuccess) {
        jitc_log(Warn, "jit_amd_init(): hipGetDeviceCount() failed: %s",
                 hipGetErrorString(result));
        return false;
    }

    if (device_count == 0) {
        jitc_log(Warn, "jit_amd_init(): no AMD/HIP devices were found.");
        return false;
    }

    hip_check(hipRuntimeGetVersion(&jitc_amd_runtime_version_v));

    for (int i = 0; i < device_count; ++i) {
        hipDeviceProp_t props;
        result = hipGetDeviceProperties(&props, i);
        if (result != hipSuccess) {
            jitc_log(Warn, "jit_amd_init(): hipGetDeviceProperties(%i) "
                           "failed: %s", i, hipGetErrorString(result));
            continue;
        }

        std::string arch = jitc_amd_normalize_arch(props.gcnArchName);
        if (arch.empty()) {
            jitc_log(Warn, "jit_amd_init(): skipping device %i because it did "
                           "not report a GCN architecture name.", i);
            continue;
        }

        AMDDevice dev;
        dev.id = i;
        dev.name = strdup(props.name);
        dev.arch = strdup(arch.c_str());
        dev.multi_processor_count = props.multiProcessorCount;
        dev.wavefront_size = props.warpSize ? props.warpSize : 32;
        dev.max_threads_per_block = props.maxThreadsPerBlock;
        dev.shared_memory_bytes = props.sharedMemPerBlock;

        hip_check(hipDeviceGet(&dev.device, i));
        hip_check(hipCtxCreate(&dev.context, 0, dev.device));
        hip_check(hipStreamCreateWithFlags(&dev.stream, hipStreamNonBlocking));
        hip_check(hipEventCreateWithFlags(&dev.event, hipEventDisableTiming));
        hip_check(hipEventCreateWithFlags(&dev.sync_stream_event,
                                          hipEventDisableTiming));

        jitc_log(Info, "jit_amd_init(): found device %i: %s (%s, %u CUs)",
                 (int) state.amd_devices.size(), props.name, dev.arch,
                 dev.multi_processor_count);

        state.amd_devices.push_back(dev);
    }

    return !state.amd_devices.empty();
}

void jitc_amd_shutdown() {
    for (AMDDevice &dev : state.amd_devices) {
        if (dev.context)
            hipCtxSetCurrent(dev.context);

        for (hipModule_t module : dev.modules)
            if (module)
                hipModuleUnload(module);

#if defined(DRJIT_ENABLE_HIPRT)
        if (dev.hiprt_context)
            hiprtDestroyContext((hiprtContext) dev.hiprt_context);
#endif

        if (dev.event)
            hipEventDestroy(dev.event);
        if (dev.sync_stream_event)
            hipEventDestroy(dev.sync_stream_event);
        if (dev.stream)
            hipStreamDestroy(dev.stream);

        if (dev.context)
            hipCtxDestroy(dev.context);

        free(dev.name);
        free(dev.arch);
    }

    state.amd_devices.clear();
}

void jitc_amd_set_device(int device_id) {
    ThreadState *ts = thread_state(JitBackend::AMD);
    if (ts->device == device_id && ts->amd_context)
        return;

    if (device_id < 0 || (size_t) device_id >= state.amd_devices.size())
        jitc_raise("jit_amd_set_device(%i): must be in the range 0..%i!",
                   device_id, (int) state.amd_devices.size() - 1);

    jitc_log(Info, "jit_amd_set_device(%i)", device_id);

    AMDDevice &device = state.amd_devices[device_id];

    if (ts->amd_stream)
        hip_check(hipStreamSynchronize(ts->amd_stream));

    hip_check(hipCtxSetCurrent(device.context));

    ts->device = device_id;
    ts->amd_raw_device = device.id;
    ts->amd_device = device.device;
    ts->amd_context = device.context;
    ts->amd_stream = device.stream;
    ts->amd_event = device.event;
    ts->amd_sync_stream_event = device.sync_stream_event;
    ts->amd_arch = device.arch;
    ts->amd_wavefront_size = device.wavefront_size;
    ts->amd_max_threads = device.max_threads_per_block;
}

void jitc_amd_sync_thread(ThreadState *ts) {
    if (!ts)
        return;

    hipStream_t stream = ts->amd_stream;
    hip_check(hipCtxSetCurrent(ts->amd_context));
    unlock_guard guard(state.lock);
    hip_check(hipStreamSynchronize(stream));
}

void jitc_amd_sync_stream(uintptr_t stream) {
    ThreadState *ts = thread_state(JitBackend::AMD);
    hipEvent_t sync_event = ts->amd_sync_stream_event;
    hip_check(hipCtxSetCurrent(ts->amd_context));
    hip_check(hipEventRecord(sync_event, ts->amd_stream));
    hip_check(hipStreamWaitEvent(stream == 2 ? hipStreamPerThread
                                             : (hipStream_t) stream,
                                 sync_event, 0));
}

void jitc_amd_sync_device(ThreadState *ts) {
    if (!ts)
        return;

    hip_check(hipCtxSetCurrent(ts->amd_context));
    unlock_guard guard(state.lock);
    hip_check(hipDeviceSynchronize());
}

JitEvent jitc_amd_event_create(bool enable_timing) {
    ThreadState *ts = thread_state(JitBackend::AMD);
    hip_check(hipCtxSetCurrent(ts->amd_context));

    EventData *event = new EventData(JitBackend::AMD, enable_timing);
    event->ts = ts;

    unsigned flags = enable_timing ? hipEventDefault : hipEventDisableTiming;
    hip_check(hipEventCreateWithFlags(&event->amd_event, flags));
    return (JitEvent) event;
}

void jitc_amd_event_destroy(JitEvent event) {
    EventData *e = (EventData *) event;
    hip_check(hipCtxSetCurrent(e->ts->amd_context));
    hip_check(hipEventDestroy(e->amd_event));
    delete e;
}

void jitc_amd_event_record(JitEvent event) {
    EventData *e = (EventData *) event;
    hip_check(hipCtxSetCurrent(e->ts->amd_context));
    hip_check(hipEventRecord(e->amd_event, e->ts->amd_stream));
}

int jitc_amd_event_query(JitEvent event) {
    EventData *e = (EventData *) event;
    hip_check(hipCtxSetCurrent(e->ts->amd_context));
    hipError_t result = hipEventQuery(e->amd_event);
    if (result == hipSuccess)
        return 1;
    if (result == hipErrorNotReady)
        return 0;
    hip_check(result);
    return 0;
}

void jitc_amd_event_wait(JitEvent event) {
    EventData *e = (EventData *) event;
    hip_check(hipCtxSetCurrent(e->ts->amd_context));
    hip_check(hipEventSynchronize(e->amd_event));
}

float jitc_amd_event_elapsed_time(JitEvent start, JitEvent end) {
    EventData *s = (EventData *) start;
    EventData *e = (EventData *) end;
    if (!s->enable_timing || !e->enable_timing)
        jitc_raise("jit_event_elapsed_time(): both events must have timing enabled");

    hip_check(hipCtxSetCurrent(s->ts->amd_context));
    float ms;
    hip_check(hipEventElapsedTime(&ms, s->amd_event, e->amd_event));
    return ms;
}

#endif
