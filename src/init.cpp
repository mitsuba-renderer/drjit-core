#include "internal.h"
#include "malloc.h"
#include "internal.h"
#include "log.h"

State state;
Buffer buffer;

#if defined(ENOKI_CUDA)
    __thread Stream *active_stream = nullptr;
#endif

/// Initialize core data structures of the JIT compiler
void jit_init() {
    if (state.initialized)
        return;

    if (!state.variables.empty())
        jit_fail("Cannot reinitialize JIT while variables are still being used!");

#if defined(ENOKI_CUDA)
    // Enumerate CUDA devices and collect suitable ones
    int n_devices = 0;
    cuda_check(cudaGetDeviceCount(&n_devices));

    jit_log(Info, "jit_init(): detecting devices ..");
    for (int i = 0; i < n_devices; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        jit_log(Info,
                " - Found CUDA device %i: \"%s\" "
                "(PCI ID %02x:%02x.%i, %i SMs, %s).",
                i, prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID,
                prop.multiProcessorCount, jit_mem_string(prop.totalGlobalMem));
        if (prop.unifiedAddressing == 0) {
            jit_log(Warn, " - Warning: device does *not* support unified addressing, skipping..");
            continue;
        } else if (prop.managedMemory == 0) {
            jit_log(Warn, " - Warning: device does *not* support managed memory, skipping..");
            continue;
        }
        if (prop.concurrentManagedAccess == 0)
            jit_log(Warn, " - Warning: device does *not* support concurrent managed access.");


        cuda_check(cudaSetDevice(i));
        state.devices.push_back(i);
    }

    // Enable P2P communication if possible
    for (int da : state.devices) {
        for (int db : state.devices) {
            if (da == db)
                continue;

            int peer_ok = 0;
            cuda_check(cudaDeviceCanAccessPeer(&peer_ok, da, db));
            if (peer_ok) {
                jit_log(Debug, " - Enabling peer access from device %i -> %i.",
                        da, db);
                cuda_check(cudaSetDevice(da));
                cuda_check(cudaDeviceEnablePeerAccess(db, 0));
            }
        }
    }
#endif

    state.scatter_gather_operand = 0;
    state.alloc_addr_mask = 0;
    state.alloc_addr_ref = nullptr;
    state.variable_index = 1;
    state.initialized = true;
}

/// Release all resources used by the JIT compiler, and report reference leaks.
void jit_shutdown() {
    if (!state.initialized)
        return;

    jit_log(Info, "jit_shutdown(): destroying streams ..");

    for (auto [key, stream] : state.streams) {
        jit_device_set(stream->device, stream->stream);
        jit_free_flush();
        {
            unlock_guard guard(state.mutex);
            cuda_check(cudaStreamSynchronize(stream->handle));
        }
        cuda_check(cudaEventDestroy(stream->event));
        cuda_check(cudaStreamDestroy(stream->handle));
        delete stream->release_chain;
        delete stream;
    }
    state.streams.clear();
    active_stream = nullptr;

    for (auto [key, value] : state.kernels) {
        free((char *) key);
        cuda_check(cuModuleUnload(value.cu_module));
    }
    state.kernels.clear();

    if (state.log_level >= Warn) {
        uint32_t n_leaked = 0;
        for (auto &var : state.variables) {
            if (n_leaked == 0)
                jit_log(Warn, "jit_shutdown(): detected variable leaks:");
            if (n_leaked < 10)
                jit_log(Warn, " - variable %u is still referenced!", var.first);
            else if (n_leaked == 10)
                jit_log(Warn, " - (skipping remainder)");
            ++n_leaked;
        }

        if (n_leaked > 0)
            jit_log(Warn, "jit_shutdown(): %u variables are still referenced!", n_leaked);
    }

    if (state.variables.empty() && !state.cse_cache.empty())
        jit_fail("jit_shutdown(): detected a common subexpression elimination cache leak!");

    jit_malloc_shutdown();
    state.devices.clear();
    state.initialized = false;

    jit_log(Info, "jit_shutdown(): done.");
}

/// Set the currently active device & stream
void jit_device_set(int32_t device, uint32_t stream) {
    if (device == -1) {
#if defined(ENOKI_CUDA)
        active_stream = nullptr;
#endif
        return;
    }

    if ((size_t) device >= state.devices.size())
        jit_raise("jit_device_set(): invalid device ID!");

#if defined(ENOKI_CUDA)
    std::pair<uint32_t, uint32_t> key(device, stream);
    auto it = state.streams.find(key);

    Stream *stream_ptr, *active_stream_ptr = active_stream;
    if (it != state.streams.end()) {
        stream_ptr = it->second;
        if (stream_ptr == active_stream_ptr)
            return;
        jit_log(Trace, "jit_device_set(device=%i, stream=%i): selecting stream.", device, stream);
        if (stream_ptr->device != active_stream_ptr->device)
            cuda_check(cudaSetDevice(state.devices[device]));
    } else {
        jit_log(Trace, "jit_device_set(device=%i, stream=%i): creating stream.", device, stream);
        cudaStream_t handle = nullptr;
        cudaEvent_t event = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking));
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        stream_ptr = new Stream();
        stream_ptr->device = device;
        stream_ptr->stream = stream;
        stream_ptr->handle = handle;
        stream_ptr->event = event;
        state.streams[key] = stream_ptr;
    }
    active_stream = stream_ptr;
#endif
}

/// Wait for all computation on the current stream to finish
void jit_sync_stream() {
#if defined(ENOKI_CUDA)
    Stream *stream = active_stream;
    if (unlikely(!stream))
        return;
    jit_log(Trace, "jit_sync_stream(): starting..");
    /* Release mutex while synchronizing */ {
        unlock_guard guard(state.mutex);
        cuda_check(cudaStreamSynchronize(stream->handle));
    }
    jit_log(Trace, "jit_sync_stream(): done.");
#endif
}

/// Wait for all computation on the current device to finish
void jit_sync_device() {
#if defined(ENOKI_CUDA)
    Stream *stream = active_stream;
    if (unlikely(!stream))
        return;
    jit_log(Trace, "jit_sync_device(): starting..");
    /* Release mutex while synchronizing */ {
        unlock_guard guard(state.mutex);
        cuda_check(cudaDeviceSynchronize());
    }
    jit_log(Trace, "jit_sync_device(): done.");
#endif
}

