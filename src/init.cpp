#include "internal.h"
#include "malloc.h"
#include "internal.h"
#include "log.h"

State state;
Buffer buffer;
__thread Stream *active_stream = nullptr;

/// Initialize core data structures of the JIT compiler
void jit_init() {
    if (state.initialized)
        return;

    if (!state.variables.empty())
        jit_fail("Cannot reinitialize JIT while variables are still being used!");

    // Enumerate CUDA devices and collect suitable ones

    jit_log(Info, "jit_init(): detecting devices ..");

    int n_devices = 0;
    bool has_cuda = jit_cuda_init();

    if (has_cuda)
        cuda_check(cudaGetDeviceCount(&n_devices));

    for (int i = 0; i < n_devices; ++i) {
        int pci_bus_id = 0, pci_dom_id = 0, pci_dev_id = 0, num_sm = 0,
            unified_addr = 0, managed = 0, concurrent_managed = 0;
        size_t mem_total = 0;
        char name[256];

        cuda_check(cuDeviceTotalMem(&mem_total, i));
        cuda_check(cuDeviceGetName(name, sizeof(name), i));
        cuda_check(cudaDeviceGetAttribute(&pci_bus_id, cudaDevAttrPciBusId, i));
        cuda_check(cudaDeviceGetAttribute(&pci_dev_id, cudaDevAttrPciDeviceId, i));
        cuda_check(cudaDeviceGetAttribute(&pci_dom_id, cudaDevAttrPciDomainId, i));
        cuda_check(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, i));
        cuda_check(cudaDeviceGetAttribute(&unified_addr, cudaDevAttrUnifiedAddressing, i));
        cuda_check(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, i));
        cuda_check(cudaDeviceGetAttribute(&concurrent_managed, cudaDevAttrConcurrentManagedAccess, i));

        jit_log(Info,
                " - Found CUDA device %i: \"%s\" "
                "(PCI ID %02x:%02x.%i, %i SMs, %s)",
                i, name, pci_bus_id, pci_dev_id, pci_dom_id, num_sm,
                jit_mem_string(mem_total));
        if (unified_addr == 0) {
            jit_log(Warn, " - Warning: device does *not* support unified addressing, skipping ..");
            continue;
        } else if (managed == 0) {
            jit_log(Warn, " - Warning: device does *not* support managed memory, skipping ..");
            continue;
        }
        if (concurrent_managed == 0)
            jit_log(Warn, " - Warning: device does *not* support concurrent managed access.");

        state.devices.push_back(Device{i, num_sm});
    }

    // Enable P2P communication if possible
    for (auto &a : state.devices) {
        for (auto &b : state.devices) {
            if (a.id == b.id)
                continue;

            int peer_ok = 0;
            cuda_check(cudaDeviceCanAccessPeer(&peer_ok, a.id, b.id));
            if (peer_ok) {
                jit_log(Debug, " - Enabling peer access from device %i -> %i",
                        a.id, b.id);
                cuda_check(cudaSetDevice(a.id));
                cudaError_t rv = cudaDeviceEnablePeerAccess(b.id, 0);
                if (rv == cudaErrorPeerAccessAlreadyEnabled)
                    continue;
                cuda_check(rv);
            }
        }
    }

    if (!state.devices.empty())
        cuda_check(cudaSetDevice(state.devices[0].id));

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

    jit_log(Info, "jit_shutdown(): done");
}

/// Set the currently active device & stream
void jit_device_set(int32_t device, uint32_t stream) {
    if (device == -1) {
        active_stream = nullptr;
        return;
    }

    if ((size_t) device >= state.devices.size())
        jit_raise("jit_device_set(): invalid device ID!");

    std::pair<uint32_t, uint32_t> key(device, stream);
    auto it = state.streams.find(key);

    Stream *stream_ptr, *active_stream_ptr = active_stream;
    if (it != state.streams.end()) {
        stream_ptr = it->second;
        if (stream_ptr == active_stream_ptr)
            return;
        jit_log(Trace, "jit_device_set(device=%i, stream=%i): selecting stream", device, stream);
        if (stream_ptr->device != active_stream_ptr->device)
            cuda_check(cudaSetDevice(state.devices[device].id));
    } else {
        jit_log(Trace, "jit_device_set(device=%i, stream=%i): creating stream", device, stream);
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
}

/// Wait for all computation on the current stream to finish
void jit_sync_stream() {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        return;

    jit_log(Trace, "jit_sync_stream(): starting ..");
    /* Release mutex while synchronizing */ {
        unlock_guard guard(state.mutex);
        cuda_check(cudaStreamSynchronize(stream->handle));
    }
    jit_log(Trace, "jit_sync_stream(): done.");
}

/// Wait for all computation on the current device to finish
void jit_sync_device() {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        return;

    jit_log(Trace, "jit_sync_device(): starting ..");
    /* Release mutex while synchronizing */ {
        unlock_guard guard(state.mutex);
        cuda_check(cudaDeviceSynchronize());
    }
    jit_log(Trace, "jit_sync_device(): done.");
}

Stream *jit_get_stream(const char *func_name) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("%s(): device and stream must be set! (call jit_device_set() beforehand)!", func_name);
    return stream;
}
