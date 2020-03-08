#include "jit.h"
#include "malloc.h"
#include "jit.h"
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
        Device device;
        device.id = (uint32_t) i;
        device.block_count = prop.multiProcessorCount * 4;
        device.thread_count = 128;
        state.devices.push_back(device);
    }

    // Enable P2P communication if possible
    for (const Device &da : state.devices) {
        for (const Device &db : state.devices) {
            if (da.id == db.id)
                continue;

            int peer_ok = 0;
            cuda_check(cudaDeviceCanAccessPeer(&peer_ok, da.id, db.id));
            if (peer_ok) {
                jit_log(Debug, " - Enabling peer access from device %i -> %i.",
                        da.id, db.id);
                cuda_check(cudaSetDevice(da.id));
                cuda_check(cudaDeviceEnablePeerAccess(db.id, 0));
            }
        }
    }
#endif
    state.scatter_gather_operand = 0;
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

    jit_malloc_shutdown();
    state.devices.clear();
    state.initialized = false;

    jit_log(Info, "jit_shutdown(): done.");
}

/// Set the currently active device & stream
void jit_device_set(uint32_t device, uint32_t stream) {
#if defined(ENOKI_CUDA)
    if (device >= state.devices.size())
        jit_raise("jit_device_set(): invalid device ID!");

    std::pair<uint32_t, uint32_t> key(device, stream);
    auto it = state.streams.find(key);

    Stream *stream_ptr, *active_stream_ptr = active_stream;
    if (it != state.streams.end()) {
        stream_ptr = it->second;
        if (stream_ptr == active_stream_ptr)
            return;
        jit_log(Trace, "jit_device_set(device=%i, stream=%i): selecting stream.", device, stream);
        if (stream_ptr->device != active_stream_ptr->device)
            cuda_check(cudaSetDevice(state.devices[device].id));
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
#else
    jit_fail("jit_device_set(): unsupported! (CUDA support was disabled.)");
#endif
}

/// Wait for all computation on the current stream to finish
void jit_stream_sync() {
#if defined(ENOKI_CUDA)
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_fail("jit_stream_sync(): device and stream must be set! "
                 "(call jit_device_set() beforehand)!");
    unlock_guard guard(state.mutex);
    jit_log(Trace, "jit_stream_sync(): starting..");
    cuda_check(cudaStreamSynchronize(stream->handle));
    jit_log(Trace, "jit_stream_sync(): done.");
#else
    jit_raise("jit_stream_sync(): unsupported! (CUDA support was disabled.)");
#endif
}

/// Wait for all computation on the current device to finish
void jit_device_sync() {
#if defined(ENOKI_CUDA)
    unlock_guard guard(state.mutex);
    jit_log(Trace, "jit_device_sync(): starting..");
    cuda_check(cudaDeviceSynchronize());
    jit_log(Trace, "jit_device_sync(): done.");
#else
    jit_raise("jit_device_sync(): unsupported! (CUDA support was disabled.)");
#endif
}

Buffer::Buffer() : m_start(nullptr), m_cur(nullptr), m_end(nullptr) {
    const size_t size = 1024;
    m_start = (char *) malloc(size);
    if (unlikely(m_start == nullptr))
        jit_fail("Buffer(): out of memory!");
    m_end = m_start + size;
    clear();
}

void Buffer::expand() {
    size_t old_alloc_size = m_end - m_start,
           new_alloc_size = 2 * old_alloc_size,
           used_size      = m_cur - m_start,
           copy_size      = std::min(used_size + 1, old_alloc_size);

    char *tmp = (char *) malloc(new_alloc_size);
    if (unlikely(m_start == nullptr))
        jit_fail("Buffer::expand() out of memory!");
    memcpy(tmp, m_start, copy_size);
    free(m_start);

    m_start = tmp;
    m_end = m_start + new_alloc_size;
    m_cur = m_start + used_size;
}

