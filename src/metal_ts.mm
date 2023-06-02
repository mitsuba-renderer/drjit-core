#include "log.h"
#include "thread_state.h"
#include "metal.h"

#define ThreadState ThreadState_
#include <Metal/Metal.h>
#undef ThreadState

extern id<MTLDevice> device;
extern id<MTLCommandQueue> queue;
extern id<MTLCommandBuffer> cbuf;

NB_TLS ThreadState* thread_state_metal = nullptr;

class MetalThreadState : public ThreadState {
public:
    // ================== ThreadState interface ==================

    /// Allocate private or shared memory accessible on host+device
    void *malloc(size_t size, bool shared) override;

    /// Associate the CPU memory address associated with an allocation
    void *host_ptr(void *ptr) override;

    /// Enqueue a memory copy operation
    void enqueue_memcpy(void *dst, const void *src, size_t size) override;

    /// Enqueue a host callback function
    void enqueue_callback(void (*fn)(void *), void *payload) override;

    /// Wait for queued computation to finish
    void sync() override;
};

ThreadState *jitc_metal_thread_state_new() {
    if (!device)
        jitc_raise("jit_metal_thread_state_new(): the Metal backend has not "
                   "been initialized.\nThere could be two reasons for "
                   "this:\n\n 1. The graphics card of your machine is too old "
                   "and unsupported.\n\n 2. The application code did not "
                   "perform the backend initialization.\n    Call `jit_init(1 "
                   "<< (int) JitBackend::Metal)` in this case.");

    return new MetalThreadState();
}

void *MetalThreadState::malloc(size_t size, bool shared) {
    id<MTLBuffer> buffer = [device newBufferWithLength: (NSUInteger) size
                                   options: shared ? MTLResourceStorageModeShared
                                                   : MTLResourceStorageModePrivate];
    return (void *) buffer;
}

void *MetalThreadState::host_ptr(void *buffer) {
    return [((id<MTLBuffer>) buffer) contents];
}

void MetalThreadState::enqueue_callback(void (*fn)(void *), void *payload) {
    [cbuf addCompletedHandler: ^(id<MTLCommandBuffer>) { fn(payload); }];
}

void MetalThreadState::enqueue_memcpy(void *dst, const void *src, size_t size) {
    id <MTLBlitCommandEncoder> enc = [cbuf blitCommandEncoder];

    [enc copyFromBuffer: (id<MTLBuffer>) src
           sourceOffset: 0
               toBuffer: (id<MTLBuffer>) dst
      destinationOffset: 0
                   size: size];

    [enc endEncoding];
    [enc release];
}

void MetalThreadState::sync() {
    [cbuf commit];
    [cbuf waitUntilCompleted];
    [cbuf release];
    cbuf = [queue commandBuffer];
}
