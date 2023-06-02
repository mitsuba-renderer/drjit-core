#include "log.h"
#include "metal.h"

#define ThreadState ThreadState_
#include <Metal/Metal.h>
#undef ThreadState

id<MTLDevice> device;
id<MTLCommandQueue> queue;
id<MTLCommandBuffer> cbuf;

bool jitc_metal_init() {
    if (device)
        return true;

    if(!(device = MTLCreateSystemDefaultDevice()))
        return false;

    if (![device supportsRaytracing] ||
        ![device hasUnifiedMemory]) {
        device = nil;
        return false;
    }

    queue = [device newCommandQueue];
    cbuf = [queue commandBuffer];

    jitc_log(Info, " - Found Metal device \"%s\"", [[device name] UTF8String]);
    return true;
}

void jitc_metal_shutdown() {
    if (!device)
        return;
    [queue release];
    [device release];
    [cbuf release];
    queue = nil;
    device = nil;
    cbuf = nil;
}

void jitc_metal_free(int /* device_id */,  bool /* shared */, void *ptr) {
    [((id<MTLBuffer>) ptr) release];
}
