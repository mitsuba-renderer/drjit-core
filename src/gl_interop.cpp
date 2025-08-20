#include <drjit-core/gl_interop.h>
#include <drjit-core/jit.h>
#include "cuda.h"
#include "cuda_api.h"
#include "internal.h"

#if !defined(DRJIT_DYNAMIC_CUDA)
#  include <cudaGL.h>
#  include <driver_types.h>
#else
#  define GL_TEXTURE_2D 0x0DE1
#endif

void *jit_register_gl_buffer(GLuint gl_buffer) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUgraphicsResource cuda_resource;
    cuda_check(cuGraphicsGLRegisterBuffer(
        &cuda_resource, gl_buffer, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

    return (void *) cuda_resource;
}

void *jit_register_gl_texture(GLuint gl_texture) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUgraphicsResource cuda_resource;
    cuda_check(
        cuGraphicsGLRegisterImage(&cuda_resource, gl_texture, GL_TEXTURE_2D,
                                  CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

    return (void *) cuda_resource;
}

void jit_unregister_cuda_resource(void *cuda_resource) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    cuda_check(
        cuGraphicsUnregisterResource((CUgraphicsResource) cuda_resource));
}

void *jit_map_graphics_resource_ptr(void *cuda_resource, size_t *n_bytes) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    cuda_check(cuGraphicsMapResources(1, (CUgraphicsResource *) &cuda_resource,
                                      ts->stream));

    void *ptr;
    cuda_check(cuGraphicsResourceGetMappedPointer(
        &ptr, n_bytes, (CUgraphicsResource) cuda_resource));

    return ptr;
}

void *jit_map_graphics_resource_array(void *cuda_resource, uint32_t array_index,
                                      uint32_t mip_level) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    void *ptr;
    cuda_check(cuGraphicsMapResources(1, (CUgraphicsResource *) &cuda_resource,
                                      ts->stream));
    cuda_check(cuGraphicsSubResourceGetMappedArray(
        (CUarray *) &ptr, (CUgraphicsResource) cuda_resource, array_index,
        mip_level));

    return ptr;
}

void jit_unmap_graphics_resource(void *cuda_resource) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    cuda_check(
        cuGraphicsUnmapResources(1, (CUgraphicsResource *) &cuda_resource, 0));
}

void jit_memcpy_2d_to_array_async(void *dst, void *src, size_t src_pitch,
                                  size_t component_size_bytes, size_t width,
                                  size_t height, bool from_host) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUDA_MEMCPY2D op;
    memset(&op, 0, sizeof(CUDA_MEMCPY2D));

    if (from_host) {
        op.srcMemoryType = CU_MEMORYTYPE_HOST;
        op.srcHost       = src;
    } else {
        op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        op.srcDevice     = (CUdeviceptr) src;
    }
    op.srcPitch      = src_pitch;
    op.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    op.dstArray      = (CUarray) dst;
    op.WidthInBytes  = width * component_size_bytes;
    op.Height        = height;

    cuda_check(cuMemcpy2DAsync(&op, ts->stream));
}

#undef cuda_check
