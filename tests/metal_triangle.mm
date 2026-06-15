// metal_triangle.mm — Metal analog of triangle.cpp.
//
// Mirrors tests/triangle.cpp (CUDA/OptiX) for the Metal backend: it builds a
// single-triangle acceleration structure with the raw (native Objective-C)
// Metal API, registers it with Dr.Jit via jit_metal_configure_scene(), then
// generates a 16x16 grid of camera rays using the high-level MetalArray<>
// wrapper, traces them all in a single batched jit_metal_ray_trace() launch,
// migrates the hit mask to the host, and prints the resulting image. The loop
// runs twice to exercise kernel caching, exactly like triangle.cpp.
//
// Unlike OptiX, Metal performs the triangle intersection inline in the compute
// kernel, so there are no miss/closest-hit shaders, program groups, or SBT.
//
// This file uses native Objective-C Metal (#import <Metal/Metal.h>) and ARC —
// the metal-cpp bindings (MTL::/NS::) the original metal_rt_test.cpp relied on
// are no longer part of the project.

#include <drjit-core/array.h>
#include <drjit-core/metal.h>
#include <cstdio>
#include <cstring>

#import <Metal/Metal.h>

namespace dr = drjit;

// NOTE: avoid the name `UInt32` for the alias — <Metal/Metal.h> pulls in
// Foundation, which already typedefs `UInt32` to `unsigned int`.
using Float = dr::MetalArray<float>;
using UIntM  = dr::MetalArray<uint32_t>;
using Mask  = dr::MetalArray<bool>;

// A single triangle, matching the one in tests/triangle.cpp.
static const float vertices[9] = {
    -0.8f, -0.8f, 0.0f,
     0.8f, -0.8f, 0.0f,
     0.0f,  0.8f, 0.0f
};
static const uint32_t indices[3] = { 0, 1, 2 };

static void demo() {
    id<MTLDevice>       device = (__bridge id<MTLDevice>)       jit_metal_device_handle();
    id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>) jit_metal_command_queue();

    // =====================================================
    //  Upload vertex / index data
    // =====================================================
    id<MTLBuffer> vb = [device newBufferWithBytes:vertices
                                           length:sizeof(vertices)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> ib = [device newBufferWithBytes:indices
                                           length:sizeof(indices)
                                          options:MTLResourceStorageModeShared];

    // =====================================================
    //  Build the bottom-level acceleration structure (BLAS)
    // =====================================================
    MTLAccelerationStructureTriangleGeometryDescriptor *geom =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geom.vertexBuffer  = vb;
    geom.vertexStride  = 12;
    geom.vertexFormat  = MTLAttributeFormatFloat3;
    geom.indexBuffer   = ib;
    geom.indexType     = MTLIndexTypeUInt32;
    geom.triangleCount = 1;
    geom.opaque        = YES;

    MTLPrimitiveAccelerationStructureDescriptor *blas_desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    blas_desc.geometryDescriptors = @[ geom ];

    MTLAccelerationStructureSizes blas_sizes =
        [device accelerationStructureSizesWithDescriptor:blas_desc];
    id<MTLAccelerationStructure> blas =
        [device newAccelerationStructureWithSize:blas_sizes.accelerationStructureSize];
    id<MTLBuffer> blas_scratch =
        [device newBufferWithLength:blas_sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc =
        [cb accelerationStructureCommandEncoder];
    [enc buildAccelerationStructure:blas
                         descriptor:blas_desc
                      scratchBuffer:blas_scratch
                scratchBufferOffset:0];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    // =====================================================
    //  Build the top-level acceleration structure (TLAS)
    // =====================================================
    MTLAccelerationStructureInstanceDescriptor inst_desc;
    memset(&inst_desc, 0, sizeof(inst_desc));
    // Identity transform (column-major 3x4)
    inst_desc.transformationMatrix.columns[0].x = 1.f;
    inst_desc.transformationMatrix.columns[1].y = 1.f;
    inst_desc.transformationMatrix.columns[2].z = 1.f;
    inst_desc.accelerationStructureIndex = 0;
    inst_desc.mask                       = 0xFF;
    inst_desc.options                    = MTLAccelerationStructureInstanceOptionOpaque;

    id<MTLBuffer> inst_buf =
        [device newBufferWithBytes:&inst_desc
                            length:sizeof(inst_desc)
                           options:MTLResourceStorageModeShared];

    MTLInstanceAccelerationStructureDescriptor *tlas_desc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    tlas_desc.instanceCount                    = 1;
    tlas_desc.instanceDescriptorBuffer         = inst_buf;
    tlas_desc.instancedAccelerationStructures  = @[ blas ];

    MTLAccelerationStructureSizes tlas_sizes =
        [device accelerationStructureSizesWithDescriptor:tlas_desc];
    id<MTLAccelerationStructure> tlas =
        [device newAccelerationStructureWithSize:tlas_sizes.accelerationStructureSize];
    id<MTLBuffer> tlas_scratch =
        [device newBufferWithLength:tlas_sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];

    cb  = [queue commandBuffer];
    enc = [cb accelerationStructureCommandEncoder];
    [enc buildAccelerationStructure:tlas
                         descriptor:tlas_desc
                      scratchBuffer:tlas_scratch
                scratchBufferOffset:0];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    // =====================================================
    //  Let Dr.Jit know about the scene
    // =====================================================
    void *rt_resources[] = { (__bridge void *) blas,
                             (__bridge void *) vb,
                             (__bridge void *) ib };

    UIntM scene = UIntM::steal(jit_metal_configure_scene(
        (__bridge void *) tlas,
        rt_resources, 3,
        /* intersection_fn_library= */ nullptr,
        /* n_ift_entries=          */ 0,
        /* ift_function_names=     */ nullptr,
        /* n_ift_buffers=          */ 0,
        /* ift_buffers=            */ nullptr,
        /* ift_buffer_slots=       */ nullptr,
        /* geometry_types_mask=    */ 0x1u));

    // Twice, to verify kernel caching (like triangle.cpp).
    for (int i = 0; i < 2; ++i) {
        // =====================================================
        //  Generate a 16x16 grid of camera rays
        // =====================================================
        int res = 16;
        UIntM index = dr::arange<UIntM>(res * res),
             x     = index % res,
             y     = index / res;

        Float ox = Float(x) * (2.0f / res) - 1.0f,
              oy = Float(y) * (2.0f / res) - 1.0f,
              oz = -1.f;
        Float dx = 0.f, dy = 0.f, dz = 1.f;
        Float mint = 0.f, maxt = 100.f;
        Mask  mask = true;

        // =====================================================
        //  Trace all rays in a single batched launch
        // =====================================================
        uint32_t ray_args[8] = {
            ox.index(), oy.index(), oz.index(),
            dx.index(), dy.index(), dz.index(),
            mint.index(), maxt.index()
        };

        uint32_t out[7];
        jit_metal_ray_trace(8, ray_args, mask.index(), out, 7, scene.index());

        Mask valid = Mask::steal(out[0]);
        for (int k = 1; k < 7; ++k)
            jit_var_dec_ref(out[k]);

        // Convert the hit mask to a 0/1 image and read it back on the host.
        UIntM hit = select(valid, UIntM(1), UIntM(0));
        jit_var_eval(hit.index());

        UIntM hit_host =
            UIntM::steal(jit_var_migrate(hit.index(), JitBackend::None));
        jit_sync_thread();

        const uint32_t *img = hit_host.data();
        printf("Iteration %d:\n", i);
        for (int r = 0; r < res; ++r) {
            for (int c = 0; c < res; ++c)
                printf("%i ", img[r * res + c]);
            printf("\n");
        }
        printf("\n");
        fflush(stdout);
    }
}

int main(int, char **) {
    jit_init(1u << (uint32_t) JitBackend::Metal);

    if (!jit_has_backend(JitBackend::Metal)) {
        fprintf(stderr, "Metal backend not available.\n");
        jit_shutdown(0);
        return 1;
    }
    if (!jit_metal_supports_ray_tracing()) {
        fprintf(stderr, "Metal ray tracing not supported on this device.\n");
        jit_shutdown(0);
        return 1;
    }

    @autoreleasepool {
        demo();
    }

    jit_shutdown(0);
    return 0;
}
