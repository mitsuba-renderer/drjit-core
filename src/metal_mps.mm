/*
    src/metal_mps.mm -- Metal Performance Shaders wrappers.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    This file must be compiled as Objective-C++ (.mm) because the MPS APIs
    use Objective-C classes. The rest of the Metal backend stays in plain
    C++ with metal-cpp wrappers and reaches the wrappers here through
    ``extern "C"`` entry points so no Obj-C headers leak into the rest of
    the codebase.

    Currently provides:
      * ``jitc_metal_mps_gemm`` -- batched GEMM via MPSMatrixMultiplication
*/

#if defined(DRJIT_ENABLE_METAL)

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdint>

extern "C" void jitc_metal_mps_gemm(
    void *mtl_device, void *mtl_queue,
    void *a_buf, size_t a_offset,
    void *b_buf, size_t b_offset,
    void *c_buf, size_t c_offset,
    uint32_t M, uint32_t N, uint32_t K,
    bool At, bool Bt,
    uint32_t a_rows, uint32_t a_cols,
    uint32_t b_rows, uint32_t b_cols,
    uint32_t tsize, int mps_data_type,
    double alpha, double beta)
{
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) mtl_device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) mtl_queue;

        MPSDataType dt = (MPSDataType) mps_data_type;

        MPSMatrixDescriptor *desc_a = [MPSMatrixDescriptor
            matrixDescriptorWithRows:a_rows
            columns:a_cols
            rowBytes:(NSUInteger)(a_cols * tsize)
            dataType:dt];

        MPSMatrixDescriptor *desc_b = [MPSMatrixDescriptor
            matrixDescriptorWithRows:b_rows
            columns:b_cols
            rowBytes:(NSUInteger)(b_cols * tsize)
            dataType:dt];

        MPSMatrixDescriptor *desc_c = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
            columns:N
            rowBytes:(NSUInteger)(N * tsize)
            dataType:dt];

        MPSMatrix *mat_a = [[MPSMatrix alloc]
            initWithBuffer:(__bridge id<MTLBuffer>) a_buf
            offset:a_offset
            descriptor:desc_a];

        MPSMatrix *mat_b = [[MPSMatrix alloc]
            initWithBuffer:(__bridge id<MTLBuffer>) b_buf
            offset:b_offset
            descriptor:desc_b];

        MPSMatrix *mat_c = [[MPSMatrix alloc]
            initWithBuffer:(__bridge id<MTLBuffer>) c_buf
            offset:c_offset
            descriptor:desc_c];

        MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc]
            initWithDevice:dev
            transposeLeft:At
            transposeRight:Bt
            resultRows:M
            resultColumns:N
            interiorColumns:K
            alpha:alpha
            beta:beta];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [gemm encodeToCommandBuffer:cb
            leftMatrix:mat_a
            rightMatrix:mat_b
            resultMatrix:mat_c];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

#endif // defined(DRJIT_ENABLE_METAL)
