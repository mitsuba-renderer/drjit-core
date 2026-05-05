// metal_rt_test.cpp — Test Metal ray tracing via the Dr.Jit API.
//
// Builds a single-triangle acceleration structure via raw Metal API,
// configures it with jit_metal_configure_scene(), then traces rays using
// jit_metal_ray_trace() and verifies the results.

#include <drjit-core/jit.h>
#include <drjit-core/metal.h>
#include <cstdio>
#include <cstring>
#include <cmath>

// metal-cpp headers for raw Metal API (accel struct construction)
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

int main() {
    jit_set_log_level_stderr(LogLevel::Trace);
    jit_set_flag(JitFlag::PrintIR, 1);
    jit_init((uint32_t) JitBackend::Metal);

    if (!jit_has_backend(JitBackend::Metal)) {
        fprintf(stderr, "Metal backend not available\n");
        return 1;
    }
    if (!jit_metal_supports_ray_tracing()) {
        fprintf(stderr, "Metal ray tracing not supported on this device\n");
        return 1;
    }

    int failures = 0;
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    auto *device = (MTL::Device *) jit_metal_device_handle();
    auto *queue  = (MTL::CommandQueue *) jit_metal_command_queue();

    // ----------------------------------------------------------------
    //  Build a triangle: vertices at (0,0,0), (1,0,0), (0,1,0)
    // ----------------------------------------------------------------
    float vertices[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    uint32_t indices[] = { 0, 1, 2 };

    auto *vb = device->newBuffer(vertices, sizeof(vertices),
                                 MTL::ResourceStorageModeShared);
    auto *ib = device->newBuffer(indices, sizeof(indices),
                                 MTL::ResourceStorageModeShared);

    // --- BLAS ---
    auto *geom = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    geom->setVertexBuffer(vb);
    geom->setVertexStride(12);
    geom->setVertexFormat(MTL::AttributeFormatFloat3);
    geom->setIndexBuffer(ib);
    geom->setIndexType(MTL::IndexTypeUInt32);
    geom->setTriangleCount(1);
    geom->setOpaque(true);

    NS::Object *geom_objs[] = { (NS::Object *) geom };
    auto *geom_arr = NS::Array::array((const NS::Object *const *) geom_objs, 1);
    auto *blas_desc = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    blas_desc->setGeometryDescriptors(geom_arr);

    auto blas_sizes = device->accelerationStructureSizes(blas_desc);
    auto *blas = device->newAccelerationStructure(blas_sizes.accelerationStructureSize);
    auto *blas_scratch = device->newBuffer(blas_sizes.buildScratchBufferSize,
                                           MTL::ResourceStorageModePrivate);

    auto *cb = queue->commandBuffer();
    auto *enc = cb->accelerationStructureCommandEncoder();
    enc->buildAccelerationStructure(blas, blas_desc, blas_scratch, 0);
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();

    // --- TLAS ---
    MTL::AccelerationStructureInstanceDescriptor inst_desc;
    memset(&inst_desc, 0, sizeof(inst_desc));
    // Identity transform (column-major 3x4)
    inst_desc.transformationMatrix.columns[0] = {1, 0, 0};
    inst_desc.transformationMatrix.columns[1] = {0, 1, 0};
    inst_desc.transformationMatrix.columns[2] = {0, 0, 1};
    inst_desc.transformationMatrix.columns[3] = {0, 0, 0};
    inst_desc.accelerationStructureIndex = 0;
    inst_desc.mask = 0xFF;
    inst_desc.options = MTL::AccelerationStructureInstanceOptionOpaque;

    auto *inst_buf = device->newBuffer(&inst_desc, sizeof(inst_desc),
                                       MTL::ResourceStorageModeShared);

    NS::Object *blas_objs[] = { (NS::Object *) blas };
    auto *blas_arr = NS::Array::array((const NS::Object *const *) blas_objs, 1);
    auto *tlas_desc = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    tlas_desc->setInstanceCount(1);
    tlas_desc->setInstanceDescriptorBuffer(inst_buf);
    tlas_desc->setInstancedAccelerationStructures(blas_arr);

    auto tlas_sizes = device->accelerationStructureSizes(tlas_desc);
    auto *tlas = device->newAccelerationStructure(tlas_sizes.accelerationStructureSize);
    auto *tlas_scratch = device->newBuffer(tlas_sizes.buildScratchBufferSize,
                                           MTL::ResourceStorageModePrivate);

    cb = queue->commandBuffer();
    enc = cb->accelerationStructureCommandEncoder();
    enc->buildAccelerationStructure(tlas, tlas_desc, tlas_scratch, 0);
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();

    // ----------------------------------------------------------------
    //  Configure Dr.Jit to use the TLAS
    // ----------------------------------------------------------------
    void *rt_resources[] = { blas, vb, ib };
    uint32_t scene_index = jit_metal_configure_scene(
        tlas, rt_resources, 3,
        /* intersection_fn_library= */ nullptr,
        /* n_ift_entries= */ 0,
        /* ift_function_names= */ nullptr,
        /* ift_buffers= */ nullptr,
        /* ift_buffer_slots= */ nullptr,
        /* geometry_types_mask= */ 0x1u);

    // ----------------------------------------------------------------
    //  Test 1: Ray hitting the triangle center
    //  Origin (0.25, 0.25, -1), direction (0, 0, 1), tmin=0, tmax=100
    //  Expected: hit at distance ~1.0, barycentrics ~(0.25, 0.25)
    // ----------------------------------------------------------------
    {
        float ox_v = 0.25f, oy_v = 0.25f, oz_v = -1.0f;
        float dx_v = 0.0f,  dy_v = 0.0f,  dz_v = 1.0f;
        float tmin_v = 0.0f, tmax_v = 100.0f;

        uint32_t ray_args[8];
        ray_args[0] = jit_var_literal(JitBackend::Metal, VarType::Float32, &ox_v);
        ray_args[1] = jit_var_literal(JitBackend::Metal, VarType::Float32, &oy_v);
        ray_args[2] = jit_var_literal(JitBackend::Metal, VarType::Float32, &oz_v);
        ray_args[3] = jit_var_literal(JitBackend::Metal, VarType::Float32, &dx_v);
        ray_args[4] = jit_var_literal(JitBackend::Metal, VarType::Float32, &dy_v);
        ray_args[5] = jit_var_literal(JitBackend::Metal, VarType::Float32, &dz_v);
        ray_args[6] = jit_var_literal(JitBackend::Metal, VarType::Float32, &tmin_v);
        ray_args[7] = jit_var_literal(JitBackend::Metal, VarType::Float32, &tmax_v);

        uint8_t mask_v = 1;
        uint32_t mask = jit_var_literal(JitBackend::Metal, VarType::Bool, &mask_v);

        uint32_t out[7];
        jit_metal_ray_trace(8, ray_args, mask, out, 7, scene_index);

        // Schedule all outputs and evaluate once
        for (int i = 0; i < 7; i++)
            jit_var_schedule(out[i]);
        jit_eval();

        printf("Test 1 — hit center:\n");
        printf("  valid:    %s\n", jit_var_str(out[0]));
        printf("  distance: %s\n", jit_var_str(out[1]));
        printf("  bary_u:   %s\n", jit_var_str(out[2]));
        printf("  bary_v:   %s\n", jit_var_str(out[3]));
        printf("  inst_id:  %s\n", jit_var_str(out[4]));
        printf("  prim_id:  %s\n", jit_var_str(out[5]));
        fflush(stdout);

        // Cleanup
        for (int i = 0; i < 8; i++) jit_var_dec_ref(ray_args[i]);
        jit_var_dec_ref(mask);
        for (int i = 0; i < 7; i++) jit_var_dec_ref(out[i]);
    }

    // ----------------------------------------------------------------
    //  Test 2: Ray missing the triangle
    //  Origin (5, 5, -1), direction (0, 0, 1)
    //  Expected: no hit (valid=false)
    // ----------------------------------------------------------------
    {
        float ox_v = 5.0f, oy_v = 5.0f, oz_v = -1.0f;
        float dx_v = 0.0f, dy_v = 0.0f, dz_v = 1.0f;
        float tmin_v = 0.0f, tmax_v = 100.0f;

        uint32_t ray_args[8];
        ray_args[0] = jit_var_literal(JitBackend::Metal, VarType::Float32, &ox_v);
        ray_args[1] = jit_var_literal(JitBackend::Metal, VarType::Float32, &oy_v);
        ray_args[2] = jit_var_literal(JitBackend::Metal, VarType::Float32, &oz_v);
        ray_args[3] = jit_var_literal(JitBackend::Metal, VarType::Float32, &dx_v);
        ray_args[4] = jit_var_literal(JitBackend::Metal, VarType::Float32, &dy_v);
        ray_args[5] = jit_var_literal(JitBackend::Metal, VarType::Float32, &dz_v);
        ray_args[6] = jit_var_literal(JitBackend::Metal, VarType::Float32, &tmin_v);
        ray_args[7] = jit_var_literal(JitBackend::Metal, VarType::Float32, &tmax_v);

        uint8_t mask_v = 1;
        uint32_t mask = jit_var_literal(JitBackend::Metal, VarType::Bool, &mask_v);

        uint32_t out[7];
        jit_metal_ray_trace(8, ray_args, mask, out, 7, scene_index);

        for (int i = 0; i < 7; i++)
            jit_var_schedule(out[i]);
        jit_eval();

        printf("Test 2 — miss:\n");
        printf("  valid: %s\n", jit_var_str(out[0]));
        fflush(stdout);

        for (int i = 0; i < 8; i++) jit_var_dec_ref(ray_args[i]);
        jit_var_dec_ref(mask);
        for (int i = 0; i < 7; i++) jit_var_dec_ref(out[i]);
    }

    // Cleanup — release the per-scene drjit handle. This drops drjit's
    // last reference on the MetalScene and frees its cached IFTs.
    jit_var_dec_ref(scene_index);

    tlas->release();
    tlas_scratch->release();
    tlas_desc->release();
    inst_buf->release();
    blas->release();
    blas_scratch->release();
    blas_desc->release();
    geom->release();
    vb->release();
    ib->release();

    pool->release();

    printf("\n%d test(s) failed.\n", failures);
    fflush(stdout);

    jit_shutdown(0);
    return failures;
}
