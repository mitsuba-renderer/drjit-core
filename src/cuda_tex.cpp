#include "cuda_tex.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "op.h"
#include <string.h>

void *jitc_cuda_tex_create(size_t ndim, const size_t *shape, size_t n_channels,
                           int filter_mode, int wrap_mode) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_create(): invalid texture dimension!");
    else if (n_channels != 1 && n_channels != 2 && n_channels != 4)
        jitc_raise("jit_cuda_tex_create(): invalid channel count! (only 1, 2, "
                   "and 4 supported)");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUarray array = nullptr;
    if (ndim == 1 || ndim == 2) {
        CUDA_ARRAY_DESCRIPTOR array_desc;
        memset(&array_desc, 0, sizeof(CUDA_ARRAY_DESCRIPTOR));
        array_desc.Width = shape[0];
        array_desc.Height = (ndim == 2) ? shape[1] : 1;
        array_desc.Format = CU_AD_FORMAT_FLOAT;
        array_desc.NumChannels = (unsigned int) n_channels;
        cuda_check(cuArrayCreate(&array, &array_desc));
    } else {
        CUDA_ARRAY3D_DESCRIPTOR array_desc;
        memset(&array_desc, 0, sizeof(CUDA_ARRAY3D_DESCRIPTOR));
        array_desc.Width = shape[0];
        array_desc.Height = shape[1];
        array_desc.Depth = shape[2];
        array_desc.Format = CU_AD_FORMAT_FLOAT;
        array_desc.NumChannels = (unsigned int) n_channels;
        cuda_check(cuArray3DCreate(&array, &array_desc));
    }

    CUDA_RESOURCE_DESC res_desc;
    memset(&res_desc, 0, sizeof(CUDA_RESOURCE_DESC));
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = array;

    CUDA_TEXTURE_DESC tex_desc;
    memset(&tex_desc, 0, sizeof(CUDA_TEXTURE_DESC));
    switch (filter_mode) {
        case 0:
            tex_desc.filterMode = tex_desc.mipmapFilterMode =
                CU_TR_FILTER_MODE_POINT;
            break;
        case 1:
            tex_desc.filterMode = tex_desc.mipmapFilterMode =
                CU_TR_FILTER_MODE_LINEAR;
            break;
        default:
            jitc_raise("jit_cuda_tex_create(): invalid filter mode!");
            break;
    }
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    for (size_t i = 0; i < 3; ++i) {
        switch (wrap_mode) {
            case 0:
                tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_WRAP;
                break;
            case 1:
                tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
                break;
            case 2:
                tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_MIRROR;
                break;
            default:
                jitc_raise("jit_cuda_tex_create(): invalid wrap mode!");
                break;
        }
    }
    tex_desc.maxAnisotropy = 1;

    CUDA_RESOURCE_VIEW_DESC view_desc;
    memset(&view_desc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));
    view_desc.width = shape[0];
    view_desc.height = (ndim >= 2) ? shape[1] : 1;
    view_desc.depth = (ndim == 3) ? shape[2] : 0;

    if (n_channels == 1)
        view_desc.format = CU_RES_VIEW_FORMAT_FLOAT_1X32;
    else if (n_channels == 2)
        view_desc.format = CU_RES_VIEW_FORMAT_FLOAT_2X32;
    else
        view_desc.format = CU_RES_VIEW_FORMAT_FLOAT_4X32;

    CUtexObject result;
    cuda_check(cuTexObjectCreate(&result, &res_desc, &tex_desc, &view_desc));
    return (void *) result;
}

void jitc_cuda_tex_memcpy_d2t(size_t ndim, const size_t *shape, size_t n_channels,
                              const void *src_ptr, void *dst_texture) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_memcpy(): invalid texture dimension!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUDA_RESOURCE_DESC res_desc;
    cuda_check(cuTexObjectGetResourceDesc(&res_desc, (CUtexObject) dst_texture));

    size_t pitch = shape[0] * n_channels * sizeof(float);

    if (ndim == 1 || ndim == 2) {
        CUDA_MEMCPY2D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY2D));
        op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        op.srcDevice = (CUdeviceptr) src_ptr;
        op.srcPitch = pitch;
        op.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        op.dstArray = res_desc.res.array.hArray;
        op.WidthInBytes = pitch;
        op.Height = (ndim == 2) ? shape[1] : 1;
        cuda_check(cuMemcpy2DAsync(&op, ts->stream));
    } else {
        CUDA_MEMCPY3D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY3D));
        op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        op.srcDevice = (CUdeviceptr) src_ptr;
        op.srcPitch = pitch;
        op.srcHeight = shape[1];
        op.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        op.dstArray = res_desc.res.array.hArray;
        op.WidthInBytes = pitch;
        op.Height = shape[1];
        op.Depth = shape[2];
        cuda_check(cuMemcpy3DAsync(&op, ts->stream));
    }
}

void jitc_cuda_tex_memcpy_t2d(size_t ndim, const size_t *shape, size_t n_channels,
                              const void *src_texture, void *dst_ptr) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_memcpy(): invalid texture dimension!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUDA_RESOURCE_DESC res_desc;
    cuda_check(cuTexObjectGetResourceDesc(&res_desc, (CUtexObject) src_texture));

    size_t pitch = shape[0] * n_channels * sizeof(float);

    if (ndim == 1 || ndim == 2) {
        CUDA_MEMCPY2D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY2D));
        op.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        op.srcArray = res_desc.res.array.hArray;
        op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        op.dstDevice = (CUdeviceptr) dst_ptr;
        op.dstPitch = pitch;
        op.WidthInBytes = pitch;
        op.Height = (ndim == 2) ? shape[1] : 1;
        cuda_check(cuMemcpy2DAsync(&op, ts->stream));
    } else {
        CUDA_MEMCPY3D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY3D));
        op.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        op.srcArray = res_desc.res.array.hArray;
        op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        op.dstDevice = (CUdeviceptr) dst_ptr;
        op.dstPitch = pitch;
        op.dstHeight = shape[1];
        op.WidthInBytes = pitch;
        op.Height = shape[1];
        op.Depth = shape[2];
        cuda_check(cuMemcpy3DAsync(&op, ts->stream));
    }
}

void jitc_cuda_tex_lookup(size_t ndim, uint32_t texture_id, const uint32_t *pos,
                          uint32_t mask, uint32_t *out) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_lookup(): invalid texture dimension!");

    // Validate input types, determine size of the operation
    uint32_t size = 0;
    for (size_t i = 0; i <= ndim; ++i) {
        uint32_t index = i < ndim ? pos[i] : mask;
        VarType ref = i < ndim ? VarType::Float32 : VarType::Bool;
        const Variable *v = jitc_var(index);
        if ((VarType) v->type != ref)
            jitc_raise("jit_cuda_tex_lookup(): type mismatch for arg. %zu (got "
                       "%s, expected %s)", i, type_name[v->type],
                       type_name[(int) ref]);
        size = std::max(size, v->size);
    }

    // Potentially apply any masks on the mask stack
    Ref valid = borrow(mask);
    {
        Ref mask_top = steal(jitc_var_mask_peek(JitBackend::CUDA));
        uint32_t size_top = jitc_var(mask_top)->size;

        // If the mask on the mask stack is compatible, merge it
        if (size_top == size || size_top == 1 || size == 1) {
            uint32_t dep[2] = { mask, mask_top };
            valid = steal(jitc_var_new_op(JitOp::And, 2, dep));
        }
    }

    uint32_t dep[3] = {
        valid,
        texture_id,
        pos[0]
    };

    if (ndim >= 2) {
        const char *stmt_1[2] = {
           ".reg.v2.f32 $r0$n"
           "mov.v2.f32 $r0, { $r1, $r2 }",
           ".reg.v4.f32 $r0$n"
           "mov.v4.f32 $r0, { $r1, $r2, $r3, $r3 }"
        };
        dep[2] = jitc_var_new_stmt(JitBackend::CUDA, VarType::Void,
                                   stmt_1[ndim - 2], 1, (unsigned int) ndim, pos);
    } else {
        jitc_var_inc_ref_ext(dep[2]);
    }

    const char *stmt_2[4] = {
        ".reg.v4.f32 $r0$n"
        "@$r1 tex.1d.v4.f32.f32 $r0, [$r2, {$r3}]$n"
        "@!$r1 mov.v4.f32 $r0, {0.0, 0.0, 0.0, 0.0}",
        ".reg.v4.f32 $r0$n"
        "@$r1 tex.2d.v4.f32.f32 $r0, [$r2, $r3]$n"
        "@!$r1 mov.v4.f32 $r0, {0.0, 0.0, 0.0, 0.0}",
        ".reg.v4.f32 $r0$n"
        "@$r1 tex.3d.v4.f32.f32 $r0, [$r2, $r3]$n"
        "@!$r1 mov.v4.f32 $r0, {0.0, 0.0, 0.0, 0.0}"
    };

    uint32_t lookup = jitc_var_new_stmt(JitBackend::CUDA, VarType::Void,
                                        stmt_2[ndim - 1], 1, 3, dep);
    jitc_var_dec_ref_ext(dep[2]);

    const char *stmt_3[4] = {
        "mov.f32 $r0, $r1.r",
        "mov.f32 $r0, $r1.g",
        "mov.f32 $r0, $r1.b",
        "mov.f32 $r0, $r1.a"
    };

    for (int i = 0; i < 4; ++i)
        out[i] = jitc_var_new_stmt(JitBackend::CUDA, VarType::Float32,
                                   stmt_3[i], 1, 1, &lookup);

    jitc_var_dec_ref_ext(lookup);
}

void jitc_cuda_tex_destroy(void *texture) {
    if (!texture)
        return;
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);
    CUDA_RESOURCE_DESC res_desc;
    cuda_check(cuTexObjectGetResourceDesc(&res_desc, (CUtexObject) texture));
    cuda_check(cuTexObjectDestroy((CUtexObject) texture));
    cuda_check(cuArrayDestroy(res_desc.res.array.hArray));
}
