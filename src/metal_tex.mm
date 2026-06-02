/*
    src/metal_tex.mm -- hardware texture support for the Metal backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_METAL)

#include "metal.h"
#include "metal_tex.h"
#include "metal_ts.h"
#include "internal.h"
#include "eval.h"
#include "malloc.h"
#include "log.h"
#include "var.h"
#include "op.h"
#include "tex.h"

// Suppress the obsolete Carbon <CarbonCore/Threads.h>, whose ThreadState collides with Dr.Jit
#define __THREADS__
#import <Metal/Metal.h>

#include <mutex>
#include <algorithm>
#include <cstring>

// ============================================================================
//  Opaque-resource handle resolution
// ============================================================================

bool jitc_metal_resource_id(void *owner, ResourceKind kind, void **value_out) {
    MTLResourceID rid;
    switch (kind) {
        case ResourceKind::Accel:
            rid = ((__bridge id<MTLAccelerationStructure>)
                       ((MetalScene *) owner)->tlas).gpuResourceID;
            break;

        case ResourceKind::Texture:
            rid = ((__bridge id<MTLTexture>)
                       ((MetalTexResource *) owner)->object).gpuResourceID;
            break;

        case ResourceKind::Sampler:
            rid = ((__bridge id<MTLSamplerState>)
                       ((MetalTexResource *) owner)->object).gpuResourceID;
            break;

        default:
            // A buffer is an ordinary pointer; an IFT is PSO-dependent and is
            // refreshed at launch. Neither resolves to a gpuResourceID here.
            return false;
    }

    uint64_t v64;
    std::memcpy(&v64, &rid, sizeof(v64));
    *value_out = (void *) (uintptr_t) v64;
    return true;
}


// ============================================================================
//  Helpers
// ============================================================================

static MTLPixelFormat metal_tex_pixel_format(int format, size_t channels_internal) {
    if ((VarType) format == VarType::Float32) {
        switch (channels_internal) {
            case 1: return MTLPixelFormatR32Float;
            case 2: return MTLPixelFormatRG32Float;
            default: return MTLPixelFormatRGBA32Float;
        }
    } else { // float16
        switch (channels_internal) {
            case 1: return MTLPixelFormatR16Float;
            case 2: return MTLPixelFormatRG16Float;
            default: return MTLPixelFormatRGBA16Float;
        }
    }
}

struct MetalTexReleasePayload {
    MetalTexture *texture;
    size_t index;
};

// ============================================================================
//  Creation / destruction
// ============================================================================

void *jitc_metal_tex_create(size_t ndim, const size_t *shape, size_t n_channels,
                            int format, int filter_mode, int wrap_mode) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_metal_tex_create(): invalid texture dimension!");
    else if (n_channels == 0)
        jitc_raise("jit_metal_tex_create(): must have at least 1 channel!");

    ThreadState *ts = thread_state(JitBackend::Metal);
    id<MTLDevice> device = (__bridge id<MTLDevice>) ts->metal_device;

    size_t type_size;
    switch ((VarType) format) {
        case VarType::Float32: type_size = sizeof(float); break;
        case VarType::Float16: type_size = sizeof(uint16_t); break;
        default: jitc_raise("jit_metal_tex_create(): invalid data type!");
    }

    MTLSamplerMinMagFilter filter;
    switch (filter_mode) {
        case 0: filter = MTLSamplerMinMagFilterNearest; break;
        case 1: filter = MTLSamplerMinMagFilterLinear; break;
        default: jitc_raise("jit_metal_tex_create(): invalid filter mode!");
    }

    MTLSamplerAddressMode wrap;
    switch (wrap_mode) {
        case 0: wrap = MTLSamplerAddressModeRepeat; break;
        case 1: wrap = MTLSamplerAddressModeClampToEdge; break;
        case 2: wrap = MTLSamplerAddressModeMirrorRepeat; break;
        default: jitc_raise("jit_metal_tex_create(): invalid wrap mode!");
    }

    MetalTexture *tex = new MetalTexture();
    tex->ndim = (uint32_t) ndim;
    tex->format = format;
    tex->type_size = type_size;
    tex->n_channels = n_channels;
    tex->n_textures = 1 + ((n_channels - 1) / 4);
    for (size_t i = 0; i < ndim; ++i)
        tex->shape[i] = shape[i];
    tex->n_referenced_textures = tex->n_textures;
    tex->records = std::make_unique<MetalTexResource[]>(tex->n_textures + 1);
    tex->textures.resize(tex->n_textures, nullptr);
    tex->indices.resize(tex->n_textures, 0);

    // A 1D texture is backed by a height-1 2D texture so that hardware linear
    // filtering works (``texture1d`` has no ``sample()`` in MSL).
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;

    @autoreleasepool {
        for (size_t i = 0; i < tex->n_textures; ++i) {
            size_t ci = tex->channels_internal(i);

            MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
            desc.textureType = (ndim == 3) ? MTLTextureType3D : MTLTextureType2D;
            desc.pixelFormat = metal_tex_pixel_format(format, ci);
            desc.width = width;
            desc.height = height;
            desc.depth = depth;
            desc.mipmapLevelCount = 1;
            desc.usage = MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModePrivate;

            id<MTLTexture> mtl_tex = [device newTextureWithDescriptor:desc];
            if (!mtl_tex)
                jitc_raise("jit_metal_tex_create(): texture allocation failed!");
            tex->textures[i] = (__bridge_retained void *) mtl_tex;

            tex->records[i].parent = tex;
            tex->records[i].object = tex->textures[i];

            tex->indices[i] = jitc_var_mem_map(JitBackend::Metal,
                                               VarType::UInt64,
                                               &tex->records[i], 1, 0);

            auto *payload = new MetalTexReleasePayload({ tex, i });
            jitc_var_set_callback(
                tex->indices[i],
                [](uint32_t, int free, void *cb) {
                    if (!free)
                        return;
                    auto *p = (MetalTexReleasePayload *) cb;
                    MetalTexture *t = p->texture;
                    size_t idx = p->index;

                    // Release this sub-texture's MTLTexture.
                    if (t->textures[idx])
                        (void) (__bridge_transfer id<MTLTexture>) t->textures[idx];
                    t->textures[idx] = nullptr;

                    if (--t->n_referenced_textures == 0) {
                        if (t->sampler)
                            (void) (__bridge_transfer id<MTLSamplerState>) t->sampler;
                        t->sampler = nullptr;
                        delete t;
                    }
                    delete p;
                },
                (void *) payload, false);
        }

        // One shared sampler for all sub-textures (filter / wrap live only here,
        // never baked into MSL — otherwise two textures differing only in mode
        // would fragment the kernel cache).
        MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = filter;
        sd.magFilter = filter;
        sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
        sd.sAddressMode = wrap;
        sd.tAddressMode = wrap;
        sd.rAddressMode = wrap;
        sd.normalizedCoordinates = YES;
        sd.supportArgumentBuffers = YES;
        id<MTLSamplerState> smp = [device newSamplerStateWithDescriptor:sd];
        if (!smp)
            jitc_raise("jit_metal_tex_create(): sampler allocation failed!");
        tex->sampler = (__bridge_retained void *) smp;

        tex->records[tex->n_textures].parent = tex;
        tex->records[tex->n_textures].object = tex->sampler;

        tex->sampler_index = jitc_var_mem_map(JitBackend::Metal,
                                              VarType::UInt64,
                                              &tex->records[tex->n_textures], 1, 0);
    }

    jitc_log(LogLevel::Debug, "jitc_metal_tex_create(): " DRJIT_PTR,
             (uintptr_t) tex);
    return (void *) tex;
}

void jitc_metal_tex_get_shape(size_t ndim, const void *texture_handle,
                              size_t *shape) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_metal_tex_get_shape(): invalid texture dimension!");
    const MetalTexture &tex = *((const MetalTexture *) texture_handle);
    for (size_t i = 0; i < ndim; ++i)
        shape[i] = tex.shape[i];
    shape[ndim] = tex.n_channels;
}

void jitc_metal_tex_get_indices(const void *texture_handle, uint32_t *indices) {
    if (!texture_handle)
        return;
    const MetalTexture &tex = *((const MetalTexture *) texture_handle);
    for (size_t i = 0; i < tex.n_textures; ++i)
        indices[i] = tex.indices[i];
    // The sampler is a separate input variable on Metal (CUDA bakes it into the
    // texture object); report it last so frozen-function recording captures it.
    indices[tex.n_textures] = tex.sampler_index;
}

void jitc_metal_tex_destroy(void *texture_handle) {
    if (!texture_handle)
        return;
    jitc_log(LogLevel::Debug, "jitc_metal_tex_destroy(" DRJIT_PTR ")",
             (uintptr_t) texture_handle);

    MetalTexture *tex = (MetalTexture *) texture_handle;

    // Release our creation references. The MetalTexture itself is deleted by
    // the per-sub-texture callback once the last sub-texture's reference count
    // reaches zero
    const size_t n_textures = tex->n_textures;
    jitc_var_dec_ref(tex->sampler_index);
    for (size_t i = 0; i < n_textures; ++i)
        jitc_var_dec_ref(tex->indices[i]);
}

// ============================================================================
//  Device <-> texture memory transfer
// ============================================================================
//
// The interleaved tensor layout (``n_channels`` components per texel) is split
// into the per-sub-texture pixel-format layout (1/2/4 components, 3 padded to
// 4). For the common single-sub-texture case with 1/2/4 channels no staging is
// needed and the copy is a direct GPU blit. The 3-channel and multi-sub-texture
// cases (de)interleave on the GPU via the ``deinterleave``/``interleave``
// utility kernels into private staging buffers — no host round-trip or sync.

static size_t metal_tex_n_texels(size_t ndim, const size_t *shape) {
    size_t n = shape[0];
    for (size_t i = 1; i < ndim; ++i)
        n *= shape[i];
    return n;
}

/// Dispatch a channel (de)interleave utility kernel (``kern`` selects the
/// direction and element width). ``src``/``dst`` are device pointers; the
/// per-encoder hazard tracking orders this against the surrounding blits.
static void metal_channel_pack(MetalThreadState *mts, MetalKernel kern,
                               const void *src, void *dst, uint32_t n_channels,
                               uint32_t ci, uint32_t tex_base, uint32_t c_valid,
                               uint32_t n_threads) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)
        state.metal_devices[mts->device].pipelines[(uint32_t) kern];
    if (!pso)
        jitc_raise("metal_channel_pack(): utility pipeline missing.");

    size_t src_off = 0, dst_off = 0;
    id<MTLBuffer> src_buf =
        (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) src, &src_off);
    id<MTLBuffer> dst_buf =
        (__bridge id<MTLBuffer>) jitc_metal_find_buffer(dst, &dst_off);
    if (!src_buf || !dst_buf)
        jitc_raise("metal_channel_pack(): buffer lookup failed.");

    struct {
        uint64_t src, dst;
        uint32_t n_channels, ci, tex_base, c_valid;
    } params;
    params.src = (uint64_t) [src_buf gpuAddress] + src_off;
    params.dst = (uint64_t) [dst_buf gpuAddress] + dst_off;
    params.n_channels = n_channels;
    params.ci = ci;
    params.tex_base = tex_base;
    params.c_valid = c_valid;

    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>) mts->ensure_compute_encoder();
    [enc setComputePipelineState:pso];
    [enc setBytes:&params length:sizeof(params) atIndex:0];
    [enc useResource:src_buf usage:MTLResourceUsageRead];
    [enc useResource:dst_buf usage:MTLResourceUsageWrite];
    uint32_t tg = std::min<uint32_t>(
        (uint32_t) pso.maxTotalThreadsPerThreadgroup, n_threads);
    if (tg == 0)
        tg = 1;
    [enc dispatchThreads:MTLSizeMake(n_threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void jitc_metal_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                               const void *src_ptr, void *dst_texture_handle) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_metal_tex_memcpy(): invalid texture dimension!");

    MetalTexture &tex = *((MetalTexture *) dst_texture_handle);
    ThreadState *ts = thread_state(JitBackend::Metal);
    MetalThreadState *mts = (MetalThreadState *) ts->actual_state();

    size_t n_texels = metal_tex_n_texels(ndim, shape);
    size_t type_size = tex.type_size;
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;

    bool needs_staging = (tex.n_textures > 1) || (tex.n_channels == 3);
    MetalKernel deint = (type_size == 2) ? MetalKernel::DeinterleaveU16
                                         : MetalKernel::DeinterleaveU32;

    std::vector<void *> packed(tex.n_textures, nullptr);
    @autoreleasepool {
        // Phase 1: deinterleave each sub-texture's channels into a packed,
        // zero-padded private staging buffer (compute). Skipped when the source
        // already matches the single sub-texture's pixel layout.
        if (needs_staging) {
            for (size_t i = 0; i < tex.n_textures; ++i) {
                size_t ci = tex.channels_internal(i);
                packed[i] = jitc_malloc(JitBackend::Metal,
                                        n_texels * ci * type_size);
                metal_channel_pack(mts, deint, src_ptr, packed[i],
                                   (uint32_t) tex.n_channels, (uint32_t) ci,
                                   (uint32_t) (i * 4), (uint32_t) tex.channels(i),
                                   (uint32_t) (n_texels * ci));
            }
        }

        // Phase 2: blit the (packed or already-matching) buffer into the texture.
        id<MTLBlitCommandEncoder> blit =
            (__bridge id<MTLBlitCommandEncoder>) mts->ensure_blit_encoder();
        for (size_t i = 0; i < tex.n_textures; ++i) {
            size_t ci = tex.channels_internal(i);
            size_t bytes_per_row = width * ci * type_size;
            size_t src_off = 0;
            id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(needs_staging ? packed[i]
                                                     : (void *) src_ptr,
                                       &src_off);
            [blit copyFromBuffer:src_buf
                    sourceOffset:src_off
               sourceBytesPerRow:bytes_per_row
             sourceBytesPerImage:bytes_per_row * height
                      sourceSize:MTLSizeMake(width, height, depth)
                       toTexture:(__bridge id<MTLTexture>) tex.textures[i]
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
        }
    }

    for (void *p : packed)
        if (p)
            jitc_free(p); // freed once the GPU work completes
}

void jitc_metal_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                               const void *src_texture_handle, void *dst_ptr) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_metal_tex_memcpy(): invalid texture dimension!");

    MetalTexture &tex = *((MetalTexture *) src_texture_handle);
    ThreadState *ts = thread_state(JitBackend::Metal);
    MetalThreadState *mts = (MetalThreadState *) ts->actual_state();

    size_t n_texels = metal_tex_n_texels(ndim, shape);
    size_t type_size = tex.type_size;
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;

    bool needs_staging = (tex.n_textures > 1) || (tex.n_channels == 3);

    if (!needs_staging) {
        @autoreleasepool {
            id<MTLBlitCommandEncoder> blit =
                (__bridge id<MTLBlitCommandEncoder>) mts->ensure_blit_encoder();
            size_t ci = tex.channels_internal(0);
            size_t off;
            id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(dst_ptr, &off);
            [blit copyFromTexture:(__bridge id<MTLTexture>) tex.textures[0]
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(width, height, depth)
                         toBuffer:dst_buf
                destinationOffset:off
           destinationBytesPerRow:width * ci * type_size
         destinationBytesPerImage:width * height * ci * type_size];
        }
        return;
    }

    // Staged path: blit each sub-texture into a packed private buffer, then
    // re-interleave it into the destination on the GPU (no host round-trip).
    MetalKernel inter = (type_size == 2) ? MetalKernel::InterleaveU16
                                         : MetalKernel::InterleaveU32;
    std::vector<void *> packed(tex.n_textures, nullptr);
    @autoreleasepool {
        // Phase 1: texture -> packed (blit).
        id<MTLBlitCommandEncoder> blit =
            (__bridge id<MTLBlitCommandEncoder>) mts->ensure_blit_encoder();
        for (size_t i = 0; i < tex.n_textures; ++i) {
            size_t ci = tex.channels_internal(i);
            packed[i] = jitc_malloc(JitBackend::Metal,
                                    n_texels * ci * type_size);
            size_t off = 0;
            id<MTLBuffer> packed_buf = (__bridge id<MTLBuffer>)
                jitc_metal_find_buffer(packed[i], &off);
            [blit copyFromTexture:(__bridge id<MTLTexture>) tex.textures[i]
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(width, height, depth)
                         toBuffer:packed_buf
                destinationOffset:off
           destinationBytesPerRow:width * ci * type_size
         destinationBytesPerImage:width * height * ci * type_size];
        }

        // Phase 2: packed -> interleaved destination (compute), one thread per
        // real channel (padding is not written back).
        for (size_t i = 0; i < tex.n_textures; ++i) {
            size_t ci = tex.channels_internal(i), c = tex.channels(i);
            metal_channel_pack(mts, inter, packed[i], dst_ptr,
                               (uint32_t) tex.n_channels, (uint32_t) ci,
                               (uint32_t) (i * 4), (uint32_t) c,
                               (uint32_t) (n_texels * c));
        }
    }

    for (void *p : packed)
        if (p)
            jitc_free(p);
}

// ============================================================================
//  Lookup / bilerp graph builders
// ============================================================================

static Variable jitc_metal_tex_check(VarType out_type, size_t ndim,
                                     const uint32_t *pos) {
    uint32_t size = 0;
    bool dirty = false, symbolic = false;

    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_metal_tex_check(): invalid texture dimension!");

    for (size_t i = 0; i < ndim; ++i) {
        const Variable *v = jitc_var(pos[i]);
        if ((VarType) v->type != VarType::Float32)
            jitc_raise("jit_metal_tex_check(): type mismatch for arg. %zu (got "
                       "%s, expected %s)", i, type_name[v->type],
                       type_name[(int) VarType::Float32]);
        size = std::max(size, v->size);
        dirty |= v->is_dirty();
        symbolic |= (bool) v->symbolic;
    }

    for (size_t i = 0; i < ndim; ++i) {
        const Variable *v = jitc_var(pos[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_metal_tex_check(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::Metal));
        for (size_t i = 0; i < ndim; ++i) {
            if (jitc_var(pos[i])->is_dirty())
                jitc_raise_dirty_error(pos[i]);
        }
    }

    Variable v;
    v.size = size;
    v.backend = (uint32_t) JitBackend::Metal;
    v.symbolic = symbolic;
    v.type = (uint32_t) out_type;
    return v;
}

/// Build a coordinate-payload ``TexData`` from the position indices (taking a
/// new reference to each), to be freed by the node's destruction callback.
static TexData *metal_tex_coord_data(size_t ndim, const uint32_t *pos) {
    TexData *td = new TexData();
    td->ndim = (uint32_t) ndim;
    for (size_t i = 0; i < ndim; ++i) {
        td->indices[i] = pos[i];
        jitc_var_inc_ref(pos[i]);
    }
    return td;
}

static void metal_tex_set_free_callback(uint32_t index, TexData *td) {
    jitc_var_set_callback(
        index,
        [](uint32_t, int free, void *ptr) {
            if (free)
                delete (TexData *) ptr;
        },
        td, true);
}

/// Build a texture sample node (``TexLookup`` / ``TexFetchBilerp``)
static uint32_t metal_tex_node(VarKind kind, const Variable &tmpl,
                               uint32_t tex_h, uint32_t smp_h,
                               uint32_t active, bool masked, TexData *td) {
    uint32_t node = masked
        ? jitc_var_new_node_3(JitBackend::Metal, kind, VarType::Float32,
              tmpl.size, tmpl.symbolic, tex_h, jitc_var(tex_h), smp_h,
              jitc_var(smp_h), active, jitc_var(active), (uintptr_t) td)
        : jitc_var_new_node_2(JitBackend::Metal, kind, VarType::Float32,
              tmpl.size, tmpl.symbolic, tex_h, jitc_var(tex_h), smp_h,
              jitc_var(smp_h), (uintptr_t) td);
    metal_tex_set_free_callback(node, td);
    return node;
}

void jitc_metal_tex_lookup(size_t ndim, const void *texture_handle,
                           const uint32_t *pos, uint32_t active, uint32_t *out) {
    MetalTexture &tex = *((MetalTexture *) texture_handle);
    Variable tmpl = jitc_metal_tex_check(VarType::Float32, ndim, pos);

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        Ref tex_h = steal(jitc_var_resource_pointer(tex.indices[ti],
                                                    ResourceKind::Texture));
        Ref smp_h = steal(jitc_var_resource_pointer(tex.sampler_index,
                                                    ResourceKind::Sampler));
        TexData *td = metal_tex_coord_data(ndim, pos);

        Ref tex_load = steal(metal_tex_node(VarKind::TexLookup, tmpl, tex_h,
                                            smp_h, active, masked, td));

        for (size_t ch = 0; ch < tex.channels(ti); ++ch)
            *out++ = jitc_var_new_node_1(
                JitBackend::Metal, VarKind::Extract, VarType::Float32,
                tmpl.size, tmpl.symbolic, tex_load, jitc_var(tex_load),
                (uint64_t) ch);
    }
}

void jitc_metal_tex_bilerp_fetch(size_t ndim, const void *texture_handle,
                                 const uint32_t *pos, uint32_t active,
                                 uint32_t *out) {
    if (ndim != 2)
        jitc_raise("jitc_metal_tex_bilerp_fetch(): only 2D textures are "
                   "supported!");

    MetalTexture &tex = *((MetalTexture *) texture_handle);
    Variable tmpl = jitc_metal_tex_check(VarType::Float32, ndim, pos);

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        // The texture / sampler handles depend only on ``ti``; share them across
        // channels (only ``td``/the fetch node below are component-specific).
        Ref tex_h = steal(jitc_var_resource_pointer(tex.indices[ti],
                                                    ResourceKind::Texture));
        Ref smp_h = steal(jitc_var_resource_pointer(tex.sampler_index,
                                                    ResourceKind::Sampler));

        for (size_t ch = 0; ch < tex.channels(ti); ++ch) {
            TexData *td = metal_tex_coord_data(ndim, pos);
            td->component = (uint32_t) ch;

            Ref tex_load = steal(metal_tex_node(VarKind::TexFetchBilerp, tmpl,
                                                tex_h, smp_h, active, masked, td));

            for (uint32_t j = 0; j < 4; ++j)
                *out++ = jitc_var_new_node_1(
                    JitBackend::Metal, VarKind::Extract, VarType::Float32,
                    tmpl.size, tmpl.symbolic, tex_load, jitc_var(tex_load),
                    (uint64_t) j);
        }
    }
}

#endif // defined(DRJIT_ENABLE_METAL)
