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

#include "metal_launch.h"

#include <mutex>
#include <algorithm>
#include <cstring>

// ============================================================================
//  Helpers
// ============================================================================

static MTLPixelFormat metal_tex_pixel_format(int format, size_t channels_storage,
                                             int srgb) {
    if ((VarType) format == VarType::UInt8) {
        if (srgb) {
            switch (channels_storage) {
                case 1: return MTLPixelFormatR8Unorm_sRGB;
                case 2: return MTLPixelFormatRG8Unorm_sRGB;
                default: return MTLPixelFormatRGBA8Unorm_sRGB;
            }
        }
        switch (channels_storage) {
            case 1: return MTLPixelFormatR8Unorm;
            case 2: return MTLPixelFormatRG8Unorm;
            default: return MTLPixelFormatRGBA8Unorm;
        }
    } else if ((VarType) format == VarType::Float32) {
        switch (channels_storage) {
            case 1: return MTLPixelFormatR32Float;
            case 2: return MTLPixelFormatRG32Float;
            default: return MTLPixelFormatRGBA32Float;
        }
    } else { // float16
        switch (channels_storage) {
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

/// Validate \c filter_mode / \c wrap_mode and map them to Metal sampler enums.
static void metal_tex_sampler_modes(int filter_mode, int wrap_mode,
                                    MTLSamplerMinMagFilter *filter,
                                    MTLSamplerAddressMode *wrap) {
    switch (filter_mode) {
        case 0: *filter = MTLSamplerMinMagFilterNearest; break;
        case 1: *filter = MTLSamplerMinMagFilterLinear; break;
        default: jitc_raise("jit_metal_tex(): invalid filter mode!");
    }
    switch (wrap_mode) {
        case 0: *wrap = MTLSamplerAddressModeRepeat; break;
        case 1: *wrap = MTLSamplerAddressModeClampToEdge; break;
        case 2: *wrap = MTLSamplerAddressModeMirrorRepeat; break;
        default: jitc_raise("jit_metal_tex(): invalid wrap mode!");
    }
}

/// Create the one shared sampler and register it as the texture's final
/// resource record (index ``n_textures``).
static void metal_tex_make_sampler(id<MTLDevice> device, MetalTexture *tex,
                                   MTLSamplerMinMagFilter filter,
                                   MTLSamplerAddressMode wrap) {
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
        jitc_raise("jit_metal_tex(): sampler allocation failed!");
    tex->sampler = (__bridge_retained void *) smp;

    tex->records[tex->n_textures].parent = tex;
    tex->records[tex->n_textures].object = tex->sampler;
    tex->sampler_index = jitc_var_mem_map(JitBackend::Metal, VarType::UInt64,
                                          &tex->records[tex->n_textures], 1, 0);
}

/// Install a callback on sub-texture ``i`` that releases the ``MTLTexture`` and
/// (once the last sub-texture is freed) the ``MetalTexture`` itself.
static void metal_tex_install_release(MetalTexture *tex, size_t i) {
    auto *payload = new MetalTexReleasePayload({ tex, i });
    jitc_var_set_callback(
        tex->indices[i],
        [](uint32_t, int free, void *cb) {
            if (!free)
                return;
            auto *p = (MetalTexReleasePayload *) cb;
            MetalTexture *t = p->texture;
            size_t idx = p->index;

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

// ============================================================================
//  Creation / destruction
// ============================================================================

void *jitc_metal_tex_create(size_t ndim, const size_t *shape, size_t n_channels,
                            int format, int filter_mode, int wrap_mode,
                            int writable, int srgb) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_metal_tex_create(): invalid texture dimension!");
    else if (n_channels == 0)
        jitc_raise("jit_metal_tex_create(): must have at least 1 channel!");
    for (size_t i = 0; i < ndim; ++i)
        if (shape[i] == 0)
            jitc_raise("jit_metal_tex_create(): texture dimensions must be nonzero!");

    ThreadState *ts = thread_state(JitBackend::Metal);
    id<MTLDevice> device = (__bridge id<MTLDevice>) ts->metal_device;

    size_t type_size;
    switch ((VarType) format) {
        case VarType::Float32: type_size = sizeof(float); break;
        case VarType::Float16: type_size = sizeof(uint16_t); break;
        case VarType::UInt8:   type_size = sizeof(uint8_t); break;
        default: jitc_raise("jit_metal_tex_create(): invalid data type!");
    }

    MTLSamplerMinMagFilter filter;
    MTLSamplerAddressMode wrap;
    metal_tex_sampler_modes(filter_mode, wrap_mode, &filter, &wrap);

    MetalTexture *tex = new MetalTexture(type_size, n_channels, writable != 0);
    tex->ndim = ndim;
    for (size_t i = 0; i < ndim; ++i)
        tex->shape[i] = shape[i];

    // Promote 1D to 2D textures as the former supports no filtering in MSL
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;

    @autoreleasepool {
        for (size_t i = 0; i < tex->n_textures; ++i) {
            size_t ci = tex->channels_storage(i);

            MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
            desc.textureType = (ndim == 3) ? MTLTextureType3D : MTLTextureType2D;
            desc.pixelFormat = metal_tex_pixel_format(format, ci, srgb);
            desc.width = width;
            desc.height = height;
            desc.depth = depth;
            desc.mipmapLevelCount = 1;
            desc.usage = MTLTextureUsageShaderRead |
                         (tex->writable ? MTLTextureUsageShaderWrite : 0);
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

            metal_tex_install_release(tex, i);
        }

        metal_tex_make_sampler(device, tex, filter, wrap);
    }

    jitc_log(LogLevel::Debug, "jitc_metal_tex_create(): " DRJIT_PTR,
             (uintptr_t) tex);
    return (void *) tex;
}

/// Infer (channels, component VarType, sRGB) from a wrappable ``MTLPixelFormat``.
static bool metal_tex_format_info(MTLPixelFormat pf, size_t *channels,
                                  int *format, int *srgb) {
    *srgb = 0;
    switch (pf) {
        case MTLPixelFormatR32Float:    *channels = 1; *format = (int) VarType::Float32; return true;
        case MTLPixelFormatRG32Float:   *channels = 2; *format = (int) VarType::Float32; return true;
        case MTLPixelFormatRGBA32Float: *channels = 4; *format = (int) VarType::Float32; return true;
        case MTLPixelFormatR16Float:    *channels = 1; *format = (int) VarType::Float16; return true;
        case MTLPixelFormatRG16Float:   *channels = 2; *format = (int) VarType::Float16; return true;
        case MTLPixelFormatRGBA16Float: *channels = 4; *format = (int) VarType::Float16; return true;
        case MTLPixelFormatR8Unorm:     *channels = 1; *format = (int) VarType::UInt8; return true;
        case MTLPixelFormatRG8Unorm:    *channels = 2; *format = (int) VarType::UInt8; return true;
        case MTLPixelFormatRGBA8Unorm:  *channels = 4; *format = (int) VarType::UInt8; return true;
        case MTLPixelFormatR8Unorm_sRGB:    *channels = 1; *format = (int) VarType::UInt8; *srgb = 1; return true;
        case MTLPixelFormatRG8Unorm_sRGB:   *channels = 2; *format = (int) VarType::UInt8; *srgb = 1; return true;
        case MTLPixelFormatRGBA8Unorm_sRGB: *channels = 4; *format = (int) VarType::UInt8; *srgb = 1; return true;
        default: return false;
    }
}

void *jitc_metal_tex_wrap(uintptr_t handle, size_t ndim, int format,
                          int writable, int filter_mode, int wrap_mode,
                          int srgb) {
    if (!handle)
        jitc_raise("jit_tex_wrap(): null texture handle!");

    id<MTLTexture> mtl_tex = (__bridge id<MTLTexture>) (void *) handle;

    size_t channels = 0;
    int tex_format = 0, tex_srgb = 0;
    if (!metal_tex_format_info(mtl_tex.pixelFormat, &channels, &tex_format,
                               &tex_srgb))
        jitc_raise("jit_tex_wrap(): unsupported pixel format; only R/RG/RGBA "
                   "8-bit unorm and 16-/32-bit float textures can be wrapped.");
    if (tex_format != format)
        jitc_raise("jit_tex_wrap(): texture component type does not match the "
                   "Dr.Jit texture's scalar type (use a Float16 texture type "
                   "for a half-precision texture, and vice versa).");
    (void) srgb; // sRGB decoding is implied by the wrapped pixel format

    size_t width = mtl_tex.width, height = 1, depth = 1, tex_ndim;
    switch (mtl_tex.textureType) {
        case MTLTextureType2D:
            tex_ndim = 2; height = mtl_tex.height; break;
        case MTLTextureType3D:
            tex_ndim = 3; height = mtl_tex.height; depth = mtl_tex.depth; break;
        default:
            jitc_raise("jit_tex_wrap(): only 2D and 3D textures can be wrapped.");
    }
    if (tex_ndim != ndim)
        jitc_raise("jit_tex_wrap(): texture dimensionality (%zuD) does not "
                   "match the Dr.Jit texture type (%zuD).", tex_ndim, ndim);

    if (writable && !(mtl_tex.usage & MTLTextureUsageShaderWrite))
        jitc_raise("jit_tex_wrap(): a writable wrap requires the texture to have "
                   "MTLTextureUsageShaderWrite.");

    MTLSamplerMinMagFilter filter;
    MTLSamplerAddressMode wrap;
    metal_tex_sampler_modes(filter_mode, wrap_mode, &filter, &wrap);

    // A wrappable pixel format has <= 4 channels, hence a single sub-texture.
    size_t type_size = (tex_format == (int) VarType::Float32) ? sizeof(float)
                                                              : sizeof(uint16_t);
    MetalTexture *tex = new MetalTexture(type_size, channels, writable != 0);
    tex->ndim = ndim;
    tex->shape[0] = width;
    tex->shape[1] = height;
    tex->shape[2] = depth;

    @autoreleasepool {
        id<MTLDevice> device =
            (__bridge id<MTLDevice>) thread_state(JitBackend::Metal)->metal_device;

        // Borrow the external texture, retaining it for the wrapper's lifetime.
        tex->textures[0] = (__bridge_retained void *) mtl_tex;
        tex->records[0].parent = tex;
        tex->records[0].object = tex->textures[0];
        tex->indices[0] = jitc_var_mem_map(JitBackend::Metal, VarType::UInt64,
                                           &tex->records[0], 1, 0);

        metal_tex_install_release(tex, 0);
        metal_tex_make_sampler(device, tex, filter, wrap);
    }

    jitc_log(LogLevel::Debug, "jitc_metal_tex_wrap(): " DRJIT_PTR,
             (uintptr_t) tex);
    return (void *) tex;
}

uintptr_t jitc_metal_tex_native_handle(const void *handle,
                                       size_t sub_index) {
    const MetalTexture &tex = *((const MetalTexture *) handle);
    if (sub_index >= tex.n_textures)
        jitc_raise("jit_tex_native_handle(): sub-texture index out of range.");
    return (uintptr_t) tex.textures[sub_index];
}

void jitc_metal_tex_get_shape(const void *handle, size_t *shape) {
    const MetalTexture &tex = *((const MetalTexture *) handle);
    for (size_t i = 0; i < tex.ndim; ++i)
        shape[i] = tex.shape[i];
    shape[tex.ndim] = tex.n_channels;
}

void jitc_metal_tex_get_indices(const void *handle, uint32_t *indices) {
    if (!handle)
        return;
    const MetalTexture &tex = *((const MetalTexture *) handle);
    for (size_t i = 0; i < tex.n_textures; ++i)
        indices[i] = tex.indices[i];
    indices[tex.n_textures] = tex.sampler_index;
}

void jitc_metal_tex_destroy(void *handle) {
    if (!handle)
        return;
    jitc_log(LogLevel::Debug, "jitc_metal_tex_destroy(" DRJIT_PTR ")",
             (uintptr_t) handle);

    MetalTexture *tex = (MetalTexture *) handle;

    const size_t n_textures = tex->n_textures;
    jitc_var_dec_ref(tex->sampler_index);
    for (size_t i = 0; i < n_textures; ++i)
        jitc_var_dec_ref(tex->indices[i]);
}

// ============================================================================
//  Device <-> texture memory transfer
// ============================================================================

static size_t metal_tex_n_texels(size_t ndim, const size_t *shape) {
    size_t n = shape[0];
    for (size_t i = 1; i < ndim; ++i)
        n *= shape[i];
    return n;
}

/// Host-side mirror of ``channel_pack_params`` in resources/metal_kernels.metal
struct ChannelPackParams {
    uint64_t src, dst;
    uint32_t n_channels, ci, tex_base, c_valid;
};

/// Launch a kernel to (de) interleave texture memory
static void metal_channel_pack(MetalThreadState *mts, MetalKernel kern,
                               const void *src, void *dst, uint32_t n_channels,
                               uint32_t ci, uint32_t tex_base, uint32_t c_valid,
                               uint32_t n_threads) {
    ChannelPackParams params;
    id<MTLBuffer> src_buf = metal_resolve(src, &params.src);
    id<MTLBuffer> dst_buf = metal_resolve(dst, &params.dst);
    if (!src_buf || !dst_buf)
        jitc_raise("metal_channel_pack(): buffer lookup failed.");
    params.n_channels = n_channels;
    params.ci = ci;
    params.tex_base = tex_base;
    params.c_valid = c_valid;

    metal_dispatch_threads(mts, metal_pipeline(mts, kern), params,
                           {{ src_buf, MTLResourceUsageRead },
                            { dst_buf, MTLResourceUsageWrite }},
                           n_threads);
}

void jitc_metal_tex_memcpy_d2t(const void *src_ptr, void *dst_handle) {
    MetalTexture &tex = *((MetalTexture *) dst_handle);
    ThreadState *ts = thread_state(JitBackend::Metal);
    MetalThreadState *mts = (MetalThreadState *) ts->actual_state();

    size_t ndim = tex.ndim;
    const size_t *shape = tex.shape;
    size_t n_texels = metal_tex_n_texels(ndim, shape);
    size_t type_size = tex.type_size;
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;

    bool needs_staging = (tex.n_textures > 1) || (tex.n_channels == 3);
    MetalKernel deint = (type_size == 1)   ? MetalKernel::DeinterleaveU8
                        : (type_size == 2) ? MetalKernel::DeinterleaveU16
                                           : MetalKernel::DeinterleaveU32;

    std::vector<void *> packed(tex.n_textures, nullptr);
    @autoreleasepool {
        // Phase 1: if required, deinterleave each sub-texture's channels into a
        // packed, zero-padded private staging buffer.
        if (needs_staging) {
            for (size_t i = 0; i < tex.n_textures; ++i) {
                size_t ci = tex.channels_storage(i);
                packed[i] = jitc_malloc(JitBackend::Metal,
                                        n_texels * ci * type_size);
                metal_channel_pack(mts, deint, src_ptr, packed[i],
                                   (uint32_t) tex.n_channels, (uint32_t) ci,
                                   (uint32_t) (i * 4), (uint32_t) tex.channels(i),
                                   (uint32_t) (n_texels * ci));
            }
        }

        // Phase 2: blit the buffer into the texture.
        id<MTLBlitCommandEncoder> blit =
            (__bridge id<MTLBlitCommandEncoder>) mts->ensure_blit_encoder();
        for (size_t i = 0; i < tex.n_textures; ++i) {
            size_t ci = tex.channels_storage(i);
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
        jitc_free(p);
}

void jitc_metal_tex_memcpy_t2d(const void *src_handle, void *dst_ptr) {
    MetalTexture &tex = *((MetalTexture *) src_handle);
    ThreadState *ts = thread_state(JitBackend::Metal);
    MetalThreadState *mts = (MetalThreadState *) ts->actual_state();

    size_t ndim = tex.ndim;
    const size_t *shape = tex.shape;
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
            size_t ci = tex.channels_storage(0);
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

    // Blit each sub-texture into a buffer, then interleave to destination
    MetalKernel inter = (type_size == 1)   ? MetalKernel::InterleaveU8
                        : (type_size == 2) ? MetalKernel::InterleaveU16
                                           : MetalKernel::InterleaveU32;
    std::vector<void *> packed(tex.n_textures, nullptr);
    @autoreleasepool {
        id<MTLBlitCommandEncoder> blit =
            (__bridge id<MTLBlitCommandEncoder>) mts->ensure_blit_encoder();
        for (size_t i = 0; i < tex.n_textures; ++i) {
            size_t ci = tex.channels_storage(i);
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

        for (size_t i = 0; i < tex.n_textures; ++i) {
            size_t ci = tex.channels_storage(i), c = tex.channels(i);
            metal_channel_pack(mts, inter, packed[i], dst_ptr,
                               (uint32_t) tex.n_channels, (uint32_t) ci,
                               (uint32_t) (i * 4), (uint32_t) c,
                               (uint32_t) (n_texels * c));
        }
    }

    for (void *p : packed)
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

/// Build a coordinate-payload ``TexData`` from the position indices.
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

void jitc_metal_tex_lookup(const void *handle,
                           const uint32_t *pos, uint32_t active, uint32_t *out) {
    MetalTexture &tex = *((MetalTexture *) handle);
    size_t ndim = tex.ndim;
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

void jitc_metal_tex_write(void *handle, const uint32_t *pos,
                          const uint32_t *value, uint32_t active) {
    MetalTexture &tex = *((MetalTexture *) handle);
    if (!tex.writable)
        jitc_raise("jit_tex_write(): texture was not created with the "
                   "'writable' flag.");

    size_t ndim = tex.ndim, n_channels = tex.n_channels;

    // Validate inputs and compute the broadcast launch size
    uint32_t size = 1;
    bool dirty = false, symbolic = false;
    auto check = [&](uint32_t idx, VarType expect, const char *what,
                        size_t which) {
        const Variable *v = jitc_var(idx);
        if (expect != VarType::Void && (VarType) v->type != expect)
            jitc_raise("jit_tex_write(): type mismatch for %s %zu (got %s, "
                       "expected %s).", what, which, type_name[v->type],
                       type_name[(int) expect]);
        if (v->size != 1) {
            if (size != 1 && v->size != size)
                jitc_raise("jit_tex_write(): incompatible argument sizes "
                           "(%u vs %u).", size, v->size);
            size = v->size;
        }
        dirty |= v->is_dirty();
        symbolic |= (bool) v->symbolic;
    };
    for (size_t i = 0; i < ndim; ++i)
        check(pos[i], VarType::UInt32, "coordinate", i);
    for (size_t c = 0; c < n_channels; ++c)
        check(value[c], VarType::Float32, "value", c);
    check(active, VarType::Void, "mask", 0);

    if (dirty)
        jitc_eval(thread_state(JitBackend::Metal));

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    // One write node per sub-texture
    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        size_t base = ti * 4, nc = tex.channels(ti);
        // Create a write handle to the same texture
        Ref tex_h = steal(jitc_var_resource_pointer(tex.indices[ti],
                                                    ResourceKind::Texture,
                                                    /*write=*/1));

        TexData *td = metal_tex_coord_data(ndim, pos);
        td->n_values = (uint32_t) nc;
        for (size_t c = 0; c < nc; ++c) {
            td->values[c] = value[base + c];
            jitc_var_inc_ref(value[base + c]);
        }

        uint32_t node =
            masked ? jitc_var_new_node_2(
                         JitBackend::Metal, VarKind::TexWrite, VarType::Void,
                         size, symbolic, tex_h, jitc_var(tex_h), active,
                         jitc_var(active), (uintptr_t) td)
                   : jitc_var_new_node_1(
                         JitBackend::Metal, VarKind::TexWrite, VarType::Void,
                         size, symbolic, tex_h, jitc_var(tex_h), (uintptr_t) td);

        metal_tex_set_free_callback(node, td);
        jitc_var_mark_side_effect(node);
    }
}

void jitc_metal_tex_bilerp_fetch(const void *handle,
                                 const uint32_t *pos, uint32_t active,
                                 uint32_t *out) {
    MetalTexture &tex = *((MetalTexture *) handle);
    if (tex.ndim != 2)
        jitc_raise("jitc_metal_tex_bilerp_fetch(): only 2D textures are "
                   "supported!");
    size_t ndim = tex.ndim;
    Variable tmpl = jitc_metal_tex_check(VarType::Float32, ndim, pos);

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
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
