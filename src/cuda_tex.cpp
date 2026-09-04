#include "cuda_tex.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "op.h"
#include "eval.h"
#include "tex.h"
#include <string.h>
#include <memory>

using StagingAreaDeleter = void (*)(void *);

/// Host-side state for one CUDA hardware texture
struct CUDATexture : TextureBase {
    std::unique_ptr<CUtexObject[]> textures; /// Array of CUDA texture objects
    std::unique_ptr<uint32_t[]> indices; /// Array of indices of texture object pointers for the JIT
    std::unique_ptr<CUarray[]> arrays; /// Array of CUDA arrays (the base level when MIP-mapped)
    std::unique_ptr<CUmipmappedArray[]> mip_arrays; /// MIP pyramids (n_levels > 1 only)
    std::unique_ptr<CUsurfObject[]> surfaces; /// Surface objects (writable only)
    std::unique_ptr<uint32_t[]> surf_indices; /// JIT handles of surface objects

    // -- State for a texture wrapping an external OpenGL texture  --
    bool external = false; /// Wraps an OpenGL texture via interop?
    bool mapped = false; /// Currently mapped for use by Dr.Jit?
    unsigned int gl_id = 0; /// The wrapped OpenGL texture id
    void *graphics_resource = nullptr; /// Registered CUgraphicsResource
    int filter_mode = 1, wrap_mode = 0; /// Sampler settings (texobject rebuild)
    int storage_format = 0; /// VarType of the storage (texobject rebuild)
    int srgb = 0; /// Decode sRGB -> linear on sampling (UInt8 textures)

    /// Allocates the per-sub-texture arrays; \ref TextureBase computes
    /// \c n_textures from the channel count.
    CUDATexture(size_t type_size, size_t n_channels, bool writable)
        : TextureBase(JitBackend::CUDA, type_size, n_channels, writable),
          textures(std::make_unique<CUtexObject[]>(n_textures)),
          indices(std::make_unique<uint32_t[]>(n_textures)),
          arrays(std::make_unique<CUarray[]>(n_textures)),
          surfaces(writable ? std::make_unique<CUsurfObject[]>(n_textures) : nullptr),
          surf_indices(writable ? std::make_unique<uint32_t[]>(n_textures) : nullptr) {}

    /**
     * \brief Releases the texture object at the given index in \ref textures
     * and returns whether or not there is at least one texture that is still
     * not released.
     */
    bool release_texture(size_t index) {
        if (state.backends & (1u << (uint32_t) JitBackend::CUDA)) {
            // Only run the following code if the CUDA context is still alive
            ThreadState *ts = thread_state(JitBackend::CUDA);
            scoped_set_context guard(ts->context);
            if (writable)
                cuda_check(cuSurfObjectDestroy(surfaces[index]));
            // A MIP pyramid owns its level arrays, including the base
            if (mip_arrays)
                cuda_check(cuMipmappedArrayDestroy(mip_arrays[index]));
            else
                cuda_check(cuArrayDestroy(arrays[index]));
            cuda_check(cuTexObjectDestroy(textures[index]));
        }

        return --n_referenced_textures > 0;
    }

    /**
     * \brief Returns the ID of a JIT variable, which encodes the address of the
     * i-th texture object.
     *
     * This function returns a JIT variable ID of a literal representing the
     * texture address. This feature is internally used to make textures
     * compatible with frozen function recording. This function returns an
     * owning reference, which the caller must release eventually.
     */
    uint32_t get_jit_pointer(uint32_t i) {
        return jitc_var_pointer(JitBackend::CUDA, jitc_var(indices[i])->data,
                                indices[i], false);
    }

    /// Like \ref get_jit_pointer, but for the i-th surface object (writable
    /// textures only). Used by ``TexWrite`` to reference the ``sust`` target.
    uint32_t get_surf_jit_pointer(uint32_t i) {
        return jitc_var_pointer(JitBackend::CUDA,
                                jitc_var(surf_indices[i])->data,
                                surf_indices[i], false);
    }
};

struct TextureReleasePayload {
    CUDATexture* texture;
    size_t index;
};

/// Build a sampling ``CUtexObject`` over \c array (or over the MIP pyramid
/// \c mip when non-null).
static CUtexObject cuda_tex_make_texobject(CUarray array, int format,
                                           size_t channels, int filter_mode,
                                           int wrap_mode, int srgb, size_t width,
                                           size_t height, size_t depth,
                                           CUmipmappedArray mip = nullptr,
                                           size_t n_levels = 1,
                                           int mip_filter = 1,
                                           size_t max_aniso = 1);

void *jitc_cuda_tex_create(size_t ndim, const size_t *shape, size_t n_channels,
                           int format, int filter_mode, int wrap_mode,
                           int writable, int srgb, size_t n_levels,
                           int mip_filter, size_t max_aniso) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_create(): invalid texture dimension!");
    else if (n_channels == 0)
        jitc_raise("jit_cuda_tex_create(): must have at least 1 channel!");
    for (size_t i = 0; i < ndim; ++i)
        if (shape[i] == 0)
            jitc_raise("jit_cuda_tex_create(): texture dimensions must be nonzero!");
    if (n_levels == 0)
        n_levels = 1;
    if (n_levels > 1 && writable)
        jitc_raise("jit_cuda_tex_create(): MIP-mapped textures cannot be "
                   "writable!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    if (filter_mode != 0 && filter_mode != 1)
        jitc_raise("jit_cuda_tex_create(): invalid filter mode!");
    if (wrap_mode < 0 || wrap_mode > 2)
        jitc_raise("jit_cuda_tex_create(): invalid wrap mode!");

    size_t array_format = 0, tsize = 0;
    switch ((VarType) format) {
        case VarType::Float32:
            array_format = CU_AD_FORMAT_FLOAT;
            tsize = sizeof(float);
            break;
        case VarType::Float16:
            array_format = CU_AD_FORMAT_HALF;
            tsize = sizeof(uint16_t);
            break;
        case VarType::UInt8:
            array_format = CU_AD_FORMAT_UNSIGNED_INT8;
            tsize = sizeof(uint8_t);
            break;
        default:
            jitc_raise("jit_cuda_tex_create(): invalid data type!");
            break;
    }

    CUDATexture *texture =
        new CUDATexture(tsize, n_channels, writable != 0);
    texture->ndim = ndim;
    texture->srgb = srgb;
    texture->n_levels = n_levels;
    for (size_t i = 0; i < ndim; ++i)
        texture->shape[i] = shape[i];
    if (n_levels > 1)
        texture->mip_arrays =
            std::make_unique<CUmipmappedArray[]>(texture->n_textures);
    for (size_t tex = 0; tex < texture->n_textures; ++tex) {
        const size_t tex_channels = texture->channels_storage(tex);

        CUarray array = nullptr;
        CUmipmappedArray mip = nullptr;
        if (n_levels > 1) {
            CUDA_ARRAY3D_DESCRIPTOR array_desc{};
            array_desc.Width = shape[0];
            array_desc.Height = (ndim >= 2) ? shape[1] : 0;
            array_desc.Depth = (ndim == 3) ? shape[2] : 0;
            array_desc.Format = (CUarray_format) array_format;
            array_desc.NumChannels = (unsigned int) tex_channels;
            cuda_check(cuMipmappedArrayCreate(&mip, &array_desc, (unsigned int) n_levels));
            texture->mip_arrays[tex] = mip;
            cuda_check(cuMipmappedArrayGetLevel(&array, mip, 0));
        } else if ((ndim == 1 || ndim == 2) && !writable) {
            CUDA_ARRAY_DESCRIPTOR array_desc{};
            array_desc.Width = shape[0];
            array_desc.Height = (ndim == 2) ? shape[1] : 0;
            array_desc.Format = (CUarray_format) array_format;
            array_desc.NumChannels = (unsigned int) tex_channels;
            cuda_check(cuArrayCreate(&array, &array_desc));
        } else {
            // A writable texture needs SURFACE_LDST, which is only expressible
            // via the 3D descriptor (also used for any genuine 3D texture).
            CUDA_ARRAY3D_DESCRIPTOR array_desc{};
            array_desc.Width = shape[0];
            array_desc.Height = (ndim >= 2) ? shape[1] : 0;
            array_desc.Depth = (ndim == 3) ? shape[2] : 0;
            array_desc.Format = (CUarray_format) array_format;
            array_desc.NumChannels = (unsigned int) tex_channels;
            array_desc.Flags = writable ? CUDA_ARRAY3D_SURFACE_LDST : 0;
            cuda_check(cuArray3DCreate(&array, &array_desc));
        }

        texture->arrays[tex] = array;
        // The resource view dimensions must match the array (0 = absent);
        // a mismatch is rejected for mipmapped resources.
        texture->textures[tex] = cuda_tex_make_texobject(
            array, format, tex_channels, filter_mode, wrap_mode, srgb, shape[0],
            (ndim >= 2) ? shape[1] : 0, (ndim == 3) ? shape[2] : 0,
            mip, n_levels, mip_filter, max_aniso);
        texture->indices[tex] =
            jitc_var_mem_map(JitBackend::CUDA, VarType::UInt64,
                             (void *) texture->textures[tex], 1, false);

        // For a writable texture, also build a surface object over the same
        // array and expose its handle for ``TexWrite``.
        if (writable) {
            CUDA_RESOURCE_DESC res_desc{};
            res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
            res_desc.res.array.hArray = array;
            cuda_check(cuSurfObjectCreate(&(texture->surfaces[tex]), &res_desc));
            texture->surf_indices[tex] =
                jitc_var_mem_map(JitBackend::CUDA, VarType::UInt64,
                                 (void *) texture->surfaces[tex], 1, false);
        }

        TextureReleasePayload *payload_ptr =
            new TextureReleasePayload({ texture, tex });

        jitc_var_set_callback(
            texture->indices[tex],
            [](uint32_t /* index */, int free, void *callback_data) {
                if (free) {
                    TextureReleasePayload& payload =
                        *((TextureReleasePayload *) callback_data);

                    CUDATexture *texture = payload.texture;
                    size_t tex = payload.index;

                    if (!texture->release_texture(tex))
                        delete texture;

                    delete &payload;
                }
            },
            (void *) payload_ptr,
            false
        );
    }

    texture->device_bytes = texture->storage_bytes();
    jitc_tex_track_usage(JitBackend::CUDA, (int64_t) texture->device_bytes, 1);

    jitc_log(LogLevel::Debug, "jitc_cuda_tex_create(): " DRJIT_PTR,
             (uintptr_t) texture);

    return (void *) texture;
}

void jitc_cuda_tex_get_shape(const void *handle, size_t *shape) {
    CUDATexture &texture = *((CUDATexture *) handle);
    for (size_t i = 0; i < texture.ndim; ++i)
        shape[i] = texture.shape[i];
    shape[texture.ndim] = texture.n_channels;
}

void jitc_cuda_tex_get_indices(const void *handle, uint32_t *indices) {
    if (!handle)
        return;

    CUDATexture &texture = *((CUDATexture *) handle);

    for (uint32_t i = 0; i < texture.n_textures; i++)
        indices[i] = texture.indices[i];

    // A writable texture also stores through per-sub-texture surface handles;
    // expose them so frozen-function recording captures and rebinds them.
    if (texture.writable)
        for (uint32_t i = 0; i < texture.n_textures; i++)
            indices[texture.n_textures + i] = texture.surf_indices[i];
}

static std::unique_ptr<void, StagingAreaDeleter>
jitc_cuda_tex_alloc_staging_area(size_t n_texels,
                                 const CUDATexture &texture) {
    // Each texture except for the last one will need exactly 4 channels
    size_t staging_area_size =
        texture.type_size * n_texels *
        ((texture.n_textures - 1) * 4 +
         texture.channels_storage(texture.n_textures - 1));
    void *staging_area = jitc_malloc(JitBackend::CUDA, staging_area_size);

    return std::unique_ptr<void, StagingAreaDeleter>(staging_area, jitc_free);
}

/*
 * \brief (De)interleave channels between a packed device buffer and the staging
 * area, whose layout is partitioned so each CUDA sub-texture's data is
 * contiguous. \c to_staging selects the direction.
 */
static void jitc_cuda_tex_stage(
    bool to_staging, ThreadState *ts,
    const std::unique_ptr<void, StagingAreaDeleter> &staging,
    size_t n_texels, const void *linear, const CUDATexture &tex) {
    scoped_set_context guard(ts->context);

    size_t texel_size = tex.n_channels * tex.type_size;
    CUdeviceptr packed = (CUdeviceptr) linear,
                staged = (CUdeviceptr) staging.get();

    for (size_t i = 0; i < tex.n_textures; ++i) {
        size_t texel_offset = i * 4 * tex.type_size;
        size_t staged_off = n_texels * texel_offset;
        size_t staged_pitch = tex.channels_storage(i) * tex.type_size;

        CUDA_MEMCPY2D op{};
        op.srcMemoryType = op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        if (to_staging) {
            op.srcDevice = packed; op.srcXInBytes = texel_offset; op.srcPitch = texel_size;
            op.dstDevice = staged; op.dstXInBytes = staged_off;   op.dstPitch = staged_pitch;
        } else {
            op.srcDevice = staged; op.srcXInBytes = staged_off;   op.srcPitch = staged_pitch;
            op.dstDevice = packed; op.dstXInBytes = texel_offset; op.dstPitch = texel_size;
        }
        op.WidthInBytes = tex.channels(i) * tex.type_size;
        op.Height = n_texels;

        cuda_check(cuMemcpy2DAsync(&op, ts->stream));
    }
}

/*
 * \brief Copy between a packed linear device buffer and a texture's array(s).
 * \c to_texture selects the direction (device->texture vs texture->device).
 * Multi-sub-texture / 3-channel layouts route through a channel-partitioned
 * staging area (deinterleaved before an upload, interleaved after a readback).
 */
static void jitc_cuda_tex_memcpy(bool to_texture, const CUDATexture &tex,
                                 const void *linear, size_t level) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    if (tex.external && !tex.mapped)
        jitc_raise("jit_tex_memcpy(): map() the OpenGL-wrapped texture first.");
    if (level >= tex.n_levels)
        jitc_raise("jit_tex_memcpy(): MIP level %zu out of range!", level);

    size_t ndim = tex.ndim;
    size_t shape[3];
    tex.level_shape(level, shape);
    size_t n_texels = shape[0];
    for (size_t dim = 1; dim < ndim; ++dim)
        n_texels *= shape[dim];

    StagingAreaDeleter noop = [](void *) {};
    std::unique_ptr<void, StagingAreaDeleter> staging(nullptr, noop);
    bool needs_staging = (tex.n_textures > 1) || (tex.n_channels == 3);
    if (needs_staging) {
        staging = jitc_cuda_tex_alloc_staging_area(n_texels, tex);
        if (to_texture)
            jitc_cuda_tex_stage(true, ts, staging, n_texels, linear, tex);
    }

    // Texture array(s) on one side; the packed buffer (or staging area) on the
    // other.
    CUdeviceptr buf = (CUdeviceptr) (needs_staging ? staging.get() : linear);

    for (size_t i = 0; i < tex.n_textures; ++i) {
        size_t pitch = shape[0] * tex.channels_storage(i) * tex.type_size;
        size_t buf_off = needs_staging ? i * n_texels * 4 * tex.type_size : 0;

        CUarray array = tex.arrays[i];
        if (level > 0)
            cuda_check(cuMipmappedArrayGetLevel(&array, tex.mip_arrays[i],
                                                (unsigned int) level));

        if (ndim == 3) {
            CUDA_MEMCPY3D op{};
            if (to_texture) {
                op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                op.srcDevice = buf; op.srcXInBytes = buf_off;
                op.srcPitch = pitch; op.srcHeight = shape[1];
                op.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                op.dstArray = array;
            } else {
                op.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                op.srcArray = array;
                op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                op.dstDevice = buf; op.dstXInBytes = buf_off;
                op.dstPitch = pitch; op.dstHeight = shape[1];
            }
            op.WidthInBytes = pitch;
            op.Height = shape[1];
            op.Depth = shape[2];
            cuda_check(cuMemcpy3DAsync(&op, ts->stream));
        } else {
            CUDA_MEMCPY2D op{};
            if (to_texture) {
                op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                op.srcDevice = buf; op.srcXInBytes = buf_off; op.srcPitch = pitch;
                op.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                op.dstArray = array;
            } else {
                op.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                op.srcArray = array;
                op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                op.dstDevice = buf; op.dstXInBytes = buf_off; op.dstPitch = pitch;
            }
            op.WidthInBytes = pitch;
            op.Height = (ndim == 2) ? shape[1] : 1;
            cuda_check(cuMemcpy2DAsync(&op, ts->stream));
        }
    }

    if (needs_staging && !to_texture)
        jitc_cuda_tex_stage(false, ts, staging, n_texels, linear, tex);
}

void jitc_cuda_tex_memcpy_d2t(const void *src_ptr, void *dst_handle,
                              size_t level) {
    jitc_cuda_tex_memcpy(true, *(const CUDATexture *) dst_handle, src_ptr,
                         level);
}

void jitc_cuda_tex_memcpy_t2d(const void *src_handle, void *dst_ptr) {
    jitc_cuda_tex_memcpy(false, *(const CUDATexture *) src_handle, dst_ptr, 0);
}

Variable jitc_cuda_tex_check(VarType out_type, size_t ndim, const uint32_t *pos,
                             size_t n_extra = 0, const uint32_t *extra = nullptr) {
    // Validate input types, determine size of the operation
    uint32_t size = 0;
    bool dirty = false, symbolic = false;
    JitBackend backend = JitBackend::None;

    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_check(): invalid texture dimension!");

    // The coordinates, followed by any further float32 operands (LOD/gradients)
    uint32_t idx[9];
    size_t n = 0;
    for (size_t i = 0; i < ndim; ++i)
        idx[n++] = pos[i];
    for (size_t i = 0; i < n_extra; ++i)
        idx[n++] = extra[i];

    for (size_t i = 0; i < n; ++i) {
        const Variable *v = jitc_var(idx[i]);
        if ((VarType) v->type != VarType::Float32)
            jitc_raise("jit_cuda_tex_check(): type mismatch for arg. %zu (got "
                       "%s, expected %s)", i, type_name[v->type],
                       type_name[(int) VarType::Float32]);
        size = std::max(size, v->size);
        dirty |= v->is_dirty();
        symbolic |= (bool) v->symbolic;
        backend = (JitBackend) v->backend;
    }

    for (size_t i = 0; i < n; ++i) {
        const Variable *v = jitc_var(idx[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_cuda_tex_check(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(backend));
        for (size_t i = 0; i < n; ++i) {
            if (jitc_var(idx[i])->is_dirty())
                jitc_raise_dirty_error(idx[i]);
        }
    }

    Variable v;
    v.size = size;
    v.backend = (uint32_t) backend;
    v.symbolic = symbolic;
    v.type = (uint32_t) out_type;
    return v;
}

void jitc_cuda_tex_lookup(const void *handle, const uint32_t *pos,
                          uint32_t active, uint32_t *out) {
    CUDATexture &tex = *((CUDATexture *) handle);
    if (tex.external && !tex.mapped)
        jitc_raise("jit_tex_lookup(): map() the OpenGL-wrapped texture before "
                   "sampling it.");
    size_t ndim = tex.ndim;
    VarType out_type = VarType::Float32;
    Variable v = jitc_cuda_tex_check(out_type, ndim, pos);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        // Perform a fetch per texture ..
        v.kind = (uint32_t) VarKind::TexLookup;
        memset(v.dep, 0, sizeof(v.dep));
        const Variable *active_v = jitc_var(active);
        if (active_v->is_literal() && active_v->literal == 1) {
            v.dep[0] = tex.get_jit_pointer((uint32_t) ti);
            v.literal = 0;
        } else {
            v.literal = 1; // encode a masked operation
            uint64_t zero_i = 0;
            Ref zero = steal(jitc_var_literal((JitBackend) v.backend, VarType::Pointer, &zero_i, 1, 0));
            uint32_t pointer = tex.get_jit_pointer((uint32_t) ti);
            v.dep[0]         = jitc_var_select(active, pointer, zero);
            jitc_var_dec_ref(pointer);
        }
        for (size_t j = 0; j < ndim; ++j) {
            v.dep[j + 1] = pos[j];
            jitc_var_inc_ref(pos[j]);
        }
        Ref tex_load = steal(jitc_var_new(v));

        // .. and then extract components
        v.kind = (uint32_t) VarKind::Extract;
        memset(v.dep, 0, sizeof(v.dep));
        for (size_t ch = 0; ch < tex.channels(ti); ++ch) {
            v.literal = (uint64_t) ch;
            v.dep[0] = tex_load;
            jitc_var_inc_ref(tex_load);
            *out++ = jitc_var_new(v);
        }
    }
}

/// Shared body of the LOD/gradient sampling entry points: emit one ``kind``
/// node per sub-texture, carrying the coordinates and ``n_extra`` further
/// float32 operands (LOD or gradients) in the ``TexData`` payload.
static void cuda_tex_sample_op(VarKind kind, const void *handle,
                               const uint32_t *pos, size_t n_extra,
                               const uint32_t *extra, uint32_t active,
                               uint32_t *out) {
    CUDATexture &tex = *((CUDATexture *) handle);
    if (tex.external && !tex.mapped)
        jitc_raise("jit_tex_lookup_lod/grad(): map() the OpenGL-wrapped "
                   "texture before sampling it.");
    if (tex.n_levels <= 1)
        jitc_raise("jit_tex_lookup_lod/grad(): texture was created without "
                   "MIP levels!");

    size_t ndim = tex.ndim;
    Variable tmpl =
        jitc_cuda_tex_check(VarType::Float32, ndim, pos, n_extra, extra);

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        Ref tex_ptr = steal(tex.get_jit_pointer((uint32_t) ti));

        TexData *td = new TexData();
        td->ndim = (uint32_t) ndim;
        for (size_t i = 0; i < ndim; ++i) {
            td->indices[i] = pos[i];
            jitc_var_inc_ref(pos[i]);
        }
        td->n_values = (uint32_t) n_extra;
        for (size_t i = 0; i < n_extra; ++i) {
            td->values[i] = extra[i];
            jitc_var_inc_ref(extra[i]);
        }

        uint32_t node =
            masked ? jitc_var_new_node_2(
                         JitBackend::CUDA, kind, VarType::Float32, tmpl.size,
                         tmpl.symbolic, tex_ptr, jitc_var(tex_ptr), active,
                         jitc_var(active), (uintptr_t) td)
                   : jitc_var_new_node_1(
                         JitBackend::CUDA, kind, VarType::Float32, tmpl.size,
                         tmpl.symbolic, tex_ptr, jitc_var(tex_ptr),
                         (uintptr_t) td);

        jitc_var_set_callback(
            node,
            [](uint32_t, int free, void *ptr) {
                if (free)
                    delete (TexData *) ptr;
            },
            td, true);

        Ref tex_load = steal(node);
        for (size_t ch = 0; ch < tex.channels(ti); ++ch)
            *out++ = jitc_var_new_node_1(
                JitBackend::CUDA, VarKind::Extract, VarType::Float32,
                tmpl.size, tmpl.symbolic, tex_load, jitc_var(tex_load),
                (uint64_t) ch);
    }
}

void jitc_cuda_tex_lookup_lod(const void *handle, const uint32_t *pos,
                              uint32_t lod, uint32_t active, uint32_t *out) {
    cuda_tex_sample_op(VarKind::TexLookupLod, handle, pos, 1, &lod, active,
                       out);
}

void jitc_cuda_tex_lookup_grad(const void *handle, const uint32_t *pos,
                               const uint32_t *ddx, const uint32_t *ddy,
                               uint32_t active, uint32_t *out) {
    size_t ndim = ((CUDATexture *) handle)->ndim;
    uint32_t extra[6];
    for (size_t i = 0; i < ndim; ++i) {
        extra[i] = ddx[i];
        extra[ndim + i] = ddy[i];
    }
    cuda_tex_sample_op(VarKind::TexLookupGrad, handle, pos, 2 * ndim, extra,
                       active, out);
}

void jitc_cuda_tex_write(void *handle, const uint32_t *pos,
                         const uint32_t *value, uint32_t active) {
    CUDATexture &tex = *((CUDATexture *) handle);
    if (!tex.writable)
        jitc_raise("jit_tex_write(): texture was not created with the "
                   "'writable' flag.");
    if (tex.external && !tex.mapped)
        jitc_raise("jit_tex_write(): map() the OpenGL-wrapped texture before "
                   "writing it.");

    size_t ndim = tex.ndim, n_channels = tex.n_channels;

    // Coordinates are integer texels (UInt32), values are Float32. Determine
    // the broadcast launch size.
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
        jitc_eval(thread_state(JitBackend::CUDA));

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    // One surface store per sub-texture (each writes its 1/2/4-channel slice).
    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        size_t base = ti * 4, nc = tex.channels(ti);
        Ref surf = steal(tex.get_surf_jit_pointer((uint32_t) ti));

        TexData *td = new TexData();
        td->ndim = (uint32_t) ndim;
        for (size_t i = 0; i < ndim; ++i) {
            td->indices[i] = pos[i];
            jitc_var_inc_ref(pos[i]);
        }
        td->n_values = (uint32_t) nc;
        for (size_t c = 0; c < nc; ++c) {
            td->values[c] = value[base + c];
            jitc_var_inc_ref(value[base + c]);
        }
        td->comp_bytes = (uint32_t) tex.type_size;

        uint32_t node =
            masked ? jitc_var_new_node_2(
                         JitBackend::CUDA, VarKind::TexWrite, VarType::Void,
                         size, symbolic, surf, jitc_var(surf), active,
                         jitc_var(active), (uintptr_t) td)
                   : jitc_var_new_node_1(
                         JitBackend::CUDA, VarKind::TexWrite, VarType::Void,
                         size, symbolic, surf, jitc_var(surf), (uintptr_t) td);

        jitc_var_set_callback(
            node,
            [](uint32_t, int free, void *ptr) {
                if (free)
                    delete (TexData *) ptr;
            },
            td, true);

        jitc_var_mark_side_effect(node);
    }
}

void jitc_cuda_tex_bilerp_fetch(const void *handle,
                                const uint32_t *pos, uint32_t active,
                                uint32_t *out) {
    CUDATexture &tex = *((CUDATexture *) handle);
    if (tex.ndim != 2)
        jitc_raise("jitc_cuda_tex_bilerp_fetch(): only 2D textures are supported!");
    if (tex.external && !tex.mapped)
        jitc_raise("jit_tex_bilerp_fetch(): map() the OpenGL-wrapped texture "
                   "before sampling it.");
    size_t ndim = tex.ndim;
    VarType out_type = VarType::Float32;
    Variable v = jitc_cuda_tex_check(out_type, ndim, pos);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        for (size_t ch = 0; ch < tex.channels(ti); ++ch) {
            // Perform a fetch per texture and channel..
            v.kind = (uint32_t) VarKind::TexFetchBilerp;
            v.literal = ch;
            memset(v.dep, 0, sizeof(v.dep));
            v.dep[0] = tex.get_jit_pointer((uint32_t) ti);
            v.dep[1] = active;
            jitc_var_inc_ref(active);
            for (size_t j = 0; j < ndim; ++j) {
                v.dep[j + 2] = pos[j];
                jitc_var_inc_ref(pos[j]);
            }
            Ref tex_load = steal(jitc_var_new(v));

            memset(v.dep, 0, sizeof(v.dep));
            v.kind = (uint32_t) VarKind::Extract;
            for (uint32_t j = 0; j < 4; ++j) {
                // .. and then extract components
                v.literal = (uint64_t) j;
                v.dep[0] = tex_load;
                jitc_var_inc_ref(tex_load);
                *out++ = jitc_var_new(v);
            }
        }
    }
}

uintptr_t jitc_cuda_tex_native_handle(const void *handle,
                                      size_t /* sub_index */) {
    const CUDATexture &tex = *((const CUDATexture *) handle);
    // On the OpenGL-interop backend the native handle is the wrapped OpenGL
    // texture id; a Dr.Jit-allocated texture has no GL identity (id 0).
    return (uintptr_t) tex.gl_id;
}

// GL texture targets (avoid pulling in an OpenGL header; mirrors gl_interop.cpp).
#define DRJIT_GL_TEXTURE_2D 0x0DE1
#define DRJIT_GL_TEXTURE_3D 0x806F

/// Build a CUtexObject sampling \c array, mirroring jitc_cuda_tex_create()'s
/// descriptor setup. Used to (re)create the texture object of an OpenGL-wrapped
/// texture on each \ref jitc_cuda_tex_map().
static CUtexObject cuda_tex_make_texobject(CUarray array, int format,
                                           size_t channels, int filter_mode,
                                           int wrap_mode, int srgb, size_t width,
                                           size_t height, size_t depth,
                                           CUmipmappedArray mip,
                                           size_t n_levels, int mip_filter,
                                           size_t max_aniso) {
    CUDA_RESOURCE_DESC res_desc{};
    if (mip) {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = mip;
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;
    }

    // UInt8 storage is read back as a normalized float in [0, 1] (the default
    // read mode), optionally decoding sRGB -> linear.
    bool is_u8 = (VarType) format == VarType::UInt8;

    CUDA_TEXTURE_DESC tex_desc{};
    tex_desc.filterMode =
        (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
    tex_desc.mipmapFilterMode = (mip && mip_filter != 0)
        ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
    // The trilinear "optimization" degrades a MIP blend to a plain bilinear
    // lookup whenever the LOD fraction is near 0 or 1, which makes hardware
    // filtering visibly diverge from the other backends. Turn it off.
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES |
                     CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION |
                     ((is_u8 && srgb) ? CU_TRSF_SRGB : 0);
    CUaddress_mode am = (wrap_mode == 0)   ? CU_TR_ADDRESS_MODE_WRAP
                        : (wrap_mode == 1) ? CU_TR_ADDRESS_MODE_CLAMP
                                           : CU_TR_ADDRESS_MODE_MIRROR;
    for (size_t i = 0; i < 3; ++i)
        tex_desc.addressMode[i] = am;
    max_aniso = max_aniso < 1 ? 1 : (max_aniso > 16 ? 16 : max_aniso);
    tex_desc.maxAnisotropy = (unsigned int) max_aniso;
    tex_desc.maxMipmapLevelClamp = (float) (n_levels - 1);

    CUDA_RESOURCE_VIEW_DESC view_desc{};
    view_desc.width = width;
    view_desc.height = height;
    view_desc.depth = depth;
    view_desc.lastMipmapLevel = (unsigned int) (n_levels - 1);
    bool is_f32 = (VarType) format == VarType::Float32;
    if (channels == 1)
        view_desc.format = is_u8  ? CU_RES_VIEW_FORMAT_UINT_1X8
                         : is_f32 ? CU_RES_VIEW_FORMAT_FLOAT_1X32
                                  : CU_RES_VIEW_FORMAT_FLOAT_1X16;
    else if (channels == 2)
        view_desc.format = is_u8  ? CU_RES_VIEW_FORMAT_UINT_2X8
                         : is_f32 ? CU_RES_VIEW_FORMAT_FLOAT_2X32
                                  : CU_RES_VIEW_FORMAT_FLOAT_2X16;
    else
        view_desc.format = is_u8  ? CU_RES_VIEW_FORMAT_UINT_4X8
                         : is_f32 ? CU_RES_VIEW_FORMAT_FLOAT_4X32
                                  : CU_RES_VIEW_FORMAT_FLOAT_4X16;

    CUtexObject texobj = 0;
    cuda_check(cuTexObjectCreate(&texobj, &res_desc, &tex_desc, &view_desc));
    return texobj;
}

void *jitc_cuda_tex_wrap(uintptr_t handle, size_t ndim, int format,
                         int writable, int filter_mode, int wrap_mode,
                         int srgb) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_tex_wrap(): invalid texture dimension!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    // Register the OpenGL texture for interop (copied from gl_interop.cpp's
    // registration). For sampling we register read/write (not write-discard,
    // which would drop the existing contents); a writable wrap additionally
    // requests surface load/store so kernels can render into the texture.
    CUgraphicsResource resource = nullptr;
    GLenum gl_target = (ndim == 3) ? DRJIT_GL_TEXTURE_3D : DRJIT_GL_TEXTURE_2D;
    unsigned int reg_flags = writable ? CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST
                                      : CU_GRAPHICS_REGISTER_FLAGS_NONE;
    cuda_check(cuGraphicsGLRegisterImage(&resource, (GLuint) handle, gl_target,
                                         reg_flags));

    // Transiently map to read back the texture's metadata.
    CUDA_ARRAY3D_DESCRIPTOR desc{};
    cuda_check(cuGraphicsMapResources(1, &resource, ts->stream));
    CUarray array = nullptr;
    cuda_check(cuGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    cuda_check(cuArray3DGetDescriptor(&desc, array));
    cuda_check(cuGraphicsUnmapResources(1, &resource, ts->stream));

    int tex_format;
    size_t comp_size;
    switch (desc.Format) {
        case CU_AD_FORMAT_FLOAT: tex_format = (int) VarType::Float32; comp_size = 4; break;
        case CU_AD_FORMAT_HALF:  tex_format = (int) VarType::Float16; comp_size = 2; break;
        case CU_AD_FORMAT_UNSIGNED_INT8: tex_format = (int) VarType::UInt8; comp_size = 1; break;
        default:
            cuda_check(cuGraphicsUnregisterResource(resource));
            jitc_raise("jit_tex_wrap(): unsupported OpenGL texture format; only "
                       "8-bit unsigned and 16-/32-bit float textures can be "
                       "wrapped.");
    }
    if (tex_format != format) {
        cuda_check(cuGraphicsUnregisterResource(resource));
        jitc_raise("jit_tex_wrap(): texture component type does not match the "
                   "Dr.Jit texture's scalar type.");
    }
    size_t tex_ndim = (desc.Depth > 0) ? 3 : (desc.Height > 0 ? 2 : 1);
    if (tex_ndim != ndim) {
        cuda_check(cuGraphicsUnregisterResource(resource));
        jitc_raise("jit_tex_wrap(): texture dimensionality (%zuD) does not match "
                   "the Dr.Jit texture type (%zuD).", tex_ndim, ndim);
    }
    size_t channels = desc.NumChannels;
    if (channels == 0 || channels > 4) {
        cuda_check(cuGraphicsUnregisterResource(resource));
        jitc_raise("jit_tex_wrap(): unsupported channel count %zu.", channels);
    }

    CUDATexture *texture =
        new CUDATexture(comp_size, channels, writable != 0);
    texture->ndim = ndim;
    texture->external = true;
    texture->gl_id = (unsigned int) handle;
    texture->graphics_resource = resource;
    texture->filter_mode = filter_mode;
    texture->wrap_mode = wrap_mode;
    texture->storage_format = tex_format;
    texture->srgb = srgb;
    texture->shape[0] = desc.Width;
    texture->shape[1] = desc.Height;
    texture->shape[2] = desc.Depth;
    texture->indices[0] = 0; // texture / surface objects only exist while mapped

    jitc_log(LogLevel::Debug, "jitc_cuda_tex_wrap(gl=%u): " DRJIT_PTR,
             texture->gl_id, (uintptr_t) texture);
    return (void *) texture;
}

void jitc_cuda_tex_map(void *handle) {
    CUDATexture *texture = (CUDATexture *) handle;
    if (!texture->external || texture->mapped)
        return; // Dr.Jit-allocated textures (and already-mapped ones) are ready.

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUgraphicsResource resource =
        (CUgraphicsResource) texture->graphics_resource;
    cuda_check(cuGraphicsMapResources(1, &resource, ts->stream));
    CUarray array = nullptr;
    cuda_check(cuGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    texture->arrays[0] = array;

    // The mapped array (and hence the texture/surface objects) may change
    // across map cycles, so (re)build them and their handle variables here.
    // Like a Dr.Jit-allocated texture, a writable wrap is both sampled via a
    // texture object and written via a surface object over the same array.
    texture->textures[0] = cuda_tex_make_texobject(
        array, texture->storage_format, texture->n_channels,
        texture->filter_mode, texture->wrap_mode, texture->srgb,
        texture->shape[0], texture->shape[1], texture->shape[2]);
    texture->indices[0] =
        jitc_var_mem_map(JitBackend::CUDA, VarType::UInt64,
                         (void *) texture->textures[0], 1, false);

    if (texture->writable) {
        CUDA_RESOURCE_DESC res_desc{};
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;
        cuda_check(cuSurfObjectCreate(&texture->surfaces[0], &res_desc));
        texture->surf_indices[0] =
            jitc_var_mem_map(JitBackend::CUDA, VarType::UInt64,
                             (void *) texture->surfaces[0], 1, false);
    }

    texture->mapped = true;
}

void jitc_cuda_tex_unmap(void *handle) {
    CUDATexture *texture = (CUDATexture *) handle;
    if (!texture->external || !texture->mapped)
        return;

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    if (texture->writable) {
        if (texture->surf_indices[0]) {
            jitc_var_dec_ref(texture->surf_indices[0]);
            texture->surf_indices[0] = 0;
        }
        if (texture->surfaces[0]) {
            cuda_check(cuSurfObjectDestroy(texture->surfaces[0]));
            texture->surfaces[0] = 0;
        }
    }
    if (texture->indices[0]) {
        jitc_var_dec_ref(texture->indices[0]);
        texture->indices[0] = 0;
    }
    if (texture->textures[0]) {
        cuda_check(cuTexObjectDestroy(texture->textures[0]));
        texture->textures[0] = 0;
    }
    texture->arrays[0] = nullptr; // owned by OpenGL; invalidated by unmap

    CUgraphicsResource resource =
        (CUgraphicsResource) texture->graphics_resource;
    cuda_check(cuGraphicsUnmapResources(1, &resource, ts->stream));
    texture->mapped = false;
}

void jitc_cuda_tex_destroy(void *handle) {
    if (!handle)
        return;

    jitc_log(LogLevel::Debug, "jitc_cuda_tex_destroy(" DRJIT_PTR ")",
             (uintptr_t) handle);

    CUDATexture *texture = (CUDATexture *) handle;

    // A texture wrapping an OpenGL texture owns no arrays/objects, just
    // release the mapping and interop resources.
    if (texture->external) {
        if (texture->mapped)
            jitc_cuda_tex_unmap(texture);
        if (texture->graphics_resource) {
            ThreadState *ts = thread_state(JitBackend::CUDA);
            scoped_set_context guard(ts->context);
            CUresult rv = cuGraphicsUnregisterResource(
                (CUgraphicsResource) texture->graphics_resource);
            if (rv != CUDA_ERROR_INVALID_GRAPHICS_CONTEXT) // OpenGL already gone
                cuda_check(rv);
        }
        delete texture;
        return;
    }

    // Fetch the count in case the struct is deleted during the traversal
    const size_t n_textures = texture->n_textures;

    // Release the surface-handle holders first
    if (texture->writable) {
        for (size_t tex = 0; tex < n_textures; ++tex)
            jitc_var_dec_ref(texture->surf_indices[tex]);
    }

    for (size_t tex = 0; tex < n_textures; ++tex)
        jitc_var_dec_ref(texture->indices[tex]);
}
