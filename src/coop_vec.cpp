/*
    src/coop_vec.cpp -- Backend-independent parts of the Cooperative Vector API

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "var.h"
#include "coop_vec.h"
#include "internal.h"
#include "log.h"
#include "op.h"
#include "optix_api.h"
#include "optix.h"
#include "cuda.h"

// Strip away loop phi nodes to recover the nested memory buffer. This is needed
// when cooperative vectors in symbolic loops/conditionals reference the buffer
// backing the matrix data, which has been converted into a symbolic variable.
static uint32_t unwrap(uint32_t index) {
    while (true) {
        const Variable *v = jitc_var(index);
        if (v->kind != (uint32_t) VarKind::LoopPhi)
            return index;
        index = borrow(v->dep[3]);
    }
}

bool jitc_coop_vec_supported(JitBackend backend) {
    if (backend == JitBackend::CUDA)
        return (jitc_cuda_version_major == 12 &&
                jitc_cuda_version_minor >= 8) ||
               jitc_cuda_version_major > 12;
    else
        return true;
}

uint32_t jitc_coop_vec_pack(uint32_t n, const uint32_t *in) {
    if (n == 0)
        jitc_raise("jit_coop_vec_pack(): vector cannot be empty!");
    if (n > 0xFFFF)
        jitc_raise("jit_coop_vec_pack(): cooperative vector is too large!");

    for (uint32_t i = 0; i < n; ++i) {
        if (in[i] == 0)
            jitc_raise("jit_coop_vec_pack(): argument %u is uninitialized!", i);
    }

    const Variable *arg_v = jitc_var(in[0]);
    if (!jitc_coop_vec_supported((JitBackend) arg_v->backend))
        jitc_raise("jit_coop_vec_pack(): The use of cooperative vectors on "
                   "the CUDA/OptiX backend requires CUDA 12.8 or newer "
                   "(which corresponds to driver version R570+).");

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecPack;
    v.type = arg_v->type;
    v.size = arg_v->size;
    v.backend = arg_v->backend;
    v.array_length = (uint16_t) n;
    v.coop_vec = true;
    v.optix = v.backend == (uint32_t) JitBackend::CUDA;

    drjit::unique_ptr<CoopVecPackData> cvid = new CoopVecPackData();
    bool is_literal = true;
    uint64_t literal = arg_v->literal;
    cvid->indices.reserve(n);

    jitc_log(Debug, "jit_coop_vec_pack(): building a cooperative vector with %u elements", n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t index = in[i];
        const Variable *v2 = jitc_var(index);
        v.size = std::max(v.size, v2->size);
        if (v2->backend != v.backend || v2->type != v.type)
            jitc_raise("jit_coop_vec_pack(): inputs must have compatible types and backends!");

        if (!v2->is_literal() || v2->literal != literal)
            is_literal = false;

        jitc_var_inc_ref(index);
        cvid->indices.push_back(index);
        jitc_log(Debug, "  - entry %u: r%u", i, index);
    }

    if (is_literal)
        return jitc_coop_vec_literal((JitBackend) v.backend, (VarType) v.type,
                                     &literal, v.size, n);

    return jitc_var_new_take_ownership(v, std::move(cvid), true);
}

void jitc_coop_vec_unpack(uint32_t index, uint32_t n, uint32_t *out) {
    Variable *vec_v = jitc_var(index);
    Variable v;

    if (!vec_v->coop_vec)
        jitc_raise("jit_coop_vec_unpack(): source must be a cooperative vector!");
    if (vec_v->array_length != n)
        jitc_raise("jit_coop_vec_unpack(): internal error, array length did not match!");

    uint32_t length = vec_v->array_length;
    if (vec_v->is_coop_vec_literal()) {
        uint64_t literal = vec_v->literal;
        Ref r = steal(jitc_var_literal((JitBackend) vec_v->backend,
                                       (VarType) vec_v->type, &literal,
                                       vec_v->size, 0));
        for (uint32_t i = 0; i < length; ++i) {
            jitc_var_inc_ref(r);
            out[i] = r;
        }
        return;
    }

    v.kind = (uint32_t) VarKind::CoopVecUnpack;
    v.type = vec_v->type;
    v.size = vec_v->size;
    v.backend = vec_v->backend;
    v.dep[0] = index;

    for (uint32_t i = 0; i < length; ++i) {
        jitc_var_inc_ref(index);
        v.literal = i;
        out[i] = jitc_var_new(v);
    }
}

uint32_t jitc_coop_vec_literal(JitBackend backend,
                               VarType type,
                               const void *value,
                               size_t size,
                               uint32_t length) {
    if (unlikely(size == 0))
        return 0;

    Variable v;
    memcpy(&v.literal, value, type_size[(uint32_t) type]);
    v.kind = (uint32_t) VarKind::CoopVecLiteral;
    v.type = (uint32_t) type;
    v.size = (uint32_t) size;
    v.backend = (uint32_t) backend;
    v.array_length = (uint16_t) length;
    v.coop_vec = true;
    v.optix = v.backend == (uint32_t) JitBackend::CUDA;

    return jitc_var_new(v);
}

uint32_t jitc_coop_vec_load(uint32_t buffer, uint32_t offset, uint32_t length) {
    VarType vt;
    JitBackend backend;
    {
        Variable *buffer_v = jitc_var(buffer);
        vt = (VarType) buffer_v->type;
        backend = (JitBackend) buffer_v->backend;
    }

    void *p = nullptr;
    Ref tmp = steal(jitc_var_data(buffer, false, &p));
    Ref buf_ptr = steal(jitc_var_pointer(backend, p, tmp, 0));

    Ref mask = steal(jitc_var_bool(backend, true));
    mask = steal(jitc_var_mask_apply(mask, 1));

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecLoad;
    v.type = (uint32_t) vt;
    v.size = 1;
    v.backend = (uint32_t) backend;
    v.array_length = (uint16_t) length;
    v.literal = offset;
    v.coop_vec = true;
    v.optix = backend == JitBackend::CUDA;
    v.dep[0] = buf_ptr;
    v.dep[1] = mask;
    jitc_var_inc_ref(buf_ptr);
    jitc_var_inc_ref(mask);

    return jitc_var_new(v);
}

uint32_t jitc_coop_vec_unary_op(JitOp op, uint32_t a0) {
    if (!a0)
        return 0;

    Variable *a0_v = jitc_var(a0);
    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecUnaryOp;
    v.literal = (uint32_t) op;
    v.type = a0_v->type;
    v.size = a0_v->size;
    v.backend = a0_v->backend;
    v.array_length = a0_v->array_length;
    v.coop_vec = true;
    v.dep[0] = a0;
    jitc_var_inc_ref(a0, a0_v);
    return jitc_var_new(v);
}

uint32_t jitc_coop_vec_cast(uint32_t index, VarType vt) {
    if (!index)
        return 0;

    Variable *prev_v = jitc_var(index);
    if ((VarType) prev_v->type == vt) {
        jitc_var_inc_ref(index, prev_v);
        return index;
    }

    /// The OptiX conversion intrinsic is currently too limited
    if ((JitBackend) prev_v->backend == JitBackend::CUDA) {
        uint32_t n = prev_v->array_length;
        if (n == 0) // just here to silence a GCC warning..
            return 0;
        uint32_t *tmp1 = (uint32_t *) alloca(sizeof(uint32_t) * n),
                 *tmp2 = (uint32_t *) alloca(sizeof(uint32_t) * n);
        jitc_coop_vec_unpack(index, n, tmp1);
        for (uint32_t i = 0; i < n; ++i)
            tmp2[i] = jitc_var_cast(tmp1[i], vt, false);
        uint32_t result = jitc_coop_vec_pack(n, tmp2);
        for (uint32_t i = 0; i < n; ++i) {
            jitc_var_dec_ref(tmp1[i]);
            jitc_var_dec_ref(tmp2[i]);
        }
        return result;
    }

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecCast;
    v.type = (uint32_t) vt;
    v.size = prev_v->size;
    v.backend = prev_v->backend;
    v.array_length = prev_v->array_length;
    v.coop_vec = true;
    v.dep[0] = index;
    jitc_var_inc_ref(index, prev_v);

    return jitc_var_new(v);
}

uint32_t jitc_coop_vec_binary_op(JitOp op, uint32_t a0, uint32_t a1) {
    if (!a0 || !a1)
        jitc_raise("jit_coop_vec_binary_op(): detected uninitialized inputs!");

    Variable *a0_v = jitc_var(a0),
             *a1_v = jitc_var(a1);

    if (a0_v->array_length != a1_v->array_length)
        jitc_raise("jit_coop_vec_binary_op(): the cooperative vectors have "
                   "incompatible lengths (%u and %u)!",
                   a0_v->array_length, a1_v->array_length);

    if (a0_v->type != a1_v->type)
        jitc_raise("jit_coop_vec_binary_op(): the cooperative vectors have "
                   "incompatible types (%s and %s)!",
                   type_name[a0_v->type], type_name[a1_v->type]);

    if (!(a0_v->size == a1_v->size || a1_v->size == 1 || a0_v->size == 1))
        jitc_raise(
            "jit_coop_vec_binary_op(): incompatible width (%u and %u)!",
            a0_v->size, a1_v->size);

    uint32_t max_size = std::max(a0_v->size, a1_v->size);

    // Exploit some basic optimization opportunities (useful for AD)
    switch (op) {
        case JitOp::Add:
            if (jitc_is_any_zero(a0_v)) { return jitc_var_resize(a1, max_size); }
            if (jitc_is_any_zero(a1_v)) { return jitc_var_resize(a0, max_size); }
            break;

        case JitOp::Mul:
            if (jitc_is_one(a0_v)) { return jitc_var_resize(a1, max_size); }
            if (jitc_is_one(a1_v)) { return jitc_var_resize(a0, max_size); }
            break;

        default:
            break;
    }

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecBinaryOp;
    v.literal = (uint32_t) op;
    v.type = a0_v->type;
    v.size = max_size;
    v.backend = a0_v->backend;
    v.array_length = a0_v->array_length;
    v.coop_vec = true;
    v.dep[0] = a0;
    v.dep[1] = a1;
    jitc_var_inc_ref(a0, a0_v);
    jitc_var_inc_ref(a1, a1_v);
    return jitc_var_new(v);
}

uint32_t jitc_coop_vec_ternary_op(JitOp op, uint32_t a0, uint32_t a1, uint32_t a2) {
    if (!a0 || !a1 || !a2)
        jitc_raise("jit_coop_vec_ternary_op(): detected uninitialized inputs!");

    Variable *a0_v = jitc_var(a0),
             *a1_v = jitc_var(a1),
             *a2_v = jitc_var(a2);

    uint32_t max_size = std::max(std::max(a0_v->size, a1_v->size), a2_v->size);

    if (a0_v->array_length != a1_v->array_length || a0_v->array_length != a2_v->array_length)
        jitc_raise("jit_coop_vec_ternary_op(): the cooperative vectors have an "
                   "incompatible size (%u, %u, and %u)!",
                   a0_v->array_length, a1_v->array_length, a2_v->array_length);

    if (a0_v->type != a1_v->type || a0_v->type != a1_v->type)
        jitc_raise("jit_coop_vec_ternary_op(): the cooperative vectors have "
                   "incompatible types (%s, %s, and %s)!",
                   type_name[a0_v->type], type_name[a1_v->type], type_name[a2_v->type]);

    if (!(a0_v->size == max_size || a0_v->size == 1) ||
        !(a1_v->size == max_size || a1_v->size == 1) ||
        !(a2_v->size == max_size || a2_v->size == 1))
        jitc_raise(
            "jit_coop_vec_ternary_op(): incompatible width (%u, %u, and %u)!",
            a0_v->size, a1_v->size, a2_v->size);

    // Exploit some basic optimization opportunities (useful for AD)
    if (op == JitOp::Fma) {
        if (jitc_is_one(a0_v)) {
            Ref result = steal(jitc_coop_vec_binary_op(JitOp::Add, a1, a2));
            return jitc_var_resize(result, max_size);
        }
        if (jitc_is_one(a1_v)) {
            Ref result = steal(jitc_coop_vec_binary_op(JitOp::Add, a0, a2));
            return jitc_var_resize(result, max_size);
        }
        if (jitc_is_any_zero(a1_v)) {
            Ref result = steal(jitc_coop_vec_binary_op(JitOp::Mul, a0, a1));
            return jitc_var_resize(result, max_size);
        }
    }

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecTernaryOp;
    v.literal = (uint32_t) op;
    v.type = a0_v->type;
    v.size = max_size;
    v.backend = a0_v->backend;
    v.array_length = a0_v->array_length;
    v.coop_vec = true;
    v.dep[0] = a0;
    v.dep[1] = a1;
    v.dep[2] = a2;
    jitc_var_inc_ref(a0, a0_v);
    jitc_var_inc_ref(a1, a1_v);
    jitc_var_inc_ref(a2, a2_v);
    return jitc_var_new(v);
}

MatrixDescr jitc_coop_vec_compute_layout(uint32_t index,
                                         const MatrixDescr *in_,
                                         MatrixLayout layout,
                                         uint32_t offset) {
    const MatrixDescr in = *in_;
    const bool is_vector = in.cols == 1;

#if defined(DRJIT_ENABLE_OPTIX)
    JitBackend backend;
    VarType vt;

    {
        const Variable *v = jitc_var(index);
        vt = (VarType) v->type;
        backend = (JitBackend) v->backend;
    }

    uint32_t tsize = type_size[(uint32_t) vt];

    if (backend == JitBackend::CUDA) {
        uint32_t offset_in_bytes = in.offset * tsize;
        if (offset_in_bytes % 64 != 0)
            jitc_raise(
                "jit_coop_vec_compute_layout(): OptiX requires input matrices "
                "to be 64-byte aligned. Encountered an input with "
                "offset %u, which is not divisible by 64.", offset_in_bytes);

        uint32_t out_align = is_vector ? 16 : 64;
        offset = (ceil_div(offset * tsize, out_align) * out_align) / tsize;
    }
#else
    (void) index;
#endif

    MatrixDescr r;
    r.dtype = in.dtype;
    r.layout = is_vector ? MatrixLayout::RowMajor : layout;
    r.rows = in.rows;
    r.cols = in.cols;
    r.offset = offset;
    r.stride = r.cols;
    r.size = (r.rows - 1) * r.stride + r.cols;

#if defined(DRJIT_ENABLE_OPTIX)
    if (backend == JitBackend::CUDA && r.layout != MatrixLayout::RowMajor) {
        OptixDeviceContext ctx = jitc_optix_context();
        uint32_t type_id = jitc_optix_coop_vec_type_id(vt),
                 layout_id = jitc_optix_coop_vec_layout_id(layout);
        size_t size = 0;

        if (vt != VarType::Float16)
            jitc_raise(
                "jit_coop_vec_compute_layout(): CUDA/OptiX conversion to "
                "optimal layout is currently limited to half precision data.");

        if (!optixCoopVecMatrixComputeSize)
            jitc_raise("jit_coop_vec_compute_layout(): Cooperative vectors are not "
                       "supported by your NVIDIA GPU driver. Please install "
                       "driver version 570 or newer.");

        jitc_optix_check(optixCoopVecMatrixComputeSize(
            ctx, in.rows, in.cols, type_id, layout_id, 0, &size));
        r.stride = 0;
        r.size = (uint32_t) size / tsize;
    }
#endif

    return r;
}

void jitc_coop_vec_pack_matrices(uint32_t count,
                                 uint32_t in,
                                 const MatrixDescr *in_descr,
                                 uint32_t out,
                                 const MatrixDescr *out_descr) {
    void *in_p = nullptr, *out_p = nullptr;
    Ref in_data = steal(jitc_var_data(in, true, &in_p));
    Ref out_data = steal(jitc_var_data(out, true, &out_p));

    JitBackend backend;
    {
        const Variable *out_v = jitc_var(out);
        jitc_log(Debug, "jit_coop_vec_pack(): packing %u %s, %u bytes, r%u -> r%u",
                 count, count == 1 ? "matrix" : "matrices", out_v->size * type_size[out_v->type], in, out);
        backend = (JitBackend) out_v->backend;
    }

    thread_state(backend)->coop_vec_pack(count, in_p, in_descr, out_p, out_descr);
}

uint32_t jitc_coop_vec_matvec(uint32_t A_index,
                              const MatrixDescr *A_descr,
                              uint32_t x_index,
                              uint32_t b_index,
                              const MatrixDescr *b_descr,
                              int transpose) {

    if (!A_index || !x_index)
        jitc_raise("jit_coop_vec_matvec(): detected uninitialized inputs!");

    VarType a_vt = VarType::Void,
            b_vt = VarType::Void,
            x_vt = VarType::Void;
    uint32_t size;
    JitBackend backend;

    uint32_t input_length  = transpose ? A_descr->rows : A_descr->cols,
             output_length = transpose ? A_descr->cols : A_descr->rows;

    drjit::unique_ptr<CoopVecMatVecData> cvmvd = new CoopVecMatVecData();
    {
        Variable *x_v = jitc_var(x_index);
        x_vt = (VarType) x_v->type;
        backend = (JitBackend) x_v->backend;
        size = x_v->size;

        if (x_v->array_length != input_length)
            jitc_raise(
                "jit_coop_vec_matvec(): incompatible shapes. Attempted to "
                "multiply a %ux%u matrix by a vector with %u elements.",
                output_length, input_length, x_v->array_length);
    }

    A_index = unwrap(A_index);
    if (b_index)
        b_index = unwrap(b_index);

    Ref a_ptr, b_ptr;
    {
        void *p = nullptr;
        Ref tmp = steal(jitc_var_data(A_index, false, &p));

        a_vt = (VarType) jitc_var(tmp)->type;
        a_ptr = steal(jitc_var_pointer(backend, p, tmp, 0));
        cvmvd->A_descr = *A_descr;

        if (backend == JitBackend::CUDA) {
            uint32_t tsize = type_size[(int) a_vt],
                     offset_in_bytes = A_descr->offset * tsize,
                     stride_in_bytes = A_descr->stride * tsize;

            if (offset_in_bytes % 64)
                jitc_raise("jit_coop_vec_matvec(): matrix offset (%u bytes) "
                           "must be 64-byte aligned.\n", offset_in_bytes);

            if (stride_in_bytes % 16)
                jitc_raise("jit_coop_vec_matvec(): matrix stride (%u bytes) "
                           "must be 16-byte aligned.\n", stride_in_bytes);
        }
    }

    if (b_index && b_descr) {
        void *p = nullptr;
        Ref tmp = steal(jitc_var_data(b_index, false, &p));

        b_vt = (VarType) jitc_var(tmp)->type;
        b_ptr = steal(jitc_var_pointer(backend, p, tmp, 0));
        cvmvd->b_descr = *b_descr;

        if (b_descr->rows != output_length || b_descr->cols != 1)
            jitc_raise(
                "jit_coop_vec_matvec(): 'b' vector has an incompatible shape "
                "(expected (%u x 1), got (%u x %u)).",
                output_length, b_descr->rows, b_descr->cols);

        if (b_descr->stride != 1)
            jitc_raise(
                "jit_coop_vec_matvec(): 'b' vector must be tightly packed.");
    }

    cvmvd->transpose = transpose;

    bool supported = false, is_llvm = backend == JitBackend::LLVM;
    supported |= a_vt == VarType::Float16 && x_vt == VarType::Float16 &&
                 (b_vt == VarType::Void || b_vt == VarType::Float16);
    supported |= is_llvm && (a_vt == VarType::Float32 && x_vt == VarType::Float32 &&
                            (b_vt == VarType::Void || b_vt == VarType::Float32));

    if (!supported)
        jitc_raise(
            "jit_coop_vec_matvec(): incompatible input types (currently, only "
            "float16 is supported on the CUDA/OptiX backend. The LLVM backend"
            "supports float16 and float32).");

    Ref mask = steal(jitc_var_bool(backend, true));
    mask = steal(jitc_var_mask_apply(mask, size));

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecMatVec;
    v.type = (uint32_t) x_vt;
    v.size = std::max(size, jitc_var(mask)->size);
    v.backend = (uint32_t) backend;
    v.array_length = (uint16_t) output_length;
    v.coop_vec = true;
    v.dep[0] = a_ptr;
    v.dep[1] = x_index;
    v.dep[2] = mask;
    v.dep[3] = b_ptr;
    jitc_var_inc_ref(a_ptr);
    jitc_var_inc_ref(x_index);
    jitc_var_inc_ref(b_ptr);
    jitc_var_inc_ref(mask);

    return jitc_var_new_take_ownership(v, std::move(cvmvd), true);
}

uint32_t jitc_coop_vec_accum(uint32_t target_, uint32_t target_size,
                             uint32_t offset, uint32_t index) {
    JitBackend backend;
    VarType vt;
    uint32_t size;
    {
        const Variable *v = jitc_var(index);
        backend = (JitBackend) v->backend;
        vt = (VarType) v->type;
        size = v->size;

        if (backend == JitBackend::CUDA &&
            !(vt == VarType::Float16 || vt == VarType::Float32))
            jitc_raise(
                "jit_coop_vec_accum(): this operation is restricted to "
                "float16 precision on the CUDA/OptiX backend.");
    }

    if (target_)
        target_ = unwrap(target_);

    Ref target = borrow(target_);
    if (!target) {
        uint64_t z = 0;
        target = steal(jitc_var_literal(backend, vt, &z, target_size, true));
    } else {
        const Variable *target_v = jitc_var(target);
        // Copy-on-Write logic. See the same line in jitc_var_scatter() for details
        if (target_v->ref_count != 2 && target_v->ref_count_stashed != 1) {
            target = steal(jitc_var_copy(target));
            target_v = jitc_var(target);
        }

        if ((VarType) target_v->type != vt)
            jitc_raise("jit_coop_vec_accum(): source/target "
                       "buffers have an incompatible type (%s vs %s)!",
                       type_name[target_v->type], type_name[(int) vt]);
    }

    void *p = nullptr;
    target = steal(jitc_var_data(target, false, &p));
    Ref target_ptr = steal(jitc_var_pointer(backend, p, target, 1));

    Ref mask = steal(jitc_var_bool(backend, true));
    mask = steal(jitc_var_mask_apply(mask, size));

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecAccum;
    v.type = (uint32_t) VarType::Void;
    v.size = std::max(size, jitc_var(mask)->size);
    v.backend = (uint32_t) backend;
    v.literal = offset;
    v.symbolic = jitc_flag(JitFlag::SymbolicScope);
    v.dep[0] = target_ptr;
    v.dep[1] = index;
    v.dep[2] = mask;
    jitc_var_inc_ref(target_ptr);
    jitc_var_inc_ref(index);
    jitc_var_inc_ref(mask);

    uint32_t result = jitc_var_new(v, true);
    jitc_var_mark_side_effect(result);
    return target.release();
}

uint32_t jitc_coop_vec_outer_product_accum(uint32_t target_,
                                           uint32_t target_size,
                                           const MatrixDescr *descr,
                                           uint32_t a, uint32_t b) {
    JitBackend backend;
    VarType vt;
    uint32_t size;

    if (target_)
        target_ = unwrap(target_);

    {
        const Variable *v_a = jitc_var(a),
                       *v_b = jitc_var(b);

        if (!v_a->coop_vec || !v_b->coop_vec || v_a->type != v_b->type)
            jitc_raise("jit_coop_vec_outer_product_accum(): 'a' and 'b' must "
                       "be cooperative vectors of a compatible type!");

        backend = (JitBackend) v_a->backend;
        vt = (VarType) v_a->type;
        size = std::max(v_a->size, v_b->size);

        if (backend == JitBackend::CUDA) {
            if (vt != VarType::Float16)
                jitc_raise(
                    "jit_coop_vec_outer_product_accum(): this operation is "
                    "restricted to float16 precision on the CUDA/OptiX backend.");
            if (descr->layout != MatrixLayout::TrainingOptimal)
                jitc_raise("jit_coop_vec_outer_product_accum(): the matrix "
                           "must be in training-optimal layout!");
        }
    }

    Ref target = borrow(target_);
    if (!target) {
        uint64_t z = 0;
        target = steal(jitc_var_literal(backend, vt, &z, target_size, true));
    } else {
        const Variable *target_v = jitc_var(target);
        // Copy-on-Write logic. See the same line in jitc_var_scatter() for details
        if (target_v->ref_count != 2 && target_v->ref_count_stashed != 1) {
            target = steal(jitc_var_copy(target));
            target_v = jitc_var(target);
        }

        if ((VarType) target_v->type != vt)
            jitc_raise("jit_coop_vec_outer_product_accum(): source/target "
                       "buffers have an incompatible type (%s vs %s)!",
                       type_name[target_v->type], type_name[(int) vt]);
    }

    void *p = nullptr;
    target = steal(jitc_var_data(target, false, &p));
    Ref target_ptr = steal(jitc_var_pointer(backend, p, target, 1));

    Ref mask = steal(jitc_var_bool(backend, true));
    mask = steal(jitc_var_mask_apply(mask, size));

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecOuterProductAccum;
    v.type = (uint32_t) VarType::Void;
    v.size = std::max(size, jitc_var(mask)->size);
    v.backend = (uint32_t) backend;
    v.symbolic = jitc_flag(JitFlag::SymbolicScope);
    v.dep[0] = target_ptr;
    v.dep[1] = a;
    v.dep[2] = b;
    v.dep[3] = mask;
    jitc_var_inc_ref(target_ptr);
    jitc_var_inc_ref(a);
    jitc_var_inc_ref(b);
    jitc_var_inc_ref(mask);

    drjit::unique_ptr<MatrixDescr> md = new MatrixDescr(*descr);

    uint32_t result = jitc_var_new_take_ownership(v, std::move(md), true);
    jitc_var_mark_side_effect(result);
    return target.release();
}
