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
#include <drjit-core/nanostl.h>

uint32_t jitc_coop_vec_new(const uint32_t *indices, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        if (indices[i] == 0)
            jitc_raise("jit_coop_vec_new(): argument %u is uninitialized!", i);
    }

    const Variable *arg_v = jitc_var(indices[0]);
    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecNew;
    v.type = arg_v->type;
    v.size = arg_v->size;
    v.backend = arg_v->backend;
    v.array_length = size;
    v.coop_vec = true;
    v.optix = v.backend == (uint32_t) JitBackend::CUDA;

    drjit::unique_ptr<CoopVecNewData> cvid = new CoopVecNewData();
    cvid->indices.reserve(size);

    for (uint32_t i = 0; i < size; ++i) {
        uint32_t index = indices[i];
        jitc_var_inc_ref(index);
        cvid->indices.push_back(index);
    }

    uint32_t result = jitc_var_new(v, true);
    jitc_var(result)->data = cvid.get();

    jitc_var_set_callback(
        result,
        [](uint32_t, int free, void *p) {
            CoopVecNewData *ld = (CoopVecNewData *) p;
            if (free)
                delete (CoopVecNewData *) ld;
        },
        cvid.release(), true);

    return result;
}

uint32_t jitc_coop_vec_get(uint32_t vec, uint32_t offset) {
    Variable *vec_v = jitc_var(vec);
    Variable v;
    if (offset >= vec_v->array_length)
        jitc_raise("jit_coop_vec_get(): element %u is beyond the end of the "
                   "cooperative vector (of size %u)", offset, vec_v->array_length);
    v.kind = (uint32_t) VarKind::CoopVecGet;
    v.literal = offset;
    v.type = vec_v->type;
    v.size = vec_v->size;
    v.backend = vec_v->backend;
    v.dep[0] = vec;
    jitc_var_inc_ref(vec, vec_v);

    return jitc_var_new(v);
}

uint32_t jitc_coop_vec_set(uint32_t vec, uint32_t offset, uint32_t value) {
    Variable *vec_v = jitc_var(vec);
    if (!value)
        jitc_raise("jit_coop_vec_set(): 'value' is unitialized!");

    Variable *value_v = jitc_var(value);
    if (!(value_v->size == vec_v->size || vec_v->size == 1 || value_v->size == 1) ||
        value_v->type != vec_v->type)
        jitc_raise("jit_coop_vec_set(): incompatible arguments!");
    if (offset >= vec_v->array_length)
        jitc_raise("jit_coop_vec_set(): element %u is beyond the end of the "
                   "cooperative vector (of size %u)", offset, vec_v->array_length);
    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecSet;
    v.literal = offset;
    v.type = vec_v->type;
    v.size = std::max(vec_v->size, value_v->size);
    v.backend = vec_v->backend;
    v.array_length = vec_v->array_length;
    v.coop_vec = true;

    v.dep[0] = vec;
    v.dep[1] = value;
    jitc_var_inc_ref(vec, vec_v);
    jitc_var_inc_ref(value, value_v);

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

uint32_t jitc_coop_vec_binary_op(JitOp op, uint32_t a0, uint32_t a1) {
    if (!a0 || !a1)
        jitc_raise("jit_coop_vec_binary_op(): detected uninitialized inputs!");

    Variable *a0_v = jitc_var(a0),
             *a1_v = jitc_var(a1);

    if (a0_v->array_length != a1_v->array_length)
        jitc_raise("jit_coop_vec_binary_op(): the cooperative vectors have "
                   "incompatible sizes (%u and %u)!",
                   a0_v->array_length, a1_v->array_length);

    if (a0_v->type != a1_v->type)
        jitc_raise("jit_coop_vec_binary_op(): the cooperative vectors have "
                   "incompatible types (%s and %s)!",
                   type_name[a0_v->type], type_name[a1_v->type]);

    if (!(a0_v->size == a1_v->size || a1_v->size == 1 || a0_v->size == 1))
        jitc_raise(
            "jit_coop_vec_binary_op(): incompatible thread count (%u and %u)!",
            a0_v->size, a1_v->size);

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecBinaryOp;
    v.literal = (uint32_t) op;
    v.type = a0_v->type;
    v.size = std::max(a0_v->size, a1_v->size);
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
            "jit_coop_vec_ternary_op(): incompatible thread count (%u, %u, and %u)!",
            a0_v->size, a1_v->size, a2_v->size);

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
    MatrixDescr in = *in_;
    JitBackend backend;
    VarType vt;

    {
        const Variable *v = jitc_var(index);
        vt = (VarType) v->type;
        backend = (JitBackend) v->backend;
    }

    bool is_vector = in.cols == 1;
    uint32_t tsize = type_size[(uint32_t) vt];

#if defined(DRJIT_ENABLE_OPTIX)
    if (backend == JitBackend::CUDA) {
        uint32_t offset_in_bytes = in.offset * tsize;
        if (offset_in_bytes % 64 != 0)
            jitc_raise(
                "jit_coop_vec_pack(): OptiX requires input matrices "
                "to be 64-byte aligned. Encountered an input with "
                "offset %u, which is not divisible by 64.", offset_in_bytes);

        uint32_t out_align = is_vector ? 16 : 64;
        offset = (ceil_div(offset * tsize, out_align) * out_align) / tsize;
    }
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

        if (!optixCoopVecMatrixComputeSize)
            jitc_raise("jit_coop_vec_pack(): Cooperative vectors are not "
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

void jitc_coop_vec_pack(uint32_t count, uint32_t in, const MatrixDescr *in_descr,
                        uint32_t out, const MatrixDescr *out_descr) {
    void *in_p = nullptr, *out_p = nullptr;
    Ref in_data = steal(jitc_var_data(in, true, &in_p));
    Ref out_data = steal(jitc_var_data(out, true, &out_p));

    JitBackend backend;
    {
        const Variable *out_v = jitc_var(out);
        jitc_log(Debug, "jit_coop_vec_pack(): packing %u matrixes (%u bytes)",
                 count, out_v->size * type_size[out_v->type]);
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
            jitc_raise("jit_coop_vec_matvec(): 'x' vector has an incompatible size "
                       "(expected %u, got %u).", input_length, x_v->array_length);
    }

    Ref a_ptr, b_ptr;
    {
        Variable *a_v = jitc_var(A_index);
        a_vt = (VarType) a_v->type;
        a_ptr = steal(jitc_var_pointer((JitBackend) a_v->backend, a_v->data, A_index, 0));
        cvmvd->A_descr = *A_descr;

        if (backend == JitBackend::CUDA) {
            uint32_t tsize = type_size[a_v->type],
                     offset_in_bytes = A_descr->offset * tsize,
                     stride_in_bytes = A_descr->stride * tsize;
            if (offset_in_bytes % 64)
                jitc_raise("jit_coop_vec_matvec(): matrix offset (%u bytes) must be 64-byte aligned.\n", offset_in_bytes);
            if (stride_in_bytes % 16)
                jitc_raise("jit_coop_vec_matvec(): matrix stride (%u bytes) must be 16-byte aligned.\n", stride_in_bytes);
        }
    }

    if (b_index && b_descr) {
        Variable *b_v = jitc_var(b_index);
        b_ptr = steal(jitc_var_pointer((JitBackend) b_v->backend, b_v->data, b_index, 0));
        b_vt = (VarType) b_v->type;
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
        jitc_raise("jit_coop_vec_matvec(): incompatible input types "
                   "(currently, only float16 is supported on the CUDA/OptiX)!");

    Variable v;
    v.kind = (uint32_t) VarKind::CoopVecMatVec;
    v.type = (uint32_t) x_vt;
    v.size = size;
    v.backend = (uint32_t) backend;
    v.array_length = output_length;
    v.coop_vec = true;
    v.dep[0] = a_ptr;
    v.dep[1] = x_index;
    v.dep[2] = b_ptr;
    jitc_var_inc_ref(a_ptr);
    jitc_var_inc_ref(x_index);
    jitc_var_inc_ref(b_ptr);

    uint32_t result = jitc_var_new(v, true);
    jitc_var(result)->data = cvmvd.get();
    jitc_var_set_callback(
        result,
        [](uint32_t, int free, void *p) {
            CoopVecMatVecData *ld = (CoopVecMatVecData *) p;
            if (free)
                delete (CoopVecMatVecData *) ld;
        },
        cvmvd.release(), true);

    return result;
}

uint32_t jitc_coop_vec_scatter_add(uint32_t index, uint32_t target_, uint32_t offset_) {
    JitBackend backend;
    uint32_t length, size;
    {
        const Variable *v = jitc_var(index);
        length = v->array_length;
        size = std::max(v->size, jitc_var(offset_)->size);
        backend = (JitBackend) v->backend;
    }

    Ref target = borrow(target_);

    if (backend == JitBackend::LLVM) {
        Ref mask   = steal(jitc_var_bool(backend, true)),
            base   = steal(offset_);

        for (uint32_t i = 0; i < length; ++i) {
            Ref v0 = steal(jitc_coop_vec_get(index, i));
            Ref offset_i = steal(jit_var_u32(backend, i)),
                offset = steal(jit_var_add(base, offset_i));

            target = steal(jitc_var_scatter(target, v0, offset, mask, ReduceOp::Add,
                                            ReduceMode::Auto));
        }
    } else {
        void *ptr = nullptr;
        target = steal(jitc_var_data(target, true, &ptr));
        Ref ptr_v = steal(jitc_var_pointer(backend, ptr, target, 0));

        /// TODO masking

        Variable v;
        v.kind = (uint32_t) VarKind::CoopVecScatterAdd;
        v.type = (uint32_t) VarType::Void;
        v.size = size;
        v.backend = (uint32_t) backend;
        v.dep[0] = index;
        v.dep[1] = offset_;
        v.dep[2] = ptr_v;
        jitc_var_inc_ref(index);
        jitc_var_inc_ref(offset_);
        jitc_var_inc_ref(ptr_v);
    }

    return target_;
}
