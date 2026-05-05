/*
    src/metal_ts.h -- ThreadState specialization for the Metal backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_METAL)

#include <cstring>
#include "log.h"
#include "var.h"

struct MetalThreadState : ThreadState {
    /// Lazily initialized in jitc_init_thread_state(). The MTL::CommandQueue*,
    /// MTL::Device*, and MTL::SharedEvent* are stored in the ThreadStateBase
    /// fields ``metal_queue`` / ``metal_device`` / ``metal_event``.
    MetalThreadState() = default;
    ~MetalThreadState();

    /// GPU addresses of buffers referenced by vcall data sections that need
    /// useResource() calls at kernel launch time.
    std::vector<void *> metal_call_resources;

    void barrier() override;

    Task *launch(Kernel kernel, KernelKey *key, XXH128_hash_t hash,
                 uint32_t size, std::vector<void *> *kernel_params,
                 const std::vector<uint32_t> *kernel_param_ids,
                 KernelHistoryEntry *kernel_history_entry) override;

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override;

    /// Reduce elements within blocks
    void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                      uint32_t block_size, const void *in, void *out) override;

    /// Implements various kinds of prefix reductions
    void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, bool exclusive, bool reverse,
                             const void *in, void *out) override;

    /// Compute a dot product of two equal-sized arrays
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override;

    /// Row-major GEMM. Implemented via Metal Performance Shaders
    /// (MPSMatrixMultiplication) on capable devices, with a fall-back
    /// hand-written kernel for devices/types that MPS does not support.
    void batched_gemm(VarType vt, bool At, bool Bt,
                      uint32_t M, uint32_t N, uint32_t K,
                      const GemmBatch *batch,
                      const void *A, const void *B, void *C) override;

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override;

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t block_mkperm(const uint32_t *values, uint32_t size,
                          uint32_t block_size, uint32_t bucket_count,
                          uint32_t *perm, uint32_t *offsets) override;

    /// Synchronous copy (blocks the calling thread)
    void memcpy(void *dst, const void *src, size_t size) override;

    /// Asynchronous copy via the Metal command queue
    void memcpy_async(void *dst, const void *src, size_t size) override;

    /// Asynchronously update a single element in memory. Metal's unified
    /// memory model lets us do this directly via a CPU write when the buffer
    /// is in shared mode, otherwise we route through a small staging copy.
    void poke(void *dst, const void *src, uint32_t size) override;

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override;

    /// Enqueue a host callback to fire when the current command buffer
    /// completes. Implemented via ``MTL::CommandBuffer::addCompletedHandler``.
    void enqueue_host_func(void (*callback)(void *), void *payload) override;

    /// Reduce a previously expanded (scatter-add) variable. Mirrors
    /// ``LLVMThreadState::reduce_expanded`` (llvm_ts.cpp:1033). The buffer
    /// at ``data`` holds ``exp`` consecutive copies of ``size`` elements;
    /// this collapses them into the first copy in-place.
    void reduce_expanded(VarType vt, ReduceOp op, void *data,
                         uint32_t exp, uint32_t size) override;

    /// Pack matrices/vectors for the cooperative vector API.
    ///
    /// Metal does not have hardware-specific "training-optimal" or
    /// "inference-optimal" matrix layouts (those are an OptiX feature
    /// driven by NVIDIA tensor-core data formats). All layouts collapse
    /// to plain row-major on Metal, so this function performs a pure
    /// data shuffle: it copies ``count`` matrices/vectors from their
    /// source positions described by ``in_d`` to destination positions
    /// described by ``out_d``, handling row-stride differences.
    ///
    /// The ``in``/``out`` arguments are GPU virtual addresses (returned
    /// by ``jitc_var_data``) — we route each copy through the existing
    /// ``memcpy_async`` path which knows how to translate them to
    /// ``MTL::Buffer*`` and dispatch a blit encoder. Logically mirrors
    /// LLVMThreadState::coop_vec_pack (llvm_ts.cpp:1058).
    void coop_vec_pack(uint32_t count, const void *in_, const MatrixDescr *in_d,
                       void *out_, const MatrixDescr *out_d) override {
        const uint8_t *in = (const uint8_t *) in_;
        uint8_t *out = (uint8_t *) out_;

        for (uint32_t i = 0; i < count; ++i) {
            const MatrixDescr &id = in_d[i], &od = out_d[i];
            uint32_t tsize = type_size[(int) id.dtype];

            if (id.stride == id.cols && od.stride == od.cols) {
                // Both densely packed: one large blit copy.
                memcpy_async(out + od.offset * tsize,
                             in  + id.offset * tsize,
                             (size_t) id.size * tsize);
            } else {
                // Strided source or destination: row-by-row copies.
                for (uint32_t j = 0; j < id.rows; ++j)
                    memcpy_async(
                        out + ((size_t) od.offset + (size_t) j * od.stride) * tsize,
                        in  + ((size_t) id.offset + (size_t) j * id.stride) * tsize,
                        (size_t) id.cols * tsize);
            }
        }
    }
};

#endif // defined(DRJIT_ENABLE_METAL)
