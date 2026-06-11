/*
    src/metal_launch.h -- Shared launch helpers for Metal utility kernels

    Helpers to resolve Dr.Jit pointers to MTLBuffers/GPU addresses and to
    encode compute dispatches with declarative resource-residency lists.
    Objective-C++ only: include this header *after* ``#import <Metal/Metal.h>``
    (with the usual ``__THREADS__`` Carbon workaround in place).

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include "metal.h"
#include "metal_ts.h"
#include "malloc.h"

#include <algorithm>
#include <initializer_list>

/// Resolve a Dr.Jit Metal pointer and return both its id<MTLBuffer> pointer
/// and the associated GPU memory address. Handles device pointers (private
/// storage, where the Dr.Jit pointer equals the GPU address) and shared
/// allocations (where it is the CPU-side ``contents`` pointer) alike.
inline id<MTLBuffer> metal_resolve(const void *ptr, uint64_t *addr_out) {
    size_t off = 0;
    id<MTLBuffer> buf =
        (__bridge id<MTLBuffer>) jitc_metal_find_buffer((void *) ptr, &off);
    *addr_out = (uint64_t) [buf gpuAddress] + off;
    return buf;
}

/// Precompiled utility-kernel pipeline (see MetalKernel) of ``ts``'s device
inline id<MTLComputePipelineState> metal_pipeline(const MetalThreadState *ts,
                                                  MetalKernel kernel) {
    return (__bridge id<MTLComputePipelineState>)
        state.metal_devices[ts->device].pipelines[(uint32_t) kernel];
}

/// A buffer accessed by a dispatched kernel, with its intended usage for
/// Metal's hazard tracking. Entries with a nil buffer are skipped.
struct MetalUse {
    id<MTLBuffer> buf;
    MTLResourceUsage usage;
};

/// Bind ``pso``, the inline ``params`` constant buffer (slot 0), and the given
/// resources on the current compute encoder; returns the encoder for dispatch.
template <typename Params>
id<MTLComputeCommandEncoder>
metal_encode(MetalThreadState *ts, id<MTLComputePipelineState> pso,
             const Params &params, std::initializer_list<MetalUse> uses) {
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>) ts->ensure_compute_encoder();
    [enc setComputePipelineState:pso];
    [enc setBytes:&params length:sizeof(Params) atIndex:0];
    for (const MetalUse &u : uses) {
        if (u.buf)
            [enc useResource:u.buf usage:u.usage];
    }
    return enc;
}

/// Encode a 1-D dispatch with one thread per element
template <typename Params>
void metal_dispatch_threads(MetalThreadState *ts,
                            id<MTLComputePipelineState> pso,
                            const Params &params,
                            std::initializer_list<MetalUse> uses,
                            uint32_t n_threads) {
    id<MTLComputeCommandEncoder> enc = metal_encode(ts, pso, params, uses);
    uint32_t tg = std::min((uint32_t) pso.maxTotalThreadsPerThreadgroup,
                           round_pow2(n_threads));
    [enc dispatchThreads:MTLSizeMake(n_threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

/// Encode a dispatch of ``grid`` threadgroups of ``tg_width`` threads,
/// optionally with ``tg_mem`` bytes of threadgroup memory in slot 0
template <typename Params>
void metal_dispatch_groups(MetalThreadState *ts,
                           id<MTLComputePipelineState> pso,
                           const Params &params,
                           std::initializer_list<MetalUse> uses,
                           MTLSize grid, uint32_t tg_width,
                           uint32_t tg_mem = 0) {
    id<MTLComputeCommandEncoder> enc = metal_encode(ts, pso, params, uses);
    if (tg_mem)
        [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
    [enc dispatchThreadgroups:grid
        threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
}
