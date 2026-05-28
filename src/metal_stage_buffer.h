/*
    src/metal_stage_buffer.h -- Suballocated staging-buffer pool for
    CPU<->GPU transfers.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Pool design (LuisaCompute-inspired). Per device, two pools — one for
    upload (CPU → GPU) staging and one for download (GPU → CPU) staging.
    Each pool owns a small set of large StorageModeShared MTL::Buffers.
    Allocations bump a per-block offset; once the block fills, it moves to a
    "draining" list and the next allocation either picks a recycled block or
    creates a new one. A block becomes recyclable once every allocation it
    handed out has been released (signalled via the caller's command-buffer
    completion handler).

    The acquire path returns a CPU-visible pointer plus the underlying
    (MTL::Buffer*, offset) tuple — caller binds the buffer slice to a blit
    encoder. The acquire path also returns an opaque ``release_token``;
    once the GPU is done using the slice (typically in the command-buffer
    completion handler), the caller MUST invoke
    ``jitc_metal_stage_release(release_token)`` so the pool can recycle.

    Allocations larger than the configured block size fall back to ad-hoc
    fresh MTL::Buffers; the release path handles both flavours via the
    token's tag bit.
*/

#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_METAL)

#include <cstddef>

struct ThreadState;

/// Acquire a contiguous block of upload-staging memory of at least ``size``
/// bytes. Returns a CPU-visible pointer; stores the underlying
/// ``MTL::Buffer*`` and offset in the out-params, plus an opaque release
/// token to be passed to ``jitc_metal_stage_release`` when the GPU is done.
extern void *jitc_metal_stage_acquire_upload(ThreadState *ts, size_t size,
                                             void **mtl_buffer_out,
                                             size_t *offset_out,
                                             void **release_token_out);

/// Counterpart for download (GPU → CPU) staging.
extern void *jitc_metal_stage_acquire_download(ThreadState *ts, size_t size,
                                               void **mtl_buffer_out,
                                               size_t *offset_out,
                                               void **release_token_out);

/// Release a previously acquired allocation. Safe to call from a Metal
/// completion handler thread.
extern void jitc_metal_stage_release(void *release_token);

/// Release every block held by every pool. Called from jitc_metal_shutdown.
extern void jitc_metal_stage_shutdown();

#endif // defined(DRJIT_ENABLE_METAL)
