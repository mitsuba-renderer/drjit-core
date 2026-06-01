/*
    src/metal_stage_buffer.mm -- Suballocated staging-buffer pool for
    CPU<->GPU transfers.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Compiled as Objective-C++ with ARC. Metal buffer handles are stored in the
    plain C++ ``Block``/``Pool`` structs as opaque ``void*`` that own a +1
    reference (bridged via ``__bridge_retained`` / released via
    ``__bridge_transfer``). The pool maps are keyed by the device handle's raw
    pointer identity.
*/

#if defined(DRJIT_ENABLE_METAL)

#include "metal_stage_buffer.h"
#include "internal.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// Imported after the project headers, with the obsolete Carbon
// <CarbonCore/Threads.h> suppressed (its ``typedef UInt16 ThreadState``
// collides with Dr.Jit's ``struct ThreadState``).
#define __THREADS__
#import <Metal/Metal.h>

namespace {

constexpr size_t kAlign            = 16;
constexpr size_t kUploadBlockSize  = 64 * 1024 * 1024;  // 64 MiB
constexpr size_t kDownloadBlockSize = 32 * 1024 * 1024; // 32 MiB

// Allocations larger than half a block size bypass the pool.
constexpr size_t kAdhocFraction   = 2;

// A single MTLBuffer used as a bump-allocate region. Lives until pool
// shutdown. ``buf`` owns a +1 reference (released in the Pool destructor).
struct Block {
    void        *buf       = nullptr;  // owned (+1) id<MTLBuffer>
    uint8_t     *contents  = nullptr;
    size_t       capacity  = 0;
    size_t       offset    = 0;
    std::atomic<int> outstanding{0};
};

struct Pool {
    std::mutex                  mtx;
    Block                      *active = nullptr;
    std::vector<Block *>        draining;
    std::vector<Block *>        free;
    void                       *device     = nullptr;  // borrowed id<MTLDevice>
    size_t                      block_size = 0;

    ~Pool() {
        auto release_all = [](std::vector<Block *> &v) {
            for (Block *b : v) {
                if (b->buf)
                    (void) (__bridge_transfer id<MTLBuffer>) b->buf;
                delete b;
            }
            v.clear();
        };
        if (active) {
            if (active->buf)
                (void) (__bridge_transfer id<MTLBuffer>) active->buf;
            delete active;
            active = nullptr;
        }
        release_all(draining);
        release_all(free);
    }
};

// Token layout: low bit indicates an ad-hoc allocation. For pooled
// allocations the token is the ``Block *``; for ad-hoc, it's an owned (+1)
// MTLBuffer pointer with the low bit set.
constexpr uintptr_t kAdhocTag = 1;

inline void *make_pooled_token(Block *b) {
    return (void *) b;
}
inline void *make_adhoc_token(id<MTLBuffer> buf) {
    void *p = (__bridge_retained void *) buf; // transfer +1 into the token
    return (void *) ((uintptr_t) p | kAdhocTag);
}
inline bool token_is_adhoc(void *t) {
    return ((uintptr_t) t & kAdhocTag) != 0;
}
inline Block *token_to_block(void *t) {
    return (Block *) t;
}
inline void *token_to_adhoc_ptr(void *t) {
    return (void *) ((uintptr_t) t & ~kAdhocTag);
}

std::mutex                                              g_mtx;
std::unordered_map<void *, std::unique_ptr<Pool>>       g_upload_pools;
std::unordered_map<void *, std::unique_ptr<Pool>>       g_download_pools;

Pool &get_pool(std::unordered_map<void *, std::unique_ptr<Pool>> &map,
               void *device, size_t block_size) {
    auto &slot = map[device];
    if (!slot) {
        slot              = std::make_unique<Pool>();
        slot->device      = device;
        slot->block_size  = block_size;
    }
    return *slot;
}

Block *new_block(void *device, size_t capacity) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>) device;
    id<MTLBuffer> buf = [dev newBufferWithLength:capacity
                                         options:MTLResourceStorageModeShared];
    if (!buf)
        jitc_fail("metal_stage_buffer: failed to allocate %zu-byte staging "
                  "block", capacity);
    Block *b    = new Block();
    b->buf      = (__bridge_retained void *) buf;
    b->contents = (uint8_t *) [buf contents];
    b->capacity = capacity;
    return b;
}

// Pool mutex must be held by the caller.
Block *acquire_block_locked(Pool &p, size_t aligned_size) {
    if (p.active && p.active->offset + aligned_size <= p.active->capacity)
        return p.active;

    if (p.active) {
        // The active block can't fit this request. Either reset it (if no
        // outstanding allocations from it) or move it to the draining list.
        if (p.active->outstanding.load(std::memory_order_acquire) == 0) {
            p.active->offset = 0;
            if (aligned_size <= p.active->capacity)
                return p.active;
            p.draining.push_back(p.active);
        } else {
            p.draining.push_back(p.active);
        }
        p.active = nullptr;
    }

    // Reclaim a draining block whose outstanding count has dropped to 0.
    for (auto it = p.draining.begin(); it != p.draining.end(); ++it) {
        if ((*it)->outstanding.load(std::memory_order_acquire) == 0 &&
            aligned_size <= (*it)->capacity) {
            p.active = *it;
            p.active->offset = 0;
            p.draining.erase(it);
            return p.active;
        }
    }

    // Reuse a fully-free block.
    for (auto it = p.free.begin(); it != p.free.end(); ++it) {
        if (aligned_size <= (*it)->capacity) {
            p.active = *it;
            p.active->offset = 0;
            p.free.erase(it);
            return p.active;
        }
    }

    // Allocate a fresh block.
    size_t cap = std::max(p.block_size, aligned_size);
    p.active   = new_block(p.device, cap);
    return p.active;
}

void *acquire_impl(
    std::unordered_map<void *, std::unique_ptr<Pool>> &pool_map,
    size_t default_block_size, ThreadState *ts, size_t size,
    void **mtl_buffer_out, size_t *offset_out, void **release_token_out) {
  @autoreleasepool {
    void *device = ts->metal_device;
    if (!device)
        jitc_fail("metal_stage_buffer: ThreadState has no metal_device.");

    size_t aligned = (size + kAlign - 1) & ~(kAlign - 1);

    // Big requests bypass the pool to avoid pinning a huge block.
    if (aligned > default_block_size / kAdhocFraction) {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) device;
        id<MTLBuffer> buf =
            [dev newBufferWithLength:aligned
                             options:MTLResourceStorageModeShared];
        if (!buf)
            jitc_fail("metal_stage_buffer: ad-hoc alloc of %zu bytes failed.",
                      aligned);
        if (mtl_buffer_out)    *mtl_buffer_out    = (__bridge void *) buf;
        if (offset_out)        *offset_out        = 0;
        if (release_token_out) *release_token_out = make_adhoc_token(buf);
        return [buf contents];
    }

    Pool *pool = nullptr;
    {
        std::lock_guard<std::mutex> g(g_mtx);
        pool = &get_pool(pool_map, device, default_block_size);
    }

    Block *b   = nullptr;
    size_t off = 0;
    {
        std::lock_guard<std::mutex> g(pool->mtx);
        b   = acquire_block_locked(*pool, aligned);
        off = b->offset;
        b->offset += aligned;
        b->outstanding.fetch_add(1, std::memory_order_acq_rel);
    }

    if (mtl_buffer_out)    *mtl_buffer_out    = b->buf;
    if (offset_out)        *offset_out        = off;
    if (release_token_out) *release_token_out = make_pooled_token(b);
    return b->contents + off;
  }
}

} // anonymous namespace

void *jitc_metal_stage_acquire_upload(ThreadState *ts, size_t size,
                                      void **mtl_buffer_out,
                                      size_t *offset_out,
                                      void **release_token_out) {
    return acquire_impl(g_upload_pools, kUploadBlockSize, ts, size,
                        mtl_buffer_out, offset_out, release_token_out);
}

void *jitc_metal_stage_acquire_download(ThreadState *ts, size_t size,
                                        void **mtl_buffer_out,
                                        size_t *offset_out,
                                        void **release_token_out) {
    return acquire_impl(g_download_pools, kDownloadBlockSize, ts, size,
                        mtl_buffer_out, offset_out, release_token_out);
}

void jitc_metal_stage_release(void *release_token) {
    if (!release_token)
        return;
    if (token_is_adhoc(release_token)) {
        // Release the +1 reference owned by the token.
        (void) (__bridge_transfer id<MTLBuffer>)
            token_to_adhoc_ptr(release_token);
        return;
    }
    Block *b = token_to_block(release_token);
    b->outstanding.fetch_sub(1, std::memory_order_acq_rel);
    // Pool reset of `offset` happens lazily on the next acquire path that
    // observes outstanding == 0.
}

void jitc_metal_stage_shutdown() {
    @autoreleasepool {
        std::lock_guard<std::mutex> g(g_mtx);
        g_upload_pools.clear();
        g_download_pools.clear();
    }
}

#endif // defined(DRJIT_ENABLE_METAL)
