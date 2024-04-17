/*
    drjit-core/hash.h -- Hash functions for small integers from MurmurHash

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/
#pragma once

#include <cstdint>
#include <cstdlib>

// fmix32 from MurmurHash by Austin Appleby (public domain)
inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

// fmix64 from MurmurHash by Austin Appleby (public domain)
inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= (uint64_t) 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= (uint64_t) 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}

struct UInt32Hasher {
    size_t operator()(uint32_t v) const {
        return (size_t) fmix32(v);
    }
};

struct UInt64Hasher {
    size_t operator()(uint64_t v) const {
        return (size_t) fmix64(v);
    }
};

struct PointerHasher {
    size_t operator()(const void *p) const {
        if constexpr (sizeof(void *) == 4)
            return (size_t) fmix32((uint32_t) (uintptr_t) p);
        else
            return (size_t) fmix64((uint64_t) (uintptr_t) p);
    }
};
