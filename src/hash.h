/*
    src/hash.h -- Functionality for hashing kernels and other data (via xxHash)

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined (_MSC_VER)
#  pragma warning (push)
#  pragma warning (disable:26451)
#endif

#include <tsl/robin_map.h>
#include <drjit-core/hash.h>
#include <xxh3.h>

#if defined (_MSC_VER)
#  pragma warning (pop)
#endif

#include <string.h>

#if !defined(likely)
#  if !defined(_MSC_VER)
#    define likely(x)   __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#  else
#    define unlikely(x) x
#    define likely(x) x
#  endif
#endif

#if defined(__GNUC__)
    __attribute__((noreturn, __format__(__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
extern void jitc_fail(const char* fmt, ...) noexcept;

struct XXH128Cmp {
    size_t operator()(const XXH128_hash_t &h1,
                      const XXH128_hash_t &h2) const {
        return std::tie(h1.high64, h1.low64) < std::tie(h2.high64, h2.low64);
    }
};

inline void hash_combine(size_t& seed, size_t value) {
    /// From CityHash (https://github.com/google/cityhash)
    const size_t mult = 0x9ddfea08eb382d69ull;
    size_t a = (value ^ seed) * mult;
    a ^= (a >> 47);
    size_t b = (seed ^ a) * mult;
    b ^= (b >> 47);
    seed = b * mult;
}

inline size_t hash(const void *ptr, size_t size, size_t seed) {
    return (size_t) XXH3_64bits_withSeed(ptr, size, (unsigned long long) seed);
}

inline size_t hash(const void *ptr, size_t size) {
    return (size_t) XXH3_64bits(ptr, size);
}

inline size_t hash_str(const char *str, size_t seed) {
    return hash(str, strlen(str), seed);
}

inline size_t hash_str(const char *str) {
    return hash(str, strlen(str));
}

inline XXH128_hash_t hash_kernel(const char *str) {
    const char *offset = strstr(str, "body:");
    if (unlikely(!offset)) {
        offset = strchr(str, '{');
        if (unlikely(!offset))
            jitc_fail("hash_kernel(): invalid input!");
    }

    return XXH128(offset, strlen(offset), 0);
}
