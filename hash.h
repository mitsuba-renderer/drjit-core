#pragma once

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <nmmintrin.h>

inline void hash_combine(size_t& seed, size_t value) {
    /// From CityHash (https://github.com/google/cityhash)
    const size_t mult = 0x9ddfea08eb382d69ull;
    size_t a = (value ^ seed) * mult;
    a ^= (a >> 47);
    size_t b = (seed ^ a) * mult;
    b ^= (b >> 47);
    seed = b * mult;
}

struct pair_hash {
    template <typename T1, typename T2>
    size_t operator()(const std::pair<T1, T2> &x) const {
        size_t result = std::hash<T1>()(x.first);
        hash_combine(result, std::hash<T2>()(x.second));
        return result;
    }
};

inline uint32_t crc32(const uint8_t *ptr, size_t size) {
    const uint8_t *end = ptr + size;
    uint64_t state64 = 0;
    while (ptr + 8 < end) {
        state64 = _mm_crc32_u64(state64, *((uint64_t *) ptr));
        ptr += 8;
    }
    uint32_t state32 = (uint32_t) state64;
    while (ptr + 4 < end) {
        state32 = _mm_crc32_u32(state32, *((uint32_t *) ptr));
        ptr += 4;
    }
    while (ptr < end)
        state32 = _mm_crc32_u8(state32, *ptr++);
    return state32;
}
