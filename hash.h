#pragma once

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

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


/// CRC32 hash function
uint32_t crc32(const void *ptr, size_t size);
