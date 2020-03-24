#pragma once

#include <tsl/robin_map.h>
#include <string.h>

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
extern uint32_t crc32(uint32_t state, const void *ptr, size_t size);
extern uint32_t crc32_64(uint32_t state, const uint64_t *ptr, size_t size);
extern uint32_t crc32_str(uint32_t state, const char *str);

extern uint32_t hash_kernel(const char *str);
