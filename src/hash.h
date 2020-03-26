#pragma once

#include <tsl/robin_map.h>
#include <string.h>

extern "C" {
    extern unsigned long long XXH64(const void *ptr, size_t size,
                                    unsigned long long seed);
}

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

inline size_t hash(const void *ptr, size_t size, size_t seed = 0) {
    return (size_t) XXH64(ptr, size, (unsigned long long) seed);
}

inline size_t hash_str(const char *str, size_t seed = 0) {
    return hash(str, strlen(str), seed);
}

inline size_t hash_kernel(const char *str, size_t seed = 0) {
    const char *offset = strchr(str, '{');
    return hash_str(offset ? offset : str, seed);
}
