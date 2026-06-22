#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/// Portable trailing zero count
inline uint32_t tzcnt_u64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long y;
    _BitScanForward64(&y, x);
    return (uint32_t) y;
#else
    return (uint32_t) __builtin_ctzll(x);
#endif
}

/// Free-slot allocator based on a packed bit representation. Allocation returns
/// the lowest set bit, which improves the overall cache and branch prediction
/// behavior of Dr.Jit. A cursor marks the lowest word that may hold a free
/// bit so that the scan can resume from there instead of from the start.
struct FreeSlots {
    std::vector<uint64_t> words; //< A set bit marks a free slot
    uint32_t cursor = 0;         //< Lowest word that may contain a free bit
    size_t count = 0;            //< Number of free slots

    bool empty() const { return count == 0; }
    size_t size() const { return count; }

    void push(uint32_t i) {
        uint32_t w = i / 64;
        if (w >= words.size())
            words.resize(w + 1, 0);
        words[w] |= (uint64_t) 1 << (i % 64);
        cursor = std::min(cursor, w);
        count++;
    }

    uint32_t pop() {
        while (words[cursor] == 0)
            cursor++;
        uint64_t bits = words[cursor];
        words[cursor] = bits & (bits - 1); // clear lowest set bit
        count--;
        return cursor * 64 + tzcnt_u64(bits);
    }
};
