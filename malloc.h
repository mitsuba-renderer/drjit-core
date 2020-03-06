#pragma once

#include "api.h"
#include "hash.h"

/// Data structure characterizing a memory allocation
struct AllocInfo {
    AllocType type;
    uint32_t device;
    size_t size;

    AllocInfo() { }

    AllocInfo(AllocType type, uint32_t device, size_t size)
        : type(type), device(device), size(size) { }

    bool operator==(const AllocInfo &at) const {
        return type == at.type && device == at.device &&
               size == at.size;
    }

    bool operator!=(const AllocInfo &at) const {
        return type != at.type || device != at.device ||
               size != at.size;
    }
};

/// Custom hasher for \ref AllocInfo
struct AllocInfoHasher {
    size_t operator()(const AllocInfo &at) const {
        size_t result = std::hash<size_t>()(at.size);
        hash_combine(result, std::hash<uint32_t>()(at.device));
        hash_combine(result, std::hash<uint32_t>()((uint32_t) at.type));
        return result;
    }
};

using AllocInfoMap = tsl::robin_map<AllocInfo, std::vector<void *>, AllocInfoHasher>;

/// Allocate the given flavor of memory
void *jit_malloc(AllocType type, size_t size) __attribute__((malloc));

/// Release the given pointer
void jit_free(void *ptr);

/// Schedule a function that will reclaim pending jit_free()s
void jit_free_flush();

/// Release all unused memory to the GPU / OS
void jit_malloc_trim();

/// Shut down the memory allocator (calls \ref jit_malloc_trim() and reports leaks)
void jit_malloc_shutdown();
