#pragma once

#include "malloc.h"
#include "cuda.h"
#include <mutex>
#include <string.h>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#if defined(ENOKI_CUDA)
struct Stream {
    /// Enoki device index associated with this stream (*not* the CUDA device ID)
    uint32_t device = 0;

    /// Index of this stream
    uint32_t stream = 0;

    /// Associated CUDA stream handle
    cudaStream_t handle = nullptr;

    /// Memory regions that will be unused once the running kernel finishes
    AllocInfoMap alloc_pending;
};

using StreamMap = tsl::robin_map<std::pair<uint32_t, uint32_t>, Stream *, pair_hash>;
#endif

enum EnokiType { Invalid = 0, Int8, UInt8, Int16, UInt16,
                 Int32, UInt32, Int64, UInt64, Float16,
                 Float32, Float64, Bool, Pointer };

/// Central variable data structure, which represents an assignment in SSA form
struct Variable {
    /// Data type of this variable
    uint32_t type = (uint32_t) EnokiType::Invalid;

    /// PTX instruction to compute it
    char *cmd = nullptr;

    /// Associated label (for debugging)
    char *label = nullptr;

    /// Number of entries
    size_t size = 0;

    /// Pointer to device memory
    void *data = nullptr;

    /// External (i.e. by Enoki) reference count
    uint32_t ref_count_ext = 0;

    /// Internal (i.e. within the PTX instruction stream) reference count
    uint32_t ref_count_int = 0;

    /// Dependencies of this instruction
    uint32_t dep[3] { 0 };

    /// Extra dependency (which is not directly used in arithmetic, e.g. scatter/gather)
    uint32_t extra_dep = 0;

    /// Does the instruction have side effects (e.g. 'scatter')
    bool side_effect = false;

    /// A variable is 'dirty' if there are pending scatter operations to it
    bool dirty = false;

    /// Free 'data' after this variable is no longer referenced?
    bool free_variable = true;

    /// Optimization: is this a direct pointer (rather than an array which stores a pointer?)
    bool direct_pointer = false;

    /// Size of the (heuristic for instruction scheduling)
    uint32_t subtree_size = 0;

    Variable() = default;
    ~Variable() {
        if (free_variable && data)
            jit_free(data);
        free(cmd);
        free(label);
    }
};

/// Records the full JIT compiler state
struct State {
    /// Must be held to access members
    std::mutex mutex;

    /// Indicates whether the state is initialized by \ref jit_init()
    bool initialized = false;

    /// Number of available devices and their CUDA IDs
    std::vector<uint32_t> devices;

#if defined(ENOKI_CUDA)
    /// Maps Enoki (device index, stream index) pairs to a Stream data structure
    StreamMap streams;
#endif

    /// Map of currently allocated memory regions
    tsl::robin_map<void *, AllocInfo> alloc_used;

    /// Map of currently unused memory regions
    AllocInfoMap alloc_free;

    /// Keep track of current memory usage and a maximum watermark
    size_t alloc_usage    [(int) AllocType::Count] { 0 },
           alloc_watermark[(int) AllocType::Count] { 0 };

    /// Stores the mapping from variable indices to variables
    tsl::robin_map<uint32_t, Variable> variables;

    /// Stores the mapping from pointer addresses to variable indices
    tsl::robin_map<const void *, uint32_t> ptr_map;

    /// Enumerates "live" (externally referenced) variables and statements with side effects
    tsl::robin_set<uint32_t> live;

    /// Current variable index
    uint32_t variable_index = 0;
};

using lock_guard = std::lock_guard<std::mutex>;

class unlock_guard {
public:
    unlock_guard(std::mutex &mutex) : m_mutex(mutex) { m_mutex.unlock(); }
    ~unlock_guard() { m_mutex.lock(); }

    unlock_guard(const unlock_guard &) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;

private:
    std::mutex &m_mutex;
};

/// Global state record shared by all threads
#if defined(ENOKI_CUDA)
  extern __thread Stream *active_stream;
#endif

extern State state;

void jit_init();
void jit_shutdown();
void jit_set_context(uint32_t device, uint32_t stream);
void jit_device_sync();
