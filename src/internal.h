/*
    src/internal.h -- Central data structure definitions

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "malloc.h"
#include "cuda_api.h"
#include "llvm_api.h"
#include "alloc.h"
#include "io.h"
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string.h>
#include <inttypes.h>

#if defined(ENOKI_ENABLE_TBB)
#  include <tbb/spin_mutex.h>
#endif

namespace tbb { class task; };

#if defined(ENOKI_ENABLE_TBB)
  using FastMutex = tbb::spin_mutex;
#else
  using FastMutex = std::mutex;
#endif

static constexpr LogLevel Disable = LogLevel::Disable;
static constexpr LogLevel Error   = LogLevel::Error;
static constexpr LogLevel Warn    = LogLevel::Warn;
static constexpr LogLevel Info    = LogLevel::Info;
static constexpr LogLevel Debug   = LogLevel::Debug;
static constexpr LogLevel Trace   = LogLevel::Trace;

#define ENOKI_PTR "<0x%" PRIxPTR ">"

#if !defined(likely)
#  if !defined(_MSC_VER)
#    define likely(x)   __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#  else
#    define unlikely(x) x
#    define likely(x) x
#  endif
#endif

/// Caches basic information about a CUDA device
struct Device {
    // CUDA device context
    CUcontext context;

    /// CUDA device ID
    int id;

    /// Device compute capability (major * 10 + minor)
    int compute_capability;

    /// Number of SMs
    uint32_t num_sm;

    /// Max. bytes of shared memory per SM
    uint32_t shared_memory_bytes;

    /// Compute a good configuration for a grid-stride loop
    void get_launch_config(uint32_t *blocks_out, uint32_t *threads_out,
                           uint32_t size, uint32_t max_threads = 1024,
                           uint32_t max_blocks_per_sm = 4) const {
        uint32_t blocks_avail  = (size + max_threads - 1) / max_threads;

        uint32_t blocks;
        if (blocks_avail < num_sm) {
            // Not enough work for 1 full wave
            blocks = blocks_avail;
        } else {
            // Don't produce more than 4 blocks per SM
            uint32_t blocks_per_sm = blocks_avail / num_sm;
            if (blocks_per_sm > max_blocks_per_sm)
                blocks_per_sm = max_blocks_per_sm;
            blocks = blocks_per_sm * num_sm;
        }

        uint32_t threads = max_threads;
        if (blocks <= 1 && size < max_threads)
            threads = size;

        if (blocks_out)
            *blocks_out  = blocks;

        if (threads_out)
            *threads_out = threads;
    }
};

/// Keeps track of asynchronous deallocations via jit_free()
struct ReleaseChain {
    AllocInfoMap entries;
    /// Pointer to next linked list entry
    ReleaseChain *next = nullptr;
};

/// Represents a single stream of a parallel comunication
struct Stream {
    /// Is this a CUDA stream?
    bool cuda = false;

    /// Enoki device index associated with this stream (*not* the CUDA device ID)
    uint32_t device = 0;

    /// Index of this stream
    uint32_t stream = 0;

    /**
     * Memory regions that were freed via jit_free(), but which might still be
     * used by a currently running kernel. They will be safe to re-use once the
     * currently running kernel has finished.
     */
    ReleaseChain *release_chain = nullptr;

    /**
     * Keeps track of variables that have to be computed when jit_eval() is
     * called, in particular: externally referenced variables and statements
     * with side effects.
     */
    std::vector<uint32_t> todo;

    /// ---------------------------- LLVM-specific ----------------------------

#if defined(ENOKI_ENABLE_TBB)
    /// Mutex protecting 'tbb_task_queue'
    std::mutex tbb_task_queue_mutex;

    /// Per-stream task queue that will be processed in FIFO order
    std::deque<tbb::task *> tbb_task_queue;

    /// Kernel task that will receive queued computation
    tbb::task *tbb_kernel_task = nullptr;

    /// Root task of the entire stream, for synchronization purposes
    tbb::task *tbb_task_root = nullptr;
#endif

    /// ---------------------------- CUDA-specific ----------------------------

    /// Associated CUDA stream handle
    CUstream handle = nullptr;

    /// A CUDA event for synchronization purposes
    CUevent event = nullptr;
};

enum ArgType {
    Register,
    Input,
    Output
};

#pragma pack(push, 1)

/// Central variable data structure, which represents an assignment in SSA form
struct Variable {
    /// External reference count (by application using Enoki)
    uint32_t ref_count_ext;

    /// Internal reference count (dependencies within computation graph)
    uint32_t ref_count_int;

    /// Dependencies of this instruction
    uint32_t dep[4];

    /// Intermediate language statement
    char *stmt;

    /// Pointer to device memory
    void *data;

    /// Number of entries
    uint32_t size;

    /// Size of the instruction subtree (heuristic for instruction scheduling)
    uint32_t tsize;

    /// Register index (temporarily used during jit_eval())
    uint32_t reg_index : 24;

    /// Argument index (temporarily used during jit_eval())
    uint32_t arg_index : 16;

    /// Data type of this variable
    uint32_t type : 4;

    /// Argument type (register: 0, input: 1, output: 2)
    uint32_t arg_type : 2;

    /// Is this variable registered with the CUDA backend?
    uint32_t cuda : 1;

    /// Don't deallocate 'data' when this variable is destructed?
    uint32_t retain_data : 1;

    /// Free the 'stmt' variables at destruction time?
    uint32_t free_stmt : 1;

    /// Was this variable labeled?
    uint32_t has_label : 1;

    /// Is this a scatter operation?
    uint32_t scatter : 1;

    /// Are there pending scatter operations to this variable?
    uint32_t pending_scatter : 1;

    /// Optimization: is this a direct pointer (rather than an array which stores a pointer?)
    uint32_t direct_pointer : 1;

    /// Do the variable contents have irregular alignment? (e.g. due to jit_var_map())
    uint32_t unaligned : 1;

    /// Is this variable marked as an output? (temporarily used during jit_eval())
    uint32_t output_flag : 1;

    /// Are we currently caching the result of a jitc_vcall()?
    uint32_t vcall_cached : 1;

    Variable() {
        memset(this, 0, sizeof(Variable));
    }
};

/// Abbreviated version of the Variable data structure
struct VariableKey {
    char *stmt;
    uint32_t size;
    uint32_t dep[4];
    uint16_t type;
    uint16_t flags;

    VariableKey(const Variable &v)
        : stmt(v.stmt), size(v.size), dep{ v.dep[0], v.dep[1], v.dep[2], v.dep[3] },
          type((uint16_t) v.type),
          flags((v.free_stmt ? 1 : 0) + (v.cuda ? 2 : 0)) { }

    bool operator==(const VariableKey &v) const {
        return strcmp(stmt, v.stmt) == 0 && size == v.size &&
               dep[0] == v.dep[0] && dep[1] == v.dep[1] &&
               dep[2] == v.dep[2] && dep[3] == v.dep[3] &&
               type == v.type && flags == v.flags;
    }
};

#pragma pack(pop)

/// Helper class to hash VariableKey instances
struct VariableKeyHasher {
    size_t operator()(const VariableKey &k) const {
        size_t state;
        if (unlikely(k.flags & 1)) {
            // Dynamically allocated string, hash its contents
            state = hash_str(k.stmt);
            state = hash(&k.size, sizeof(VariableKey) - sizeof(char *), state);
        } else {
            // Statically allocated string, hash its address
            state = hash(&k, sizeof(VariableKey));
        }
        return state;
    }
};

/// Cache data structure for common subexpression elimination
using CSECache =
    tsl::robin_map<VariableKey, uint32_t, VariableKeyHasher,
                   std::equal_to<VariableKey>,
                   std::allocator<std::pair<VariableKey, uint32_t>>,
                   /* StoreHash = */ true>;

/// Maps from variable ID to a Variable instance
using VariableMap =
    tsl::robin_map<uint32_t, Variable, std::hash<uint32_t>,
                   std::equal_to<uint32_t>,
                   aligned_allocator<std::pair<uint32_t, Variable>, 64>,
                   /* StoreHash = */ false>;

/// Key data structure for kernel source code & device ID
struct KernelKey {
    char *str = nullptr;
    int device = 0;

    KernelKey(char *str, int device) : str(str), device(device) { }

    bool operator==(const KernelKey &k) const {
        return strcmp(k.str, str) == 0 && device == k.device;
    }
};

/// Helper class to hash KernelKey instances
struct KernelHash {
    size_t operator()(const KernelKey &k) const {
        return compute_hash(hash_kernel(k.str), k.device);
    }

    static size_t compute_hash(size_t kernel_hash, int device) {
        size_t hash = kernel_hash;
        hash_combine(hash, size_t(device) + 1);
        return hash;
    }
};

/// Data structure, which maps from kernel source code to compiled kernels
using KernelCache =
    tsl::robin_map<KernelKey, Kernel, KernelHash, std::equal_to<KernelKey>,
                   std::allocator<std::pair<KernelKey, Kernel>>,
                   /* StoreHash = */ true>;

// Key associated with a pointer registerered in Enoki's pointer registry
struct RegistryKey {
    const char *domain;
    uint32_t id;
    RegistryKey(const char *domain, uint32_t id) : domain(domain), id(id) { }

    bool operator==(const RegistryKey &k) const {
        return id == k.id && strcmp(domain, k.domain) == 0;
    }
};

/// Helper class to hash RegistryKey instances
struct RegistryKeyHasher {
    size_t operator()(const RegistryKey &k) const {
        return hash_str(k.domain, k.id);
    }
};

using RegistryFwdMap = tsl::robin_map<RegistryKey, void *, RegistryKeyHasher,
                                      std::equal_to<RegistryKey>,
                                      std::allocator<std::pair<RegistryKey, void *>>,
                                      /* StoreHash = */ true>;

using RegistryRevMap = tsl::robin_pg_map<const void *, RegistryKey>;

// Maps (device ID, stream ID) to a Stream instance
using StreamMap = tsl::robin_map<std::pair<uint32_t, uint32_t>, Stream *, pair_hash>;

/// Records the full JIT compiler state (most frequently two used entries at top)
struct State {
    /// Must be held to access members
    FastMutex mutex;

    /// Must be held to access 'stream->release_chain' and 'state.alloc_free'
    FastMutex malloc_mutex;

    /// Stores the mapping from variable indices to variables
    VariableMap variables;

    /// Must be held to execute jit_eval()
    std::mutex eval_mutex;

    /// Log level (stderr)
    LogLevel log_level_stderr = LogLevel::Info;

    /// Log level (callback)
    LogLevel log_level_callback = LogLevel::Disable;

    /// Callback for log messages
    LogCallback log_callback = nullptr;

    /// Was the LLVM backend successfully initialized?
    bool has_llvm = false;

    /// Was the CUDA backend successfully initialized?
    bool has_cuda = false;

    /// Available devices and their CUDA IDs
    std::vector<Device> devices;

    /// Maps Enoki (device index, stream index) pairs to a Stream data structure
    StreamMap streams;

    /// Two-way mapping that can be used to associate pointers with unique 32 bit IDs
    RegistryFwdMap registry_fwd;
    RegistryRevMap registry_rev;

    /// Map of currently allocated memory regions
    tsl::robin_pg_map<const void *, AllocInfo> alloc_used;

    /// Map of currently unused memory regions
    AllocInfoMap alloc_free;

    /// Host memory regions mapped into unified memory & pending release
    std::vector<std::pair<bool, void *>> alloc_unmap;

    /// Keep track of current memory usage and a maximum watermark
    size_t alloc_usage    [(int) AllocType::Count] { 0 },
           alloc_watermark[(int) AllocType::Count] { 0 };

    /// Maps from pointer addresses to variable indices
    tsl::robin_pg_map<const void *, uint32_t> variable_from_ptr;

    /// Maps from variable indices to (optional) descriptive labels
    tsl::robin_map<uint32_t, char *> labels;

    /// Maps from variable indices to cached results from jitc_vcall()
    tsl::robin_map<uint32_t, std::pair<uint32_t, VCallBucket *>> vcall_cache;

    /// Maps from a key characterizing a variable to its index
    CSECache cse_cache;

    /// Current variable index
    uint32_t variable_index = 1;

    /// Limit the output of jitc_var_str()?
    uint32_t print_limit = 20;

    /// Dispatch to multiple streams that run concurrently?
    bool parallel_dispatch = true;

    /// Cache of previously compiled kernels
    KernelCache kernel_cache;
};

/// RAII helper for locking a mutex (like std::lock_guard)
template <typename T> class lock_guard_t {
public:
    lock_guard_t(T &mutex) : m_mutex(mutex) { m_mutex.lock(); }
    ~lock_guard_t() { m_mutex.unlock(); }
    lock_guard_t(const lock_guard_t &) = delete;
    lock_guard_t &operator=(const lock_guard_t &) = delete;
private:
    T &m_mutex;
};

/// RAII helper for *unlocking* a mutex
template <typename T> class unlock_guard_t {
public:
    unlock_guard_t(T &mutex) : m_mutex(mutex) { m_mutex.unlock(); }
    ~unlock_guard_t() { m_mutex.lock(); }
    unlock_guard_t(const unlock_guard_t &) = delete;
    unlock_guard_t &operator=(const unlock_guard_t &) = delete;
private:
    T &m_mutex;
};

using lock_guard   =   lock_guard_t<FastMutex>;
using unlock_guard = unlock_guard_t<FastMutex>;


struct Buffer {
public:
    Buffer(size_t size);

    // Disable copy/move constructor and assignment
    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) = delete;

    ~Buffer() {
        free(m_start);
    }

    const char *get() { return m_start; }

    void clear() {
        m_cur = m_start;
        m_start[0] = '\0';
    }

    /// Append a string to the buffer
    void put(const char *str) {
        do {
            char* cur = (char*) memccpy(m_cur, str, '\0', m_end - m_cur);

            if (likely(cur)) {
                m_cur = cur - 1;
                break;
            }

            expand();
        } while (true);
    }

    /// Append a string with the specified length
    void put(const char *str, size_t size) {
        if (unlikely(m_cur + size >= m_end))
            expand(size);
        memcpy(m_cur, str, size); m_cur += size;
        *m_cur = '\0';
    }

    /// Append an unsigned 32 bit integer
    void put_uint32(uint32_t value) {
        if (unlikely(m_cur + 10 >= m_end))
            expand(10);

        const char *num = "0123456789";
        char buf[11];
        buf[10] = '\0';
        int i = 9;
        do {
            buf[--i] = num[value % 10];
            value /= 10;
        } while (value);
        put(buf + i, 9 - i);
    }

    /// Append a single character to the buffer
    void putc(char c) {
        if (unlikely(m_cur + 1 == m_end))
            expand();
        *m_cur++ = c;
        *m_cur = '\0';
    }

    /// Remove the last 'n' characters
    void rewind(size_t n) {
        m_cur -= n;
        if (m_cur < m_start)
            m_cur = m_start;
    }

    /// Check if the buffer contains a given substring
    bool contains(const char *needle) const {
        return strstr(m_start, needle) != nullptr;
    }

    /// Append a formatted (printf-style) string to the buffer
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
    size_t fmt(const char *format, ...);

    /// Like \ref fmt, but specify arguments through a va_list.
    size_t vfmt(const char *format, va_list args_);

    size_t size() const { return m_cur - m_start; }

    void swap(Buffer &b) {
        std::swap(m_start, b.m_start);
        std::swap(m_cur, b.m_cur);
        std::swap(m_end, b.m_end);
    }
private:
    void expand(size_t minval = 2);

private:
    char *m_start, *m_cur, *m_end;
};

/// Global state record shared by all threads
#if defined(_MSC_VER)
  extern __declspec(thread) Stream* active_stream;
#else
  extern __thread Stream* active_stream;
#endif

extern State state;
extern Buffer buffer;

#if !defined(_WIN32)
  extern char *jit_temp_path;
#else
  extern wchar_t *jit_temp_path;
#endif

/// Initialize core data structures of the JIT compiler
extern void jit_init(int llvm, int cuda);

/// Release all resources used by the JIT compiler, and report reference leaks.
extern void jit_shutdown(int light);

/// Set the currently active device & stream
extern void jit_set_device(int32_t device, uint32_t stream);

/// Wait for all computation on the current stream to finish
extern void jit_sync_stream();

/// Wait for all computation on the current device to finish
extern void jit_sync_device();

/// Search for a shared library and dlopen it if possible
void *jit_find_library(const char *fname, const char *glob_pat,
                       const char *env_var);
