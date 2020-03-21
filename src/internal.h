#pragma once

#include "malloc.h"
#include "cuda_api.h"
#include "llvm_api.h"
#include "alloc.h"
#include <mutex>
#include <condition_variable>
#include <string.h>
#include <inttypes.h>

static constexpr LogLevel Disable = LogLevel::Disable;
static constexpr LogLevel Error   = LogLevel::Error;
static constexpr LogLevel Warn    = LogLevel::Warn;
static constexpr LogLevel Info    = LogLevel::Info;
static constexpr LogLevel Debug   = LogLevel::Debug;
static constexpr LogLevel Trace   = LogLevel::Trace;

#define ENOKI_PTR "<0x%" PRIxPTR ">"

#if !defined(likely)
#  define likely(x)   __builtin_expect(!!(x), 1)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/// Caches basic information about a CUDA device
struct Device {
    // CUDA device context
    CUcontext context;

    /// CUDA device ID
    int id;

    /// Number of SMs
    int num_sm;
};

/// Keeps track of asynchronous deallocations via jit_free()
struct ReleaseChain {
    AllocInfoMap entries;
    /// Pointer to next linked list entry
    ReleaseChain *next = nullptr;
};

/// Represents a single CUDA stream and events to synchronize with others
struct Stream {
    /// Enoki device index associated with this stream (*not* the CUDA device ID)
    uint32_t device = 0;

    /// Index of this stream
    uint32_t stream = 0;

    /// Associated CUDA stream handle
    CUstream handle = nullptr;

    /// A CUDA event for synchronization purposes
    CUevent event = nullptr;

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
};

enum ArgType {
    Register,
    Input,
    Output
};

using StreamMap = tsl::robin_map<std::pair<uint32_t, uint32_t>, Stream *, pair_hash>;

#pragma pack(push)
#pragma pack(1)

/// Central variable data structure, which represents an assignment in SSA form
struct Variable {
    /// External reference count (by application using Enoki)
    uint32_t ref_count_ext;

    /// Internal reference count (dependencies within computation graph)
    uint32_t ref_count_int;

    /// Dependencies of this instruction
    uint32_t dep[3];

    /// Extra dependency (which is not directly used in arithmetic, e.g. scatter/gather)
    uint32_t extra_dep;

    /// Number of entries
    uint32_t size;

    /// Intermediate language statement
    char *stmt;

    /// Pointer to device memory
    void *data;

    /// Size of the instruction subtree (heuristic for instruction scheduling)
    uint32_t tsize;

    /// Register index (temporarily used during jit_eval())
    uint32_t reg_index : 24;

    /// Argument index (temporarily used during jit_eval())
    uint32_t arg_index : 16;

    /// Data type of this variable
    uint32_t type : 4;

    /// Argument type (register: 0, input: 1, output: 2)
    ArgType arg_type : 2;

    /// Does the instruction have side effects (e.g. 'scatter')
    bool side_effect : 1;

    /// Don't deallocate 'data' when this variable is destructed?
    bool retain_data : 1;

    /// Free the 'stmt' variables at destruction time?
    bool free_stmt : 1;

    /// Was this variable labeled?
    bool has_label : 1;

    /// A variable is 'dirty' if there are pending scatter operations to it
    bool dirty : 1;

    /// Optimization: is this a direct pointer (rather than an array which stores a pointer?)
    bool direct_pointer : 1;

    /// Is this variable registered with the CUDA backend?
    bool cuda : 1;

    Variable() {
        memset(this, 0, sizeof(Variable));
    }
};

#pragma pack(pop)

/// Abbreviated version of the Variable data structure
struct VariableKey {
    char *stmt;
    uint32_t size;
    uint32_t dep[3];
    uint32_t extra_dep;
    uint16_t type;
    uint16_t free_stmt;

    VariableKey(const Variable &v)
        : stmt(v.stmt), size(v.size), dep{ v.dep[0], v.dep[1], v.dep[2] },
          extra_dep(v.extra_dep), type((uint16_t) v.type),
          free_stmt(v.free_stmt ? 1 : 0) { }

    bool operator==(const VariableKey &v) const {
        return strcmp(stmt, v.stmt) == 0 && size == v.size &&
               dep[0] == v.dep[0] && dep[1] == v.dep[1] && dep[2] == v.dep[2] &&
               extra_dep == v.extra_dep && type == v.type &&
               free_stmt == v.free_stmt;
    }
};

struct VariableKeyHasher {
    size_t operator()(const VariableKey &k) const {
        uint32_t state;
        if (likely(k.free_stmt == 0)) {
            state = crc32_64(0, (const uint64_t *) &k.stmt, 4);
        } else {
            state = crc32_str(0, k.stmt);
            state = crc32_64(state, (const uint64_t *) &k.size, 3);
        }
        return state;
    }
};

using VariableMap =
    tsl::robin_map<uint32_t, Variable, std::hash<uint32_t>,
                   std::equal_to<uint32_t>,
                   aligned_allocator<std::pair<uint32_t, Variable>, 64>,
                   /* StoreHash = */ false>;

using CSECache =
    tsl::robin_map<VariableKey, uint32_t, VariableKeyHasher,
                   std::equal_to<VariableKey>,
                   std::allocator<std::pair<VariableKey, uint32_t>>,
                   /* StoreHash = */ true>;

using KernelCache =
    tsl::robin_map<const char *, Kernel, string_hash, string_eq,
                   std::allocator<std::pair<const char *, Kernel>>,
                   /* StoreHash = */ true>;

/// Records the full JIT compiler state (most frequently two used entries at top)
struct State {
    /// Must be held to access members
    std::mutex mutex;

    /// Stores the mapping from variable indices to variables
    VariableMap variables;

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

    /// Two-way mapping that associates every allocation with a unique 32 bit ID
    tsl::robin_pg_map<void *, uint32_t> alloc_id_rev;
    tsl::robin_map   <uint32_t, void *> alloc_id_fwd;
    uint32_t alloc_id_ctr = 1;

    /// Map of currently allocated memory regions
    tsl::robin_pg_map<const void *, AllocInfo> alloc_used;

    /// Map of currently unused memory regions
    AllocInfoMap alloc_free;

    /// Keep track of current memory usage and a maximum watermark
    size_t alloc_usage    [(int) AllocType::Count] { 0 },
           alloc_watermark[(int) AllocType::Count] { 0 };

    /// Maps from pointer addresses to variable indices
    tsl::robin_pg_map<const void *, uint32_t> variable_from_ptr;

    /// Maps from variable indices to (optional) descriptive labels
    tsl::robin_map<uint32_t, char *> labels;

    /// Maps from a key characterizing a variable to its index
    CSECache cse_cache;

    /// Current variable index
    uint32_t variable_index = 1;

    /// Current operand for scatter/gather operations
    uint32_t scatter_gather_operand = 0;

    /// Dispatch to multiple streams that run concurrently?
    bool parallel_dispatch = true;

    /// Cache of previously compiled kernels
    KernelCache kernel_cache;

    /**
     * Keeps track of variables that have to be computed when jit_eval() is
     * called, in particular: externally referenced variables and statements
     * with side effects.
     */
    std::vector<uint32_t> todo_host;
};

/// RAII helper for locking a mutex (like std::lock_guard)
class lock_guard {
public:
    lock_guard(std::mutex &mutex) : m_mutex(mutex) { m_mutex.lock(); }
    ~lock_guard() { m_mutex.unlock(); }
    lock_guard(const lock_guard &) = delete;
    lock_guard &operator=(const lock_guard &) = delete;
private:
    std::mutex &m_mutex;
};

/// RAII helper for *unlocking* a mutex
class unlock_guard {
public:
    unlock_guard(std::mutex &mutex) : m_mutex(mutex) { m_mutex.unlock(); }
    ~unlock_guard() { m_mutex.lock(); }
    unlock_guard(const unlock_guard &) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;
private:
    std::mutex &m_mutex;
};

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
            char *cur = (char *) memccpy(m_cur, str, '\0', m_end - m_cur);

            if (likely(cur)) {
                m_cur = cur - 1;
                break;
            }

            expand();
        } while (true);
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
    void expand();

private:
    char *m_start, *m_cur, *m_end;
};

/// Global state record shared by all threads
extern __thread Stream *active_stream;

extern State state;
extern Buffer buffer;

/// Initialize core data structures of the JIT compiler
extern void jit_init(int llvm, int cuda);

/// Release all resources used by the JIT compiler, and report reference leaks.
extern void jit_shutdown(int light);

/// Set the currently active device & stream
extern void jit_device_set(int32_t device, uint32_t stream);

/// Wait for all computation on the current stream to finish
extern void jit_sync_stream();

/// Wait for all computation on the current device to finish
extern void jit_sync_device();

/// Search for a shared library and dlopen it if possible
void *jit_find_library(const char *fname, const char *glob_pat,
                       const char *env_var);
