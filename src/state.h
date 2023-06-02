#include "var.h"
#include "registry.h"
#include "malloc.h"

struct OptixPipelineData;
struct OptixShaderBindingTable;

/// Key data structure for kernel source code & device ID
struct KernelKey {
    char *str = nullptr;
    int device = 0;
    uint64_t flags = 0;

    KernelKey(char *str, int device, uint64_t flags) : str(str), device(device), flags(flags) { }

    bool operator==(const KernelKey &k) const {
        return strcmp(k.str, str) == 0 && device == k.device && flags == k.flags;
    }
};

/// Helper class to hash KernelKey instances
struct KernelHash {
    size_t operator()(const KernelKey &k) const {
        return compute_hash(hash_kernel(k.str).high64, k.device, k.flags);
    }

    static size_t compute_hash(size_t kernel_hash, int device, uint64_t flags) {
        size_t hash = kernel_hash;
        hash_combine(hash, (size_t) flags + size_t(device + 1));
        return hash;
    }
};

/// Represents a compiled kernel for the various different backends
struct Kernel {
    void *data;
    uint32_t size;
    union {
#if defined(DRJIT_ENABLE_CUDA)
        struct {
            /// Compiled module
            void *mod; // CUmodule

            /// Main kernel entry point
            void *func; // CUfunction

            // Preferred block size to maximize occupancy
            uint32_t block_size;
        } cuda;
#endif

#if defined(DRJIT_ENABLE_OPTIX)
        struct {
            void *mod; // OptixModule
            void **pg; // OptixProgramGroup
            void *pipeline; // OptixPipeline
            uint8_t *sbt_record;
            uint32_t pg_count;
        } optix;
#endif

#if defined(DRJIT_ENABLE_LLVM)
        struct {
            /// Relocation table, the first element is the kernel entry point
            void **reloc;

            /// Length of the 'reloc' table
            uint32_t n_reloc;

#if defined(DRJIT_ENABLE_ITTNOTIFY)
            void *itt;
#endif
        } llvm;
#endif
    };
};

/// Data structure, which maps from kernel source code to compiled kernels
using KernelCache =
    tsl::robin_map<KernelKey, Kernel, KernelHash, std::equal_to<KernelKey>,
                   std::allocator<std::pair<KernelKey, Kernel>>,
                   /* StoreHash = */ true>;

/// Caches basic information about a CUDA device
struct Device {
#if defined(DRJIT_ENABLE_CUDA)
    // CUDA device context
    CUcontext context;

    /// Associated CUDA stream handle
    CUstream stream = nullptr;

    /// A CUDA event for synchronization purposes
    CUevent event = nullptr;

    /// CUDA device ID
    int id;

    /// Device compute capability (major * 10 + minor)
    uint32_t compute_capability;

    /// Targeted PTX version (major * 10 + minor)
    uint32_t ptx_version;

    /// Number of SMs
    uint32_t num_sm;

    /// Max. bytes of shared memory per SM
    uint32_t shared_memory_bytes;

    // Support for stream-ordered memory allocations (async alloc/free)
    bool memory_pool;

    /** \brief If preemptable is false, long-running kernels might freeze
     * the OS GUI and time out after 2 sec */
    bool preemptable;

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
#endif

#if defined(DRJIT_ENABLE_OPTIX)
    /// OptiX device context
    void *optix_context = nullptr;
#endif
};

/// Data structure to store a history of the launched kernels
struct KernelHistory {
    KernelHistory();
    ~KernelHistory();

    void append(const KernelHistoryEntry &entry);
    KernelHistoryEntry *get();
    void clear();

private:
    KernelHistoryEntry *m_data;
    size_t m_size;
    size_t m_capacity;
};

/// Records the full JIT compiler state (most frequently two used entries at top)
struct State {
    /// Must be held to access members
    Lock lock;

    /// Must be held to access 'state.alloc_free'
    Lock alloc_free_lock;

    /// Stores the mapping from variable indices to variables
    VariableMap variables;

    /// Counter to create variable scopes that enforce a variable ordering
    uint32_t scope_ctr = 0;

    /// Maps from a key characterizing a variable to its index
    LVNMap lvn_map;

    /// Must be held to execute jitc_eval()
    Lock eval_lock;

    /// Log level (stderr)
    LogLevel log_level_stderr = LogLevel::Info;

    /// Log level (callback)
    LogLevel log_level_callback = LogLevel::Disable;

    /// Callback for log messages
    LogCallback log_callback = nullptr;

    /// Bit-mask of successfully initialized backends
    uint32_t backends = 0;

    /// Available devices and their CUDA IDs
    std::vector<Device> devices;

    /// State associated with each DrJit thread
    std::vector<ThreadState *> tss;

    /// Pointer registries for LLVM and CUDA backends
    Registry registry_cpu, registry_gpu;

    /// Map of currently allocated memory regions
    AllocUsedMap alloc_used;

    /// Map of currently unused memory regions
    AllocInfoMap alloc_free;

    /// Keep track of current memory usage and a maximum watermark
    size_t alloc_usage    [2 * (int) JitBackend::Count] { 0 },
           alloc_allocated[2 * (int) JitBackend::Count] { 0 },
           alloc_watermark[2 * (int) JitBackend::Count] { 0 };

    /// Keep track of the number of created JIT variables
    uint32_t variable_watermark = 0;

    /// Maps from variable ID to extra information for a fraction of variables
    ExtraMap extra;

    /// Current variable index
    uint32_t variable_index = 1;

    /// Limit the output of jit_var_str()?
    uint32_t print_limit = 20;

    /// Statistics on kernel launches
    size_t kernel_hard_misses = 0;
    size_t kernel_soft_misses = 0;
    size_t kernel_hits = 0;
    size_t kernel_launches = 0;

    /// Cache of previously compiled kernels
    KernelCache kernel_cache;

    /// Kernel launch history
    KernelHistory kernel_history;

#if !defined(_WIN32)
    char *temp_path = nullptr;
#else
    wchar_t *temp_path = nullptr;
#endif

    /// Return a pointer to the registry corresponding to the specified backend
    Registry *registry(JitBackend backend) {
        return backend == JitBackend::CUDA ? &registry_gpu : &registry_cpu;
    }

    State() {
        lock_init(lock);
        lock_init(alloc_free_lock);
        lock_init(eval_lock);
    }

    ~State() {
        lock_destroy(lock);
        lock_destroy(alloc_free_lock);
        lock_destroy(eval_lock);
        free(temp_path);
    }
};

extern State state;
