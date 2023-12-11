/*
    src/internal.h -- Central data structure definitions

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "malloc.h"
#include "cuda.h"
#include "llvm.h"
#include "alloc.h"
#include "io.h"
#include <queue>
#include <string.h>
#include <inttypes.h>
#include <nanothread/nanothread.h>

/// List of operations in Dr.Jit-Core's intermediate representation (see ``Variable::kind``)
enum VarKind : uint32_t {
    // Invalid operation (default initialization in the Variable class)
    Invalid,

    // An evaluated variable represented by a device memory region
    Evaluated,

    // Undefined/uninitialized memory
    Undefined,

    // A literal constant. Standard variable types ("nodes") start after this entry.
    Literal,

    // A no-op (generates no code)
    Nop,

    // Common unary operations
    Neg, Not, Sqrt, Abs,

    // Common binary arithmetic operations
    Add, Sub, Mul, Div, Mod,

    // High multiplication
    Mulhi,

    // Fused multiply-add
    Fma,

    // Minimum, maximum
    Min, Max,

    // Rounding operations
    Ceil, Floor, Round, Trunc,

    // Comparisons
    Eq, Neq, Lt, Le, Gt, Ge,

    // Ternary operator
    Select,

    // Bit-level counting operations
    Popc, Clz, Ctz,

    // Bit-wise operations
    And, Or, Xor,

    // Shifts
    Shl, Shr,

    // Fast approximations
    Rcp, Rsqrt,

    // Multi-function generator (CUDA)
    Sin, Cos, Exp2, Log2,

    // Casts
    Cast, Bitcast,

    // Ensure that an index is within the array bounds
    BoundsCheck,

    // Memory-related operations
    Gather, Scatter, ScatterInc, ScatterKahan,

    // Counter node to determine the current lane ID
    Counter,

    // Default mask used to ignore out-of-range SIMD lanes (LLVM)
    DefaultMask,

    // A polymorphic function call
    Call,

    // Specialized nodes for calls
    CallMask, CallSelf,

    // Input argument to a function call
    CallInput,

    // Output of a function call
    CallOutput,

    // Perform a standard texture lookup (CUDA)
    TexLookup,

    // Load all texels used for bilinear interpolation (CUDA)
    TexFetchBilerp,

    // Perform a ray tracing call
    TraceRay,

    // Extract a component from an operation that produced multiple results
    Extract,

    // Variable marking the start of a loop
    LoopStart,

    // Variable marking the loop condition
    LoopCond,

    // Variable marking the end of a loop
    LoopEnd,

    // SSA Phi variable at start of loop
    LoopPhi,

    // SSA Phi variable at end of loop
    LoopOutput,

    // Variable marking the start of a conditional statement
    CondStart,

    // Variable marking the start of the 'false' branch
    CondMid,

    // Variable marking the end of a conditional statement
    CondEnd,

    // SSA Phi variable marking an output of a conditional statement
    CondOutput,

    // Denotes the number of different node types
    Count
};

/// Central variable data structure, which represents an assignment in SSA form
struct Variable {
    /// Zero-initialize by default
    Variable() { memset((void *) this, 0, sizeof(Variable)); }

    // =================  Reference count, dependencies, scope ================

    /// Number of times that this variable is referenced elsewhere
    uint32_t ref_count;

    /// Identifier of the basic block containing this variable
    uint32_t scope;

    /// Up to 4 dependencies of this instruction
    uint32_t dep[4];

    // ======  Size & encoded instruction (IR statement, literal, data) =======

    /// The 'kind' field determines which entry of the following union is used
    union {
        // Floating point/integer value, reinterpreted as u64
        uint64_t literal;

        /// Pointer to device memory. Used when kind == VarKind::Evaluated
        void *data;
    };

    /// Number of entries
    uint32_t size;

    /// Unused, to be eventually used to upgrade to 64 bit array sizes
    uint32_t unused;

    /// How many times has this variable entry been (re-) used?
    uint32_t counter;

    // ================  Essential flags used in the LVN key  =================

    // Variable kind (IR statement / literal constant / data)
    uint32_t kind : 8;

    /// Backend associated with this variable
    uint32_t backend : 2;

    /// Variable type (Bool/Int/Float/....)
    uint32_t type : 4;

    /// Is this a pointer variable that is used to write to some array?
    uint32_t write_ptr : 1;

    // =======================  Miscellaneous fields =========================

    /// If set, 'data' will not be deallocated when the variable is destructed
    uint32_t retain_data : 1;

    /// Does this variable represent symbolic computation?
    uint32_t symbolic : 1;

    /// Must be set if 'data' is not properly aligned in memory
    uint32_t unaligned : 1;

    /// Does this variable perform an OptiX operation?
    uint32_t optix : 1;

    /// If set, evaluation will have side effects on other variables
    uint32_t side_effect : 1;

    /// Tracks if an SSA (LLVM) implicit f32 cast has been performed during assembly
    uint32_t ssa_f32_cast : 1;

    // =========== Entries that are temporarily used in jitc_eval() ============

    /// Argument type
    uint32_t param_type : 2;

    /// Is this variable marked as an output?
    uint32_t output_flag : 1;

    /// Consumed bit for operations that should only be executed once
    uint32_t consumed : 1;

    /// Unused for now
    uint32_t unused_2 : 7;

    /// Offset of the argument in the list of kernel parameters
    uint32_t param_offset;

    /// Register index
    uint32_t reg_index;

    // ========================  Side effect tracking  =========================

    /// Number of queued side effects
    uint32_t ref_count_se;

    /// If nonzero, references an associated 'VariableExtra' field in 'state.extra'
    uint32_t extra;

    // =========================   Helper functions   ==========================

    bool is_evaluated() const { return kind == (uint32_t) VarKind::Evaluated; }
    bool is_literal()   const { return kind == (uint32_t) VarKind::Literal; }
    bool is_undefined() const { return kind == (uint32_t) VarKind::Undefined; }
    bool is_node()      const { return (uint32_t) kind > VarKind::Literal; }
    bool is_dirty()     const { return ref_count_se > 0; }
};

/// This record represents additional information that can *optionally* be
/// associated with a 'Variable' instance. These are factored out into a struct
/// since only very few variables need these fields, and to to ensure that
/// ``sizeof(Variable) == 64`` (i.e. a variable fits into a L1 cache line)
struct VariableExtra {
    /// A human-readable variable label (for GraphViz visualizations, debugging LLVM/PTX IR)
    char *label = nullptr;

    /// Callback to be invoked when the variable is evaluated/deallocated
    void (*callback)(uint32_t, int, void *) = nullptr;

    /// An opaque pointer that will be passed to the callback
    void *callback_data = nullptr;

    /// Set to 'true' if the central mutex should be released before invoking 'callback'
    bool callback_internal = false;
};

#pragma pack(push, 1)

/// Abbreviated version of the Variable data structure for Local Value Numbering (LVN)
struct VariableKey {
    uint32_t size;
    uint32_t scope;
    uint32_t dep[4];
    uint32_t kind      : 8;
    uint32_t backend   : 2;
    uint32_t type      : 4;
    uint32_t write_ptr : 1;
    uint32_t unused    : 17;
    uint64_t literal;

    VariableKey(const Variable &v) {
        size = v.size;
        scope = v.scope;
        for (int i = 0; i < 4; ++i)
            dep[i] = v.dep[i];
        literal = v.literal;

        kind = v.kind;
        backend = v.backend;
        type = v.type;
        write_ptr = v.write_ptr;
        unused = 0;
    }

    bool operator==(const VariableKey &v) const {
        return memcmp((const void *) this, (const void *) &v, 9 * sizeof(uint32_t)) == 0;
    }
};

#pragma pack(pop)

/// Helper class to hash VariableKey instances
struct VariableKeyHasher {
    size_t operator()(const VariableKey &k) const {
        return hash((const void *) &k, 9 * sizeof(uint32_t), 0);
    }
};

/// Cache data structure for local value numbering
using LVNMap =
    tsl::robin_map<VariableKey, uint32_t, VariableKeyHasher,
                   std::equal_to<VariableKey>,
                   std::allocator<std::pair<VariableKey, uint32_t>>,
                   /* StoreHash = */ true>;

/// Caches basic information about a CUDA device
struct Device {
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

#if defined(DRJIT_ENABLE_OPTIX)
    /// OptiX device context
    void *optix_context = nullptr;
#endif

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

/// A few forward declarations for OptiX
#if defined(DRJIT_ENABLE_OPTIX)
using OptixProgramGroup = void *;

struct OptixShaderBindingTable {
    void* raygenRecord;
    void* exceptionRecord;
    void* missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    void* hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    void* callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
};

struct OptixPipelineData {
    OptixPipelineCompileOptions compile_options;
    OptixModule module;
    std::vector<OptixProgramGroup> program_groups;
};
#endif

struct WeakRef {
    uint32_t index;
    uint32_t counter;
    WeakRef() : index(0), counter(0) { }
    WeakRef(uint32_t index, uint32_t counter)
        : index(index), counter(counter) { }
};


/// Represents a single stream of a parallel communication
struct ThreadState {
    /// Backend type
    JitBackend backend;

    /**
     * List of variables that are scheduled for evaluation (via
     * jitc_var_schedule()) that will take place at the next call to jitc_eval().
     */
    std::vector<WeakRef> scheduled;

    /**
     * List of special variables of type VarType::Void, whose evaluation will
     * cause side effects that modify other variables. They will be evaluated
     * at the next call to jitc_eval().
     */
    std::vector<uint32_t> side_effects;

    /// When recording loops or virtual function calls, side effects go here.
    std::vector<uint32_t> side_effects_symbolic;

    /**
     * Stack of variable indices indicating the list of active SIMD lanes.
     * This is used to constrain the behavior of gather/scatter operations.
     */
    std::vector<uint32_t> mask_stack;

    /// Stack of variable name prefixes, mainly useful for GraphViz exports
    std::vector<char *> prefix_stack;

    /// Stack of symbolic recording sessions
    std::vector<std::string> record_stack;

    /// Combined version of the elements of 'prefix_stack'
    char *prefix = nullptr;

    /// Identifier associated with the current basic block
    uint32_t scope = 0;

    /// Registry index of the 'self' pointer of the call being recorded
    uint32_t call_self_value = 0;

    /// .. and the JIT variable that it will be mapped to
    uint32_t call_self_index = 0;

    /// ---------------------------- CUDA-specific ----------------------------

    /// Redundant copy of the device context
    CUcontext context = nullptr;

    /// Associated CUDA stream handle
    CUstream stream = nullptr;

    /// A CUDA event for synchronization purposes
    CUevent event = nullptr;

    /**
     * \brief DrJit device ID associated with this device
     *
     * This value may differ from the CUDA device ID if the machine contains
     * CUDA devices that are incompatible with DrJit.
     *
     * Equals -1 for LLVM ThreadState instances.
     */
    int device = 0;

    /// Device compute capability (major * 10 + minor)
    uint32_t compute_capability = 0;

    /// Targeted PTX version (major * 10 + minor)
    uint32_t ptx_version = 0;

    // Support for stream-ordered memory allocations (async alloc/free)
    bool memory_pool = false;

#if defined(DRJIT_ENABLE_OPTIX)
    /// OptiX pipeline associated with the next kernel launch
    OptixPipelineData *optix_pipeline = nullptr;
    OptixShaderBindingTable *optix_sbt = nullptr;
#endif
};

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

/// Data structure, which maps from kernel source code to compiled kernels
using KernelCache =
    tsl::robin_map<KernelKey, Kernel, KernelHash, std::equal_to<KernelKey>,
                   std::allocator<std::pair<KernelKey, Kernel>>,
                   /* StoreHash = */ true>;

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

using UnusedPQ = std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>;

/// Records the full JIT compiler state (most frequently two used entries at top)
struct State {
    /// Must be held to access members of this data structure
    Lock lock;

    /// Must be held to access 'state.alloc_free'
    Lock alloc_free_lock;

    /// A flat list of variable data structures, including unused one
    std::vector<Variable> variables{ 1 };

    /// A priority queue of indices into 'variables' that are currently unused
    UnusedPQ unused_variables;

    /// Maps from a key characterizing a variable to its index
    LVNMap lvn_map;

    /// A flast list of variable VariableExtra data structures (see its
    /// definition for documentation). Includes unused ones.
    std::vector<VariableExtra> extra { 1 };

    // A priority queu of indices into 'extra' that are currently unused
    UnusedPQ unused_extra;

    /// Counter to create variable scopes that enforce a variable ordering
    uint32_t scope_ctr = 0;
    size_t variable_counter = 0;

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

    /// Map of currently allocated memory regions
    AllocUsedMap alloc_used;

    /// Map of currently unused memory regions
    AllocInfoMap alloc_free;

    /// Keep track of current memory usage and a maximum watermark
    size_t alloc_usage    [(int) AllocType::Count] { 0 },
           alloc_allocated[(int) AllocType::Count] { 0 },
           alloc_watermark[(int) AllocType::Count] { 0 };

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
    KernelHistory kernel_history = KernelHistory();

#if defined(DRJIT_ENABLE_OPTIX)
    /// Default OptiX pipeline for testcases etc.
    OptixPipelineData *optix_default_pipeline = nullptr;
    /// Default OptiX Shader Binding Table for testcases etc.
    OptixShaderBindingTable *optix_default_sbt = nullptr;
    /// Index of the JIT variable handling the lifetime of the default Optix SBT
    uint32_t optix_default_sbt_index = 0;
#endif

    State() {
        lock_init(lock);
        lock_init(alloc_free_lock);
        lock_init(eval_lock);
    }

    ~State() {
        lock_destroy(lock);
        lock_destroy(alloc_free_lock);
        lock_destroy(eval_lock);
    }
};

enum ParamType { Register, Input, Output };

/// State specific to threads
#if defined(_MSC_VER)
  extern __declspec(thread) ThreadState* thread_state_llvm;
  extern __declspec(thread) ThreadState* thread_state_cuda;
  extern __declspec(thread) JitBackend default_backend;
#else
  extern __thread ThreadState* thread_state_llvm;
  extern __thread ThreadState* thread_state_cuda;
  extern __thread JitBackend default_backend;
#endif

extern ThreadState *jitc_init_thread_state(JitBackend backend);

inline ThreadState *thread_state(JitBackend backend) {
    ThreadState *result = (backend == JitBackend::CUDA) ? thread_state_cuda : thread_state_llvm;
    if (unlikely(!result))
        result = jitc_init_thread_state(backend);
    return result;
}

inline ThreadState *thread_state(uint32_t backend) {
    return thread_state((JitBackend) backend);
}

extern State state;

#if !defined(_WIN32)
  extern char *jitc_temp_path;
#else
  extern wchar_t *jitc_temp_path;
#endif

/// Initialize core data structures of the JIT compiler
extern void jitc_init(uint32_t backends);

/// Release all resources used by the JIT compiler, and report reference leaks.
extern void jitc_shutdown(int light);

/// Set the currently active device & stream
extern void jitc_cuda_set_device(int device);

/// Wait for all computation on the current stream to finish
extern void jitc_sync_thread();

/// Wait for all computation on the current stream to finish
extern void jitc_sync_thread(ThreadState *stream);

/// Wait for all computation on the current device to finish
extern void jitc_sync_device();

/// Wait for all computation on *all devices* to finish
extern void jitc_sync_all_devices();

/// Search for a shared library and dlopen it if possible
void *jitc_find_library(const char *fname, const char *glob_pat,
                        const char *env_var);

/// Return a pointer to the CUDA stream associated with the currently active device
extern void* jitc_cuda_stream();

/// Return a pointer to the CUDA context associated with the currently active device
extern void* jitc_cuda_context();

/// Push a new CUDA context to the currently active device
extern void jitc_cuda_push_context(void *);

/// Pop the current CUDA context and return it
extern void* jitc_cuda_pop_context();

extern void jitc_set_flags(uint32_t flags);

extern uint32_t jitc_flags();

/// Push a new label onto the prefix stack
extern void jitc_prefix_push(JitBackend backend, const char *label);

/// Pop a label from the prefix stack
extern void jitc_prefix_pop(JitBackend backend);

JIT_MALLOC inline void* malloc_check(size_t size) {
    void *ptr = malloc(size);
    if (unlikely(!ptr)) {
        fprintf(stderr, "malloc_check(): failed to allocate %zu bytes!", size);
        abort();
    }
    return ptr;
}

JIT_MALLOC inline void* malloc_check_zero(size_t size) {
    void *ptr = malloc_check(size);
    memset(ptr, 0, size);
    return ptr;
}

JIT_MALLOC inline void* realloc_check(void *orig, size_t size) {
    void *ptr = realloc(orig, size);
    if (unlikely(!ptr)) {
        fprintf(stderr, "realloc_check(): could not resize memory region to %zu bytes!", size);
        abort();
    }
    return ptr;
}


// ===========================================================================
// Helper functions to classify different variable types
// ===========================================================================

inline bool jitc_is_arithmetic(VarType type) {
    return type != VarType::Void && type != VarType::Bool;
}

inline bool jitc_is_float(VarType type) {
    return type == VarType::Float16 ||
           type == VarType::Float32 ||
           type == VarType::Float64;
}

inline bool jitc_is_half(VarType type) { return type == VarType::Float16; }
inline bool jitc_is_single(VarType type) { return type == VarType::Float32; }
inline bool jitc_is_double(VarType type) { return type == VarType::Float64; }
inline bool jitc_is_bool(VarType type) { return type == VarType::Bool; }

inline bool jitc_is_sint(VarType type) {
    return type == VarType::Int8 ||
           type == VarType::Int16 ||
           type == VarType::Int32 ||
           type == VarType::Int64;
}

inline bool jitc_is_uint(VarType type) {
    return type == VarType::UInt8 ||
           type == VarType::UInt16 ||
           type == VarType::UInt32 ||
           type == VarType::UInt64;
}

inline bool jitc_is_int(VarType type) {
    return jitc_is_sint(type) || jitc_is_uint(type);
}

inline bool jitc_is_void(VarType type) {
    return type == VarType::Void;
}

inline bool jitc_is_arithmetic(const Variable *v) { return jitc_is_arithmetic((VarType) v->type); }
inline bool jitc_is_float(const Variable *v) { return jitc_is_float((VarType) v->type); }
inline bool jitc_is_half(const Variable* v) { return jitc_is_half((VarType) v->type); }
inline bool jitc_is_single(const Variable *v) { return jitc_is_single((VarType) v->type); }
inline bool jitc_is_double(const Variable *v) { return jitc_is_double((VarType) v->type); }
inline bool jitc_is_sint(const Variable *v) { return jitc_is_sint((VarType) v->type); }
inline bool jitc_is_uint(const Variable *v) { return jitc_is_uint((VarType) v->type); }
inline bool jitc_is_int(const Variable *v) { return jitc_is_int((VarType) v->type); }
inline bool jitc_is_void(const Variable *v) { return jitc_is_void((VarType) v->type); }
inline bool jitc_is_bool(const Variable *v) { return jitc_is_bool((VarType) v->type); }
inline bool jitc_is_zero(Variable *v) { return v->is_literal() && v->literal == 0; }

inline bool jitc_is_one(Variable *v) {
    if (!v->is_literal())
        return false;

    uint64_t one;
    switch ((VarType) v->type) {
        case VarType::Float16: one = 0x3c00ull; break;
        case VarType::Float32: one = 0x3f800000ull; break;
        case VarType::Float64: one = 0x3ff0000000000000ull; break;
        default: one = 1; break;
    }

    return v->literal == one;
}

extern const char *var_kind_name[(int) VarKind::Count];
