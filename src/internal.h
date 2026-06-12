/*
    src/internal.h -- Central data structure definitions

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include <inttypes.h>
#include "malloc.h"
#if defined(DRJIT_ENABLE_CUDA)
#  include "cuda.h"
#endif
#include "llvm.h"
#include "alloc.h"
#include "io.h"
#include <string.h>
#include <new>
#include <nanothread/nanothread.h>

/// List of operations in Dr.Jit-Core's intermediate representation (see ``Variable::kind``)
enum class VarKind : uint32_t {
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
    Neg, Not, Sqrt, SqrtApprox, Abs,

    // Common binary arithmetic operations
    Add, Sub, Mul, Div, DivApprox, Mod,

    // High integer multiplication
    MulHi,

    // Wide integer multiplication
    MulWide,

    // Fused multiply-add (integers & floats)
    Fma,

    // Minimum, maximum
    Min, Max,

    // Rounding operations
    Ceil, Floor, Round, Trunc,

    // Comparisons
    Eq, Neq, Lt, Le, Gt, Ge,

    // Ternary operator
    Select, ArraySelect,

    // Bit-level counting operations
    Popc, Clz, Ctz, Brev,

    // Bit-wise operations
    And, Or, Xor,

    // Shifts
    Shl, Shr,

    // Fast approximations
    Rcp, RcpApprox, RSqrtApprox,

    // Multi-function generator (CUDA)
    Sin, Cos, Exp2, Log2, Tanh,

    // Casts
    Cast, Bitcast,

    // Ensure that an index is within the array bounds
    BoundsCheck,

    // Memory-related operations
    Gather, Scatter, ScatterInc, ScatterKahan, ScatterCAS, ScatterExch,

    // Gather multiple contiguous values at once
    PacketGather,

    // Scatter multiple contiguous values at once
    PacketScatter,

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

    // Write to a hardware texture / surface (a side effect)
    TexWrite,

    // Memory read starting at different base pointers per lane (CUDA)
    VectorLoad,

    // Perform a ray tracing call
    TraceRay,

    // Extract a component from an operation that produced multiple results
    Extract,

    /// Retrieve the index of the current thread (LLVM mode)
    ThreadIndex,

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

    // SSA Phi variable at end of loop (special version for array variables)
    ArrayPhi,

    // Variable marking the start of a conditional statement
    CondStart,

    // Variable marking the start of the 'false' branch
    CondMid,

    // Variable marking the end of a conditional statement
    CondEnd,

    // SSA Phi variable marking an output of a conditional statement
    CondOutput,

    // Create an uninitialized array
    Array,

    // Initialize the entries of a variable array with a literal constant
    ArrayInit,

    // Read an element from a variable array
    ArrayRead,

    // Write an element to a variable array
    ArrayWrite,

    // Cooperative Vector API
    CoopVecLiteral,
    CoopVecPack,
    CoopVecUnpack,
    CoopVecLoad,
    CoopVecCast,
    CoopVecUnaryOp,
    CoopVecBinaryOp,
    CoopVecTernaryOp,
    CoopVecMatVec,
    CoopVecAccum,
    CoopVecOuterProductAccum,

    // Shader execution reordering (OptiX)
    ReorderThread,

    // Denotes the number of different node types
    Count
};

/// Temporary state value of variable arrays during compilation
enum class ArrayState : uint32_t {
    /// This variable is not an array
    Invalid,

    /// Newly created or copied array
    Clean,

    /// The array has been modified by a write operation. Subsequent
    /// reads/writes will cause it to enter 'Conflicted' mode
    Modified,

    /// Conflicting reads/writes have been detected for this array
    Conflicted
};

/// Classifies an opaque GPU resource referenced by a pointer-literal handle.
enum class ResourceKind : uint8_t {
    Buffer  = 0, ///< Ordinary device buffer pointer (not an opaque resource)
    Accel   = 1, ///< Acceleration structure
    IFT     = 2, ///< Intersection-function table
    Texture = 3, ///< Texture object
    Sampler = 4  ///< Sampler object
};

/// Central variable data structure, which represents an assignment in SSA form
struct alignas(64) Variable {
    /// Zero-initialize by default
    Variable() { memset((void *) this, 0, sizeof(Variable)); }

    // =================  Reference count, dependencies, scope ================
    // (6*4 = 24 bytes)

    /// Number of times that this variable is referenced elsewhere
    uint32_t ref_count;

    /// Identifier of the basic block containing this variable
    uint32_t scope;

    /// Up to 4 dependencies of this instruction
    uint32_t dep[4];

    // ======  Size & encoded instruction (IR statement, literal, data) =======
    // (8 + 3*4 = 20 bytes)

    /// The 'kind' field determines which entry of the following union is used
    union {
        // Floating point/integer value, reinterpreted as u64
        uint64_t literal;

        /// Pointer to device memory. Used when kind == VarKind::Evaluated
        void *data;
    };

    /// Number of entries
    uint32_t size;

    /// How many times has this variable entry been (re-) used?
    uint32_t counter;

    /// Free-use field temporarily used by various parts of Dr.Jit
    uint32_t scratch;

    // ================  Essential flags used in the LVN key  =================
    // (+16 bits)

    // Variable kind (IR statement / literal constant / data)
    uint32_t kind : 7;

    /// Backend associated with this variable
    uint32_t backend : 2;

    /// Variable type (Bool/Int/Float/....)
    uint32_t type : 5;

    /// Is this a pointer variable that tracks a pending write? It holds a side
    /// effect reference that keeps the target marked dirty until it expires.
    uint32_t write_ptr : 1;

    /// Is this a pointer variable that a kernel writes through? Unlike
    /// 'write_ptr', this is a pure annotation without lifetime semantics,
    /// used by backends with explicit hazard tracking (Metal).
    uint32_t written : 1;

    // =======================  Miscellaneous flags ==========================
    // (+6 bits)

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

    /// Is this a cooperative vector?
    uint32_t coop_vec : 1;

    // =========== Entries that are temporarily used in jitc_eval() ============
    // (+10 bits -> 32 bits with all the preceding individiual bits = 4 bytes)
    // (+2*4 = 8 bytes)

    /// Argument type
    uint32_t param_type : 2;

    /// Is this variable marked as an output?
    uint32_t output_flag : 1;

    /// Consumed bit for operations that should only be executed once
    uint32_t consumed : 1;

    /// This field is used in two roles depending on context
    /// 1. Expanded evaluated arrays (ReduceOp.Expand for scatters on LLVM)
    /// 2. Tagged pointer arrays (to indicate GPU resource types on Metal)
    /// Access via the \ref reduce_op() and \ref resource_kind() functions below
    uint32_t reduce_op_or_resource_kind : 3;

    /// State of the variable array during compilation (see \ref ArrayState)
    uint32_t array_state : 3;

    /// Offset of the argument in the list of kernel parameters
    uint32_t param_offset;

    /// Register index
    uint32_t reg_index;

    // ========================  Side effect tracking  =========================
    // (+2*4 = 8 bytes)

    /// Number of queued side effects
    uint16_t ref_count_se;

    union {
        /// Reference count stash, see \ref jit_var_stash_ref()
        uint16_t ref_count_stashed;

        /// Variable arrays (is_array() == 1) and cooperative vectors
        /// (coop_vec == 1) store their array length here
        uint16_t array_length;
    };

    /// If nonzero, references an associated 'VariableExtra' field in 'state.extra'
    uint32_t extra;

    // =========================   Helper functions   ==========================

    bool is_evaluated() const { return kind == (uint32_t) VarKind::Evaluated; }
    bool is_literal()   const { return kind == (uint32_t) VarKind::Literal; }
    bool is_undefined() const { return kind == (uint32_t) VarKind::Undefined; }
    bool is_node()      const { return (uint32_t) kind > (uint32_t) VarKind::Literal; }
    bool is_dirty()     const { return ref_count_se > 0; }
    bool is_array()     const { return array_state != (uint32_t) ArrayState::Invalid; }
    bool is_coop_vec_literal()   const { return kind == (uint32_t) VarKind::CoopVecLiteral; }

    ReduceOp reduce_op() const { return (ReduceOp) reduce_op_or_resource_kind; }
    void set_reduce_op(ReduceOp op) { reduce_op_or_resource_kind = (uint32_t) op; }
    ResourceKind resource_kind() const { return (ResourceKind) reduce_op_or_resource_kind; }
    void set_resource_kind(ResourceKind rk) { reduce_op_or_resource_kind = (uint32_t) rk; }
};

static_assert(sizeof(Variable) == 64);

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
    // The first four members are laid out in *exactly* the same order as the
    // corresponding fields of 'Variable'. Copy with a single vmovdqu 32-byte copy.
    uint32_t scope;
    uint32_t dep[4];
    uint64_t literal;
    uint32_t size;

    /// Bits 0..15 hold (kind|backend|type|write_ptr|written), and bits 16..31
    /// hold the array length for array/coop-vec variables.
    uint32_t packed;

    VariableKey(const Variable &v) {
        memcpy((void *) this, (const void *) &v.scope, 32);
        uint32_t array_length =
            (v.is_array() || v.coop_vec) ? (uint32_t) v.array_length : 0u;
        // The bitfields kind:7|backend:2|type:5|write_ptr:1|written:1 occupy
        // bits 0..15 of the 32-bit word immediately after 'scratch' (LSB-first
        // packing), which is exactly the layout 'packed' wants. Reading the raw
        // word and masking avoids extracting and re-assembling each field
        // (~14 dependent insns).
        uint32_t flag_word;
        memcpy(&flag_word, &v.scratch + 1, sizeof(flag_word));
        packed = (flag_word & 0xFFFFu) | (array_length << 16);
    }

    bool operator==(const VariableKey &v) const {
        return memcmp((const void *) this, (const void *) &v,
                      9 * sizeof(uint32_t)) == 0;
    }
};

/// Check condition required for fast VariableKey construction
static_assert(
    offsetof(VariableKey, scope)   == offsetof(Variable, scope)   - offsetof(Variable, scope) &&
    offsetof(VariableKey, dep)     == offsetof(Variable, dep)     - offsetof(Variable, scope) &&
    offsetof(VariableKey, literal) == offsetof(Variable, literal) - offsetof(Variable, scope) &&
    offsetof(VariableKey, size)    == offsetof(Variable, size)    - offsetof(Variable, scope),
    "The layout of Variable and VariableKey are incompatible (compiler aligment/packing issue)!");


#pragma pack(pop)

/// Helper class to hash VariableKey instances
struct VariableKeyHasher {
    size_t operator()(const VariableKey &k) const {
        // 'mix' is based on wyhash: a 64x64->128 bit multiply with folded halves
        auto mix = [](uint64_t a, uint64_t b) -> uint64_t {
            XXH128_hash_t r = XXH_mult64to128(a, b);
            return r.low64 ^ r.high64;
        };

        uint64_t w[4];
        memcpy(w, &k, sizeof(w));

        // Perform three independent 128-bit multiplies. This is one of the most
        // costly steps of tracing, and the computation is arranged this way to
        // permit overlapping their latency. XOR-ing each input with a constant
        // avoids degenerate all-zero multiplies. We reuse xxHash's 64-bit primes.
        uint64_t a = mix(w[0] ^ PRIME64_1, w[1] ^ PRIME64_2);
        uint64_t b = mix(w[2] ^ PRIME64_3, w[3] ^ PRIME64_4);
        uint64_t c = mix((uint64_t) k.packed ^ PRIME64_5, PRIME64_1);
        return (size_t) (a ^ b ^ c);
    }
};

/// Cache data structure for local value numbering
using LVNMap =
    tsl::robin_map<VariableKey, uint32_t, VariableKeyHasher,
                   std::equal_to<VariableKey>,
                   std::allocator<std::pair<VariableKey, uint32_t>>,
                   /* StoreHash = */ true>;

static_assert(
    sizeof(VariableKey) == 9 * sizeof(uint32_t),
    "VariableKey: incorrect size, likely an issue with padding/packing!");

#if defined(DRJIT_ENABLE_CUDA)
/// Caches basic information about a CUDA device
struct CUDADevice {
    // CUDA device context
    CUcontext context;

    /// Associated CUDA stream handle
    CUstream stream = nullptr;

    /// A CUDA event for synchronization purposes
    CUevent event = nullptr;

    /// A CUDA event for synchronization with external streams
    CUevent sync_stream_event = nullptr;

    /// CUDA device ID
    int id;

    /// Device compute capability (major * 10 + minor)
    uint32_t compute_capability;

    /// Targeted PTX version (major * 10 + minor)
    uint32_t ptx_version;

    /// Number of SMs
    uint32_t sm_count;

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

    // Compute a good launch configuration. This heuristic has three "regimes"
    //
    // 1. For very small workloads with just a few warps, try to maximize
    //    the number of used SMs by generating single-warp blocks.
    //
    // 2. As the workload gets larger, maintain a number of blocks
    //    matching the hardware SM count and progressively add warps.
    //
    // 3. At some point, the maximum number of warps per block is reached,
    //    and we switch over to generating more blocks instead. The method
    //    tries to maintain a block count that is a multiple of the number of
    //    SMs when the block count is less than 4*#SM count, to avoid
    //    imbalance.
    //
    //    The maximum # of warps per block for a particular kernel is not the
    //    hardware maximum but rather inferred by the occupancy-optimizing
    //    function ``cuOccupancyMaxPotentialBlockSize()``.
    void get_launch_config(uint32_t *blocks_out, uint32_t *threads_out,
                           uint32_t size, uint32_t max_threads = 1024,
                           uint32_t max_blocks_per_sm = 0) const {

        uint32_t warp_size           = 32,
                 warp_count          = (size + warp_size - 1) / warp_size,
                 max_warps_per_block = (max_threads + warp_size - 1) / warp_size;

        uint32_t block_count, warps_per_block;
        if (warp_count <= sm_count) {
            block_count = warp_count;
            warps_per_block = 1;
        } else {
            block_count = sm_count;
            warps_per_block = (warp_count + block_count - 1) / block_count;

            if (warps_per_block > max_warps_per_block) {
                // Compute the needed number of blocks given the max. warps per block
                block_count = (warp_count + max_warps_per_block - 1) / max_warps_per_block;

                // Ideally, the block count should be a multiple of the SM count, in
                // case they all take a very similar amount of time. Let's just do this
                // when the # of blocks is still small-ish.
                if (block_count < sm_count * 4)
                    block_count = (block_count + sm_count - 1) / sm_count * sm_count;

                uint32_t max_blocks = max_blocks_per_sm * sm_count;
                if (max_blocks && block_count > max_blocks) {
                    // Optional: the caller can upper-bound the number of blocks
                    // per SM. In that case, we can't generate a configuration with
                    // a thread per element, and the caller will need some kind of
                    // grid-stride loop or similar.
                    block_count = max_blocks;
                    warps_per_block = max_warps_per_block;
                } else {
                    // Given the block count, we can now compute the number of warps per block
                    warps_per_block = (warp_count + block_count - 1) / block_count;

                    // Some blocks may no longer be needed following this computation, remove them
                    block_count = (warp_count + warps_per_block - 1) / warps_per_block;
                }
            }
        }

        if (block_count * warps_per_block < warp_count && !max_blocks_per_sm) {
            fprintf(stderr,
                    "get_launch_config(): internal error for size=%u, "
                    "max_threads=%u, max_blocks_per_sm=%u.\n",
                    size, max_threads, max_blocks_per_sm);
            abort();
        }

        if (blocks_out)
            *blocks_out  = block_count;

        if (threads_out)
            *threads_out = warps_per_block * warp_size;
    }
};
#endif

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
    int allowOpacityMicromaps;
    int allowClusteredGeometry; // OptiX 9.0 ABI
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

struct KernelKey;

#if defined(DRJIT_ENABLE_METAL)
/// Identifies one of the block (prefix) reduction kernel families in
/// metal_kernels.metal; used to look up lazily created pipelines.
enum class MetalReduceKind : uint32_t {
    Small,     // block_reduce_small_*: one thread serially reduces one block
    Chunk,     // block_reduce_*: one threadgroup reduces one chunk of a block
    WideChunk, // block_reduce_wide_*: as above, but writing the accumulator
               // type (f16 input -> f32 chunk totals, for prefix reductions)
    Scan,      // block_prefix_reduce_*: one threadgroup prefix-reduces one chunk
    Dot,       // reduce_dot_*: fused dot-product chunk reduction (op ignored)
    Count
};

/// Kind of the command encoder currently open on a Metal thread state
enum class MetalEncoderKind : uint32_t {
    None = 0,
    Compute,
    Blit
};

/// Enumeration of the precompiled utility kernels in metal_kernels.metal. The
/// order must match ``metal_kernel_names`` in metal_core.mm.
enum class MetalKernel : uint32_t {
    CompressScatter,
    MkpermPhase1,
    MkpermPhase3,
    MkpermDetectOffsets,
    MkpermPhase1Tiny,
    MkpermPhase4Tiny,
    Aggregate,
    MemsetU16,
    MemsetU32,
    MemsetU64,
    ConvertF32F16,
    DeinterleaveU16,
    DeinterleaveU32,
    InterleaveU16,
    InterleaveU32,
    Count
};

/// Caches basic information about a Metal device
struct MetalDevice {
    /// MTLDevice handle (opaque pointer to keep Objective-C / Metal types out of shared headers)
    void *device;

    /// id<MTLCommandQueue> used to submit work to the GPU
    void *queue;

    /// id<MTLLibrary> holding the precompiled utility kernels
    void *utility_lib;

    /// Precompiled compute pipeline states (owned +1), indexed by MetalKernel
    void *pipelines[(uint32_t) MetalKernel::Count];

    /// Lazily created block (prefix) reduction pipelines (owned +1), indexed
    /// by [MetalReduceKind][reduction][type]. See
    /// jitc_metal_block_reduce_pipeline().
    void *reduce_pipelines[(uint32_t) MetalReduceKind::Count]
                          [(uint32_t) ReduceOp::Count]
                          [(uint32_t) VarType::Count];

    /// Maximum number of threads per threadgroup
    uint32_t max_threads_per_threadgroup;

    uint32_t simd_width;

    /// Maximum threadgroup (shared) memory available to a kernel, in bytes
    uint32_t threadgroup_memory_bytes;

    /// True if the device supports hardware ray tracing acceleration
    bool supports_ray_tracing;

    /// True if the device supports Metal 4 (tensor ops / cooperative vectors)
    bool supports_metal4;

    /// Cached human-readable device name (owned, freed at shutdown)
    char *name;
};
#endif

/// Represents a single stream of a parallel communication
struct ThreadStateBase {
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
    uint32_t scope = 2;

    /// Registry index of the 'self' pointer of the call being recorded
    uint32_t call_self_value = 0;

    /// .. and the JIT variable that it will be mapped to
    uint32_t call_self_index = 0;

    /// Indicates, if the thread state is used to record or replay frozen
    /// functions.
    KernelRecordingMode recording_mode = KernelRecordingMode::Inactive;

    /**
     * \brief DrJit device ID associated with this device
     *
     * This value may differ from the CUDA device ID if the machine contains
     * CUDA devices that are incompatible with DrJit. For CUDA it indexes
     * \ref state.devices, for Metal \ref state.metal_devices.
     *
     * Equals -1 for LLVM ThreadState instances.
     */
    int device = 0;

    /// ---------------------------- CUDA-specific ----------------------------

#if defined(DRJIT_ENABLE_CUDA)
    /// Redundant copy of the device context
    CUcontext context = nullptr;

    /// Associated CUDA stream handle
    CUstream stream = nullptr;

    /// A CUDA event for synchronization purposes
    CUevent event = nullptr;

    /// A CUDA event for synchronization with external streams
    CUevent sync_stream_event = nullptr;

    /// Device compute capability (major * 10 + minor)
    uint32_t compute_capability = 0;

    /// Targeted PTX version (major * 10 + minor)
    uint32_t ptx_version = 0;

    // Support for stream-ordered memory allocations (async alloc/free)
    bool memory_pool = false;
#endif

#if defined(DRJIT_ENABLE_OPTIX)
    /// OptiX pipeline associated with the next kernel launch
    OptixPipelineData *optix_pipeline = nullptr;
    OptixShaderBindingTable *optix_sbt = nullptr;
#endif

    /// ---------------------------- Metal-specific ----------------------------

#if defined(DRJIT_ENABLE_METAL)
    /// MTLDevice handle — opaque to keep Metal types out of shared headers
    void *metal_device = nullptr;

    /// id<MTLCommandQueue> used to submit work to the GPU
    void *metal_queue = nullptr;

    /// A id<MTLCommandBuffer> with pending work
    void *metal_cb = nullptr;

    /// Current Metal command encoder and its kind
    void *metal_encoder = nullptr;
    MetalEncoderKind metal_encoder_kind = MetalEncoderKind::None;

    /// SIMD execution width (typically 32)
    uint32_t metal_simd_width = 32;

    /// Maximum threads per threadgroup
    uint32_t metal_max_threads = 1024;
#endif
};


struct ThreadState : public ThreadStateBase {
    virtual ~ThreadState();
    ThreadState() = default;
    ThreadState(const ThreadState &other) = default;

    /**
     * Schedules a barrier taks.
     *
     * \brief The barrier ensures that subsequently launched kernels can't start
     * to run until all kernels in the current launch have finished. This should
     * be called after a set of concurrent kernels have been launched.
     */
    virtual void barrier();

    virtual Task *launch(Kernel kernel, KernelKey &key, XXH128_hash_t hash,
                         uint32_t size, std::vector<void *> &kernel_params,
                         const std::vector<uint32_t> &kernel_param_ids,
                         KernelHistoryEntry *kernel_history_entry) = 0;

    /// Fill a device memory region with constants of a given type
    virtual void memset_async(void *ptr, uint32_t size, uint32_t isize,
                              const void *src) = 0;

    /// Reduce elements within blocks
    virtual void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                              uint32_t block_size, const void *in,
                              void *out) = 0;

    /// Implements various kinds of prefix reductions
    virtual void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                                     uint32_t block_size, bool exclusive,
                                     bool reverse, const void *in,
                                     void *out) = 0;

    /// Compute a dot product of two equal-sized arrays
    virtual void reduce_dot(VarType type, const void *ptr_1,
                            const void *ptr_2,
                            uint32_t size, void *out) = 0;

    /**
     * \brief Compute ``C = op_A(A) @ op_B(B)`` for row-major matrices.
     *
     * ``op_X`` is the identity or the transpose, selected by the ``At`` /
     * ``Bt`` flags. Logical shapes after applying the flags are
     * ``A : (M, K)``, ``B : (K, N)``, ``C : (M, N)``. The element type is
     * given by ``type`` (one of ``Float16``, ``Float32``, ``Float64``,
     * ``Int32``, ``UInt32``); half-precision inputs accumulate in single
     * precision.
     *
     * If ``batch`` is non-null and describes a non-empty batch
     * (``n_bdims + n_rdims > 0``), the call is dispatched as a batched
     * GEMM: the first ``n_bdims`` batch dims iterate across distinct
     * output tiles (grid dims), while the remaining ``n_rdims`` dims are
     * contracted into the same output tile (reduce dims). Zero strides in
     * ``a_stride`` / ``b_stride`` encode broadcasts of the corresponding
     * operand. See \ref GemmBatch for a detailed description of the
     * indexing convention.
     */
    virtual void batched_gemm(VarType type, bool At, bool Bt,
                              uint32_t M, uint32_t N, uint32_t K,
                              const GemmBatch *batch,
                              const void *A, const void *B, void *C) = 0;

    /// Mask compression
    virtual uint32_t compress(const uint8_t *in, uint32_t size,
                              uint32_t *out) = 0;

    /// Compute a permutation to reorder an integer array into discrete groups
    virtual uint32_t block_mkperm(const uint32_t *values, uint32_t size,
                                    uint32_t block_size, uint32_t bucket_count,
                                    uint32_t *perm, uint32_t *offsets) = 0;

    /// Perform a synchronous copy operation
    virtual void memcpy(void *dst, const void *src, size_t size) = 0;

    /// Perform an assynchronous copy operation
    virtual void memcpy_async(void *dst, const void *src, size_t size) = 0;

    /// Asynchronously update a single element in memory
    virtual void poke(void *dst, const void *src, uint32_t size) = 0;

    virtual void aggregate(void *dst, AggregationEntry *agg, uint32_t size) = 0;

    // Enqueue a function to be run on the host once backend computation is done
    virtual void enqueue_host_func(void (*callback)(void *), void *payload) = 0;

    /// LLVM: Notify the thread state, that a variable has been expanded using
    /// \c jitc_var_expand. This is required to record the ThreadState.
    virtual void notify_expand(uint32_t index);

    /// LLVM: reduce a variable that was previously expanded due to dr.ReduceOp.Expand
    virtual void reduce_expanded(VarType vt, ReduceOp op, void *data,
                                 uint32_t exp, uint32_t size);

    /// Narrow a float32 buffer to float16 (For Metal float16 scatter-reductions)
    virtual void narrow_f32_to_f16(void *dst, const void *src, uint32_t size);

    /// Pack a set of matrices/vectors for use with the cooperative vector API
    virtual void coop_vec_pack(uint32_t count, const void *in,
                               const MatrixDescr *in_d, void *out,
                               const MatrixDescr *out_d) = 0;

    /// Reduces an array of booleans by filling trailing elements and applying a
    /// UInt32 reduction.
    virtual void block_reduce_bool(uint8_t *values, uint32_t size, uint8_t *out,
                                   ReduceOp op);

    /// Some kernels use the width of an array in a computation. When using the
    /// kernel freezing feature, this requires special precautions to ensure
    /// that the resulting capture remains usable with different array sizes.
    /// This notification function exists so that this special-case handling can
    /// be realized.
    virtual void notify_opaque_width(uint32_t index, uint32_t width_index);

    /// Notifies the thread state that an allocation should not be initialized
    /// as part of the evaluation of an undefined variable. This is required for
    /// frozen functions to handle undefined variables.
    virtual void notify_init_undefined(uint32_t index);

    /// Notify the \c ThreadState that \c jitc_free has been called on a pointer.
    /// This is required for kernel freezing.
    virtual void notify_free(const void *ptr);

    // The DisabledThreadState/RecordThreadState override this to return the
    // wrapped thread state
    virtual ThreadState *actual_state();

    /// Reset internal dynamic state
    void reset_state();
};

/// RAII helper for scoped reset dynamic thread state
class scoped_reset_thread_state {
public:
    scoped_reset_thread_state(ThreadState* ts) :ts(ts), cached_ts(*ts) {
        ts->reset_state();
    }

    ~scoped_reset_thread_state() {
        *ts = std::move(cached_ts);
    }
    scoped_reset_thread_state(const scoped_reset_thread_state &) = delete;
    scoped_reset_thread_state &operator=(const scoped_reset_thread_state &) = delete;
private:
    ThreadStateBase* ts = nullptr;
    ThreadStateBase cached_ts;
};

/// Key data structure for kernel source code & device ID
struct KernelKey {
    char *str = nullptr;
    int device = 0;
    uint64_t flags = 0;
    /// 128-bit source code hash; Dr.Jit treats these hashes as collision-free
    /// kernel identifiers, hence key equality compares hashes, not source code.
    XXH128_hash_t hash{ 0, 0 };

    KernelKey(char *str, XXH128_hash_t hash, int device, uint64_t flags)
        : str(str), device(device), flags(flags), hash(hash) {}

    bool operator==(const KernelKey &k) const {
        return hash.low64 == k.hash.low64 && hash.high64 == k.hash.high64 &&
               device == k.device && flags == k.flags;
    }
};

/// Helper class to hash KernelKey instances
struct KernelHash {
    size_t operator()(const KernelKey &k) const {
        size_t hash = k.hash.high64;
        hash_combine(hash, (size_t) k.flags + size_t(k.device + 1));
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

using PointerMap = tsl::robin_map<const void *, uint32_t, PointerHasher>;

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

/// Records the full JIT compiler state (most frequently two used entries at top)
struct State {
    /// Must be held to access members of this data structure
    Lock lock;

    /// Must be held to access 'state.alloc_free'
    Lock alloc_free_lock;

    /// A flat list of variable data structures, including unused one
    std::vector<Variable> variables;

    /// Bit-packed free list of indices into 'variables' that are currently unused
    FreeSlots unused_variables;

    /// Maps from a key characterizing a variable to its index
    LVNMap lvn_map;

    /// A flast list of variable VariableExtra data structures (see its
    /// definition for documentation). Includes unused ones.
    std::vector<VariableExtra> extra;

    /// LIFO free list of indices into 'extra' that are currently unused
    std::vector<uint32_t> unused_extra;

    /// Counter to create variable scopes that enforce a variable ordering
    uint32_t scope_ctr = 2;
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

#if defined(DRJIT_ENABLE_CUDA)
    /// Available devices and their CUDA IDs
    std::vector<CUDADevice> devices;
#endif

#if defined(DRJIT_ENABLE_METAL)
    /// Available Metal devices (Apple Silicon only)
    std::vector<MetalDevice> metal_devices;
#endif

    /// State associated with each DrJit thread
    std::vector<ThreadState *> tss;

    /// Map of currently allocated memory regions
    AllocUsedMap alloc_used;

    /// Map of currently unused memory regions
    AllocInfoMap alloc_free;

    /// Keep track of current memory usage and a maximum watermark
    size_t alloc_usage    [(int) JitBackend::Count] { 0 },
           alloc_allocated[(int) JitBackend::Count] { 0 },
           alloc_watermark[(int) JitBackend::Count] { 0 };

    /// Limit the output of jit_var_str()?
    uint32_t print_limit = 20;

    /// Statistics on kernel launches
    size_t kernel_hard_misses = 0;
    size_t kernel_soft_misses = 0;
    size_t kernel_hits = 0;
    size_t kernel_launches = 0;

    /// Cache of previously compiled kernels
    KernelCache kernel_cache;

    /// Generation counter, incremented whenever the kernel cache is cleared.
    uint32_t kernel_cache_generation = 0;

    /// Kernel launch history
    KernelHistory kernel_history = KernelHistory();

    /// Print variable leak warnings?
    bool leak_warnings = true;

#if defined(DRJIT_ENABLE_OPTIX)
    /// Default OptiX pipeline for testcases etc.
    OptixPipelineData *optix_default_pipeline = nullptr;
    /// Default OptiX Shader Binding Table for testcases etc.
    OptixShaderBindingTable *optix_default_sbt = nullptr;
    /// Index of the JIT variable handling the lifetime of the default Optix SBT
    uint32_t optix_default_sbt_index = 0;
#endif

#ifndef NDEBUG
    /// Mapping from pointers that are managed by variables to their variable
    /// indices. This is used for debugging purposes in frozen functions.
    PointerMap ptr_to_variable;
#endif


    State() {
        variables.resize(1);
        extra.resize(1);
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

/// Thread-local Dr.Jit state
struct ThreadLocal {
    ThreadState *ts_llvm = nullptr;
#if defined(DRJIT_ENABLE_CUDA)
    ThreadState *ts_cuda = nullptr;
#endif
#if defined(DRJIT_ENABLE_METAL)
    ThreadState *ts_metal = nullptr;
#endif

    // Compilation flags
    uint32_t flags = (uint32_t) JitFlag::Default;

    // Default backend
    JitBackend def_backend = JitBackend::None;
};

#if defined(_MSC_VER)
  extern __declspec(thread) ThreadLocal tls;
#else
  extern __thread ThreadLocal tls;
#endif

// Compatibility aliases that access fields in 'tls' using the previous naming convention
#define thread_state_llvm (tls.ts_llvm)
#if defined(DRJIT_ENABLE_CUDA)
#  define thread_state_cuda (tls.ts_cuda)
#endif
#if defined(DRJIT_ENABLE_METAL)
#  define thread_state_metal (tls.ts_metal)
#endif
#define jitc_flags_v      (tls.flags)
#define default_backend   (tls.def_backend)

// Map TLS accesses through std::launder() (optimization boundary) to force the compiler
// to materialize the lookup once instead of potentially replicating it across branches.
JIT_INLINE ThreadLocal &jitc_thread_local() { return *std::launder(&tls); }

extern ThreadState *jitc_init_thread_state(JitBackend backend);

inline ThreadState *thread_state(JitBackend backend) {
    ThreadState *result;
    switch (backend) {
#if defined(DRJIT_ENABLE_CUDA)
        case JitBackend::CUDA:
            result = thread_state_cuda;
            break;
#endif
#if defined(DRJIT_ENABLE_METAL)
        case JitBackend::Metal:
            result = thread_state_metal;
            break;
#endif
        case JitBackend::LLVM:
            result = thread_state_llvm;
            break;

        default:
            result = nullptr;
            break;
    }

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

#if defined(DRJIT_ENABLE_CUDA)
/// Set the currently active device & stream
extern void jitc_cuda_set_device(int device);
#endif

/// Wait for all computation on the current stream to finish. If \c hold_lock
/// is true, \c state.lock is kept through the wait, preventing other threads
/// from enqueuing new work in the meantime.
extern void jitc_sync_thread(bool hold_lock = false);
extern void jitc_sync_thread(ThreadState *stream, bool hold_lock = false);

/// Wait for all computation on the current device to finish
extern void jitc_sync_device();

/// Wait for all computation on *all devices* to finish
extern void jitc_sync_all_devices();

/// Backend predicates that fold to a compile-time ``false`` when the backend
/// is not built, allowing the compiler to remove dead branches.
template <typename T> inline bool jitc_is_cuda(T b) {
#if defined(DRJIT_ENABLE_CUDA)
    return (JitBackend) b == JitBackend::CUDA;
#else
    (void) b; return false;
#endif
}

template <typename T> inline bool jitc_is_metal(T b) {
#if defined(DRJIT_ENABLE_METAL)
    return (JitBackend) b == JitBackend::Metal;
#else
    (void) b; return false;
#endif
}

template <typename T> inline bool jitc_is_llvm(T b) {
    return (JitBackend) b == JitBackend::LLVM;
}

/// Returns true if the given backend uses GPU device memory (CUDA or Metal)
template <typename T> inline bool jitc_is_device_backend(T b) {
    return jitc_is_cuda(b) || jitc_is_metal(b);
}

/// Search for a shared library and dlopen it if possible
void *jitc_find_library(const char *fname, const char *glob_pat,
                        const char *env_var);

#if defined(DRJIT_ENABLE_CUDA)
/// Return a pointer to the CUDA stream associated with the currently active device
extern void* jitc_cuda_stream();

/// Return a pointer to the CUDA context associated with the currently active device
extern void* jitc_cuda_context();

/// Push a new CUDA context to the currently active device
extern void jitc_cuda_push_context(void *);

/// Pop the current CUDA context and return it
extern void* jitc_cuda_pop_context();
#endif

extern void jitc_set_flags(uint32_t flags);

extern uint32_t jitc_flags();

/// Selectively enables/disables flags
extern void jitc_set_flag(JitFlag flag, int enable);

/// Checks whether a given flag is active. Returns zero or one.
extern int jitc_flag(JitFlag flag);

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

/// Returns true if the value is represented using 8 bits, excluding booleans.
inline bool jitc_is_b8(VarType type) {
    return type == VarType::Int8 || type == VarType::UInt8;
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
inline bool jitc_is_any_zero(Variable *v) {
    if (!v->is_literal() && !v->is_coop_vec_literal())
        return false;

    switch ((VarType) v->type) {
        case VarType::Float16: return v->literal == 0x8000ull || v->literal == 0;
        case VarType::Float32: return v->literal == 0x80000000ull || v->literal == 0;
        case VarType::Float64: return v->literal == 0x8000000000000000ull || v->literal == 0;
        default: return v->literal == 0;
    }
}

inline bool jitc_is_one(Variable *v) {
    if (!v->is_literal() && !v->is_coop_vec_literal())
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

extern bool jitc_is_max(Variable *v);
extern bool jitc_is_min(Variable *v);
inline void jitc_var_set_data(Variable &v, void *data) { v.data = data; }

// ====================================================================
//                         Event data structure
// ====================================================================

struct EventData {
    JitBackend backend;
    bool enable_timing;
    ThreadState* ts;
    union {
#if defined(DRJIT_ENABLE_CUDA)
        CUevent cuda_event;
#endif
        Task* llvm_task;
#if defined(DRJIT_ENABLE_METAL)
        // For Metal we store a (id<MTLSharedEvent>, value) pair encoded into
        // two 64-bit slots. The pointer is held in metal_event and the
        // monotonic counter associated with the recorded signal/wait is in
        // metal_value.
        struct {
            void *metal_event;
            uint64_t metal_value;
        };
#endif
    };

    EventData(JitBackend backend, bool enable_timing)
        : backend(backend), enable_timing(enable_timing), ts(nullptr) {
#if defined(DRJIT_ENABLE_CUDA)
        if (jitc_is_cuda(backend))
            cuda_event = nullptr;
        else
#endif
#if defined(DRJIT_ENABLE_METAL)
        if (jitc_is_metal(backend)) {
            metal_event = nullptr;
            metal_value = 0;
        }
        else
#endif
            llvm_task = nullptr;
    }

    ~EventData() = default;
};

extern const char *var_kind_name[(int) VarKind::Count];
