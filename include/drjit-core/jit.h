/*
    drjit-core/jit.h -- Self-contained JIT compiler for CUDA & LLVM.

    This library implements a self-contained tracing JIT compiler that supports
    both CUDA PTX and LLVM IR as intermediate representations. It takes care of
    many tricky aspects, such as recording of arithmetic and higher-level
    operations (loops, virtual function calls), asynchronous memory allocation
    and release, kernel caching and reuse, constant propagation, common
    subexpression elimination, etc.

    While the library is internally implemented using C++17, this header file
    provides a compact C99-compatible API that can be used to access all
    functionality. The library is thread-safe: multiple threads can
    simultaneously dispatch computation to one or more CPUs/GPUs.

    As an alternative to the fairly low-level API defined here, you may prefer
    the interface in 'include/drjit-core/array.h', which provides a header-only
    C++ array abstraction with operator overloading that dispatches to the C
    API. The Dr.Jit parent project (https://github.com/mitsuba-renderer/drjit)
    can also be interpreted as continuation of this kind of abstraction, which
    adds further components like a library of transcendental mathematical
    operations, automatic differentiation support, Python bindings, etc.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include "macros.h"

#if defined(__cplusplus)
extern "C" {
#endif

// ====================================================================
//         Initialization, device enumeration, and management
// ====================================================================

/**
 * \brief List of backends that can be targeted by Dr.Jit
 *
 * Dr.Jit can perform computation using one of several computational
 * backends. Before use, a backend must be initialized via \ref jit_init().
 */
#if defined(__cplusplus)
enum class JitBackend : uint32_t {
    None = 0,

    /// CUDA backend (requires CUDA >= 10, generates PTX instructions)
    CUDA = (1 << 0),

    /// LLVM backend targeting the CPU (generates LLVM IR)
    LLVM = (1 << 1)
};
#else
enum JitBackend {
    JitBackendNone = 0,
    JitBackendCUDA = (1 << 0),
    JitBackendLLVM = (1 << 1)
};
#endif

/**
 * \brief Initialize a JIT compiler backend
 *
 * The function <tt>jit_init()</tt> must be called before using the JIT
 * compiler. It takes a bit-wise OR of elements of the \ref JitBackend
 * enumeration and tries to initialize each specified backend. Query \ref
 * jit_has_backend() following this operation to check if a backend was
 * initialized successfully. This function does nothing when initialization has
 * already occurred. It is possible to re-initialize the JIT following a call
 * to \ref jit_shutdown(), which can be useful to reset the state, e.g., in
 * testcases.
 */
extern JIT_EXPORT void
jit_init(uint32_t backends JIT_DEF((uint32_t) JitBackend::CUDA |
                                   (uint32_t) JitBackend::LLVM));

/**
 * \brief Launch an asynchronous thread that will execute jit_init() and
 * return immediately
 *
 * On machines with several GPUs, \ref jit_init() will set up a CUDA
 * environment on all devices when <tt>cuda=true</tt> is specified. This can be
 * a rather slow operation (e.g. 1 second). This function provides a convenient
 * alternative to hide this latency, for instance when importing this library
 * from an interactive Python session which doesn't need the JIT right away.
 *
 * The \c llvm and \c cuda arguments should be set to \c 1 to initialize the
 * corresponding backend, and \c 0 otherwise.
 *
 * Note that it is safe to call <tt>jit_*</tt> API functions following
 * initialization via \ref jit_init_async(), since it acquires a lock to the
 * internal data structures.
 */
extern JIT_EXPORT void
jit_init_async(uint32_t backends JIT_DEF((uint32_t) JitBackend::CUDA |
                                         (uint32_t) JitBackend::LLVM));

/// Check whether the LLVM backend was successfully initialized
extern JIT_EXPORT int jit_has_backend(JIT_ENUM JitBackend backend);

/**
 * \brief Release resources used by the JIT compiler, and report reference leaks.
 *
 * If <tt>light=1</tt>, this function performs a "light" shutdown, which
 * flushes any still running computation and releases unused memory back to the
 * OS or GPU. It will also warn about leaked variables and memory allocations.
 *
 * If <tt>light=0</tt>, the function furthermore completely unloads the LLVM and
 * CUDA backends. This frees up more memory but means that a later call to \ref
 * jit_init() or \ref jit_init_async() will be slow.
 */
extern JIT_EXPORT void jit_shutdown(int light JIT_DEF(0));

/**
 * \brief Wait for all computation scheduled by the current thread to finish
 *
 * Each thread using Dr.Jit will issue computation to an independent queue.
 * This function only synchronizes with computation issued to the queue of the
 * calling thread.
 */
extern JIT_EXPORT void jit_sync_thread();

/// Wait for all computation on the current device to finish
extern JIT_EXPORT void jit_sync_device();

/// Wait for all computation on the *all devices* to finish
extern JIT_EXPORT void jit_sync_all_devices();

// ====================================================================
//                    CUDA/LLVM-specific functionality
// ====================================================================

/// Return the no. of available CUDA devices that are compatible with Dr.Jit.
extern JIT_EXPORT int jit_cuda_device_count();

/**
 * \brief Set the active CUDA device.
 *
 * The argument must be between 0 and <tt>jit_cuda_device_count() - 1</tt>,
 * which only accounts for Dr.Jit-compatible devices. This is a per-thread
 * property: independent threads can optionally issue computation to different
 * GPUs.
 */
extern JIT_EXPORT void jit_cuda_set_device(int device);

/**
 * \brief Return the CUDA device ID associated with the current thread
 *
 * The result is in the range of 0 and <tt>jit_cuda_device_count() - 1</tt>.
 * When the machine contains CUDA devices that are incompatible with Dr.Jit (due
 * to a lack of 64-bit addressing, uniform address space, or managed memory),
 * this number may differ from the default CUDA device ID. Use
 * <tt>jit_cuda_device_raw()</tt> in that case.
 */
extern JIT_EXPORT int jit_cuda_device();

/// Return the raw CUDA device associated with the current thread
extern JIT_EXPORT int jit_cuda_device_raw();

/// Return the CUDA stream associated with the current thread
extern JIT_EXPORT void* jit_cuda_stream();

/// Return the CUDA context associated with the current thread
extern JIT_EXPORT void* jit_cuda_context();

/// Push a new CUDA context to be associated with the current thread
extern JIT_EXPORT void jit_cuda_push_context(void *);

/// Pop the CUDA context associated to the current thread and return it
extern JIT_EXPORT void* jit_cuda_pop_context();

/// Query the compute capability of the current device (e.g. '52')
extern JIT_EXPORT int jit_cuda_compute_capability();

/**
 * \brief Override generated PTX version and compute capability
 *
 * Dr.Jit generates code that runs on a wide variety of platforms supporting
 * at least the PTX version and compute capability of 60, and 50, respectively.
 * Those versions can both be bumped via this function---there is no
 * performance advantage in doing so, though some more recent features (e.g.
 * atomic operations involving double precision values) require specifying a
 * newer compute capability.
 */
extern JIT_EXPORT void jit_cuda_set_target(uint32_t ptx_version,
                                           uint32_t compute_capability);

/// Look up an CUDA driver function by name
extern JIT_EXPORT void *jit_cuda_lookup(const char *name);

/**
 * \brief Add CUDA event synchronization between thread state's and external
 * CUDA stream.
 *
 * An event will be recorded into the thread's states stream and the external stream
 * will wait on the event before performing any subsequent work.
 *
 * \param stream The CUstream handle of the external stream
 */
extern JIT_EXPORT void jit_cuda_sync_stream(uintptr_t stream);

/**
 * \brief Override the target CPU, features, and vector width of the LLVM backend
 *
 * The LLVM backend normally generates code for the detected native hardware
 * architecture akin to compiling with <tt>-march=native</tt>. This function
 * can be used to change the following code generation-related parameters:
 *
 * \param target_cpu
 *     Target CPU (e.g. <tt>haswell</tt>)
 *
 * \param target_features
 *     Comma-separated list of LLVM feature flags (e.g. <tt>+avx512f</tt>).
 *     This should be set to <tt>nullptr</tt> if you do not wish to specify
 *     individual features.
 *
 * \param vector_width
 *     Width of vector registers (e.g. 8 for AVX). Must be a power of two, and
 *     can be a multiple of the hardware register size to enable unrolling.
 */
extern JIT_EXPORT void jit_llvm_set_target(const char *target_cpu,
                                           const char *target_features,
                                           uint32_t vector_width);

/// Get the CPU that is currently targeted by the LLVM backend
extern JIT_EXPORT const char *jit_llvm_target_cpu();

/// Get the list of CPU features currently used by the LLVM backend
extern JIT_EXPORT const char *jit_llvm_target_features();

/// Get the major, minor and patch version of the LLVM library
extern JIT_EXPORT void jit_llvm_version(int *major, int *minor, int *patch);

/// Get the vector width of the LLVM backend
extern JIT_EXPORT uint32_t jit_llvm_vector_width();

/// Specify the number of threads that are used to parallelize the computation
extern JIT_EXPORT void jit_llvm_set_thread_count(uint32_t size);

/// Return the number of threads that are used to parallelize the computation
extern JIT_EXPORT uint32_t jit_llvm_thread_count();

/// Specify the number of SIMD packets that form one parallel work item
extern JIT_EXPORT void jit_llvm_set_block_size(uint32_t size);

/// Return the number of SIMD packets that form one parallel work item
extern JIT_EXPORT uint32_t jit_llvm_block_size();

// ====================================================================
//                        Logging infrastructure
// ====================================================================

#if defined(__cplusplus)
enum class LogLevel : uint32_t {
    Disable, Error, Warn, Info, InfoSym, Debug, Trace
};
#else
enum LogLevel {
    LogLevelDisable, LogLevelError, LogLevelWarn, LogLevelInfo,
    LogLevelInfoSym, LogLevelDebug, LogLevelTrace
};
#endif

/**
 * \brief Control the destination of log messages (stderr)
 *
 * By default, this library prints all log messages to the console (\c stderr).
 * This function can be used to control the minimum log level for such output
 * or prevent it entirely. In the latter case, you may wish to enable logging
 * via a callback in \ref jit_set_log_level_callback(). Both destinations can also
 * be enabled simultaneously, potentially using different log levels.
 */
extern JIT_EXPORT void jit_set_log_level_stderr(JIT_ENUM LogLevel level);

/// Return the currently set minimum log level for output to \c stderr
extern JIT_EXPORT JIT_ENUM LogLevel jit_log_level_stderr();


/**
 * \brief Control the destination of log messages (callback)
 *
 * This function can be used to specify an optional callback that will be
 * invoked with the contents of library log messages, whose severity matches or
 * exceeds the specified \c level.
 */
typedef void (*LogCallback)(JIT_ENUM LogLevel, const char *);
extern JIT_EXPORT void jit_set_log_level_callback(JIT_ENUM LogLevel level,
                                                  LogCallback callback);

/// Return the currently set minimum log level for output to a callback
extern JIT_EXPORT JIT_ENUM LogLevel jit_log_level_callback();

/// Print a log message with the specified log level and message
extern JIT_EXPORT void jit_log(JIT_ENUM LogLevel level, const char* fmt, ...);

/// Raise an exception message with the specified message
extern JIT_EXPORT void jit_raise(const char* fmt, ...) JIT_NORETURN_FORMAT;

/// Terminate the application due to a non-recoverable error
extern JIT_EXPORT void jit_fail(const char* fmt, ...) JIT_NOEXCEPT JIT_NORETURN_FORMAT;

// ====================================================================
//                         Memory allocation
// ====================================================================

#if defined(__cplusplus)
enum class AllocType : uint32_t {
    /**
     * Memory that is located on the host (i.e., the CPU). When allocated via
     * \ref jit_malloc(), host memory is immediately ready for use, and
     * its later release via \ref jit_free() also occurs instantaneously.
     *
     * Note, however, that released memory is kept within a cache and not
     * immediately given back to the operating system. Call \ref
     * jit_flush_malloc_cache() to also flush this cache.
     */
    Host,

    /**
     * Like \c Host memory, except that it may only be used *asynchronously*
     * within a computation performed by drjit-core.
     *
     * In particular, host-asynchronous memory obtained via \ref jit_malloc()
     * should not be written to directly (i.e. outside of drjit-core), since it
     * may still be used by a currently running kernel. Releasing
     * host-asynchronous memory via \ref jit_free() also occurs
     * asynchronously.
     *
     * This type of memory is used internally when running code via the LLVM
     * backend, and when this process is furthermore parallelized using Dr.Jit's
     * internal thread pool.
     */
    HostAsync,

    /**
     * Memory on the host that is "pinned" and thus cannot be paged out.
     * Host-pinned memory is accessible (albeit slowly) from CUDA-capable GPUs
     * as part of the unified memory model, and it also can be a source or
     * destination of asynchronous host <-> device memcpy operations.
     *
     * Host-pinned memory has asynchronous semantics similar to \c HostAsync.
     */
    HostPinned,

    /**
     * Memory that is located on a device (i.e., one of potentially several
     * GPUs).
     *
     * Device memory has asynchronous semantics similar to \c HostAsync.
     */
    Device,

    /// Number of possible allocation types
    Count
};
#else
enum AllocType {
    AllocTypeHost,
    AllocTypeHostPinned,
    AllocTypeDevice,
    AllocTypeCount
};
#endif

/**
 * \brief Allocate memory of the specified type
 *
 * Under the hood, Dr.Jit implements a custom allocation scheme that tries to
 * reuse allocated memory regions instead of giving them back to the OS/GPU.
 * This eliminates inefficient synchronization points in the context of CUDA
 * programs, and it can also improve performance on the CPU when working with
 * large allocations.
 *
 * The returned pointer is guaranteed to be sufficiently aligned for any kind
 * of use.
 *
 */
extern JIT_EXPORT void *jit_malloc(JIT_ENUM AllocType type, size_t size)
    JIT_MALLOC;

/**
 * \brief Release a given pointer asynchronously
 *
 * For CPU-only arrays (\ref AllocType::Host), <tt>jit_free()</tt> is
 * synchronous and very similar to <tt>free()</tt>, except that the released
 * memory is placed in Dr.Jit's internal allocation cache instead of being
 * returned to the OS. The function \ref jit_flush_malloc_cache() can optionally
 * be called to also clear this cache.
 *
 * When \c ptr is an asynchronous host pointer (\ref AllocType::HostAsync) or
 * GPU-accessible pointer (\ref AllocType::Device, \ref AllocType::HostPinned),
 * the associated memory region is possibly still being used by a running
 * kernel, and it is therefore merely *scheduled* to be reclaimed once this
 * kernel finishes.
 *
 * Kernel launches and memory-related operations (malloc, free) occur
 * asynchronously but using a linear ordering when they are scheduled by the
 * same thread (they will be placed into the same <i>stream</i> in CUDA
 * terminology). Extra care must be taken in the context of multi-threaded
 * software: it is not permissible to e.g. allocate memory on one thread,
 * launch a kernel using it, then immediately release that memory from a
 * different thread, because a valid ordering is not guaranteed in that case.
 * Operations like \ref jit_sync_thread(), \ref jit_sync_device(), and \ref
 * jit_sync_all_devices() can be used to defuse such situations.
 */
extern JIT_EXPORT void jit_free(void *ptr);

/// Release all currently unused memory to the GPU / OS
extern JIT_EXPORT void jit_flush_malloc_cache();

/// Clear the peak memory usage statistics
extern JIT_EXPORT void jit_malloc_clear_statistics();

/// Flush internal kernel cache
extern JIT_EXPORT void jit_flush_kernel_cache();

/// Query the flavor of a memory allocation made using \ref jit_malloc()
extern JIT_EXPORT JIT_ENUM AllocType jit_malloc_type(void *ptr);

/// Query the device associated with a memory allocation made using \ref jit_malloc()
extern JIT_EXPORT int jit_malloc_device(void *ptr);

/**
 * \brief Asynchronously change the flavor of an allocated memory region and
 * return the new pointer
 *
 * The operation is *always* asynchronous and, hence, will need to be followed
 * by an explicit synchronization via \ref jit_sync_thread() if memory is
 * migrated from the GPU to the CPU and expected to be accessed on the CPU
 * before the transfer has finished. Nothing needs to be done in the other
 * direction, e.g. when migrating memory that is subsequently accessed by
 * a GPU kernel.
 *
 * When no migration is necessary, the function simply returns the input
 * pointer. If migration is necessary, the behavior depends on the supplied
 * <tt>move</tt> parameter. When <tt>move==0</tt>, the implementation schedules
 * an asynchronous copy and leaves the old pointer undisturbed. If
 * <tt>move==1</tt>, the old pointer is asynchronously freed once the copy
 * operation finishes.
 *
 * When both source and target are of type \ref AllocType::Device, and
 * when the currently active device (determined by the last call to \ref
 * jit_set_device()) does not match the device associated with the allocation,
 * a peer-to-peer migration is performed.
 */
extern JIT_EXPORT void *jit_malloc_migrate(void *ptr, JIT_ENUM AllocType type,
                                           int move JIT_DEF(1));

// ====================================================================
//                          Pointer registry
// ====================================================================

/**
 * \brief Register a pointer with Dr.Jit's pointer registry
 *
 * Dr.Jit provides a central registry that maps registered pointer values to
 * low-valued 32-bit IDs. The main application is efficient virtual function
 * dispatch via \ref jit_var_call(), through the registry could be used for other
 * applications as well.
 *
 * This function registers the specified pointer \c ptr with the registry,
 * returning the associated ID value, which is guaranteed to be unique within
 * the specified domain \c domain. The domain is normally an identifier that is
 * associated with the "flavor" of the pointer (e.g. instances of a particular
 * class), and which ensures that the returned ID values are as low as
 * possible.
 *
 * Caution: for reasons of efficiency, the \c domain parameter is assumed to a
 * static constant that will remain alive. The RTTI identifier
 * <tt>typeid(MyClass).name()<tt> is a reasonable choice that satisfies this
 * requirement.
 *
 * Raises an exception when ``ptr`` is ``nullptr``, or when it has already been
 * registered with *any* domain.
 */
extern JIT_EXPORT uint32_t jit_registry_put(JIT_ENUM JitBackend backend,
                                            const char *domain, void *ptr);

/**
 * \brief Remove a pointer from the registry
 *
 * Throws an exception if the pointer is not currently registered.
 */
extern JIT_EXPORT void jit_registry_remove(const void *ptr);

/// Return the instance ID associated with the pointer, or 0 if it is ``NULL``.
extern JIT_EXPORT uint32_t jit_registry_id(const void *ptr);

/// Return the largest instance ID for the given domain
extern JIT_EXPORT uint32_t jit_registry_id_bound(JitBackend backend,
                                                 const char *domain);

/// Return the pointer value associated with a given instance ID
extern JIT_EXPORT void *jit_registry_ptr(JitBackend backend,
                                         const char *domain, uint32_t id);

/// Return an arbitrary pointer value associated with a given domain
extern JIT_EXPORT void *jit_registry_peek(JitBackend backend, const char *domain);

/// Disable any instances that are currently registered in the registry
extern JIT_EXPORT void jit_registry_clear();

// ====================================================================
//                        Variable management
// ====================================================================

#if defined(__cplusplus)
/**
 * \brief Variable types supported by the JIT compiler.
 *
 * A type promotion routine in the Dr.Jit Python bindings depends on on this
 * exact ordering, so please don't change.
 */
enum class VarType : uint32_t {
    Void, Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32,
    Int64, UInt64, Pointer, Float16, Float32, Float64, Count
};
#else
enum VarType {
    VarTypeVoid, VarTypeBool, VarTypeInt8, VarTypeUInt8,
    VarTypeInt16, VarTypeUInt16, VarTypeInt32, VarTypeUInt32,
    VarTypeInt64, VarTypeUInt64, VarTypePointer, VarTypeFloat16,
    VarTypeFloat32, VarTypeFloat64, VarTypeCount
};
#endif

/**
 * \brief Create a variable representing a literal constant
 *
 * <b>Advanced usage</b>: When \c eval is nonzero, the variable is directly
 * created in evaluated form, which means that subsequent usage will access the
 * contents via memory instead of including the actual constant value in
 * generated PTX/LLVM code.
 */
extern JIT_EXPORT uint32_t jit_var_literal(JIT_ENUM JitBackend backend,
                                           JIT_ENUM VarType type,
                                           const void *value,
                                           size_t size JIT_DEF(1),
                                           int eval JIT_DEF(0));


// Short-hand versions for making scalar literals
extern JIT_EXPORT uint32_t jit_var_u32(JitBackend backend, uint32_t value);
extern JIT_EXPORT uint32_t jit_var_i32(JitBackend backend, int32_t value);

NAMESPACE_BEGIN(drjit)
struct half;
NAMESPACE_END(drjit)
extern JIT_EXPORT uint32_t jit_var_f16(JitBackend backend, drjit::half value);
extern JIT_EXPORT uint32_t jit_var_f32(JitBackend backend, float value);
extern JIT_EXPORT uint32_t jit_var_f64(JitBackend backend, double value);

extern JIT_EXPORT uint32_t jit_var_u64(JitBackend backend, uint64_t value);
extern JIT_EXPORT uint32_t jit_var_i64(JitBackend backend, int64_t value);

extern JIT_EXPORT uint32_t jit_var_bool(JitBackend backend, bool value);
extern JIT_EXPORT uint32_t jit_var_class(JitBackend backend, void *value);

// Create an array representing uninitialized memory. The actual allocation is delayed.
extern JIT_EXPORT uint32_t jit_var_undefined(JitBackend backend, VarType type,
                                             size_t size);

/**
 * \brief Create a counter variable
 *
 * This operation creates a variable of type \ref VarType::UInt32 that will
 * evaluate to <tt>0, ..., size - 1</tt>.
 */
extern JIT_EXPORT uint32_t jit_var_counter(JIT_ENUM JitBackend backend,
                                           size_t size);

#if defined(__cplusplus)
/// List of operations supported by \ref jit_var_new_op()
enum class JitOp : uint32_t {
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
    Popc, Clz, Ctz, Brev,

    /// Bit-wise operations
    And, Or, Xor,

    // Shifts
    Shl, Shr,

    // Fast approximations
    Rcp, Rsqrt,

    // Multi-function generator (CUDA)
    Sin, Cos, Exp2, Log2,

    // Total number of operations
    Count
};
#else
enum JitOp {
    JitOpNeg, JitOpNot, JitOpSqrt, JitOpAbs, JitOpAdd, JitOpSub, JitOpMul,
    JitOpDiv, JitOpMod, JitOpMulhi, JitOpFma, JitOpMin, JitOpMax, JitOpCeil,
    JitOpFloor, JitOpRound, JitOpTrunc, JitOpEq, JitOpNeq, JitOpLt, JitOpLe,
    JitOpGt, JitOpGe, JitOpSelect, JitOpPopc, JitOpClz, JitOpCtz, JitOpAnd,
    JitOpOr, JitOpXor, JitOpShl, JitOpShr, JitOpRcp, JitOpRsqrt, JitOpSin,
    JitOpCos, JitOpExp2, JitOpLog2, JitOpCount
};
#endif


/**
 * \brief Perform an arithmetic operation dynamically
 *
 * This function dynamically dispatches to the set of unary, binary, and
 * ternary arithmetic arithmetic operations supported by Dr.Jit. The following
 * are equivalent:
 *
 * ```
 * // Perform operation statically (preferred)
 * result = jit_var_add(a0, a1);
 *
 * // Perform operation dynamically
 * uint32_t deps[2] = { a0, a1 };
 * result = jit_var_op(JitOp::Add, deps);
 * ```
 *
 * This function exists to handle situations where the static style
 * is inconvenient.
 *
 * \param op
 *    The operation to be performed
 *
 * \param dep
 *    Operand list. It is the developer's responsibility to supply an array
 *    containing the right number of operands for the requested operation.
 */
extern JIT_EXPORT uint32_t jit_var_op(JIT_ENUM JitOp op, const uint32_t *dep);

/// Compute `-a0` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_neg(uint32_t a0);

/// Compute `~a0` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_not(uint32_t a0);

/// Compute `sqrt(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_sqrt(uint32_t a0);

/// Compute `abs(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_abs(uint32_t a0);

/// Compute `a0 + a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_add(uint32_t a0, uint32_t a1);

/// Compute `a0 - a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_sub(uint32_t a0, uint32_t a1);

/// Compute `a0 * a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_mul(uint32_t a0, uint32_t a1);

/// Compute `a0 / a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_div(uint32_t a0, uint32_t a1);

/// Compute `a0 % a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_mod(uint32_t a0, uint32_t a1);

/// Compute the high part of `a0 * a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_mulhi(uint32_t a0, uint32_t a1);

/// Compute `a0 * a1 + a2` (fused) and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_fma(uint32_t a0, uint32_t a1, uint32_t a2);

/// Compute `min(a0, a1)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_min(uint32_t a0, uint32_t a1);

/// Compute `max(a0, a1)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_max(uint32_t a0, uint32_t a1);

/// Compute `ceil(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_ceil(uint32_t a0);

/// Compute `floor(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_floor(uint32_t a0);

/// Compute `round(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_round(uint32_t a0);

/// Compute `trunc(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_trunc(uint32_t a0);

/// Compute `a0 == a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_eq(uint32_t a0, uint32_t a1);

/// Compute `a0 != a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_neq(uint32_t a0, uint32_t a1);

/// Compute `a0 < a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_lt(uint32_t a0, uint32_t a1);

/// Compute `a0 <= a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_le(uint32_t a0, uint32_t a1);

/// Compute `a0 > a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_gt(uint32_t a0, uint32_t a1);

/// Compute `a0 >= a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_ge(uint32_t a0, uint32_t a1);

/// Compute `a0 ? a1 : a2` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_select(uint32_t a0, uint32_t a1, uint32_t a2);

/// Compute the population count of `a0` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_popc(uint32_t a0);

/// Count leading zeros of `a0` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_clz(uint32_t a0);

/// Count trailing zeros of `a0` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_ctz(uint32_t a0);

/// Reverse the bits of `a0` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_brev(uint32_t a0);

/// Compute `a0 & a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_and(uint32_t a0, uint32_t a1);

/// Compute `a0 | a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_or(uint32_t a0, uint32_t a1);

/// Compute `a0 ^ a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_xor(uint32_t a0, uint32_t a1);

/// Compute `a0 << a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_shl(uint32_t a0, uint32_t a1);

/// Compute `a0 >> a1` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_shr(uint32_t a0, uint32_t a1);

/// Approximate `1 / a0` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_rcp(uint32_t a0);

/// Approximate `1 / sqrt(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_rsqrt(uint32_t a0);

/// Approximate `sin(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_sin_intrinsic(uint32_t a0);

/// Approximate `cos(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_cos_intrinsic(uint32_t a0);

/// Approximate `exp2(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_exp2_intrinsic(uint32_t a0);

/// Approximate `log2(a0)` and return a variable representing the result
extern JIT_EXPORT uint32_t jit_var_log2_intrinsic(uint32_t a0);

/// Return a variable indicating valid lanes within a function call
extern JIT_EXPORT uint32_t jit_var_call_mask(JitBackend backend);

/**
 * \brief Perform an ordinary or reinterpreting cast of a variable
 *
 * This function casts the variable \c index to a different type
 * \c target_type. When \c reinterpret is zero, this is like a C-style cast (i.e.,
 * <tt>new_value = (Type) old_value;</tt>). When \c reinterpret is nonzero, the
 * value is reinterpreted without converting the value (i.e.,
 * <tt>memcpy(&new_value, &old_value, sizeof(Type));</tt>), which requires that
 * source and target type are of the same size.
 */
extern JIT_EXPORT uint32_t jit_var_cast(uint32_t index,
                                        JIT_ENUM VarType target_type,
                                        int reinterpret);

/**
 * \brief Create a variable that refers to a memory region
 *
 * This function creates a 64 bit unsigned integer literal that refers to a
 * memory region. Optionally, if \c dep is nonzero, the created variable will
 * hold a reference to the variable \c dep until the pointer is destroyed,
 * which is useful when implementing operations that access global memory.
 *
 * A nonzero value should be passed to the \c write parameter if the pointer is
 * going to be used to perform write operations. Dr.Jit needs to know about
 * this to infer whether a future scatter operation to \c dep requires making a
 * backup copy first.
 */
extern JIT_EXPORT uint32_t jit_var_pointer(JIT_ENUM JitBackend backend,
                                               const void *value,
                                               uint32_t dep,
                                               int write);
/**
 * \brief Create a variable that reads from another variable
 *
 * This operation creates a variable that performs a <em>masked gather</em>
 * operation equivalent to <tt>mask ? source[index] : 0</tt>. The variable
 * \c index must be an integer array, and \c mask must be a boolean array.
 */
extern JIT_EXPORT uint32_t jit_var_gather(uint32_t source, uint32_t index,
                                          uint32_t mask);

/**
 * \brief Gather a contiguous packet of values
 *
 * This function is analogous to (but generally more efficient than)
 * four separate gathers from indices ``index*n+ [0,1,.., n-1]``.
 * The number ``n`` must be a power of two.
 *
 * The output variable indices are written to ``out``, which must point
 * to a memory region of size ``sizeof(uint32_t)*n``.
 */
extern JIT_EXPORT void jit_var_gather_packet(size_t n, uint32_t source,
                                             uint32_t index, uint32_t mask,
                                             uint32_t *out);

/// Reverse the order of a JIT array
extern JIT_EXPORT uint32_t jit_var_reverse(uint32_t index);

#if defined(__cplusplus)
/// Reduction operations for \ref jit_var_scatter() \ref jit_reduce()
enum class ReduceOp : uint32_t {
    // Plain scatter, overwrites the previous element
    Identity,

    // Combine additively
    Add,

    /// Combine multiplicatively
    Mul,

    /// Combine via min(old, new)
    Min,

    /// Combine via max(old, new)
    Max,

    /// Binary AND
    And,

    /// Binary OR
    Or,

    // This isn't an operation, it just tracks the total number of supported strategies
    Count
};

/// For scatter-reductions, this enumeration specifies the strategy to be used
enum class ReduceMode : uint32_t {
    /// Inspect the JIT flags to choose between 'ScatterReduceLocal' (the
    /// default) and 'Atomic'
    Auto,

    /// Insert an atomic operation into the program
    Direct,

    /// Preprocess scatters going to the same address within the warp (CUDA) or
    /// packet (SSE/AVX/AVX512/..) to issue fewer atomic memory transactions.
    Local,

    /// The caller guarantees that there are no conflicts (scatters targeting
    /// the same elements). This means that the generated code can safely
    /// perform an ordinary gather, update the value, and then write back the
    /// result.
    NoConflicts,

    /// Temporarily expand the target array to a much larger size to avoid
    /// write conflicts, which enables the use of non-atomic operations (i.e.,
    /// the 'NoConflicts' mode). This is particularly helpful for small (e.g.
    /// scalar) arrays. For bigger arrays and on machines with many cores (which
    /// increases the amount of replication needed), the resulting storage
    /// costs can be prohibitive.
    ///
    /// This feature is only supported on the LLVM backend. Other backends
    /// interpret this flag as if 'Auto' had been specified.
    Expand,

    /// When setting this mode, the caller guarantees that there will be no
    /// conflicts, and that every entry is written exactly single time using an
    /// index vector representing a permutation (it's fine this permutation is
    /// accomplished by multiple separate write operations, but there should be no
    /// more than 1 write to each element).
    ///
    /// This mode primarily exists to enable internal optimizations that
    /// Dr.Jit uses when differentiating vectorized function calls and compressed
    /// loops.
    ///
    /// Giving 'Permute' as an argument to a (nominally read-only) gather
    /// operation is helpful because we then know that the reverse-mode derivative
    /// of this operation can be a plain scatter instead of a more costly
    /// atomic scatter-add.
    ///
    /// Giving 'Permute' as an argument to a scatter operation is helpful
    /// because we then know that the forward-mode derivative does not depend
    /// on any prior derivative values associated with that array, as all
    /// current entries will be overwritten.
    Permute
};
#else
enum ReduceOp {
    ReduceOpIdentity, ReduceOpAdd, ReduceOpMul, ReduceOpMin, ReduceOpMax,
    ReduceOpAnd, ReduceOpOr
};
enum ReduceMode {
    ReduceModeAuto, ReduceModeDirect, ReduceModeLocal,
    ReduceReplicate
};
#endif

/**
 * \brief Schedule a scatter or atomic scatter-reduction
 *
 * When ``op == ReduceOp::None``, this function performs an ordinary
 * scatter operation, i.e. a vectorized verson of the statement ``if (mask)
 * target[index] = value``. The variable ``index`` must be an integer array,
 * and ``mask`` must be a boolean array. The ``mode`` parameter is
 * ignored in this case.
 *
 * A direct write may not be safe (e.g. if unevaluated computation references
 * the array ``target``). The function thus returns the index of a new array
 * (which may happen to be identical to ``target``) and increases its reference
 * count by 1.
 *
 * For performance reasons, sequences involving multiple scatters to the same
 * array are assumed not to be in conflict with each other. They may execute in
 * an arbitrary order due to the underlying parallelization. This is fine if
 * the written addresses do not overlap. Otherwise, explicit evaluation via
 * ``jit_var_eval()`` is necessary to ensure a fixed ordering.
 *
 * If ``op != ReduceOp::None``, an atomic read-modify-write operation of
 * the desired type will be used instead of simply overwriting entries of
 * \c target. The ``mode`` parameter selects a compilation strategy in this ase.
 */
extern JIT_EXPORT uint32_t jit_var_scatter(uint32_t target, uint32_t value,
                                           uint32_t index, uint32_t mask,
                                           JIT_ENUM ReduceOp op JIT_DEF(ReduceOp::Identity),
                                           JIT_ENUM ReduceMode mode JIT_DEF(ReduceMode::Auto));

/**
 * \brief Scatter or scatter-reduce a contiguous packet of values
 *
 * This function is analogous to (but generally more efficient than)
 * four separate scatters from indices ``index*n+ [0,1,.., n-1]``.
 * The number ``n`` must be a power of two.
 *
 * The input variable indices are given via ``values``, which must point
 * to a memory region of size ``sizeof(uint32_t)*n``.
 */
extern JIT_EXPORT uint32_t jit_var_scatter_packet(size_t n, uint32_t target, const uint32_t *values,
                                                  uint32_t index, uint32_t mask,
                                                  JIT_ENUM ReduceOp op JIT_DEF(ReduceOp::Identity),
                                                  JIT_ENUM ReduceMode mode JIT_DEF(ReduceMode::Auto));

/// Check if a backend supports the specified type of scatter-reduction
extern JIT_EXPORT int jit_can_scatter_reduce(JitBackend backend, VarType vt,
                                             ReduceOp op);

/**
 * \brief Schedule a Kahan-compensated floating point atomic scatter-write
 *
 * This operation is just like ``jit_var_scatter()`` invoked with a floating
 * point operands and ``op=ReduceOp::Add``. The difference is that it
 * simultaneously adds to two different target buffers using the Kahan
 * summation algorithm.
 *
 * The implementation may overwrite the 'target_1' / 'target_2' pointers
 * if a copy needs to be made (for example, if another variable elsewhere
 * references the same variable).
 */
extern JIT_EXPORT void jit_var_scatter_add_kahan(uint32_t *target_1,
                                                 uint32_t *target_2,
                                                 uint32_t value,
                                                 uint32_t index,
                                                 uint32_t mask);

/**
 * \brief Atomically increment a counter and return the old value
 *
 * This operation is just like ``jit_var_scatter`` invoked with 32-bit unsigned
 * integer operands, the value ``1``, and op=ReduceOp::Add.
 *
 * The main difference is that this variant returns the *old* value before the
 * atomic write (in contrast to the more general scatter reduction, where doing
 * so would be rather complicated).
 *
 * This operation is a building block for stream compaction: threads can
 * scatter-increment a global counter to request a spot in an array.
 */
extern JIT_EXPORT uint32_t jit_var_scatter_inc(uint32_t *target,
                                               uint32_t index,
                                               uint32_t mask);

/**
 * \brief Create an identical copy of the given variable
 *
 * This function creates an exact copy of the variable \c index and returns the
 * index of the copy, whose reference count is initialized to 1.
 */
extern JIT_EXPORT uint32_t jit_var_copy(uint32_t index);


/**
 * Register an existing memory region as a variable in the JIT compiler, and
 * return its index. Its reference count is initialized to \c 1.
 *
 * \param type
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param cuda
 *    Is this a CUDA variable?
 *
 * \param ptr
 *    Point of the memory region
 *
 * \param size
 *    Number of elements (and *not* the size in bytes)
 *
 * \param free
 *    If free != 0, the JIT compiler will free the memory region via
 *    \ref jit_free() once its reference count reaches zero.
 *
 * \sa jit_var_mem_copy()
 */
extern JIT_EXPORT uint32_t jit_var_mem_map(JIT_ENUM JitBackend backend,
                                           JIT_ENUM VarType type, void *ptr,
                                           size_t size, int free);

/**
 * Copy a memory region onto the device and return its variable index. Its
 * reference count is initialized to \c 1.
 *
 * \param atype
 *    Enumeration characterizing the "flavor" of the source memory.
 *
 * \param cuda
 *    Is this a CUDA variable?
 *
 * \param vtype
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param ptr
 *    Point of the memory region
 *
 * \param size
 *    Number of elements (and *not* the size in bytes)
 *
 * \sa jit_var_mem_map()
 */
extern JIT_EXPORT uint32_t jit_var_mem_copy(JIT_ENUM JitBackend backend,
                                            JIT_ENUM AllocType atype,
                                            JIT_ENUM VarType vtype,
                                            const void *ptr,
                                            size_t size);

/// Increase the reference count of a given variable
extern JIT_EXPORT uint32_t jit_var_inc_ref_impl(uint32_t index) JIT_NOEXCEPT;

/// Decrease the reference count of a given variable
extern JIT_EXPORT void jit_var_dec_ref_impl(uint32_t index) JIT_NOEXCEPT;

#if defined(__GNUC__)
JIT_INLINE uint32_t jit_var_inc_ref(uint32_t index) JIT_NOEXCEPT {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to jit_var_dec_ref */
    if (__builtin_constant_p(index))
        return 0;
    else
        return jit_var_inc_ref_impl(index);
}

JIT_INLINE void jit_var_dec_ref(uint32_t index) JIT_NOEXCEPT {
    if (!__builtin_constant_p(index))
        jit_var_dec_ref_impl(index);
}
#else
#define jit_var_dec_ref jit_var_dec_ref_impl
#define jit_var_inc_ref jit_var_inc_ref_impl
#endif

/// Query the a variable's reference count (used by the test suite)
extern JIT_EXPORT uint32_t jit_var_ref(uint32_t index);

/**
 * \brief Temporarily stash the reference count of a variable
 *
 * The explanation of this operation needs a bit more context: Operations like
 * dr.if_stmt(), dr.while_loop(), etc., create temporary copies of their
 * inputs, which are needed to roll back changes (otherwise, the 'true_fn' of
 * dr.if_stmt() could, e.g., modify an input argument that is subsequently
 * accessed by 'false_fn')
 *
 * These copies have no actual cost since Dr.Jit's employs a copy-on-write
 * (COW) mechanism to simply reference the inputs multiple times.
 *
 * However, these extra copies cause problems when such a symbolic operation
 * has side effects (e.g., via dr.scatter). COW now requires the creation of a
 * copy in device memory, and that often turns out to be unncecessary
 * specifically for side effects.
 *
 * This operation provides a solution to this problem: when entering a symbolic
 * region, the reference counts of inputs are temporarily stashed, and these
 * stashed counts are subsequently used to make COW-related decision.
 *
 * It returns a handle that should be used to later undo the operation via \ref
 * jit_var_unstash_ref() when leaving the symbolic region.
 */
extern JIT_EXPORT uint64_t jit_var_stash_ref(uint32_t index);

/// Undo the change performed by \ref jit_var_stash_ref()
extern JIT_EXPORT void jit_var_unstash_ref(uint64_t handle);

/**
 * \brief Potentially evaluate a variable and return a pointer to its address
 * in device memory.
 *
 * This operation combines the following steps:
 *
 * - If the variable ``index`` is unevaluated or has pending side effects, then
 *   these taken care of first.
 *
 * - The function returns a pointer via the ``ptr_out`` parameter, which
 *   points to the array in device memory.
 *
 * - If ``index`` represents a literal constant or an undefined memory region,
 *   then the function returns a reference to a *new* variable with the
 *   evaluated result.
 *
 *   This is done because literal constant variables in Dr.Jit tend to be
 *   shared by many expressions. Evaluting such shared variables might
 *   therefore negatively impact the performance elsewhere. Otherwise, the
 *   function returns a new reference to the original ``index``.
 */
extern JIT_EXPORT uint32_t jit_var_data(uint32_t index, void **ptr_out);

/// Query the size of a given variable
extern JIT_EXPORT size_t jit_var_size(uint32_t index);

/// Query the type of a given variable
extern JIT_EXPORT JIT_ENUM VarType jit_var_type(uint32_t index);

/// Extract a label identifying the variable kind
extern JIT_EXPORT const char *jit_var_kind_name(uint32_t index);

/// Check if a variable has pending side effects
extern JIT_EXPORT int jit_var_is_dirty(uint32_t index);


#if defined(__cplusplus)

// Enumeration describing possible evaluation states of a Dr.Jit variable
enum class VarState : uint32_t {
    /// The variable has length 0 and effectively does not exist
    Invalid,

    /// An undefined memory region. Does not (yet) consume device memory.
    Undefined,

    /// A literal constant. Does not consume device memory.
    Literal,

    /// An ordinary unevaluated variable that is neither a literal constant nor symbolic.
    Unevaluated,

    /// An evaluated variable backed by a device memory region.
    Evaluated,

    /// An evaluated variable backed by a device memory region, with pending side effects.
    Dirty,

    /// A symbolic variable that could take on various inputs. Cannot be evaluated.
    Symbolic,

    /// This is a nested array, and the components have mixed states
    Mixed
};
#else
enum VarState {
    VarStateInvalid,
    VarStateUndefined,
    VarStateLiteral,
    VarStateUnevaluated,
    VarStateEvaluated,
    VarStateSymbolic,
    VarStateMixed
};
#endif

/// Check if a variable is a literal constant
extern JIT_EXPORT JIT_ENUM VarState jit_var_state(uint32_t index);

/// Check if a variable is a literal zero
extern JIT_EXPORT int jit_var_is_zero_literal(uint32_t index);

/// Check if a variable represents a normal (not NaN/infinity) literal
extern JIT_EXPORT int jit_var_is_finite_literal(uint32_t index);

/**
 * \breif Check if the data field of a variable is unaligned
 *
 * This function returns true if the data pointer of the variable is not
 * properly aligned in memory. Otherwise it returns false.
 */
extern JIT_EXPORT int jit_var_is_unaligned(uint32_t index);

/**
 * \brief Resize a scalar variable to a new size
 *
 * This function takes a scalar variable as input and changes its size to \c
 * size, potentially creating a new copy in case something already depends on
 * \c index. The returned copy is unevaluated.
 *
 * The function increases the reference count of the returned value.
 * When \c index is not a scalar variable and its size exactly matches \c size,
 * the function does nothing and just increases the reference count of
 * \c index. Otherwise, it fails.
 */
extern JIT_EXPORT uint32_t jit_var_resize(uint32_t index, size_t size);

/// Create a view of an existing variable that has a smaller size
extern JIT_EXPORT uint32_t jit_var_shrink(uint32_t index, size_t size);

/**
 * \brief Asynchronously migrate a variable to a different flavor of memory
 *
 * Returns the resulting variable index and increases its reference
 * count by one. When source and target type are identical, this function does
 * not perform a migration and simply returns the input index (though it
 * increases the reference count even in this case). When the source and target
 * types are different, the implementation schedules an asynchronous copy and
 * generates a new variable index.
 *
 * When both source & target are of type \ref AllocType::Device, and if the
 * current device (\ref jit_set_device()) does not match the device associated
 * with the allocation, a peer-to-peer migration is performed.
 */
extern JIT_EXPORT uint32_t jit_var_migrate(uint32_t index,
                                           JIT_ENUM AllocType type);

/// Query the current (or future, if unevaluated) allocation flavor of a variable
extern JIT_EXPORT JIT_ENUM AllocType jit_var_alloc_type(uint32_t index);

/// Query the device (or future, if not yet evaluated) associated with a variable
extern JIT_EXPORT int jit_var_device(uint32_t index);

/**
 * \brief Mark a variable as a scatter operation
 *
 * This function informs the JIT compiler that the variable 'index' has side
 * effects. It then steals a reference, includes the variable in the
 * next kernel launch, and de-references it following execution.
 */
extern JIT_EXPORT void jit_var_mark_side_effect(uint32_t index);

/**
 * \brief Return a human-readable summary of the contents of a variable
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JIT_EXPORT const char *jit_var_str(uint32_t index);

/**
 * \brief Read a single element of a variable and write it to 'dst'
 *
 * This function fetches a single entry from the variable with \c index at
 * offset \c offset and writes it to the CPU output buffer \c dst.
 *
 * This function is convenient to spot-check entries of an array, but it should
 * never be used to extract complete array contents due to its low performance
 * (every read will be performed via an individual transaction). This operation
 * fully synchronizes the host CPU & device.
 */
extern JIT_EXPORT void jit_var_read(uint32_t index, size_t offset,
                                    void *dst);

/**
 * \brief Copy 'dst' to a single element of a variable
 *
 * This function implements the reverse of jit_var_read(). This function is
 * convenient for testing, and to change localized entries of an array, but it
 * should never be used to access the complete contents of an array due to its
 * low performance (every write will be performed via an individual
 * asynchronous transaction).
 *
 * A direct write may not be safe (e.g. if unevaluated computation references
 * the array \c index). The function thus returns the index of a new array
 * (which may happen to be identical to \c index), whose reference count is
 * increased by 1.
 */
extern JIT_EXPORT uint32_t jit_var_write(uint32_t index, size_t offset,
                                         const void *src);

/**
 * \brief Create a new variable representing an array containing a specific
 * attribute associated with a specific domain in the registry.
 *
 * This function is very similar to jit_registry_attr_data but returns
 * a variable instead of a data pointer.
 *
 * \sa jit_registry_attr_data
 */
extern JIT_EXPORT uint32_t jit_var_registry_attr(JIT_ENUM JitBackend backend,
                                                 JIT_ENUM VarType type,
                                                 const char *domain,
                                                 const char *name);

// ====================================================================
//                 Kernel compilation and evaluation
// ====================================================================

/** \brief Schedule a variable \c index for future evaluation
 *
 * The function ignores literal constant arrays, because literal they tend to
 * be shared by many expressions. Evaluting such shared variables might
 * therefore negatively impact the performance elsewhere. Use the \ref
 * jit_var_schedule_force() interface to evaluate such variables into a *new*
 * representation.
 *
 * The function also ignores evaluated arrays and undefined memory regions
 * (created via \ref jit_var_undefined), since there is nothing to evaluate in
 * that case (that said, you may use \ref jit_var_schedule_force() to turn
 * undefined variables into a raw memory buffer).
 *
 * The function raises an error when \c index represents a *symbolic* variable,
 * which may never be evaluated.
 *
 * Returns \c 1 if anything was scheduled, and \c 0 otherwise. In the former
 * case, the caller should eventually invoke \ref jit_eval() to complete the
 * variable evaluation process.
 */
extern JIT_EXPORT int jit_var_schedule(uint32_t index);

/**
 * \brief Forcefully schedule a variable \c index for future evaluation via
 * \ref jit_eval()
 *
 * This function is is similar to \ref jit_var_schedule() but more aggressive.
 * In addition to the set of variable types handled by the former, it also
 * evaluates undefined arrays (by allocating uninitialized memory for them) and
 * literal constant arrays (by allocating memory and filling it with copies of
 * the constant). Both of these cases are handled immediately, rather than
 * postponing them to the next kernel compilation (\ref jit_eval()) since no
 * code generation is needed. In both of these cases, it returns a reference to
 * a new array representing the result. Otherwise, it returns a new reference
 * to ``index``.
 *
 * The function sets the \c rv output parameter to \c 1 when variables were
 * scheduled for later evaluation, which means that the caller should
 * eventually invoke \ref jit_eval() to complete the process. Otherwise, it
 * sets \c rv to \c 0.
 */
extern JIT_EXPORT uint32_t jit_var_schedule_force(uint32_t index, int *rv);

/**
 * \brief Evaluate the variable \c index right away, if it is unevaluated/dirty.
 *
 * This is a convenient shortcut to writing
 *
 * ```
 * if (jit_var_schedule(index))
 *    jit_eval();
 * ```
 *
 * Returns \c 1 if anything was evaluated, and \c 0 otherwise.
 */
extern JIT_EXPORT int jit_var_eval(uint32_t index);

/// Evaluate computation scheduled via \ref jit_var_schedule() and \ref
/// jit_var_schedule_force()
extern JIT_EXPORT void jit_eval();

/**
 * \brief Assign a callback function that is invoked when the variable is
 * evaluated or freed.
 *
 * The provided function should have the signature <tt>void callback(uint32_t
 * index, int free, void *callback_data)</tt>, where \c index is the variable
 * index, \c free == 0 indicates that the variable is evaluated, \c free == 1
 * indicates that it is freed, and \c callback_data is a user-specified value
 * that will additionally be supplied to the callback.
 *
 * Passing \c callback == nullptr will remove a previously set callback if any.
 */
extern JIT_EXPORT void
jit_var_set_callback(uint32_t index, void (*callback)(uint32_t, int, void *),
                     void *callback_data);

// ====================================================================
//      Functionality for debug output and GraphViz visualizations
// ====================================================================

/**
 * \brief Assign a descriptive label to a given variable
 *
 * The label is shown in the output of \ref jit_var_whos() and \ref
 * jit_var_graphviz()
 */
extern JIT_EXPORT uint32_t jit_var_set_label(uint32_t index, size_t nargs, ...);

/// Query the descriptive label associated with a given variable
extern JIT_EXPORT const char *jit_var_label(uint32_t index);

/**
 * \brief Return a human-readable summary of registered variables
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JIT_EXPORT const char *jit_var_whos();

/**
 * \brief Return a GraphViz representation of registered variables and their
 * dependencies
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JIT_EXPORT const char *jit_var_graphviz();

/**
 * \brief Push a string onto the label prefix stack
 *
 * Dr.Jit maintains a per-thread stack that is initially empty. If values are
 * pushed onto it, they will be concatenated to generate a prefix associated
 * with any subsequently created variables.
 *
 * The function \ref jit_var_graphviz() uses these prefixes de-clutter large
 * graph visualizations by drawing boxes around variables with a common prefix.
 */
extern JIT_EXPORT void jit_prefix_push(JIT_ENUM JitBackend backend,
                                       const char *value);

/// Pop a string from the label stack
extern JIT_EXPORT void jit_prefix_pop(JIT_ENUM JitBackend backend);

/// Query the current prefix. Returns \c nullptr if inactive.
extern JIT_EXPORT const char *jit_prefix(JIT_ENUM JitBackend);

// ====================================================================
//  JIT compiler status flags
// ====================================================================

/**
 * \brief Status flags to adjust/inspect the eagerness of the JIT compiler
 *
 * Certain Dr.Jit operations can operate in two different ways: they can be
 * executed at once, or they can be recorded to postpone evaluation to a later
 * point. The latter is generally more efficient because it enables fusion of
 * multiple operations that will then exchange information via registers
 * instead of global memory. The downside is that recording computation is
 * generally more complex/fragile and less suitable to interactive software
 * development (one e.g. cannot simply print array contents while something is
 * being recorded). The following list of flags can be used to control the
 * behavior of these features.
 */
#if defined(__cplusplus)
enum class JitFlag : uint32_t {
    /// Debug mode: annotates generated PTX/LLVM IR with Python line number
    /// information (if available)
    Debug = 1 << 0,

    /// Disable this flag to keep the Dr.Jit from reusing variable indices
    /// (helpful for low-level debugging of the Dr.Jit implementation)
    ReuseIndices = 1 << 1,

    /// Constant propagation: don't generate code for arithmetic involving
    /// literal constants
    ConstantPropagation = 1 << 2,

    /// Fast math (analogous to -ffast-math in C)
    FastMath = 1 << 3,

    /// Local value numbering: a cheap form of common subexpression elimination
    ValueNumbering = 1 << 4,

    /// Capture loops symbolically instead of unrolling and evaluating them
    /// iteratively
    SymbolicLoops = 1 << 5,

    /// Simplify loops by removing constant loop state variables. This also
    /// propagates literal constants into loops, which is useful for autodiff.
    OptimizeLoops = 1 << 6,

    /// Compress the loop state when executing evaluated loops?
    CompressLoops = 1 << 7,

    /// Capture function calls symbolically instead of evaluating their inputs,
    /// grouping them by instance ID, and then lauching a kernel per group
    SymbolicCalls = 1 << 8,

    /// Propagate constants through function calls and remove
    OptimizeCalls = 1 << 9,

    /// Merge functions produced by Dr.Jit when they have a compatible structure
    MergeFunctions = 1 << 10,

    /// Capture conditionals symbolically instead of evaluating both branches
    /// and combining their results.
    SymbolicConditionals = 1 << 11,

    /// Turn gathers that access 4 adjacent elements into packet loads?
    PacketOps = 1 << 12,

    /// Force execution through OptiX even if a kernel doesn't use ray tracing
    ForceOptiX = 1 << 13,

    /// Print the intermediate representation of generated programs
    PrintIR = 1 << 14,

    /// Maintain a history of kernel launches. Useful for profiling Dr.Jit code.
    KernelHistory = 1 << 15,

    /* Force synchronization after every kernel launch. This is useful to
       isolate crashes to a specific kernel, and to benchmark kernel runtime
       along with the KernelHistory feature. */
    LaunchBlocking = 1 << 16,

    /// Perform a local (warp/SIMD) reduction before issuing global atomics
    ScatterReduceLocal = 1 << 17,

    /// Set to \c true when Dr.Jit is capturing symbolic computation. This flag
    /// is managed automatically and should not be set by application code.
    SymbolicScope = 1 << 18,

    /// Freeze functions annotated with dr.freeze
    KernelFreezing = 1 << 19,

    /// Set to \c true when Dr.Jit is recording a frozen function
    FreezingScope = 1 << 20,

    /// Default flags
    Default = (uint32_t) ConstantPropagation | (uint32_t) ValueNumbering |
              (uint32_t) FastMath | (uint32_t) SymbolicLoops |
              (uint32_t) OptimizeLoops | (uint32_t) SymbolicCalls |
              (uint32_t) MergeFunctions | (uint32_t) OptimizeCalls |
              (uint32_t) SymbolicConditionals | (uint32_t) ReuseIndices |
              (uint32_t) ScatterReduceLocal | (uint32_t) PacketOps |
              (uint32_t) KernelFreezing,

    // Deprecated aliases, will be removed in a future version of Dr.Jit
    LoopRecord = SymbolicLoops,
    LoopOptimize = OptimizeLoops,
    VCallRecord = SymbolicCalls,
    VCallDeduplicate = MergeFunctions,
    VCallOptimize = OptimizeCalls,
    Recording = SymbolicScope
};
#else
enum JitFlag {
    JitFlagDebug = 1 << 0,
    JitFlagReuseIndices = 1 << 1,
    JitFlagConstantPropagation = 1 << 2,
    JitFlagFastMath = 1 << 3,
    JitFlagValueNumbering = 1 << 4,
    JitFlagSymbolicLoops = 1 << 5,
    JitFlagOptimizeLoops = 1 << 6,
    JitFlagCopmressLoops = 1 << 7,
    JitFlagSymbolicCalls = 1 << 8,
    JitFlagOptimizeCalls = 1 << 9,
    JitFlagMergeFunctions = 1 << 10,
    JitFlagSymbolicConditionals = 1 << 11,
    JitFlagPacketOps = 1 << 12,
    JitFlagForceOptiX = 1 << 13,
    JitFlagPrintIR = 1 << 14,
    JitFlagKernelHistory = 1 << 15,
    JitFlagLaunchBlocking = 1 << 16,
    JitFlagScatterReduceLocal = 1 << 17,
    JitFlagSymbolic = 1 << 18
    KernelFreezing = 1 << 19,
    FreezingScope = 1 << 20,
};
#endif

/// Set the JIT compiler status flags (see \ref JitFlags)
extern JIT_EXPORT void jit_set_flags(uint32_t flags);

/// Retrieve the JIT compiler status flags (see \ref JitFlags)
extern JIT_EXPORT uint32_t jit_flags();

/// Selectively enables/disables flags
extern JIT_EXPORT void jit_set_flag(JIT_ENUM JitFlag flag, int enable);

/// Checks whether a given flag is active. Returns zero or one.
extern JIT_EXPORT int jit_flag(JIT_ENUM JitFlag flag);

// ====================================================================
//  Advanced JIT usage: recording loops, virtual function calls, etc.
// ====================================================================

/**
 * \brief Begin a recording session
 *
 * Dr.Jit can record virtual function calls and loops to preserve them 1:1 in
 * the generated code. This function indicates to Dr.Jit that the program is
 * starting to record computation. The function sets \ref JitFlag.Recording and
 * returns information that will later enable stopping or canceling a recording
 * session via \ref jit_record_end().
 *
 * Recording sessions can be nested.
 */
extern JIT_EXPORT uint32_t jit_record_begin(JIT_ENUM JitBackend backend, const char *name);

/// Return a checkpoint within a recorded computation for resumption via jit_record_end
extern JIT_EXPORT uint32_t jit_record_checkpoint(JIT_ENUM JitBackend backend);

/**
 * \brief End a recording session
 *
 * The parameter \c state should be the return value from a prior call to \ref
 * jit_record_begin(). This function cleans internal data structures and
 * recovers the previous setting of the flag \ref JitFlag.Recording.
 */
extern JIT_EXPORT void jit_record_end(JIT_ENUM JitBackend backend,
                                      uint32_t state, int cleanup);

/**
 * \brief Begin recording a symbolic loop
 *
 * \param name
 *    A descriptive name
 *
 * \param symbolic
 *    Does this loop occur within a symbolic execution context? (For example,
 *    is this a symbolic loop nested within another symbolic loop?)
 *
 * \param n_indices
 *    Number of loop state variables
 *
 * \param indices
 *    Variable indices of the loop state variables
 *
 * \return
 *    Produces an opaque loop handle that holds a reference to all initial loop
 *    state variables.
 *
 * \remark
 *    Upon return, the function will additionally have modified the \c indices
 *    array to store borrowed references representing symbolic variables that
 *    should be used to record the computation performed by loop body.
 *
 *    In case of an inconsistency (e.g., incompatible state variable sizes),
 *    the function reverts any changes and raises an exception.
 */
extern JIT_EXPORT uint32_t jit_var_loop_start(const char *name,
                                              bool symbolic,
                                              size_t n_indices,
                                              uint32_t *indices);

/**
 * \brief Update the inner_in field of the symbolic loop
 *
 * \param loop
 *    A symbolic loop handle produced by ``jit_var_loop_start()``.
 *
 * \param indices
 *    The indices to update the field with
 *
 * \remark
 *    Once the jit_var_loop_start is called and the phi variables are written
 *    to their fields, this can be used to update the inner_in field of the loop
 *    by reading back the loop state variables.
 *    This is necessary, to catch cases where a variable is added to the
 * loop-state twice.
 */
extern JIT_EXPORT void jit_var_loop_update_inner_in(uint32_t loop,
                                                    uint32_t *indices);

/**
 * \brief Create a node representing the loop condition of a symbolic loop.
 *
 * \param loop
 *    A symbolic loop handle produced by ``jit_var_loop_start()``.
 *
 * \param active
 *    A boolean-valued Dr.Jit array with the loop's condition expression.
 *
 * \return
 *     Returns an handle that holds a reference to ``active``. This object
 *     should be passed to ``jit_var_loop_end``.
 */
extern JIT_EXPORT uint32_t jit_var_loop_cond(uint32_t loop, uint32_t active);

/**
 * \brief Finish symbolic recording of a loop
 *
 * \param loop
 *    A symbolic loop handle produced by ``jit_var_loop_start()``.
 *
 * \param cond
 *    A loop condition handle produced by ``jit_var_loop_cond()``.
 *
 * \param indices
 *    The indices array should contain the loop variable state following
 *    execution of the loop body.
 *
 * \param checkpoint
 *    A checkpoint handle obtained by jit_record_begin()
 *
 * \return
 *    When the function returns ``1``, the loop recording process has concluded.
 *    When the function returns ``0``, Dr.Jit identified an optimization opportunity
 *    to simplify unnecessary loop state variables, and the caller should record
 *    the loop body once more (with initial state values provided by ``indices``).
 *
 * \remark
 *    Upon successful termination (return value ``1``), the function will
 *    additionally have modified the \c indices array to store new references
 *    representing the variable state following termination of the loop.
 */
extern JIT_EXPORT int jit_var_loop_end(uint32_t loop, uint32_t cond,
                                       uint32_t *indices, uint32_t checkpoint);

/**
 * \brief Begin symbolic recording of an ``if`` statement
 *
 * \param name
 *    A descriptive name
 *
 * \param symbolic
 *    Does this conditional occur within a symbolic execution context? (For
 *    example, is it nested within a symbolic loop?)
 *
 * \param cond_t
 *    Variable index of a boolean array specifying whether the 'true' branch
 *    should be executed. Set this to (condition & loop mask)
 *
 * \param cond_f
 *    Variable index of a boolean array specifying whether the 'false' branch
 *    should be executed. Set this to (!condition & loop mask)
 *
 * \return
 *    A temporary handle that should be passed to the later routines.
 *    The caller has the responsibility of reducing the reference count
 *    of this variable once the conditional statement has been created.
 */
extern JIT_EXPORT uint32_t jit_var_cond_start(const char *name, bool symbolic,
                                              uint32_t cond_t, uint32_t cond_f);

/**
 * \brief Append a case to the ``if`` statement.
 *
 * This function should be called exactly twice: first, to specify the
 * return values of the ``if`` block. Second, to specify the return
 * values of the associated ``else`` block.
 *
 * \param index
 *    A symbolic handle produced by ``jit_var_cond_start()``.
 *
 * \param rv
 *    Pointer to an array of return value variable indices
 *
 * \param count
 *    Specifies the number of return values (i.e., the size of the array ``rv``)
 *
 * \param cond
 *    Variable index of a boolean array specifying the conditional expression
 *
 * \return
 *    A temporary handle. The caller has the responsibility of reducing the
 *    reference count of this variable once the conditional statement has been
 *    created.
 */
extern JIT_EXPORT uint32_t jit_var_cond_append(uint32_t index,
                                               const uint32_t *rv,
                                               size_t count);

/**
 * \brief Finish symbolic recording of a conditional expression
 *
 * \param index
 *    A symbolic handle produced by ``jit_var_cond_start()``.
 *
 * \param rv_out
 *    Output array that will receive the combined return values. The
 *    size must match the ``count`` argument of the preceding calls to
 *    ``jit_var_cond_append()``.
 */
extern JIT_EXPORT void jit_var_cond_end(uint32_t index, uint32_t *rv_out);

/**
 * \brief Wrap an input variable of a virtual function call before recording
 * computation
 *
 * Creates a copy of a virtual function call input argument. The copy has a
 * 'symbolic' bit set that propagates into any computation referencing it.
 */
extern JIT_EXPORT uint32_t jit_var_call_input(uint32_t index);

/**
 * \brief Inform the JIT compiler about the current instance while
 * o virtual function calls
 *
 * Following a call to \ref jit_var_set_self(), the JIT compiler will
 * intercept literal constants referring to the instance ID 'value'. In this
 * case, it will return the variable ID 'index'.
 *
 * This feature is crucial to avoid merging instance IDs into generated code.
 */
extern JIT_EXPORT void jit_var_set_self(JIT_ENUM JitBackend backend,
                                          uint32_t value, uint32_t index);

/// Query the information set via \ref jit_var_set_self
extern JIT_EXPORT void jit_var_self(JIT_ENUM JitBackend backend,
                                    uint32_t *value, uint32_t *index);

/**
 * \brief Record a virtual function call
 *
 * This function inserts a virtual function call into into the computation
 * graph. This works like a giant demultiplexer-multiplexer pair: depending on
 * the value of the \c self argument, information will flow through one of \c
 * n_inst computation graphs that are provided via the `out_nested` argument.
 *
 * \param name
 *     A descriptive name that will be used to label various created nodes in
 *     the computation graph.
 *
 * \param symbolic
 *     Is this operation nested into another symbolic operation?
 *
 * \param self
 *     Instance index (a variable of type <tt>VarType::UInt32</tt>), where
 *     0 indicates that the function call should be masked. All outputs
 *     will be zero-valued in that case.
 *
 * \param n_inst
 *     The number of instances (must be >= 1)
 *
 * \param max_inst_id
 *     Maximum instance index that might be referenced
 *
 * \param inst_id
 *     Pointer to an array of all possible instance indices that might be
 *     referenced
 *
 * \param n_in
 *     The number of input variables
 *
 * \param in
 *     Pointer to an array of input variable indices of size \c n_in
 *
 * \param n_out_nested
 *     Total number of output variables, where <tt>n_out_nested = (# of
 *     outputs) * n_inst</tt>
 *
 * \param out_nested
 *     Pointer to an array of output variable indices of size \c n_out_nested
 *
 * \param se_offset
 *     Indicates the size of the side effects queue (obtained from \ref
 *     jit_side_effects_scheduled()) before each instance call, and
 *     after the last one. <tt>n_inst + 1</tt> entries.
 *
 * \param out
 *     The final output variables representing the result of the operation
 *     are written into this argument (size <tt>n_out_nested / n_inst</tt>)
 */
extern JIT_EXPORT void jit_var_call(const char *name, int symbolic,
                                    uint32_t self, uint32_t mask,
                                    uint32_t n_inst, uint32_t max_inst_id,
                                    const uint32_t *inst_id, uint32_t n_in,
                                    const uint32_t *in, uint32_t n_out_nested,
                                    const uint32_t *out_nested,
                                    const uint32_t *se_offset, uint32_t *out);

/**
 * \brief Pushes a new mask variable onto the mask stack
 *
 * In advanced usage of Dr.Jit (e.g. recorded loops, virtual function calls,
 * etc.), it may be necessary to mask scatter and gather operations to prevent
 * undefined behavior and crashes. This function can be used to push a mask
 * onto a mask stack.  While on the stack, Dr.Jit will hold a reference to \c
 * index to keep it from being freed.
 */
extern JIT_EXPORT void jit_var_mask_push(JIT_ENUM JitBackend backend, uint32_t index);

/// Pop the mask stack
extern JIT_EXPORT void jit_var_mask_pop(JIT_ENUM JitBackend backend);

/**
 * \brief Return the top entry of the mask stack and increase its external
 * reference count. Returns zero when the stack is empty.
 */
extern JIT_EXPORT uint32_t jit_var_mask_peek(JIT_ENUM JitBackend backend);

/// Return the default mask for a wavefront of the given \c size
extern JIT_EXPORT uint32_t jit_var_mask_default(JIT_ENUM JitBackend backend,
                                                size_t size);

/**
 * \brief Combine the given mask 'index' with the mask stack
 *
 * On the LLVM backend, a default mask will be created when the mask stack is empty.
 * The \c size parameter determines the size of the associated wavefront.
 */
extern JIT_EXPORT uint32_t jit_var_mask_apply(uint32_t index, uint32_t size);

/// Compress a sparse boolean array into an index array of the active indices
extern JIT_EXPORT uint32_t jit_var_compress(uint32_t index);

// ====================================================================
//                          Horizontal reductions
// ====================================================================

/// Reduce (And) a boolean array to a single value (synchronous).
extern JIT_EXPORT int jit_var_all(uint32_t index);

/// Reduce (Or) a boolean array to a single value (synchronous).
extern JIT_EXPORT int jit_var_any(uint32_t index);

/// Reduce a variable to a single value (asynchronous)
extern JIT_EXPORT uint32_t jit_var_reduce(JIT_ENUM JitBackend backend,
                                          JIT_ENUM VarType vt,
                                          JIT_ENUM ReduceOp op,
                                          uint32_t index);

/// Reduce a avariable within blocks of size 'block_size'
extern JIT_EXPORT uint32_t jit_var_block_reduce(JIT_ENUM ReduceOp op, uint32_t index,
                                                uint32_t block_size, int symbolic);

/// Replicate values of an array into larger blocks
extern JIT_EXPORT uint32_t jit_var_tile(uint32_t index, uint32_t count);

/// Perform a dot product reduction of two compatible arrays
extern JIT_EXPORT uint32_t jit_var_reduce_dot(uint32_t index_1,
                                              uint32_t index_2);

/// Compute an exclusive (exclusive == 1) or inclusive (exclusive == 0) blocked prefix sum (asynchronous)
extern JIT_EXPORT uint32_t jit_var_block_prefix_reduce(JIT_ENUM ReduceOp op,
                                                       uint32_t index,
                                                       uint32_t block_size,
                                                       int exclusive,
                                                       int reverse);

// ====================================================================
//  Assortment of tuned kernels for initialization, reductions, etc.
// ====================================================================

/**
 * \brief Fill a device memory region with constants of a given type
 *
 * This function writes \c size values of size \c isize to the output array \c
 * ptr. The specific value is taken from \c src, which must be a CPU pointer to
 * a single int, float, double, etc. (\c isize can be 1, 2, 4, or 8).
 * Runs asynchronously.
 */
extern JIT_EXPORT void jit_memset_async(JIT_ENUM JitBackend backend, void *ptr, uint32_t size,
                                        uint32_t isize, const void *src);

/// Perform a synchronous copy operation
extern JIT_EXPORT void jit_memcpy(JIT_ENUM JitBackend backend, void *dst, const void *src, size_t size);

/// Perform an asynchronous copy operation
extern JIT_EXPORT void jit_memcpy_async(JIT_ENUM JitBackend backend, void *dst, const void *src,
                                        size_t size);

/**
 * \brief Reduce the given array to a single value
 *
 * This operation reads \c size values of type \type from the input array \c
 * ptr and performs an specified operation (e.g., addition, multiplication,
 * etc.) to combine them into a single value that is written to the device
 * variable \c out.
 *
 * Runs asynchronously.
 */
extern JIT_EXPORT void jit_reduce(JIT_ENUM JitBackend backend, JIT_ENUM VarType type,
                                  JIT_ENUM ReduceOp op,
                                  const void *in, uint32_t size, void *out);

/**
 * \brief Reduce elements within blocks of the given input array
 *
 * This function reduces contiguous blocks from \c in with size \c block_size
 * and writes the result to to \c out. For example, sum-reducing <tt>a, b, c, d,
 * e, f</tt> with <tt>block_size == 2</tt> produces <tt>a+b, c+d, e+f</tt>.
 *
 * By setting <tt>block_size == size</tt>, the function reduces the entire
 * array.
 *
 * It is legal for \c size to be *indivisible* by \c block_size, in which case
 * the reduction over the last block considers fewer elements.
 *
 * The input array must have size \c size, and the output array must have
 * size <tt>(size + block_size - 1) / block_size</tt>.
 */
extern JIT_EXPORT void jit_block_reduce(JIT_ENUM JitBackend backend,
                                        JIT_ENUM VarType type,
                                        JIT_ENUM ReduceOp op,
                                        uint32_t size,
                                        uint32_t block_size,
                                        const void *in,
                                        void *out);

/**
 * \brief Prefix-reduce elements within blocks of the given input array
 *
 * This function prefix-reduces contiguous blocks from \c in with size
 * \c block_size and writes the result to to \c out. For example, an inclusive
 * sum-reduction of <tt>a, b, c, d, e, f</tt> with <tt>block_size == 2</tt>
 * produces <tt>a, a+b, c, c+d, e, e+f</tt>.
 *
 * Both exclusive and inclusive variants are supported. If desired, the
 * reduction can be performed in-place (i.e., <tt>out == in</tt>).
 *
 * By setting <tt>block_size == size</tt>, the function reduces the entire
 * array.
 *
 * It is legal for \c size to be *indivisible* by \c block_size, in which case
 * the reduction over the last block considers fewer elements.
 *
 * Both the input and output array are expected to have \c size elements.
 */
extern JIT_EXPORT void jit_block_prefix_reduce(JIT_ENUM JitBackend backend,
                                               JIT_ENUM VarType type,
                                               JIT_ENUM ReduceOp op,
                                               uint32_t block_size,
                                               uint32_t size,
                                               int exclusive,
                                               int reverse,
                                               const void *in,
                                               void *out);
/**
 * \brief Compress a mask into a list of nonzero indices
 *
 * This function takes an 8-bit mask array \c in with size \c size as input,
 * whose entries are required to equal either zero or one. It then writes the
 * indices of nonzero entries to \c out (in increasing order), and it
 * furthermore returns the total number of nonzero mask entries.
 *
 * The internals resemble \ref jit_prefix_sum_u32(), and the CUDA implementation may
 * similarly access regions beyond the end of \c in and \c out.
 *
 * This function internally performs a synchronization step.
 */
extern JIT_EXPORT uint32_t jit_compress(JIT_ENUM JitBackend backend, const uint8_t *in,
                                        uint32_t size, uint32_t *out);


/**
 * \brief Compute a permutation to reorder an integer array into a sorted
 * configuration
 *
 * Given an unsigned integer array \c values of size \c size with entries in
 * the range <tt>0 .. bucket_count - 1</tt>, compute a permutation that can be
 * used to reorder the inputs into a sorted (but non-stable) configuration.
 * When <tt>bucket_count</tt> is relatively small (e.g. < 10K), the
 * implementation is much more efficient than the alternative of actually
 * sorting the array.
 *
 * \param perm
 *     The permutation is written to \c perm, which must point to a buffer in
 *     device memory having size <tt>size * sizeof(uint32_t)</tt>.
 *
 * \param offsets
 *     When \c offset is non-NULL, the parameter should point to a host (LLVM)
 *     or host-pinned (CUDA) memory region with a size of at least
 *     <tt>(bucket_count * 4 + 1) * sizeof(uint32_t)<tt> bytes that will be
 *     used to record the details of non-empty buckets. It will contain
 *     quadruples <tt>(index, start, size, unused)<tt> where \c index is the
 *     bucket index, and \c start and \c end specify the associated entries of
 *     the \c perm array. The 'unused' field is padding for 16 byte alignment.
 *
 * \return
 *     When \c offsets != NULL, the function returns the number of unique
 *     values found in \c values. Otherwise, it returns zero.
 */
extern JIT_EXPORT uint32_t jit_mkperm(JIT_ENUM JitBackend backend, const uint32_t *values,
                                      uint32_t size, uint32_t bucket_count,
                                      uint32_t *perm, uint32_t *offsets);

/// Helper data structure used to initialize the data block consumed by a vcall
struct AggregationEntry {
    int32_t size;
    uint32_t offset;
    const void *src;
};

/**
 * \brief Aggregate memory from different sources and write it to an output
 * array
 *
 * This function writes to the device memory address ``dst`` following the
 * instructions provided in the form of ``size`` separate ``AggregationEntry``
 * values reachable through the ``agg``` pointer.
 *
 * For each entry, it does the following:
 *
 * - if ``AggregationEntry::size`` is positive, it interprets
 *   ``AggregationEntry::src`` as an integer and copies its
 *   lowest ``size`` bytes to ``dst+offset``.
 *
 * - if ``AggregationEntry::size`` is negative, it interprets
 *   ``AggregationEntry::src`` as a ponter and copies
 *   ``-size`` bytes from this address to ``dst+offset``.
 *
 * Only size = +/- 1, 2, 4, and 8 are supported. The function runs
 * asynchronously.
 */
extern JIT_EXPORT void jit_aggregate(JitBackend backend, void *dst,
                                     AggregationEntry *agg, uint32_t size);

/// Helper data structure for vector method calls, see \ref jit_var_call()
struct CallBucket {
    /// Resolved pointer address associated with this bucket
    void *ptr;

    /// Variable index of a uint32 array storing a partial permutation
    uint32_t index;

    /// Original instance ID
    uint32_t id;
};

/**
 * \brief Compute a permutation to reorder an array of registered pointers or
 * indices in preparation for a vectorized function call
 *
 * This function expects an array of integers, whose entries correspond to
 * pointers that have previously been registered by calling \ref
 * jit_registry_put() with domain \c domain. It then invokes \ref jit_mkperm()
 * to compute a permutation that reorders the array into coherent buckets. The
 * buckets are returned using an array of type \ref CallBucket, which contains
 * both the resolved pointer address (obtained via \ref
 * jit_registry_get_ptr()) and the variable index of an unsigned 32 bit array
 * containing the corresponding entries of the input array. The total number of
 * buckets is returned via the \c bucket_count_inout argument.
 *
 * Alternatively, this function can be used to to dispatch using an arbitrary
 * index list. In this case, \c domain should be set to \c nullptr and the
 * function will expects an array of integers that correspond to the indices of
 * the callable to execute. The largest possible value in the array of indices
 * has to be passed via the \c bucket_count_inout argument, which will then be
 * overwritten with the total number of buckets.
 *
 * The memory region accessible via the \c CallBucket pointer will remain
 * accessible until the variable \c index is itself freed (i.e. when its
 * reference count becomes equal to zero). Until then, additional calls to \ref
 * jit_var_call() will return the previously computed result. This is an
 * important optimization in situations where multiple calls target the same
 * set of instances.
 */
extern JIT_EXPORT struct CallBucket *
jit_var_call_reduce(JIT_ENUM JitBackend backend, const char *domain,
                     uint32_t index, uint32_t *bucket_count_inout);

/**
 * \brief Insert a function call to a ray tracing functor into the LLVM program
 *
 * The \c args list should contain a list of variable indices corresponding to
 * the 13 required function arguments
 * - active_mask (32 bit integer with '-1' for active, and '0' for inactive)
 * - ox, oy, oz
 * - tmin
 * - dx, dy, dz
 * - time
 * - tfar
 * - mask, id, flags
 * </tt>.
 */
extern JIT_EXPORT void jit_llvm_ray_trace(uint32_t func, uint32_t scene,
                                          int shadow_ray, const uint32_t *in,
                                          uint32_t *out);

/**
 * \brief Set a new scope identifier to limit the effect of common
 * subexpression elimination
 *
 * Dr.Jit implements a very basic approximation of common subexpression
 * elimination based on local value numbering (LVN): an attempt to create a
 * variable, whose statement and dependencies match a previously created
 * variable will sidestep creation and instead reuse the old variable via
 * reference counting. However, this approach of collapsing variables does not
 * play well with more advanced constructs like recorded loops, where variables
 * in separate scopes should be kept apart.
 *
 * This function sets a unique scope identifier (a simple 32 bit integer)
 * isolate the effects of this optimization.
 */
extern JIT_EXPORT uint32_t jit_new_scope(JIT_ENUM JitBackend backend);

/// Queries the scope identifier (see \ref jit_new_scope())
extern JIT_EXPORT uint32_t jit_scope(JIT_ENUM JitBackend backend);

/// Manually sets a scope identifier (see \ref jit_new_scope())
extern JIT_EXPORT void jit_set_scope(JIT_ENUM JitBackend backend, uint32_t domain);

// ====================================================================
//                            Kernel History
// ====================================================================

/**
 * \brief List of kernel types that can be launched by Dr.Jit
 *
 * Dr.Jit sometimes launches kernels that are not generated by the JIT itself
 * (e.g. precompiled CUDA kernels for horizontal reductions). The kernel history
 * identifies them using a field of type \c KernelType.
 */
enum KernelType : uint32_t {
    /// JIT-compiled kernel
    JIT,

    /// Kernel responsible for a horizontal reduction operation (e.g. hsum)
    Reduce,

    /// Permutation kernel produced by \ref jit_mkperm()
    CallReduce,

    /// Any other kernel
    Other
};

/// Data structure for preserving kernel launch information (debugging, testing)
struct KernelHistoryEntry {
    /// Jit backend, for which the kernel was compiled
    JitBackend backend;

    /// Kernel type
    KernelType type;

    /// Stores the low/high 64 bits of the 128-bit hash kernel identifier
    uint64_t hash[2];

    /// Copy of the kernel IR string buffer
    char *ir;

    /// Does the kernel contain any OptiX (ray tracing) operations?
    int uses_optix;

    /// Whether the kernel was reused from the kernel cache
    int cache_hit;

    /// Whether the kernel was loaded from the cache on disk
    int cache_disk;

    /// Launch width / number of array entries that were processed
    uint32_t size;

    /// Number of input arrays
    uint32_t input_count;

    /// Number of output arrays + side effects
    uint32_t output_count;

    /// Number of IR operations
    uint32_t operation_count;

    /// Time (ms) spent generating the kernel intermediate representation
    float codegen_time;

    /// Time (ms) spent compiling the kernel (\c 0 if \c cache_hit is \c true)
    float backend_time;

    /// Time (ms) spent executing the kernel
    float execution_time;

    // Dr.Jit internal portion, will be cleared by jit_kernel_history()
    // ================================================================

    /// CUDA events for measuring the runtime of the kernel
    void *event_start, *event_end;

    /// nanothread task handle
    void *task;
};

/// Clear the kernel history
extern JIT_EXPORT void jit_kernel_history_clear();

/**
 * \brief Return a pointer to the first entry of the kernel history
 *
 * When \c JitFlag.KernelHistory is set to \c true, every kernel launch will add
 * and entry in the history which can be accessed via this function.
 *
 * The caller is responsible for freeing the returned data structure via the
 * following construction:
 *
 *     KernelHistoryEntry *data = jit_kernel_history();
 *     KernelHistoryEntry *e = data;
 *     while (e->backend) {
 *         free(e->ir);
 *         e++;
 *     }
 *     free(data);
 *
 * When the kernel history is empty, the function will return a null pointer.
 * Otherwise, the size of the kernel history can be inferred by iterating over
 * the entries until one reaches a entry with an invalid \c backend (e.g.
 * initialized to \c 0).
 */
extern JIT_EXPORT struct KernelHistoryEntry *jit_kernel_history();

// ====================================================================
//                        Profiling (NVTX, etc.)
// ====================================================================

/// Generate a profiling mark (single event)
extern JIT_EXPORT void jit_profile_mark(const char *message);

/// Inform the profiler (if present) about the start of a tracked range
extern JIT_EXPORT void jit_profile_range_push(const char *message);

/// Inform the profiler (if present) about the end of a tracked range
extern JIT_EXPORT void jit_profile_range_pop();

// ====================================================================
//                               Other
// ====================================================================

/// Query/set the status of variable leak warnings
extern JIT_EXPORT int jit_leak_warnings();
extern JIT_EXPORT void jit_set_leak_warnings(int value);

/// Return the item size of a JIT variable type
extern JIT_EXPORT size_t jit_type_size(JIT_ENUM VarType type) JIT_NOEXCEPT;

/// Return a name (e.g., "float32") for a given a JIT variable type
extern JIT_EXPORT const char *jit_type_name(JIT_ENUM VarType type) JIT_NOEXCEPT;

/// Return value of \ref jit_set_backend()
struct VarInfo {
    JitBackend backend;
    VarType type;
    size_t size;
    bool is_array;
};

/**
 * \brief Query details about a variable and remember its backend
 *
 * This operations serves a dual role:
 *
 * 1. It returns several pieces of information characterizing the given
 * variable: its Jit backend, variable type, and size (number of entries).
 *
 * 2. It stashes the variable's backend in a thread-local variable.
 *
 * The second step is useful in a few places: Dr.Jit operations can be called
 * with a backend value of ``JitBackend::None``, in which case the default
 * backend set via this operation takes precedence. This is useful to implement
 * generic code that works on various different backends.
 */
extern JIT_EXPORT VarInfo jit_set_backend(uint32_t index) JIT_NOEXCEPT;

/**
 * \brief Inform Dr.Jit about the current source code location
 *
 * The Python bindings use this function in combination with the JitFlag::Debug
 * flag. In this case, a tracing callback regularly updates e file and line
 * number information, which is then propagated into the 'label' field of newly
 * created variables.
 */
extern JIT_EXPORT void jit_set_source_location(const char *fname,
                                               size_t lineno) JIT_NOEXCEPT;


/**
 * \brief Enqueue a callback to be run on the host once all previously launched
 * backend operations have finished.
 *
 * Note that the callback might run right away if the backend's queue is empty.
 */
extern JIT_EXPORT void jit_enqueue_host_func(JitBackend backend,
                                             void (*callback)(void *),
                                             void *payload);

/**
 * \brief Set the target array size threshold where ``ReduceMode::Expand`` is
 * no longer used. Set to "1" by default.
 */
extern JIT_EXPORT void jit_llvm_set_expand_threshold(size_t size);
extern JIT_EXPORT size_t jit_llvm_expand_threshold() JIT_NOEXCEPT;

/// Return the identity element of a particular type of reduction
extern JIT_EXPORT uint64_t jit_reduce_identity(VarType vt, ReduceOp op);

/**
 * \brief Create a variable array
 *
 * This operation creates a memory region that is local to each thread of the
 * compiled program.
 *
 * Variable arrays may only be accessed using the ``jit_array_*`` API, as well
 * as ``jit_var_inc_ref()``,``jit_var_dec_ref()``, ``jit_var_schedule()``, and
 * ``jit_var_eval()``.
 *
 * \param backend
 *    The JIT backend in which the variable should be created
 *
 * \param vt
 *    The variable type of the allocation
 *
 * \param size
 *    Specifies the *size*, which refers to the number of threads of the
 *    underlying parallel computation.
 *
 * \param length
 *    Specifies the *length*, which refers to the number of array elements per
 *    thread.
 *
 * \return
 *    The index of the created variable
 */
extern JIT_EXPORT uint32_t jit_array_create(JitBackend backend, VarType vt,
                                            size_t size, size_t length);

/**
 * \brief Initialize an array variable
 *
 * This function sets entries an array variable to the value of a provided (non-array)
 * variable. It return a new array variable representing the result.
 */
extern JIT_EXPORT uint32_t jit_array_init(uint32_t target, uint32_t value);

/// Return the length of the specified array
extern JIT_EXPORT size_t jit_array_length(uint32_t index);

/**
 * \brief Write to a variable array
 *
 * \param target
 *     Index of a variable array
 *
 * \param offset
 *     Offset of the write. Must be an unsigned 32-bit integer.
 *
 * \param value
 *     Value to be written.
 *
 * \return
 *     A new index representing the variable array following the write.
 */
extern JIT_EXPORT uint32_t jit_array_write(uint32_t target, uint32_t offset,
                                           uint32_t value, uint32_t mask);

/**
 * \brief Read from a variable array
 *
 * \param source
 *     Index of a variable array
 *
 * \param offset
 *     Offset of the read. Must be an unsigned 32-bit integer.
 *
 * \return
 *     Index of a variable representing the result of the read operation.
 */
extern JIT_EXPORT uint32_t jit_array_read(uint32_t source, uint32_t offset,
                                          uint32_t mask);

#if defined(__cplusplus)
}
#endif
