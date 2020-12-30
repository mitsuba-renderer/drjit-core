/*
    enoki-jit/jit.h -- Self-contained JIT compiler for CUDA & LLVM.

    This library implements a self-contained tracing JIT compiler that supports
    both CUDA PTX and LLVM IR as intermediate representations. It takes care of
    many tricky aspects, such as asynchronous memory allocation and release,
    multi-device computation, kernel caching and reuse, common subexpression
    elimination, etc.

    While the library is internally implemented using C++11, this header file
    provides a compact C99-compatible API that can be used to access all
    functionality. The library is thread-safe: multiple threads can
    simultaneously dispatch computation to one or more CPUs/GPUs.

    As an alternative to the fairly low-level API defined here, you may prefer
    to use the functionality in 'enoki/cuda.h' or 'enoki/llvm.h', which
    provides a header-only C++ array class with operator overloading, which
    dispatches to the C API.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdlib.h>
#include <stdint.h>

#if defined(_MSC_VER)
#  if defined(ENOKI_JIT_BUILD)
#    define JITC_EXPORT    __declspec(dllexport)
#  else
#    define JITC_EXPORT    __declspec(dllimport)
#  endif
#  define JITC_MALLOC
#  define JITC_INLINE    __forceinline
#  define JITC_NOINLINE  __declspec(noinline)
#else
#  define JITC_EXPORT    __attribute__ ((visibility("default")))
#  define JITC_MALLOC    __attribute__((malloc))
#  define JITC_INLINE    __attribute__ ((always_inline)) inline
#  define JITC_NOINLINE  __attribute__ ((noinline)) inline
#endif

#if defined(__cplusplus)
#  define JITC_CONSTEXPR constexpr
#  define JITC_DEF(x) = x
#  define JITC_NOEXCEPT noexcept(true)
#  define JITC_ENUM ::
#else
#  define JITC_CONSTEXPR inline
#  define JITC_DEF(x)
#  define JITC_NOEXCEPT
#  define JITC_ENUM enum
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// ====================================================================
//         Initialization, device enumeration, and management
// ====================================================================

/**
 * \brief List of backends that can be targeted by Enoki-JIT
 *
 * Enoki-JIT can perform computation using one of several computational
 * backends. Before use, a backend must be initialized via \ref jit_init().
 */
#if defined(__cplusplus)
enum class JitBackend : uint32_t {
    /// CUDA backend (requires CUDA >= 10, generates PTX instructions)
    CUDA = (1 << 0),

    /// LLVM backend targeting the CPU (generates LLVM IR)
    LLVM = (1 << 1)
};
#else
enum JitBackend {
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
extern JITC_EXPORT void
jit_init(uint32_t backends JITC_DEF((uint32_t) JitBackend::CUDA |
                                    (uint32_t) JitBackend::LLVM));

/**
 * \brief Launch an ansynchronous thread that will execute jit_init() and
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
extern JITC_EXPORT void
jit_init_async(uint32_t backends JITC_DEF((uint32_t) JitBackend::CUDA |
                                          (uint32_t) JitBackend::LLVM));

/// Check whether the LLVM backend was successfully initialized
extern JITC_EXPORT int jit_has_backend(JITC_ENUM JitBackend backend);

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
extern JITC_EXPORT void jit_shutdown(int light JITC_DEF(0));

/**
 * \brief Wait for all computation scheduled by the current thread to finish
 *
 * Each thread using Enoki-JIT will issue computation to an independent queue.
 * This function only synchronizes with computation issued to the queue of the
 * calling thread.
 */
extern JITC_EXPORT void jit_sync_thread();

/// Wait for all computation on the current device to finish
extern JITC_EXPORT void jit_sync_device();

/// Wait for all computation on the *all devices* to finish
extern JITC_EXPORT void jit_sync_all_devices();

// ====================================================================
//       Advanced JIT usage: recording programs, loops, etc.
// ====================================================================

#if defined(__cplusplus)
/**
 * \brief Status flags to adjust/inspect the eagerness of the JIT compiler
 *
 * Certain Enoki operations can operate in two different ways: they can be
 * executed at once, or they can be recorded to postpone evaluation to a later
 * point. The latter is generally more efficient because it enables
 * optimizations (fusion of multiple operations, exchange of information via
 * registers instead of global memory, etc.). The downside is that recording
 * computation is generally more complex/fragile and less suitable to
 * interactive software development (one e.g. cannot simply print array
 * contents while something is being recorded).
 *
 * The following list of flags can be used to control the behavior of the JIT
 * compiler. The enoki-jit library actually doesn't do very much with this
 * flag: the main effect is that \ref jit_eval() will throw an exception when
 * it is called while the <tt>RecordingLoop</tt> and <tt>RecordingVCall<tt>
 * flags are set. The main behavioral differences will typically be in found in
 * code using enoki-jit that queries this flag.
 */
enum class JitFlag : uint32_t {
    // Default (eager) execute loops and virtual function calls at once
    Default        = 0,

    // Record loops to postpone their evaluation
    RecordLoops    = 1,

    // Status flag indicating that a loop is *currently* being recorded
    RecordingLoop  = 2,

    // Record virtual function calls to postpone their evaluation
    RecordVCalls   = 4,

    // Status flag indicating that a vcall call is *currently* being recorded
    RecordingVCall = 8,

    // Try to optimize the calling conventions of vcalls
    OptimizeVCalls = 16,

    // A loop is currently being recorded
    Recording = (uint32_t) RecordingLoop | (uint32_t) RecordingVCall
};
#else
enum JitFlag {
    JitFlagDefault = 0,
    JitFlagRecordLoops = 1,
    JitFlagRecordingLoop = 2,
    JitFlagRecordVCall = 4,
    JitFlagRecordingVCall = 8,
    JitFlagRecording = JitFlagRecordingLoop | JitFlagRecordingVCall
};
#endif

/// Set the JIT compiler status flags (see \ref JitFlags)
extern JITC_EXPORT void jit_set_flags(uint32_t flags);

/// Retrieve the JIT compiler status flags (see \ref JitFlags)
extern JITC_EXPORT uint32_t jit_flags();

/// Equivalent to <tt>jit_set_flags(jit_flags() | flag)</tt>
extern JITC_EXPORT void jit_enable_flag(JITC_ENUM JitFlag flag);

/// Equivalent to <tt>jit_set_flags(jit_flags() & ~flag)</tt>
extern JITC_EXPORT void jit_disable_flag(JITC_ENUM JitFlag flag);

/**
 * \brief Returns the number of operations with side effects (specifically,
 * scatters) scheduled for evaluation on the current thread.
 *
 * This function can be used to easily detect whether or not some piece of
 * code involves side effects. It is used to as part of the mechanism
 * that records loops, virtual function calls, etc.
 */
extern JITC_EXPORT uint32_t jit_side_effects_scheduled(JitBackend backend);

// ====================================================================
//                    CUDA/LLVM-specific functionality
// ====================================================================

/// Return the no. of available CUDA devices that are compatible with Enoki.
extern JITC_EXPORT int jit_cuda_device_count();

/**
 * \brief Set the active CUDA device.
 *
 * The argument must be between 0 and <tt>jit_cuda_device_count() - 1</tt>,
 * which only accounts for Enoki-compatible devices. This is a per-thread
 * property: independent threads can issue computation to different GPUs.
 */
extern JITC_EXPORT void jit_cuda_set_device(int device);

/**
 * \brief Return the CUDA device ID associated with the current thread
 *
 * The result is in the range of 0 and <tt>jit_cuda_device_count() - 1</tt>.
 * When the machine contains CUDA devices that are incompatible with Enoki (due
 * to a lack of 64-bit addressing, uniform address space, or managed memory),
 * this number may differ from the default CUDA device ID. Use
 * <tt>jit_cuda_device_raw()</tt> in that case.
 */
extern JITC_EXPORT int jit_cuda_device();

/// Return the raw CUDA device associated with the current thread
extern JITC_EXPORT int jit_cuda_device_raw();

/// Return the CUDA stream associated with the current thread
extern JITC_EXPORT void* jit_cuda_stream();

/// Return the CUDA context associated with the current thread
extern JITC_EXPORT void* jit_cuda_context();

/// Query the compute capability of the current device (e.g. '52')
extern JITC_EXPORT int jit_cuda_compute_capability();

/**
 * \brief Override generated PTX version and compute capability
 *
 * Enoki-JIT generates code that runs on a wide variety of platforms supporting
 * at least the PTX version and compute capability of 60, and 50, respectively.
 * Those versions can both be bumped via this function---there is no
 * performance advantage in doing so, though some more recent features (e.g.
 * atomic operations involving double precision values) require specifying a
 * newer compute capability.
 */
extern JITC_EXPORT void jit_cuda_set_target(uint32_t ptx_version,
                                            uint32_t compute_capability);

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
 *     individual featureas.
 *
 * \param vector_width
 *     Width of vector registers (e.g. 8 for AVX). Must be a power of two, and
 *     can be a multiple of the hardware register size to enable unrolling.
 */
extern JITC_EXPORT void jit_llvm_set_target(const char *target_cpu,
                                            const char *target_features,
                                            uint32_t vector_width);

/// Get the CPU that is currently targeted by the LLVM backend
extern JITC_EXPORT const char *jit_llvm_target_cpu();

/// Get the list of CPU features currently used by the LLVM backend
extern JITC_EXPORT const char *jit_llvm_target_features();

/// Return the major version of the LLVM library
extern JITC_EXPORT int jit_llvm_version_major();

/**
 * \brief Convenience function for intrinsic function selection
 *
 * Returns \c 1 if the current vector width is is at least as large as a
 * provided value, and when the host CPU provides a given target feature (e.g.
 * "+avx512f").
 */
extern JITC_EXPORT int jit_llvm_if_at_least(uint32_t vector_width,
                                             const char *feature);


/**
 * \brief Returns the variable index of a boolean array designating
 * currently active SIMD lanes
 *
 * This function returns code that correctly computes the active mask in
 * general situations. In more specific cases (e.g. branching), the return
 * value can be modified by pushing and popping masks via the next two
 * functions.
 *
 * This function returns a new reference
 */
extern JITC_EXPORT uint32_t jit_llvm_active_mask();

/// Push a new mask value onto the stack (increases the ref. count)
extern JITC_EXPORT void jit_llvm_active_mask_push(uint32_t index);

/// Pop the stack of active mask values, and dereference it
extern JITC_EXPORT void jit_llvm_active_mask_pop();

// ====================================================================
//                        Logging infrastructure
// ====================================================================

#if defined(__cplusplus)
enum class LogLevel : uint32_t {
    Disable, Error, Warn, Info, Debug, Trace
};
#else
enum LogLevel {
    LogLevelDisable, LogLevelError, LogLevelWarn,
    LogLevelInfo, LogLevelDebug, LogLevelTrace
};
#endif

/**
 * \brief Control the destination of log messages (stderr)
 *
 * By default, this library prints all log messages to the console (\c stderr).
 * This function can be used to control the minimum log level for such output
 * or prevent it entirely. In the latter case, you may wish to enable logging
 * via a callback in \ref jit_set_log_level_callback(). Both destinations can also
 * be enabled simultaneously, pontentially using different log levels.
 */
extern JITC_EXPORT void jit_set_log_level_stderr(JITC_ENUM LogLevel level);

/// Return the currently set minimum log level for output to \c stderr
extern JITC_EXPORT JITC_ENUM LogLevel jit_log_level_stderr();


/**
 * \brief Control the destination of log messages (callback)
 *
 * This function can be used to specify an optional callback that will be
 * invoked with the contents of library log messages, whose severity matches or
 * exceeds the specified \c level.
 */
typedef void (*LogCallback)(JITC_ENUM LogLevel, const char *);
extern JITC_EXPORT void jit_set_log_level_callback(JITC_ENUM LogLevel level,
                                                    LogCallback callback);

/// Return the currently set minimum log level for output to a callback
extern JITC_EXPORT JITC_ENUM LogLevel jit_log_level_callback();

/// Print a log message with the specified log level and message
extern JITC_EXPORT void jit_log(JITC_ENUM LogLevel level, const char* fmt, ...);

/// Raise an exception message with the specified message
extern JITC_EXPORT void jit_raise(const char* fmt, ...);

/// Terminate the application due to a non-recoverable error
extern JITC_EXPORT void jit_fail(const char* fmt, ...);

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
     * jit_malloc_trim() to also flush this cache.
     */
    Host,

    /**
     * Like \c Host memory, except that it may only be used *asynchronously*
     * within a computation performed by enoki-jit.
     *
     * In particular, host-asynchronous memory obtained via \ref jit_malloc()
     * should not be written to directly (i.e. outside of enoki-jit), since it
     * may still be used by a currently running kernel. Releasing
     * host-asynchronous memory via \ref jit_free() also occurs
     * asynchronously.
     *
     * This type of memory is used internally when running code via the LLVM
     * backend, and when this process is furthermore parallelized using Enoki's
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

    /**
     * Memory that is mapped in the address space of the host & all GPUs.
     *
     * Managed memory has asynchronous semantics similar to \c HostAsync.
     */
    Managed,

    /**
     * Like \c Managed, but more efficient when accesses are mostly reads. In
     * this case, the system will distribute multiple read-only copies instead
     * of moving memory back and forth.
     *
     * This type of memory has asynchronous semantics similar to \c HostAsync.
     */
    ManagedReadMostly,

    /// Number of possible allocation types
    Count
};
#else
enum AllocType {
    AllocTypeHost,
    AllocTypeHostPinned,
    AllocTypeDevice,
    AllocTypeManaged,
    AllocTypeManagedReadMostly,
    AllocTypeCount
};
#endif

/**
 * \brief Allocate memory of the specified type
 *
 * Under the hood, Enoki implements a custom allocation scheme that tries to
 * reuse allocated memory regions instead of giving them back to the OS/GPU.
 * This eliminates inefficient synchronization points in the context of CUDA
 * programs, and it can also improve performance on the CPU when working with
 * large allocations.
 *
 * The returned pointer is guaranteed to be sufficiently aligned for any kind
 * of use.
 *
 */
extern JITC_EXPORT void *jit_malloc(JITC_ENUM AllocType type, size_t size)
    JITC_MALLOC;

/**
 * \brief Release a given pointer asynchronously
 *
 * For CPU-only arrays (\ref AllocType::Host), <tt>jit_free()</tt> is
 * synchronous and very similar to <tt>free()</tt>, except that the released
 * memory is placed in Enoki's internal allocation cache instead of being
 * returned to the OS. The function \ref jit_malloc_trim() can optionally be
 * called to also clear this cache.
 *
 * When \c ptr is an asynchronous host pointer (\ref AllocType::HostAsync) or
 * GPU-accessible pointer (\ref AllocType::Device, \ref AllocType::HostPinned,
 * \ref AllocType::Managed, \ref AllocType::ManagedReadMostly), the associated
 * memory region is possibly still being used by a running kernel, and it is
 * therefore merely *scheduled* to be reclaimed once this kernel finishes.
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
extern JITC_EXPORT void jit_free(void *ptr);

/// Release all currently unused memory to the GPU / OS
extern JITC_EXPORT void jit_malloc_trim();

/**
 * \brief Asynchronously prefetch a managed memory region allocated using \ref
 * jit_malloc() so that it is available on a specified device
 *
 * This operation prefetches a memory region so that it is available on the CPU
 * (<tt>device==-1</tt>) or specified CUDA device (<tt>device&gt;=0</tt>). This
 * operation only make sense for allocations of type <tt>AllocType::Managed<tt>
 * and <tt>AllocType::ManagedReadMostly</tt>. In the former case, the memory
 * region will be fully migrated to the specified device, and page mappings
 * established elswhere are cleared. For the latter, a read-only copy is
 * created on the target device in addition to other copies that may exist
 * elsewhere.
 *
 * The function also takes a special argument <tt>device==-2</tt>, which
 * creates a read-only mapping on *all* available GPUs.
 *
 * The prefetch operation is enqueued on the current device and thread and runs
 * asynchronously with respect to the CPU, hence a \ref jit_sync_thread()
 * operation is advisable if data is <tt>target==-1</tt> (i.e. prefetching into
 * CPU memory).
 */
extern JITC_EXPORT void jit_malloc_prefetch(void *ptr, int device);

/// Query the flavor of a memory allocation made using \ref jit_malloc()
extern JITC_EXPORT JITC_ENUM AllocType jit_malloc_type(void *ptr);

/// Query the device associated with a memory allocation made using \ref jit_malloc()
extern JITC_EXPORT int jit_malloc_device(void *ptr);

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
extern JITC_EXPORT void *jit_malloc_migrate(void *ptr, JITC_ENUM AllocType type,
                                             int move JITC_DEF(1));

// ====================================================================
//                          Pointer registry
// ====================================================================

/**
 * \brief Register a pointer with Enoki's pointer registry
 *
 * Enoki provides a central registry that maps registered pointer values to
 * low-valued 32-bit IDs. The main application is efficient virtual function
 * dispatch via \ref jit_vcall(), through the registry could be used for other
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
 * Returns zero if <tt>ptr == nullptr</tt> and throws if the pointer is already
 * registered (with *any* domain).
 */
extern JITC_EXPORT uint32_t jit_registry_put(const char *domain, void *ptr);

/**
 * \brief Remove a pointer from the registry
 *
 * No-op if <tt>ptr == nullptr</tt>. Throws an exception if the pointer is not
 * currently registered.
 */
extern JITC_EXPORT void jit_registry_remove(void *ptr);

/**
 * \brief Query the ID associated a registered pointer
 *
 * Returns 0 if <tt>ptr==nullptr</tt> and throws if the pointer is not known.
 */
extern JITC_EXPORT uint32_t jit_registry_get_id(const void *ptr);

/**
 * \brief Query the domain associated a registered pointer
 *
 * Returns \c nullptr if <tt>ptr==nullptr</tt> and throws if the pointer is not
 * known.
 */
extern JITC_EXPORT const char *jit_registry_get_domain(const void *ptr);

/**
 * \brief Query the pointer associated a given domain and ID
 *
 * Returns \c nullptr if <tt>id==0</tt>, or when the (domain, ID) combination
 * is not known.
 */
extern JITC_EXPORT void *jit_registry_get_ptr(const char *domain, uint32_t id);

/// Provide a bound (<=) on the largest ID associated with a domain
extern JITC_EXPORT uint32_t jit_registry_get_max(const char *domain);

/**
 * \brief Compact the registry and release unused IDs and attributes
 *
 * It's a good idea to call this function following a large number of calls to
 * \ref jit_registry_remove().
 */
extern JITC_EXPORT void jit_registry_trim();

/**
 * \brief Set a custom per-pointer attribute
 *
 * The pointer registry can optionally associate one or more read-only
 * attribute with each pointer that can be set using this function. Such
 * pointer attributes provide an efficient way to avoid expensive vectorized
 * method calls (via \ref jitc_vcall()) for simple getter-like functions. In
 * particular, this feature would be used in conjunction with \ref
 * jit_registry_attr_data(), which returns a pointer to a linear array
 * containing all attributes. A vector of 32-bit IDs (returned by \ref
 * jit_registry_put() or \ref jit_registry_get_id()) can then be used to
 * gather from this address.
 *
 * \param ptr
 *     Pointer, whose attribute should be set. Must have been previously
 *     registered using \ref jit_registry_put()
 *
 * \param name
 *     Name of the attribute to be set.
 *
 * \param value
 *     Pointer to the attribute value (in CPU memory)
 *
 * \param size
 *     Size of the pointed-to region.
 */
extern JITC_EXPORT void jit_registry_set_attr(void *ptr,
                                              const char *name,
                                              const void *value,
                                              size_t size);

/**
 * \brief Return a pointer to a contiguous array containing a specific
 * attribute associated with a specific domain
 *
 * \sa jit_registry_set_attr
 */
extern JITC_EXPORT const void *jit_registry_attr_data(const char *domain,
                                                      const char *name);

// ====================================================================
//                        Variable management
// ====================================================================

#if defined(__cplusplus)
/**
 * \brief Variable types supported by the JIT compiler.
 *
 * A type promotion routine in the Enoki Python bindings depends on on this
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
 * generated PTX/LLVM code. This is particularly useful for loops: suppose a
 * loop references a literal constant that keeps changing (e.g. an iteration
 * counter). This change causes each iteration to generate different code,
 * requiring repeated compilation steps. By preemptively evaluating this
 * constant, Enoki-JIT can reuse a single kernel for all steps.
 */
extern JITC_EXPORT uint32_t jit_var_new_literal(JitBackend backend,
                                                JITC_ENUM VarType type,
                                                const void *value,
                                                uint32_t size JITC_DEF(1),
                                                int eval JITC_DEF(0));

/**
 * \brief Create a counter variable
 *
 * This operation creates a variable of type \ref VarType::UInt32 that will
 * evaluate to <tt>0, ..., size - 1</tt>.
 */
extern uint32_t jit_var_new_counter(JitBackend backend, uint32_t size);

/**
 * \brief Create a new variable representing the result of a LLVM/PTX statement
 *
 * This function takes a statement in an intermediate representation (CUDA PTX or
 * LLVM IR) and registers it in the global variable list. It returns the index
 * of the variable that will store the result of the statement, whose external
 * reference count is initialized to \c 1.
 *
 * You will probably want to access this function through the wrappers \ref
 * jit_var_new_stmt_0() to \ref jit_var_new_stmt_4() that take an explicit
 * list of parameter indices and assume that it's not necessary to make a copy
 * of \c stmt (i.e. <tt>stmt_static == 1</tt>).
 *
 * The string \c stmt may contain special dollar-prefixed expressions
 * (<tt>$rN</tt>, <tt>$tN</tt>, or <tt>$bN</tt>, where <tt>N</tt> ranges from
 * 0-4) to refer to operands and their types. During compilation, these will
 * then be rewritten into a register name of the variable (<tt>r</tt>), its
 * type (<tt>t</tt>), or a generic binary type of matching size (<tt>b</tt>).
 * Index <tt>0</tt> refers to the variable being generated, while indices
 * <tt>1<tt>-<tt>3</tt> refer to the operands. For instance, a PTX integer
 * addition would be encoded as follows:
 *
 * \code
 * uint32_t result = jit_var_new_stmt_2(1, VarType::Int32,
 *                                       "add.$t0 $r0, $r1, $r2",
 *                                       op1, op2);
 * \endcode
 *
 * \param cuda
 *    Specifies whether 'stmt' contains a CUDA PTX (<tt>cuda == 1</tt>) or LLVM
 *    IR (<tt>cuda == 0</tt>) instruction.
 *
 * \param vt
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param stmt
 *    Intermediate language statement.
 *
 * \param stmt_static
 *    When 'stmt' is a static string stored in the data segment of the
 *    executable, it is not necessary to make a copy. In this case, set
 *    <tt>stmt_static == 1</tt>, and <tt>0</tt> otherwise.
 *
 * \param n_dep
 *    Number of dependencies (between 0 and 4)
 *
 * \param dep
 *    Pointer to a list of \c n_dep valid variable indices
 */
extern JITC_EXPORT uint32_t jit_var_new_stmt(JitBackend backend,
                                             JITC_ENUM VarType vt,
                                             const char *stmt,
                                             int stmt_static,
                                             uint32_t n_dep,
                                             const uint32_t *dep);

// Create a new variable with 0 dependencies (wraps \c jit_var_new_stmt())
inline uint32_t jit_var_new_stmt_0(JitBackend backend, JITC_ENUM VarType vt,
                                   const char *stmt) {
    return jit_var_new_stmt(backend, vt, stmt, 1, 0, nullptr);
}

// Create a new variable with 1 dependency (wraps \c jit_var_new_stmt())
inline uint32_t jit_var_new_stmt_1(JitBackend backend, JITC_ENUM VarType vt,
                                   const char *stmt, uint32_t dep0) {
    return jit_var_new_stmt(backend, vt, stmt, 1, 1, &dep0);
}

// Create a new variable with 2 dependencies (wraps \c jit_var_new_stmt())
inline uint32_t jit_var_new_stmt_2(JitBackend backend, JITC_ENUM VarType vt,
                                   const char *stmt, uint32_t dep0,
                                   uint32_t dep1) {
    const uint32_t dep[] = { dep0, dep1 };
    return jit_var_new_stmt(backend, vt, stmt, 1, 2, dep);
}

// Create a new variable with 3 dependencies (wraps \c jit_var_new_stmt())
inline uint32_t jit_var_new_stmt_3(JitBackend backend, JITC_ENUM VarType vt,
                                   const char *stmt, uint32_t dep0,
                                   uint32_t dep1, uint32_t dep2) {
    const uint32_t dep[] = { dep0, dep1, dep2 };
    return jit_var_new_stmt(backend, vt, stmt, 1, 3, dep);
}

// Create a new variable with 4 dependencies (wraps \c jit_var_new_stmt())
inline uint32_t jit_var_new_stmt_4(JitBackend backend, JITC_ENUM VarType vt,
                                   const char *stmt, uint32_t dep0,
                                   uint32_t dep1, uint32_t dep2,
                                   uint32_t dep3) {
    const uint32_t dep[] = { dep0, dep1, dep2, dep3 };
    return jit_var_new_stmt(backend, vt, stmt, 1, 4, dep);
}

/// List of operations supported by \ref jit_var_new_op()
enum class OpType : uint32_t {
    // ---- Unary ----
    Not, Neg, Abs, Sqrt, Rcp, Rsqrt, Ceil, Floor, Round, Trunc, Exp2, Log2,
    Popc, Clz, Ctz,
    // ---- Binary ----
    Add, Sub, Mul, Div, Mod, Min, Max, And, Or, Xor, Shl, Shr,
    // ---- Comparisons ----
    Eq, Neq, Lt, Le, Gt, Ge,
    // ---- Ternary ----
    Fmadd, Select,

    Count
};

#if EK_OPNAME==1
const char *op_name[(int) OpType::Count] {
    "Not", "Neg", "Abs", "Sqrt", "Rcp", "Rsqrt", "Ceil", "Floor", "Round", "Trunc", "Exp2", "Log2",

    "Popc", "Clz", "Ctz",

    // ---- Binary ----
    "Add", "Sub", "Mul", "Div", "Mod", "Min", "Max", "And", "Or",
    "Xor", "Shl", "Shr",

    // ---- Comparisons ----
    "Eq", "Neq", "Lt", "Le", "Gt", "Ge",

    // ---- Ternary ----
    "Fmadd", "Select"
};
#endif


/**
 * \brief Perform an arithmetic operation involving one or more variables
 *
 * This function can perform a large range of unary, binary, and ternary
 * arithmetic operations. It automatically infers the necessary LLVM or PTX
 * instructions and performs constant propagation if possible (when one or
 * more input are literals).
 *
 * You will probably want to access this function through the wrappers \ref
 * jit_var_new_op_0() to \ref jit_var_new_op_4() that take an explicit
 * list of parameter indices.
 *
 * \param op
 *    The operation to be performed
 *
 * \param n_dep
 *    Number of dependencies (between 0 and 4)
 *
 * \param dep
 *    Pointer to a list of \c n_dep valid variable indices
 */
extern JITC_EXPORT uint32_t jit_var_new_op(JITC_ENUM OpType op,
                                           uint32_t n_dep, const uint32_t *dep);

// Perform an operation with 1 input (wraps \c jit_var_new_op())
inline uint32_t jit_var_new_op_1(JITC_ENUM OpType op,
                                 uint32_t dep0) {
    return jit_var_new_op(op, 1, &dep0);
}

// Perform an operation with 2 inputs (wraps \c jit_var_new_op())
inline uint32_t jit_var_new_op_2(JITC_ENUM OpType op, uint32_t dep0,
                                 uint32_t dep1) {
    const uint32_t dep[] = { dep0, dep1 };
    return jit_var_new_op(op, 2, dep);
}

// Perform an operation with 3 inputs (wraps \c jit_var_new_op())
inline uint32_t jit_var_new_op_3(JITC_ENUM OpType op, uint32_t dep0,
                                 uint32_t dep1, uint32_t dep2) {
    const uint32_t dep[] = { dep0, dep1, dep2 };
    return jit_var_new_op(op, 3, dep);
}

// Perform an operation with 4 inputs (wraps \c jit_var_new_op())
inline uint32_t jit_var_new_op_4(JITC_ENUM OpType op, uint32_t dep0,
                                 uint32_t dep1, uint32_t dep2, uint32_t dep3) {
    const uint32_t dep[] = { dep0, dep1, dep2, dep3 };
    return jit_var_new_op(op, 4, dep);
}

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
extern JITC_EXPORT uint32_t jitc_var_new_cast(uint32_t index,
                                              VarType target_type,
                                              int reinterpret);

/**
 * \brief Create an identical copy of the given variable
 *
 * This function creates an exact copy of the variable \c index and returns the
 * index of the copy, whose external reference count is initialized to 1.
 */
extern JITC_EXPORT uint32_t jit_var_copy(uint32_t index);


/**
 * Register an existing memory region as a variable in the JIT compiler, and
 * return its index. Its external reference count is initialized to \c 1.
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
extern JITC_EXPORT uint32_t jit_var_mem_map(JitBackend backend, JITC_ENUM VarType type,
                                            void *ptr, uint32_t size, int free);


/**
 * Copy a memory region onto the device and return its variable index. Its
 * external reference count is initialized to \c 1.
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
extern JITC_EXPORT uint32_t jit_var_mem_copy(JitBackend backend,
                                             JITC_ENUM AllocType atype,
                                             JITC_ENUM VarType vtype,
                                             const void *ptr,
                                             uint32_t size);

/**
 * \brief Return the combined number of internal and external references to a
 * variable
 *
 * If desired, references due to pending operations with side effects (i.e.
 * scatters) can be ignored by specifying a nonzero second argument.
 */
extern JITC_EXPORT uint32_t jit_var_refs(uint32_t index, int ignore_side_effects JITC_DEF(0));

/// Increase the external reference count of a given variable
extern JITC_EXPORT void jit_var_inc_ref_ext_impl(uint32_t index) JITC_NOEXCEPT;

/// Decrease the external reference count of a given variable
extern JITC_EXPORT void jit_var_dec_ref_ext_impl(uint32_t index) JITC_NOEXCEPT;

#if defined(__GNUC__)
JITC_INLINE void jit_var_inc_ref_ext(uint32_t index) JITC_NOEXCEPT {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to jit_var_dec_ref_ext */
    if (!__builtin_constant_p(index) || index != 0)
        jit_var_inc_ref_ext_impl(index);
}
JITC_INLINE void jit_var_dec_ref_ext(uint32_t index) JITC_NOEXCEPT {
    if (!__builtin_constant_p(index) || index != 0)
        jit_var_dec_ref_ext_impl(index);
}
#else
#define jit_var_dec_ref_ext jit_var_dec_ref_ext_impl
#define jit_var_inc_ref_ext jit_var_inc_ref_ext_impl
#endif

/// Query the pointer variable associated with a given variable
extern JITC_EXPORT void *jit_var_ptr(uint32_t index);

/// Query the size of a given variable
extern JITC_EXPORT uint32_t jit_var_size(uint32_t index);

/**
 * \brief Resize a scalar variable to a new size
 *
 * This function takes a scalar variable as input and changes its size
 * to \c size, potentially creating a new copy in case something already
 * depends on \c index. The function increases the external reference count of
 * the returned value. When \c index is not a scalar variable and its size
 * exactly matches \c size, the function does nothing and just increases
 * the external reference count of \c index. Otherwise, it fails.
 */
extern JITC_EXPORT uint32_t jit_var_resize(uint32_t index, uint32_t size);

/// Query the type of a given variable
extern JITC_EXPORT JITC_ENUM VarType jit_var_type(uint32_t index);

/// Assign a descriptive label to a given variable
extern JITC_EXPORT void jit_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern JITC_EXPORT const char *jit_var_label(uint32_t index);

/// Assign a callback function that is invoked when the given variable is freed
extern JITC_EXPORT void jit_var_set_free_callback(uint32_t index,
                                                   void (*callback)(void *),
                                                   void *payload);

/**
 * \brief Asynchronously migrate a variable to a different flavor of memory
 *
 * Returns the resulting variable index and increases its external reference
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
extern JITC_EXPORT uint32_t jit_var_migrate(uint32_t index, JITC_ENUM AllocType type);

/// Query the current (or future, if not yet evaluated) allocation flavor of a variable
extern JITC_EXPORT JITC_ENUM AllocType jit_var_alloc_type(uint32_t index);

/// Query the device (or future, if not yet evaluated) associated with a variable
extern JITC_EXPORT int jit_var_device(uint32_t index);

/**
 * \brief Mark a variable as a scatter operation
 *
 * This function should be used to inform the JIT compiler when the memory region
 * underlying a variable \c target is modified by a scatter operation with
 * index \c index. It will mark the target variable as dirty to ensure that
 * future reads from this variable (while still in dirty state) will trigger
 * an evaluation via \ref jit_eval().
 */
extern JITC_EXPORT void jit_var_mark_side_effect(uint32_t index, uint32_t target);

/**
 * \brief Return a human-readable summary of registered variables
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JITC_EXPORT const char *jit_var_whos();

/**
 * \brief Return a GraphViz representation of registered variables and their
 * dependencies
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JITC_EXPORT const char *jit_var_graphviz();

/**
 * \brief Return a human-readable summary of the contents of a variable
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JITC_EXPORT const char *jit_var_str(uint32_t index);

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
extern JITC_EXPORT void jit_var_read(uint32_t index, uint32_t offset,
                                     void *dst);

/**
 * \brief Copy 'dst' to a single element of a variable
 *
 * This function implements the reverse of jit_var_read(). This function is
 * convenient for testing, and to change localized entries of an array, but it
 * should never be used to access the complete contents of an array due to its
 * low performance (every write will be performed via an individual
 * asynchronous transaction).
 */
extern JITC_EXPORT void jit_var_write(uint32_t index, uint32_t offset,
                                      const void *src);

/**
 * \brief Print the specified variable contents from the kernel
 *
 * This function inserts a print statement directly into the kernel being
 * generated. Note that this may produce a very large volume of output.
 * Up to 3 variables can be referenced at once.
 *
 * Example: <tt>jit_var_printf(1, "Hello world: %f\n", 1, &my_variable_id);</tt>
 */
extern JITC_EXPORT void jit_var_printf(JitBackend backend, const char *fmt,
                                       uint32_t narg, const uint32_t *arg);

/**
 */
extern JITC_EXPORT void
jit_var_vcall(JitBackend backend, const char *domain, const char *name, uint32_t self,
              uint32_t n_inst, const uint32_t *inst_ids,
              const uint64_t *inst_hash, uint32_t n_in, const uint32_t *in,
              uint32_t n_out, uint32_t *out, const uint32_t *need_in,
              const uint32_t *need_out, uint32_t n_extra, const uint32_t *extra,
              const uint32_t *extra_offset, int side_effects);

// ====================================================================
//                 Kernel compilation and evaluation
// ====================================================================

/**
 * \brief Schedule a variable \c index for future evaluation via \ref jit_eval()
 *
 * Returns \c 1 if anything was scheduled, and \c 0 otherwise.
 */
extern JITC_EXPORT int jit_var_schedule(uint32_t index);

/**
 * \brief Evaluate the variable \c index right away, if it is unevaluated/dirty.
 *
 * Returns \c 1 if anything was evaluated, and \c 0 otherwise.
 */
extern JITC_EXPORT int jit_var_eval(uint32_t index);

/// Evaluate all scheduled computation
extern JITC_EXPORT void jit_eval();

// ====================================================================
//  Assortment of tuned kernels for initialization, reductions, etc.
// ====================================================================

#if defined(__cplusplus)
/// Potential reduction operations for \ref jit_reduce
enum class ReductionType : uint32_t { Add, Mul, Min, Max, And, Or, Count };
#else
enum ReductionType {
    ReductionTypeAdd, ReductionTypeMul, ReductionTypeMin,
    ReductionTypeMax, ReductionTypeAnd, ReductionTypeOr,
    ReductionTypeCount
};
#endif

/**
 * \brief Fill a device memory region with constants of a given type
 *
 * This function writes \c size values of size \c isize to the output array \c
 * ptr. The specific value is taken from \c src, which must be a CPU pointer to
 * a single int, float, double, etc. (\c isize can be 1, 2, 4, or 8).
 * Runs asynchronously.
 */
extern JITC_EXPORT void jit_memset_async(JitBackend backend, void *ptr, uint32_t size,
                                         uint32_t isize, const void *src);

/// Perform a synchronous copy operation
extern JITC_EXPORT void jit_memcpy(JitBackend backend, void *dst, const void *src, size_t size);

/// Perform an asynchronous copy operation
extern JITC_EXPORT void jit_memcpy_async(JitBackend backend, void *dst, const void *src,
                                         size_t size);

/**
 * \brief Reduce the given array to a single value
 *
 * This operation reads \c size values of type \type from the input array \c
 * ptr and performs an specified operation (e.g., addition, multplication,
 * etc.) to combine them into a single value that is written to the device
 * variable \c out.
 *
 * Runs asynchronously.
 */
extern JITC_EXPORT void jit_reduce(JitBackend backend, JITC_ENUM VarType type,
                                   JITC_ENUM ReductionType rtype,
                                   const void *ptr, uint32_t size, void *out);

/**
 * \brief Perform an exclusive scan / prefix sum over an unsigned 32 bit integer
 * array
 *
 * If desired, the scan can be performed in-place (i.e. <tt>in == out</tt>).
 * Note that the CUDA implementation will round up \c size to the maximum of
 * the following three values for performance reasons:
 *
 * - 4
 * - the next highest power of two (when size <= 4096),
 * - the next highest multiple of 2K (when size > 4096),
 *
 * For this reason, the the supplied memory regions must be sufficiently large
 * to avoid both out-of-bounds reads and writes. This is not an issue for
 * memory obtained using \ref jit_malloc(), which internally rounds
 * allocations to the next largest power of two and enforces a 64 byte minimum
 * allocation size.
 *
 * Runs asynchronously.
 */
extern JITC_EXPORT void jit_scan_u32(JitBackend backend, const uint32_t *in,
                                     uint32_t size, uint32_t *out);

/**
 * \brief Compress a mask into a list of nonzero indices
 *
 * This function takes an 8-bit mask array \c in with size \c size as input,
 * whose entries are required to equal either zero or one. It then writes the
 * indices of nonzero entries to \c out (in increasing order), and it
 * furthermore returns the total number of nonzero mask entries.
 *
 * The internals resemble \ref jit_scan_u32(), and the CUDA implementation may
 * similarly access regions beyond the end of \c in and \c out.
 *
 * This function internally performs a synchronization step.
 */
extern JITC_EXPORT uint32_t jit_compress(JitBackend backend, const uint8_t *in,
                                         uint32_t size, uint32_t *out);

/**
 * \brief Reduce an array of boolean values to a single value (AND case)
 *
 * When \c size is not a multiple of 4, the implementation will initialize up
 * to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
 * reduction algorithm can be used. This is fine for allocations made using
 * \ref jit_malloc(), which allow for this.
 *
 * Runs synchronously.
 */
extern JITC_EXPORT uint8_t jit_all(JitBackend backend, uint8_t *values, uint32_t size);

/**
 * \brief Reduce an array of boolean values to a single value (OR case)
 *
 * When \c size is not a multiple of 4, the implementation will initialize up
 * to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
 * reduction algorithm can be used. This is fine for allocations made using
 * \ref jit_malloc(), which allow for this.
 *
 * Runs synchronously.
 */
extern JITC_EXPORT uint8_t jit_any(JitBackend backend, uint8_t *values, uint32_t size);


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
extern JITC_EXPORT uint32_t jit_mkperm(JitBackend backend, const uint32_t *values,
                                       uint32_t size, uint32_t bucket_count,
                                       uint32_t *perm, uint32_t *offsets);

/// Helper data structure for vector method calls, see \ref jit_vcall()
struct VCallBucket {
    /// Resolved pointer address associated with this bucket
    void *ptr;

    /// Variable index of a uint32 array storing a partial permutation
    uint32_t index;

    /// Padding
    uint32_t unused;
};

/**
 * \brief Compute a permutation to reorder an array of registered pointers
 * in preparation for a vectorized method call
 *
 * This function expects an array of integers, whose entries correspond to
 * pointers that have previously been registered by calling \ref
 * jit_registry_put() with domain \c domain. It then invokes \ref jitc_mkperm()
 * to compute a permutation that reorders the array into coherent buckets. The
 * buckets are returned using an array of type \ref VCallBucket, which contains
 * both the resolved pointer address (obtained via \ref
 * jit_registry_get_ptr()) and the variable index of an unsigned 32 bit array
 * containing the corresponding entries of the input array. The total number of
 * buckets is returned via the \c bucket_count_out argument.
 *
 * The memory region accessible via the \c VCallBucket pointer will remain
 * accessible until the variable \c index is itself freed (i.e. when its
 * internal and external reference counts both become equal to zero). Until
 * then, additional calls to \ref jit_vcall() will return the previously
 * computed result. This is an important optimization in situations where
 * multiple vector function calls are executed on the same set of instances.
 */
extern JITC_EXPORT struct VCallBucket *jit_vcall(JitBackend backend, const char *domain,
                                                 uint32_t index,
                                                 uint32_t *bucket_count_out);

/**
 * \brief Replicate individual input elements across larger blocks
 *
 * This function copies each element of the input array \c to a contiguous
 * block of size \c block_size in the output array \c out. For example, <tt>a,
 * b, c</tt> turns into <tt>a, a, b, b, c, c</tt> when the \c block_size is set
 * to \c 2. The input array must contain <tt>size</tt> elements, and the output
 * array must have space for <tt>size * block_size</tt> elements.
 */
extern JITC_EXPORT void jit_block_copy(JitBackend backend, JITC_ENUM VarType type,
                                       const void *in, void *out,
                                       uint32_t size, uint32_t block_size);

/**
 * \brief Sum over elements within blocks
 *
 * This function adds all elements of contiguous blocks of size \c block_size
 * in the input array \c in and writes them to \c out. For example, <tt>a, b,
 * c, d, e, f</tt> turns into <tt>a+b, c+d, e+f</tt> when the \c block_size is
 * set to \c 2. The input array must contain <tt>size * block_size</tt> elements,
 * and the output array must have space for <tt>size</tt> elements.
 */
extern JITC_EXPORT void jit_block_sum(JitBackend backend, JITC_ENUM VarType type,
                                      const void *in, void *out, uint32_t size,
                                      uint32_t block_size);

#define ENOKI_USE_ALLOCATOR(Type)                                              \
    void *operator new(size_t size) { return jit_malloc(Type, size); }        \
    void *operator new(size_t size, std::align_val_t) {                        \
        return jit_malloc(Type, size);                                        \
    }                                                                          \
    void *operator new[](size_t size) { return jit_malloc(Type, size); }      \
    void *operator new[](size_t size, std::align_val_t) {                      \
        return jit_malloc(Type, size);                                        \
    }                                                                          \
    void operator delete(void *ptr) { jit_free(ptr); }                        \
    void operator delete(void *ptr, std::align_val_t) { jit_free(ptr); }      \
    void operator delete[](void *ptr) { jit_free(ptr); }                      \
    void operator delete[](void *ptr, std::align_val_t) { jit_free(ptr); }

#if defined(__cplusplus)
}
#endif
