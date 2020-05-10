/*
    enoki-jit/jit.h -- Self-contained JIT compiler for CUDA & LLVM.

    This library implements a self-contained tracing JIT compiler that supports
    both CUDA PTX and LLVM IR as intermediate languages. It takes care of many
    tricky aspects, such as asynchronous memory allocation and release,
    multi-device computation, kernel caching and reuse, common subexpression
    elimination, etc.

    While the library is internally implemented using C++11, this header file
    provides a compact C99-compatible API that can be used to access all
    functionality. The library is thread-safe: multiple threads can
    simultaneously dispatch computation to one or more CPUs/GPUs.

    As an alternative to the fairly low-level API defined here, you may prefer
    to use the functionality in 'enoki/jitvar.h', which provides a header-only
    C++ array class with operator overloading, which dispatches to the C API.

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
#else
#  define JITC_EXPORT    __attribute__ ((visibility("default")))
#  define JITC_MALLOC    __attribute__((malloc))
#endif

#if defined(__cplusplus)
#  define JITC_CONSTEXPR constexpr
#else
#  define JITC_CONSTEXPR inline
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// ====================================================================
//         Initialization, device enumeration, and management
// ====================================================================

/**
 * \brief Initialize core data structures of the JIT compiler
 *
 * This function must be called before using any of the remaining API. It
 * detects the available devices and initializes them for later use. It does
 * nothing when initialization has already occurred. Note that it is possible
 * to re-initialize the JIT following a call to \ref jitc_shutdown(), which can
 * be useful to start from a known state, e.g., in testcases.
 *
 * The \c llvm and \c cuda arguments should be set to \c 1 to initialize the
 * corresponding backend, and \c 0 otherwise.
 */
extern JITC_EXPORT void jitc_init(int llvm, int cuda);

/**
 * \brief Launch an ansynchronous thread that will execute jitc_init() and
 * return immediately
 *
 * On machines with several GPUs, \ref jitc_init() will set up a CUDA
 * environment on all devices when <tt>cuda=true</tt> is specified. This can be
 * a rather slow operation (e.g. 1 second). This function provides a convenient
 * alternative to hide this latency, for instance when importing this library
 * from an interactive Python session which doesn't need the JIT right away.
 *
 * The \c llvm and \c cuda arguments should be set to \c 1 to initialize the
 * corresponding backend, and \c 0 otherwise.
 *
 * Note that it is safe to call <tt>jitc_*</tt> API functions following
 * initialization via \ref jitc_init_async(), since it acquires a lock to the
 * internal data structures.
 */
extern JITC_EXPORT void jitc_init_async(int llvm, int cuda);

/// Check whether the LLVM backend was successfully initialized
extern JITC_EXPORT int jitc_has_llvm();

/// Check whether the CUDA backend was successfully initialized
extern JITC_EXPORT int jitc_has_cuda();

/**
 * \brief Release resources used by the JIT compiler, and report reference leaks.
 *
 * If <tt>light=1</tt>, this function performs a "light" shutdown, which
 * flushes any still running computation and releases unused memory back to the
 * OS or GPU. It will also warn about leaked variables and memory allocations.
 *
 * If <tt>light=0</tt>, the function furthermore completely unloads the LLVM or
 * CUDA backends. This frees up more memory but means that a later call to \ref
 * jitc_init() or \ref jitc_init_async() will be slow.
 */
extern JITC_EXPORT void jitc_shutdown(int light);

/**
 * \brief Return the number of target devices
 *
 * This function returns the number of available devices. Note that this refers
 * to the number of compatible CUDA devices, excluding the host CPU.
 */
extern JITC_EXPORT int32_t jitc_device_count();

/**
 * Set the currently active device and stream
 *
 * \param device
 *     Specifies the device index, a number between -1 and
 *     <tt>jitc_device_count() - 1</tt>. The number <tt>-1</tt> indicates that
 *     execution should take place on the host CPU (via LLVM). <tt>0</tt> is
 *     the first GPU (execution via CUDA), <tt>1</tt> is the second GPU, etc.
 *
 * \param stream
 *     When accessing the JIT compiler in a multi-threaded program, each thread
 *     msut run in a separate stream that is provided via this argument. The
 *     streams don't have to be numbered in any particular way, but they need
 *     to be just be unique per thread.
 */
extern JITC_EXPORT void jitc_set_device(int32_t device, uint32_t stream);

/**
 * \brief Override the target CPU, features, and vector witdth of the LLVM backend
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
extern JITC_EXPORT void jitc_llvm_set_target(const char *target_cpu,
                                             const char *target_features,
                                             uint32_t vector_width);

/// Get the CPU that is currently targeted by the LLVM backend
extern JITC_EXPORT const char *jitc_llvm_target_cpu();

/// Get the list of CPU features currently used by the LLVM backend
extern JITC_EXPORT const char *jitc_llvm_target_features();

/// Return the major version of the LLVM library
extern JITC_EXPORT int jitc_llvm_version_major();

/**
 * \brief Convenience function for intrinsic function selection
 *
 * Returns \c 1 if the current vector width is is at least as large as a
 * provided value, and when the host CPU provides a given target feature (e.g.
 * "+avx512f").
 */
extern JITC_EXPORT int jitc_llvm_if_at_least(uint32_t vector_width,
                                             const char *feature);

/**
 * \brief Dispatch computation to multiple CUDA streams or CPU cores?
 *
 * This parameter has slightly different meanings depending on the backend
 * that is used:
 *
 * <ul>
 * <li><b>CUDA backend</b>: The JIT compiler attempts to fuse all queued
 * computation into a single kernel to maximize efficiency, and this kernel
 * runs in parallel using all streaming multiprocessors available on the GPU.
 * But computation involving arrays of different size must necessarily run in
 * separate kernels, which means that it is serialized if taking place within
 * the same device and stream. In some cases, kernels only involve very small
 * inputs and outputs, in which case scheduling overheads can become
 * significant. If desired, jitc_eval() can detect this and dispatch multiple
 * kernels to separate streams that execute in parallel <em>in addition to the
 * parallel execution of individual kernels</em>.</li>
 *
 * <li><b>LLVM backend</b>: If Enoki-JIT is compiled with Intel's Thread
 * Building Blocks (TBB), execution will parallelize over all available cores
 * if this parameter is set to \c 1. Otherwise, execution will be
 * serialized.</li>
 * </ul>
 *
 * The default is \c 1 (i.e. to enable parallel dispatch) for both LLVM/CUDA
 * backends.
 */
extern JITC_EXPORT void jitc_set_parallel_dispatch(int enable);

/// Return whether or not parallel dispatch is enabled. Returns \c 0 or \c 1.
extern JITC_EXPORT int jitc_parallel_dispatch();

/// Wait for all computation on the current stream to finish
extern JITC_EXPORT void jitc_sync_stream();

/// Wait for all computation on the current device to finish
extern JITC_EXPORT void jitc_sync_device();

/// Wait for all computation on the *all devices* to finish
extern JITC_EXPORT void jitc_sync_all_devices();


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
 * via a callback in \ref jitc_set_log_level_callback(). Both destinations can also
 * be enabled simultaneously, pontentially using different log levels.
 */
extern JITC_EXPORT void jitc_set_log_level_stderr(enum LogLevel level);

/// Return the currently set minimum log level for output to \c stderr
extern JITC_EXPORT enum LogLevel jitc_log_level_stderr();


/**
 * \brief Control the destination of log messages (callback)
 *
 * This function can be used to specify an optional callback that will be
 * invoked with the contents of library log messages, whose severity matches or
 * exceeds the specified \c level.
 */
typedef void (*LogCallback)(enum LogLevel, const char *);
extern JITC_EXPORT void jitc_set_log_level_callback(enum LogLevel level, LogCallback callback);

/// Return the currently set minimum log level for output to a callback
extern JITC_EXPORT enum LogLevel jitc_log_level_callback();

/// Print a log message with the specified log level and message
extern JITC_EXPORT void jitc_log(enum LogLevel level, const char* fmt, ...);

/// Raise an exception message with the specified message
extern JITC_EXPORT void jitc_raise(const char* fmt, ...);

/// Terminate the application due to a non-recoverable error
extern JITC_EXPORT void jitc_fail(const char* fmt, ...);

// ====================================================================
//                         Memory allocation
// ====================================================================

#if defined(__cplusplus)
enum class AllocType : uint32_t {
    /**
     * Memory that is located on the host (i.e., the CPU). When allocated via
     * \ref jitc_malloc(), host memory is immediately ready for use, and
     * its later release via \ref jitc_free() also occurs instantaneously.
     *
     * Note, however, that released memory is kept within a cache and not
     * immediately given back to the operating system. Call \ref
     * jitc_malloc_trim() to also flush this cache.
     */
    Host,

    /**
     * Like \c Host memory, except that it may only be used *asynchronously*
     * within the current computation stream (ref see \ref jitc_set_device()).
     *
     * In particular, host-asynchronous memory obtained via \ref jitc_malloc()
     * should not be written to directly, since it may still be used by a
     * currently running kernel. It is legal to write to it from asynchronous
     * computation that is enqueued in the computation stream, however.
     * Similarly, releasing memory via \ref jitc_free() also occurs
     * asynchronously in the context of the computation stream.
     *
     * This type of memory is used internally when running code via the LLVM
     * backend, and when this process is furthermore parallelized using Intels'
     * Thread Building Blocks (TBB).
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

    /// Number of AllocType entries
    Count
};
#else
enum AllocType {
    AllocTypeHost,
    AllocTypeHostPinned,
    AllocTypeDevice,
    AllocTypeManaged,
    AllocTypeManagedReadMostly,
    AllocTypeCount,
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
extern JITC_EXPORT void *jitc_malloc(enum AllocType type, size_t size)
    JITC_MALLOC;

/**
 * \brief Release a given pointer asynchronously
 *
 * For CPU-only arrays (\ref AllocType::Host), <tt>jitc_free()</tt> is
 * synchronous and very similar to <tt>free()</tt>, except that the released
 * memory is placed in Enoki's internal allocation cache instead of being
 * returned to the OS. The function \ref jitc_malloc_trim() can optionally be
 * called to also clear this cache.
 *
 * When \c ptr is an asynchronous host pointer (\ref AllocType::HostAsync) or
 * GPU-accessible pointer (\ref AllocType::Device, \ref AllocType::HostPinned,
 * \ref AllocType::Managed, \ref AllocType::ManagedReadMostly), the associated
 * memory region is quite likely still being used by a running kernel, and it
 * is therefore merely *scheduled* to be reclaimed once this kernel finishes.
 * All memory-related operations thus occur in the context of a stream, and
 * this also means that extra care must be taken in the context of programs
 * that use multiple streams: it is not permissible to e.g. allocate memory in
 * one stream, launch a kernel using it, then immediately switch to a different
 * device and/or stream via \ref jitc_set_device() and release the memory
 * region there. Calling \ref jitc_sync_stream() before context switching
 * streams defuses this situation.
 */
extern JITC_EXPORT void jitc_free(void *ptr);

/**
 * \brief Asynchronously change the flavor of an allocated memory region and
 * return the new pointer
 *
 * The operation is *always* asynchronous and, hence, will need to be followed
 * by an explicit synchronization via \ref jitc_sync_stream() if memory is
 * migrated from the GPU to the CPU and expected to be accessed on the CPU
 * before the transfer has finished. Nothing needs to be done in the other
 * direction, e.g. when migrating memory that is subsequently accessed by
 * a GPU kernel.
 *
 * When no migration is necessary, the function simply returns the input
 * pointer. Otherwise, it returns a new pointer and asynchronously frees the
 * old one via (via \ref jitc_free()). When both source and target are of
 * type \ref AllocType::Device, and when the currently active device
 * (determined by the last call to \ref jitc_set_device()) does not match the
 * device associated with the allocation, a peer-to-peer migration is
 * performed.
 */
extern JITC_EXPORT void* jitc_malloc_migrate(void *ptr, enum AllocType type);

/// Release all currently unused memory to the GPU / OS
extern JITC_EXPORT void jitc_malloc_trim();

/**
 * \brief Asynchronously prefetch a memory region allocated using \ref
 * jitc_malloc() so that it is available on a specified device
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
 * The prefetch operation is enqueued on the current device and stream and runs
 * asynchronously with respect to the CPU, hence a \ref jitc_sync_stream()
 * operation is advisable if data is <tt>target==-1</tt> (i.e. prefetching into
 * CPU memory).
 */
extern JITC_EXPORT void jitc_malloc_prefetch(void *ptr, int device);


// ====================================================================
//                          Pointer registry
// ====================================================================

/**
 * \brief Register a pointer with Enoki's pointer registry
 *
 * Enoki provides a central registry that maps registered pointer values to
 * low-valued 32-bit IDs. The main application is efficient virtual function
 * dispatch via \ref jitc_vcall(), through the registry could be used for other
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
extern JITC_EXPORT uint32_t jitc_registry_put(const char *domain, void *ptr);

/**
 * \brief Remove a pointer from the registry
 *
 * No-op if <tt>ptr == nullptr</tt>. Throws an exception if the pointer is not
 * currently registered.
 */
extern JITC_EXPORT void jitc_registry_remove(void *ptr);

/**
 * \brief Query the ID associated a registered pointer
 *
 * Returns 0 if <tt>ptr==nullptr</tt> and throws if the pointer is not known.
 */
extern JITC_EXPORT uint32_t jitc_registry_get_id(const void *ptr);

/**
 * \brief Query the domain associated a registered pointer
 *
 * Returns \c nullptr if <tt>ptr==nullptr</tt> and throws if the pointer is not
 * known.
 */
extern JITC_EXPORT const char *jitc_registry_get_domain(const void *ptr);

/**
 * \brief Query the pointer associated a given domain and ID
 *
 * Returns \c nullptr if <tt>id==0</tt> and throws if the (domain, ID)
 * combination is not known.
 */
extern JITC_EXPORT void *jitc_registry_get_ptr(const char *domain, uint32_t id);

/// Provide a bound (<=) on the largest ID associated with a domain
extern JITC_EXPORT uint32_t jitc_registry_get_max(const char *domain);

/**
 * \brief Compact the registry and release unused IDs and attributes
 *
 * It's a good idea to call this function following a large number of calls to
 * \ref jitc_registry_remove().
 */
extern JITC_EXPORT void jitc_registry_trim();

/**
 * \brief Set a custom per-pointer attribute
 *
 * The pointer registry can optionally associate one or more read-only
 * attribute with each pointer that can be set using this function. Such
 * pointer attributes provide an efficient way to avoid expensive vectorized
 * method calls (via \ref jit_vcall()) for simple getter-like functions. In
 * particular, this feature would be used in conjunction with \ref
 * jitc_registry_attr_data(), which returns a pointer to a linear array
 * containing all attributes. A vector of 32-bit IDs (returned by \ref
 * jitc_registry_put() or \ref jitc_registry_get_id()) can then be used to
 * gather from this address.
 *
 * \param ptr
 *     Pointer, whose attribute should be set. Must have been previously
 *     registered using \ref jitc_registry_put()
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
extern JITC_EXPORT void jitc_registry_set_attr(void *ptr,
                                               const char *name,
                                               const void *value,
                                               size_t size);

/**
 * \brief Return a pointer to a contiguous array containing a specific
 * attribute associated with a specific domain
 *
 * \sa jitc_registry_set_attr
 */
extern JITC_EXPORT const void *jitc_registry_attr_data(const char *domain,
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
    Invalid, Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32,
    Int64, UInt64, Float16, Float32, Float64, Pointer, Count
};
#else
enum VarType {
    VarTypeInvalid, VarTypeBool, VarTypeInt8, VarTypeUInt8, VarTypeInt16,
    VarTypeUInt16, VarTypeInt32, VarTypeUInt32, VarTypeInt64, VarTypeUInt64,
    VarTypeFloat16, VarTypeFloat32, VarTypeFloat64, VarTypePointer, VarTypeCount
};
#endif

/// Convenience function to check for an integer operand
JITC_CONSTEXPR int jitc_is_integral(enum VarType type) {
#if defined(__cplusplus)
    return ((uint32_t) type >= (uint32_t) VarType::Int8 &&
            (uint32_t) type <= (uint32_t) VarType::UInt64) ? 1 : 0;
#else
    return ((uint32_t) type >= (uint32_t) VarTypeInt8 &&
            (uint32_t) type <= (uint32_t) VarTypeUInt64) ? 1 : 0;
#endif
}

/// Convenience function to check for a floating point operand
JITC_CONSTEXPR uint32_t jitc_is_floating_point(enum VarType type) {
#if defined(__cplusplus)
    return ((uint32_t) type >= (uint32_t) VarType::Float16 &&
            (uint32_t) type <= (uint32_t) VarType::Float64) ? 1 : 0;
#else
    return ((uint32_t) type >= (uint32_t) VarTypeFloat16 &&
            (uint32_t) type <= (uint32_t) VarTypeFloat64) ? 1 : 0;
#endif
}

/// Convenience function to check for an arithmetic operand
JITC_CONSTEXPR uint32_t jitc_is_arithmetic(enum VarType type) {
#if defined(__cplusplus)
    return ((uint32_t) type >= (uint32_t) VarType::Int8 &&
            (uint32_t) type <= (uint32_t) VarType::Float64) ? 1 : 0;
#else
    return ((uint32_t) type >= (uint32_t) VarTypeInt8 &&
            (uint32_t) type <= (uint32_t) VarTypeFloat64) ? 1 : 0;
#endif
}

/// Convenience function to check for an unsigned operand
JITC_CONSTEXPR uint32_t jitc_is_unsigned(enum VarType type) {
#if defined(__cplusplus)
    return ((uint32_t) type == (uint32_t) VarType::UInt8 ||
            (uint32_t) type == (uint32_t) VarType::UInt16 ||
            (uint32_t) type == (uint32_t) VarType::UInt32 ||
            (uint32_t) type == (uint32_t) VarType::UInt64);
#else
    return ((uint32_t) type == (uint32_t) VarTypeUInt8 ||
            (uint32_t) type == (uint32_t) VarTypeUInt16 ||
            (uint32_t) type == (uint32_t) VarTypeUInt32 ||
            (uint32_t) type == (uint32_t) VarTypeUInt64);
#endif
}

/// Convenience function to check for a mask operand
JITC_CONSTEXPR int jitc_is_mask(enum VarType type) {
#if defined(__cplusplus)
    return type == VarType::Bool;
#else
    return type == VarTypeBool;
#endif
}

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
 *    \ref jitc_free() once it goes out of scope.
 *
 * \sa jitc_var_copy()
 */
extern JITC_EXPORT uint32_t jitc_var_map(enum VarType type, int cuda,
                                         void *ptr, uint32_t size, int free);


/**
 * Copy a memory region onto the device and return its variable index. Its
 * external reference count is initialized to \c 1.
 *
 * \param atype
 *    Enumeration characterizing the "flavor" of the source memory
 *
 * \param vtype
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
 * \sa jitc_var_map()
 */
extern JITC_EXPORT uint32_t jitc_var_copy(enum AllocType atype,
                                          enum VarType vtype,
                                          int cuda,
                                          const void *ptr,
                                          uint32_t size);

/**
 * Register a pointer literal as a variable within the JIT compiler
 *
 * When working with memory (gathers, scatters) using the JIT compiler, we must
 * often refer to memory addresses. These addresses change over time and should
 * not baked into the JIT-compiled code, since this would impede the re-use of
 * previously compiled kernels.
 *
 * This function registers a pointer literal that accomplishes this. It is
 * functionally equivalent to
 *
 * \code
 * void *my_ptr = ...;
 * uint32_t index_out = jitc_var_copy(VarType::Pointer, &my_ptr, 1);
 * \endcode
 *
 * but generates code that is more efficient.
 *
 * As an optional extra feature, the variable index of the underlying storage
 * region can be specified via the \c index argument. The system will then
 * guarantee that this variable is kept alive while the created variable (i.e.
 * the returned variable index) is alive. Specifying <tt>index=0</tt> disables
 * this behavior.
 */
extern JITC_EXPORT uint32_t jitc_var_copy_ptr(const void *ptr, uint32_t index);

/**
 * \brief Append a statement to the instruction trace.
 *
 * This function takes a statement in an intermediate language (CUDA PTX or
 * LLVM IR) and appends it to the list of currently queued operations. It
 * returns the index of the variable that will store the result of the
 * statement, whose external reference count is initialized to \c 1.
 *
 * This function assumes that the operation does not access any operands. See
 * the other <tt>jitc_var_*</tt> functions for IR statements with 1 to 4
 * additional operands. In these latter versions, the string \c stmt may
 * contain special dollar-prefixed expressions (<tt>$rN</tt>, <tt>$tN</tt>, or
 * <tt>$bN</tt>, where <tt>N</tt> ranges from 0-4) to refer to operands and
 * their types. During compilation, these will then be rewritten into a
 * register name of the variable (<tt>r</tt>), its type (<tt>t</tt>), or a
 * generic binary type of matching size (<tt>b</tt>). Index <tt>0</tt> refers
 * to the variable being generated, while indices <tt>1<tt>-<tt>3</tt> refer to
 * the operands. For instance, a PTX integer addition would be encoded as
 * follows:
 *
 * \code
 * uint32_t result = jitc_var_new_2(VarType::Int32,
 *                                       "add.$t0 $r0, $r1, $r2",
 *                                       1, op1, op2);
 * \endcode
 *
 * \param type
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
 * \param cuda
 *    Specifies whether 'stmt' contains a CUDA PTX (<tt>cuda == 1</tt>) or LLVM
 *    IR (<tt>cuda == 0</tt>) instruction.
 *
 * \param size
 *    Size of the resulting variable. The size is automatically inferred from
 *    the operands and must only be specified for the zero-argument form.
 */
extern JITC_EXPORT uint32_t jitc_var_new_0(enum VarType type,
                                           const char *stmt,
                                           int cuda,
                                           int stmt_static,
                                           uint32_t size);

/// Append a variable to the instruction trace (1 operand)
extern JITC_EXPORT uint32_t jitc_var_new_1(enum VarType type,
                                           const char *stmt,
                                           int cuda,
                                           int stmt_static,
                                           uint32_t op1);

/// Append a variable to the instruction trace (2 operands)
extern JITC_EXPORT uint32_t jitc_var_new_2(enum VarType type,
                                           const char *stmt,
                                           int stmt_static,
                                           int cuda,
                                           uint32_t op1,
                                           uint32_t op2);

/// Append a variable to the instruction trace (3 operands)
extern JITC_EXPORT uint32_t jitc_var_new_3(enum VarType type,
                                           const char *stmt,
                                           int stmt_static,
                                           int cuda,
                                           uint32_t op1,
                                           uint32_t op2,
                                           uint32_t op3);

/// Append a variable to the instruction trace (4 operands)
extern JITC_EXPORT uint32_t jitc_var_new_4(enum VarType type,
                                           const char *stmt,
                                           int stmt_static,
                                           int cuda,
                                           uint32_t op1,
                                           uint32_t op2,
                                           uint32_t op3,
                                           uint32_t op4);

/// Append a new variable storing 'size' entries of a literal constant
extern JITC_EXPORT uint32_t jitc_var_new_literal(enum VarType type,
                                                 int cuda,
                                                 uint64_t value,
                                                 uint32_t size);

/// Return the external reference count of a given variable
extern JITC_EXPORT uint32_t jitc_var_ext_ref(uint32_t index);

/// Return the internal reference count of a given variable
extern JITC_EXPORT uint32_t jitc_var_int_ref(uint32_t index);

/// Increase the external reference count of a given variable
extern JITC_EXPORT void jitc_var_inc_ref_ext(uint32_t index);

/// Decrease the external reference count of a given variable
extern JITC_EXPORT void jitc_var_dec_ref_ext(uint32_t index);

/// Query the pointer variable associated with a given variable
extern JITC_EXPORT void *jitc_var_ptr(uint32_t index);

/// Query the size of a given variable
extern JITC_EXPORT uint32_t jitc_var_size(uint32_t index);

// Try to resize a scalar variable, potentially returns a new index
extern JITC_EXPORT uint32_t jitc_var_set_size(uint32_t index, uint32_t size);

/// Assign a descriptive label to a given variable
extern JITC_EXPORT void jitc_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern JITC_EXPORT const char *jitc_var_label(uint32_t index);

/**
 * \brief Asynchronously migrate a variable to a different flavor of memory
 *
 * The operation is asynchronous and, hence, will need to be followed by \ref
 * jitc_sync_stream() if managed memory is subsequently accessed on the CPU.
 *
 * When both source & target are of type \ref AllocType::Device, and if the
 * current device (\ref jitc_set_device()) does not match the device associated
 * with the allocation, a peer-to-peer migration is performed.
 */
extern JITC_EXPORT void jitc_var_migrate(uint32_t index, enum AllocType type);

/**
 * \brief Mark a variable as a scatter operation
 *
 * This function must be used to inform the JIT compiler when the memory region
 * underlying a variable \c target is modified by a scatter operation with
 * index \c index. It will mark the target variable as dirty to ensure that
 * future reads from this variable (while still in dirty state) will trigger
 * an evaluation via \ref jitc_eval().
 */
extern JITC_EXPORT void jitc_var_mark_scatter(uint32_t index, uint32_t target);

/**
 * \brief Is the given variable a literal that equals zero?
 *
 * This function can be used to implement very rudimentary constant propagation
 * of arithmetic and masks, which can often significantly reduce the size of
 * the IR representation that is passed onto the backend for further
 * optimization. Note that this function can only detect matching variables if
 * they have not been explicitly evaluated.
 */
extern JITC_EXPORT int jitc_var_is_literal_zero(uint32_t index);

/**
 * \brief Is the given variable a literal that equals one?
 *
 * This function can be used to implement very rudimentary constant propagation
 * of arithmetic and masks, which can often significantly reduce the size of
 * the IR representation that is passed onto the backend for further
 * optimization. Note that this function can only detect matching variables if
 * they have not been explicitly evaluated.
 */
extern JITC_EXPORT int jitc_var_is_literal_one(uint32_t index);

/**
 * \brief Return a human-readable summary of registered variables
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jitc_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JITC_EXPORT const char *jitc_var_whos();

/**
 * \brief Return a GraphViz representation of registered variables and their
 * dependencies
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jitc_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JITC_EXPORT const char *jitc_var_graphviz();

/**
 * \brief Return a human-readable summary of the contents of a variable
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jitc_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JITC_EXPORT const char *jitc_var_str(uint32_t index);

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
extern JITC_EXPORT void jitc_var_read(uint32_t index, uint32_t offset,
                                      void *dst);

/**
 * \brief Copy 'dst' to a single element of a variable
 *
 * This function implements the reverse of jitc_var_read(). This function is
 * convenient for testing, and to change localized entries of an array, but it
 * should never be used to access the complete contents of an array due to its
 * low performance (every write will be performed via an individual
 * asynchronous transaction).
 */
extern JITC_EXPORT void jitc_var_write(uint32_t index, uint32_t offset,
                                       const void *src);

// ====================================================================
//                 Kernel compilation and evaluation
// ====================================================================

/// Schedule a variable \c index for future evaluation via \ref jitc_eval()
extern JITC_EXPORT void jitc_var_schedule(uint32_t index);

/// Evaluate the variable \c index right away, if it is unevaluated/dirty.
extern JITC_EXPORT void jitc_var_eval(uint32_t index);

/// Evaluate all computation that is scheduled on the current stream
extern JITC_EXPORT void jitc_eval();

// ====================================================================
//  Assortment of tuned kernels for initialization, reductions, etc.
// ====================================================================

#if defined(__cplusplus)
/// Potential reduction operations for \ref jitc_reduce
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
extern JITC_EXPORT void jitc_memset_async(void *ptr, uint32_t size,
                                          uint32_t isize, const void *src);

/// Perform a synchronous copy operation
extern JITC_EXPORT void jitc_memcpy(void *dst, const void *src, size_t size);

/// Perform an asynchronous copy operation
extern JITC_EXPORT void jitc_memcpy_async(void *dst, const void *src,
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
extern JITC_EXPORT void jitc_reduce(enum VarType type, enum ReductionType rtype,
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
 * memory obtained using \ref jitc_malloc(), which internally rounds
 * allocations to the next largest power of two and enforces a 64 byte minimum
 * allocation size.
 *
 * Runs asynchronously.
 */
extern JITC_EXPORT void jitc_scan_u32(const uint32_t *in, uint32_t size,
                                      uint32_t *out);

/**
 * \brief Compress a mask into a list of nonzero indices
 *
 * This function takes an 8-bit mask array \c in with size \c size as input,
 * whose entries are required to equal either zero or one. It then writes the
 * indices of nonzero entries to \c out (in increasing order), and it
 * furthermore stores the total number of nonzero mask entries in
 * <tt>*count_out</tt>. The parameter \c out should point to a conservatively
 * sized device-allocated array.
 *
 * The internals resemble \ref jitc_scan_u32(), and the CUDA implementation may
 * similarly access regions beyond the end of \c in and \c out.
 *
 * Runs asynchronously. Accessing \c count_out on the host (e.g. when using the
 * LLVM+TBB backend, or when the address points to host-pinned memory allocated
 * via CUDA) requires prior synchronization.
 */
extern JITC_EXPORT void jitc_compress(const uint8_t *in, uint32_t size,
                                      uint32_t *out, uint32_t *count_out);

/**
 * \brief Reduce an array of boolean values to a single value (AND case)
 *
 * When \c size is not a multiple of 4, the implementation will initialize up
 * to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
 * reduction algorithm can be used. This is fine for allocations made using
 * \ref jitc_malloc(), which allow for this.
 *
 * Runs synchronously.
 */
extern JITC_EXPORT uint8_t jitc_all(uint8_t *values, uint32_t size);

/**
 * \brief Reduce an array of boolean values to a single value (OR case)
 *
 * When \c size is not a multiple of 4, the implementation will initialize up
 * to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
 * reduction algorithm can be used. This is fine for allocations made using
 * \ref jitc_malloc(), which allow for this.
 *
 * Runs synchronously.
 */
extern JITC_EXPORT uint8_t jitc_any(uint8_t *values, uint32_t size);


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
extern JITC_EXPORT uint32_t jitc_mkperm(const uint32_t *values, uint32_t size,
                                        uint32_t bucket_count, uint32_t *perm,
                                        uint32_t *offsets);

/// Helper data structure for vector method calls, see \ref jitc_vcall()
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
 * pointers that have previously been registered using the function \ref
 * jitc_registry_put() using the domain \c domain. It then invokes \ref
 * jit_mkperm() to compute a permutation that reorders the array into coherent
 * buckets. The buckets are returned using an array of type \ref VCallBucket,
 * which contains both the resolved pointer address (obtained via \ref
 * jitc_registry_get_ptr()) and the variable index of an unsigned 32 bit array
 * containing the corresponding entries of the input array. The total number of
 * buckets is returned via the \c bucket_count_out argument.
 *
 * Note: zero-valued array entries are allowed and will be excised by the the
 * returned pertmutation.
 */
extern JITC_EXPORT struct VCallBucket *
jitc_vcall(const char *domain, uint32_t index, uint32_t *bucket_count_out);

/**
 * \brief Replicate individual input elements across larger blocks
 *
 * This function copies each element of the input array \c to a contiguous
 * block of size \c block_size in the output array \c out. For example, <tt>a,
 * b, c</tt> turns into <tt>a, a, b, b, c, c</tt> when the \c block_size is set
 * to \c 2. The input array must contain <tt>size</tt> elements, and the output
 * array must have space for <tt>size * block_size</tt> elements.
 */
extern JITC_EXPORT void jitc_block_copy(enum VarType type, const void *in,
                                        void *out, uint32_t size,
                                        uint32_t block_size);

/**
 * \brief Sum over elements within blocks
 *
 * This function adds all elements of contiguous blocks of size \c block_size
 * in the input array \c in and writes them to \c out. For example, <tt>a, b,
 * c, d, e, f</tt> turns into <tt>a+b, c+d, e+f</tt> when the \c block_size is
 * set to \c 2. The input array must contain <tt>size * block_size</tt> elements,
 * and the output array must have space for <tt>size</tt> elements.
 */
extern JITC_EXPORT void jitc_block_sum(enum VarType type, const void *in,
                                       void *out, uint32_t size,
                                       uint32_t block_size);


#define ENOKI_USE_ALLOCATOR(Type)                                              \
    void *operator new(size_t size) { return jitc_malloc(Type, size); }        \
    void *operator new(size_t size, std::align_val_t) {                        \
        return jitc_malloc(Type, size);                                        \
    }                                                                          \
    void *operator new[](size_t size) { return jitc_malloc(Type, size); }      \
    void *operator new[](size_t size, std::align_val_t) {                      \
        return jitc_malloc(Type, size);                                        \
    }                                                                          \
    void operator delete(void *ptr) { jitc_free(ptr); }                        \
    void operator delete(void *ptr, std::align_val_t) { jitc_free(ptr); }      \
    void operator delete[](void *ptr) { jitc_free(ptr); }                      \
    void operator delete[](void *ptr, std::align_val_t) { jitc_free(ptr); }

#if defined(__cplusplus)
}
#endif
