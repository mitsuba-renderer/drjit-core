/*
    enoki/jit.h -- Self-contained JIT compiler for CUDA & LLVM.

    This library implements a self-contained tracing JIT compiler that supports
    both CUDA PTX and LLVM IR as intermediate languages. It takes care of many
    tricky aspects, such as asynchronous memory allocation and release,
    multi-device computation, kernel caching and reuse, common subexpression
    elimination, etc.

    While the library is internally implemented using C++17, this header file
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

#define JITC_EXPORT __attribute__ ((visibility("default")))

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
 * \brief Launch an ansynchronous thread that will execute jit_init() and
 * return immediately
 *
 * On machines with several GPUs, \ref jit_init() sets up a CUDA environment on
 * all devices, which can be rather low (e.g. 1 second). This function
 * provides a convenient alternative to hide this latency, for instance when
 * importing this library from an interactive Python session which doesn't
 * actually need the JIT right away.
 *
 * The \c llvm and \c cuda arguments should be set to \c 1 to initialize the
 * corresponding backend, and \c 0 otherwise.
 *
 * It is safe to call jitc_* API functions following \ref jitc_init_async(),
 * since it acquires a lock to the internal data structures.
 */
extern JITC_EXPORT void jitc_init_async(int llvm, int cuda);

/// Check whether the LLVM backend was successfully initialized
extern JITC_EXPORT int jitc_has_llvm();

/// Check whether the CUDA backend was successfully initialized
extern JITC_EXPORT int jitc_has_cuda();

/// Release all resources used by the JIT compiler, and report reference leaks.
extern JITC_EXPORT void jitc_shutdown();

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
 *     CUDA devices can concurrently execute computation from multiple streams.
 *     When accessing the JIT compiler in a multi-threaded program, each thread
 *     should specify a separate stream to exploit this additional opportunity
 *     for parallelization. When executing on the host CPU
 *     (<tt>device==-1</tt>), this argument is ignored.
 */
extern JITC_EXPORT void jitc_device_set(int32_t device, uint32_t stream);

/**
 * \brief Dispatch computation to multiple parallel streams?
 *
 * The JIT compiler attempts to fuse all queued computation into a single
 * kernel to maximize efficiency. But computation involving arrays of different
 * size must necessarily run in separate kernels, which means that it is
 * serialized if taking place within the same device and stream. If desired,
 * jitc_eval() can detect this and dispatch multiple kernels to separate
 * streams that execute in parallel. The default is \c 1 (i.e. to enable
 * parallel dispatch).
 */
extern JITC_EXPORT void jitc_parallel_dispatch_set(int enable);

/// Return whether or not parallel dispatch is enabled. Returns \c 0 or \c 1.
extern JITC_EXPORT int jitc_parallel_dispatch();

/**
 * \brief Wait for all computation on the current stream to finish
 *
 * No-op when the target device is the host CPU.
 */
extern JITC_EXPORT void jitc_sync_stream();

/**
 * \brief Wait for all computation on the current device to finish
 *
 * No-op when the target device is the host CPU.
 */
extern JITC_EXPORT void jitc_sync_device();

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
 * via a callback in \ref jitc_log_callback_set(). Both destinations can also
 * be enabled simultaneously, pontentially using different log levels.
 */
extern JITC_EXPORT void jitc_log_stderr_set(enum LogLevel level);

/// Return the currently set minimum log level for output to \c stderr
extern JITC_EXPORT enum LogLevel jitc_log_stderr();


/**
 * \brief Control the destination of log messages (callback)
 *
 * This function can be used to specify an optional callback that will be
 * invoked with the contents of library log messages, whose severity matches or
 * exceeds the specified \c level.
 */
typedef void (*LogCallback)(LogLevel, const char *);
extern JITC_EXPORT void jitc_log_callback_set(LogLevel level, LogCallback callback);

/// Return the currently set minimum log level for output to a callback
extern JITC_EXPORT enum LogLevel jitc_log_callback();

/// Print a log message with the specified log level and message
extern JITC_EXPORT void jitc_log(LogLevel level, const char* fmt, ...);

/// Raise an exception message with the specified message
extern JITC_EXPORT void jitc_raise(const char* fmt, ...);

/// Terminate the application due to a non-recoverable error
extern JITC_EXPORT void jitc_fail(const char* fmt, ...);

// ====================================================================
//                         Memory allocation
// ====================================================================

#if defined(__cplusplus)
enum class AllocType : uint32_t {
    /// Memory that is located on the host (i.e., the CPU)
    Host,

    /**
     * Memory on the host that is "pinned" and thus cannot be paged out.
     * Host-pinned memory is accessible (albeit slowly) from CUDA-capable GPUs
     * as part of the unified memory model, and it also can be a source or
     * destination of asynchronous host <-> device memcpy operations.
     */
    HostPinned,

    /// Memory that is located on a device (i.e., one of potentially several GPUs)
    Device,

    /// Memory that is mapped in the address space of both host & all GPU devices
    Managed,

    /// Like \c Managed, but more efficient when almost all accesses are reads
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
    __attribute__((malloc));

/**
 * \brief Release a given pointer asynchronously
 *
 * For CPU-only arrays (\ref AllocType::Host), <tt>jitc_free()</tt> is
 * synchronous and very similar to <tt>free()</tt>, except that the released
 * memory is placed in Enoki's internal allocation cache instead of being
 * returned to the OS. The function \ref jitc_malloc_trim() can optionally be
 * called to also clear this cache.
 *
 * When \c ptr is a GPU-accessible pointer (\ref AllocType::Device, \ref
 * AllocType::HostPinned, \ref AllocType::Managed, \ref
 * AllocType::ManagedReadMostly), the associated memory region is quite likely
 * still being used by a running kernel, and it is therefore merely *scheduled*
 * to be reclaimed once this kernel finishes. Allocation thus runs in the
 * execution context of a CUDA device, i.e., it is asynchronous with respect to
 * the CPU. This means that some care must be taken in the context of programs
 * that use multiple streams or GPUs: it is not permissible to e.g. allocate
 * memory in one context, launch a kernel using it, then immediately switch
 * context to another GPU or stream on the same GPU via \ref jitc_set_device()
 * and release the memory region there. Calling \ref jitc_sync_stream() or
 * \ref jitc_sync_device() before context switching defuses this situation.
 */
extern JITC_EXPORT void jitc_free(void *ptr);

/** e
 * \brief Asynchronously change the flavor of an allocated memory region and
 * return the new pointer
 *
 * The operation is asynchronous and, hence, will need to be followed by \ref
 * jitc_sync_stream() if managed memory is subsequently accessed on the CPU.
 * Nothing needs to be done in the other direction, e.g. when migrating from
 * host-pinned to device or managed memory.
 *
 * When both source & target are of type \ref AllocType::Device, and if the
 * current device (\ref jitc_device_set()) does not match the device associated
 * with the allocation, a peer-to-peer migration is performed.
 *
 * Note: Migrations involving AllocType::Host are currently not supported.
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

/**
 * \brief Query the unique ID associated with an allocation
 *
 * The allocator assigns a unique ID to each pointer allocated via
 * \ref jitc_malloc(). This function queries this mapping for a given
 * pointer value, and \ref jit_malloc_from_id() goes the other way.
 *
 * Returns \c 0 when the pointer could not be found. Valid IDs are always
 * nonzero.
 */
extern JITC_EXPORT uint32_t jitc_malloc_to_id(void *ptr);

/**
 * \brief Query the allocation associated with a unique ID
 *
 * The allocator assigns a unique ID to each pointer allocated via
 * \ref jitc_malloc(). This function queries this mapping for a given
 * ID value, and \ref jit_malloc_to_id() goes the other way.
 *
 * Returns \c nullptr when the ID could not be found.
 */
extern JITC_EXPORT void *jitc_malloc_from_id(uint32_t id);

// ====================================================================
//                        Variable management
// ====================================================================

#if defined(__cplusplus)
/// Variable types supported by the JIT compiler
enum class VarType : uint32_t {
    Invalid, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
    Float16, Float32, Float64, Bool, Pointer, Count
};
#else
enum VarType {
    VarTypeInvalid, VarTypeInt8, VarTypeUInt8, VarTypeInt16, VarTypeUInt16,
    VarTypeInt32, VarTypeUInt32, VarTypeInt64, VarTypeUInt64, VarTypeFloat16,
    VarTypeFloat32, VarTypeFloat64, VarTypeBool, VarTypePointer, VarTypeCount
};
#endif

/**
 * Copy a memory region from the host to onto the device and return its
 * variable index. Its external reference count is initialized to \c 1.
 *
 * \param type
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param ptr
 *    Point of the memory region
 *
 * \param size
 *    Number of elements, rather than the size in bytes
 *
 * \param free
 *    If free != 0, the JIT compiler will free the memory region via
 *    \ref jitc_free() once it goes out of scope.
 */
extern JITC_EXPORT uint32_t jitc_var_copy_to_device(enum VarType type,
                                                    const void *ptr,
                                                    size_t size);

/**
 * Register an existing memory region as a variable in the JIT compiler, and
 * return its index. Its external reference count is initialized to \c 1.
 *
 * \param type
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param ptr
 *    Point of the memory region
 *
 * \param size
 *    Number of elements, rather than the size in bytes
 *
 * \param free
 *    If free != 0, the JIT compiler will free the memory region via
 *    \ref jitc_free() once it goes out of scope.
 */
extern JITC_EXPORT uint32_t jitc_var_register(enum VarType type,
                                              void *ptr,
                                              size_t size,
                                              int free);

/**
 * Register a pointer literal as a variable within the JIT compiler
 *
 * When working with memory (gathers, scatters) using the JIT compiler, we must
 * often refer to memory addresses. These addresses should not bakied nto the
 * JIT-compiled code, since they change over time, which limits the ability to
 * re-use compiled kernels.
 *
 * This function registers a pointer literal that accomplishes this. It is
 * functionally equivalent to
 *
 * \code
 * void *my_ptr = ...;
 * uint32_t index = jitc_var_copy_to_device(VarType::Pointer, &my_ptr, 1);
 * \endcode
 *
 * but results in more efficient generated code.
 */
extern JITC_EXPORT uint32_t jitc_var_register_ptr(const void *ptr);

/**
 * \brief Append a statement to the instruction trace.
 *
 * This function takes a statement in an intermediate language (CUDA PTX or
 * LLVM IR) and appends it to the list of currently queued operations. It
 * returns the index of the variable that will store the result of the
 * statement, whose external reference count is initialized to \c 1.
 *
 * This function assumes that the operation does not access any operands. See
 * the other <tt>jit_trace_*</tt> functions for IR statements with 1 to 3
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
 * uint32_t result = jit_trace_append_2(VarType::Int32,
 *                                      "add.$t0 $r0, $r1, $r2",
 *                                      1, op1, op2);
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
 */
extern JITC_EXPORT uint32_t jitc_trace_append_0(enum VarType type,
                                                const char *stmt,
                                                int stmt_static);

/// Append a variable to the instruction trace (1 operand)
extern JITC_EXPORT uint32_t jitc_trace_append_1(enum VarType type,
                                                const char *stmt,
                                                int stmt_static,
                                                uint32_t op1);

/// Append a variable to the instruction trace (2 operands)
extern JITC_EXPORT uint32_t jitc_trace_append_2(enum VarType type,
                                                const char *stmt,
                                                int stmt_static,
                                                uint32_t op1,
                                                uint32_t op2);

/// Append a variable to the instruction trace (3 operands)
extern JITC_EXPORT uint32_t jitc_trace_append_3(enum VarType type,
                                                const char *stmt,
                                                int stmt_static,
                                                uint32_t op1,
                                                uint32_t op2,
                                                uint32_t op3);

/// Increase the external reference count of a given variable
extern JITC_EXPORT void jitc_var_ext_ref_inc(uint32_t index);

/// Decrease the external reference count of a given variable
extern JITC_EXPORT void jitc_var_ext_ref_dec(uint32_t index);

/// Query the pointer variable associated with a given variable
extern JITC_EXPORT void *jitc_var_ptr(uint32_t index);

/// Query the size of a given variable
extern JITC_EXPORT size_t jitc_var_size(uint32_t index);

/**
 * Set the size of a given variable (if possible, otherwise throw an
 * exception.)
 *
 * \param index
 *     Index of the variable, whose size should be modified
 *
 * \param size
 *     Target size value
 *
 * \param copy
 *     When the variable has already been evaluated and is a scalar, Enoki can
 *     optionally perform a copy instead of failing if <tt>copy != 0</tt>.
 *
 * Returns the ID of the changed or new variable
 */
extern JITC_EXPORT uint32_t jitc_var_set_size(uint32_t index,
                                              size_t size,
                                              int copy);

/// Assign a descriptive label to a given variable
extern JITC_EXPORT void jitc_var_label_set(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern JITC_EXPORT const char *jitc_var_label(uint32_t index);

/**
 * \brief Asynchronously migrate a variable to a different flavor of memory
 *
 * The operation is asynchronous and, hence, will need to be followed by \ref
 * jitc_sync_stream() if managed memory is subsequently accessed on the CPU.
 *
 * When both source & target are of type \ref AllocType::Device, and if the
 * current device (\ref jitc_device_set()) does not match the device associated
 * with the allocation, a peer-to-peer migration is performed.
 *
 * Note: Migrations involving AllocType::Host are currently not supported.
 */
extern JITC_EXPORT void jitc_var_migrate(uint32_t index, enum AllocType type);

/// Indicate that evaluation of the given variable causes side effects
extern JITC_EXPORT void jitc_var_mark_side_effect(uint32_t index);

/// Mark variable as dirty, e.g. because of pending scatter operations
extern JITC_EXPORT void jitc_var_mark_dirty(uint32_t index);

/**
 * \brief Return a human-readable summary of registered variables
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jitc_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JITC_EXPORT const char *jitc_var_whos();

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
 * never be used to extract complete array contents due to its low performance.
 * This operation fully synchronizes the host CPU & device.
 */
extern JITC_EXPORT void jitc_var_read(uint32_t index, size_t offset,
                                      void *dst);

/**
 * \brief Copy 'dst' to a single element of a variable
 *
 * This function implements the reverse of jit_var_read(). This function is
 * convenient to change localized entries of an array, but it should never be
 * used to extract complete array contents due to its low performance.
 */
extern JITC_EXPORT void jitc_var_write(uint32_t index, size_t offset,
                                       const void *src);

// ====================================================================
//                 Kernel compilation and evaluation
// ====================================================================

/// Evaluate all computation that is queued on the current stream
extern JITC_EXPORT void jitc_eval();

/// Call jitc_eval() only if the variable 'index' requires evaluation
extern JITC_EXPORT void jitc_var_eval(uint32_t index);

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
 * This function writes \c size values of type \c type to the output array \c
 * ptr. The specific value is taken from \c src, which must be a CPU pointer to
 * a single int, float, double, etc (depending on \c type).
 */
extern JITC_EXPORT void jitc_fill(enum VarType type, void *ptr, size_t size,
                                  const void *src);

/**
 * \brief Reduce the given array to a single value
 *
 * This operation reads \c size values of type \type from the input array \c
 * ptr and performs an specified operation (e.g., addition, multplication,
 * etc.) to combine them into a single value that is written to the device
 * variable \c out.
 */
extern JITC_EXPORT void jitc_reduce(enum VarType type, enum ReductionType rtype,
                                    const void *ptr, size_t size, void *out);

#if defined(__cplusplus)
}
#endif
