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
#define JITC_ENUM_CLASS class
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
 */
extern JITC_EXPORT void jitc_init();

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
 * It is safe to call jitc_* API functions following \ref jitc_init_async(),
 * since it acquires a lock to the internal data structures.
 */
extern JITC_EXPORT void jitc_init_async();

/// Release all resources used by the JIT compiler, and report reference leaks.
extern JITC_EXPORT void jitc_shutdown();

/**
 * \brief Set the minimum log level for messages
 *
 * Log output will be written to \c stderr. The following log levels are
 * available:
 * <ul>
 *   <li>0: error</li>
 *   <li>1: warning</li>
 *   <li>2: info</li>
 *   <li>3: debug</li>
 *   <li>4: trace</li>
 * <ul>
 */
extern JITC_EXPORT void jitc_set_log_level(uint32_t log_level);

/// Return the log level for messages
extern JITC_EXPORT uint32_t jitc_get_log_level();

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
 * \brief Wait for all computation on the current stream to finish
 *
 * No-op when the target device is the host CPU.
 */
extern JITC_EXPORT void jitc_stream_sync();

/**
 * \brief Wait for all computation on the current device to finish
 *
 * No-op when the target device is the host CPU.
 */
extern JITC_EXPORT void jitc_device_sync();

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
 * \brief Release the given pointer
 *
 * When \c ptr is a GPU-accessible pointer (\ref AllocType::Device,
 * \ref AllocType::HostPinned, \ref AllocType::Managed,
 * \ref AllocType::ManagedReadMostly), the memory is potentially still being used
 * by a running kernel and merely scheduled to be reclaimed once this
 * program finishes.
 *
 * For this reason, it is crucial that \ref jitc_free() is executed in the
 * right context chosen via \ref jitc_device_set().
 */
extern JITC_EXPORT void jitc_free(void *ptr);

/**
 * \brief Asynchronously change the flavor of an allocated memory region and
 * return the new pointer
 *
 * The operation is asynchronous and, hence, will need to be followed by \ref
 * jitc_stream_sync() if managed memory is subsequently accessed on the CPU.
 *
 * When both source & target are of type \ref AllocType::Device, and if the
 * current device (\ref jitc_device_set()) does not match the device associated
 * with the allocation, a peer-to-peer migration is performed.
 *
 * Note: Migrations involving AllocType::Host are currently not supported.
 */
extern JITC_EXPORT void* jitc_malloc_migrate(void *ptr, enum AllocType type);

/// Release all unused memory to the GPU / OS
extern JITC_EXPORT void jitc_malloc_trim();

// ====================================================================
//                        Variable management
// ====================================================================

#if defined(__cplusplus)
/// Variable types supported by the JIT compiler
enum class VarType : uint32_t {
    Invalid, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
    Float16, Float32, Float64, Bool, Pointer
};
#else
enum VarType {
    VarTypeInvalid, VarTypeInt8, VarTypeUInt8, VarTypeInt16, VarTypeUInt16,
    VarTypeInt32, VarTypeUInt32, VarTypeInt64, VarTypeUInt64, VarTypeFloat16,
    VarTypeFloat32, VarTypeFloat64, VarTypeBool, VarTypePointer
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
 * operands.
 *
 * \param type
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param stmt
 *    Intermediate language statement.
 */
extern JITC_EXPORT uint32_t jitc_trace_append_0(enum VarType type,
                                                const char *stmt);

/// Append a variable to the instruction trace (1 operand)
extern JITC_EXPORT uint32_t jitc_trace_append_1(enum VarType type,
                                                const char *stmt,
                                                uint32_t arg1);

/// Append a variable to the instruction trace (2 operands)
extern JITC_EXPORT uint32_t jitc_trace_append_2(enum VarType type,
                                                const char *stmt,
                                                uint32_t arg1,
                                                uint32_t arg2);

/// Append a variable to the instruction trace (3 operands)
extern JITC_EXPORT uint32_t jitc_trace_append_3(enum VarType type,
                                                const char *stmt,
                                                uint32_t arg1,
                                                uint32_t arg2,
                                                uint32_t arg3);

/// Increase the internal reference count of a given variable
extern JITC_EXPORT void jitc_inc_ref_int(uint32_t index);

/// Decrease the internal reference count of a given variable
extern JITC_EXPORT void jitc_dec_ref_int(uint32_t index);

/// Increase the external reference count of a given variable
extern JITC_EXPORT void jitc_inc_ref_ext(uint32_t index);

/// Decrease the external reference count of a given variable
extern JITC_EXPORT void jitc_dec_ref_ext(uint32_t index);

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
extern JITC_EXPORT void jitc_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern JITC_EXPORT const char *jitc_var_label(uint32_t index);

/**
 * \brief Asynchronously migrate a variable to a different flavor of memory
 *
 * The operation is asynchronous and, hence, will need to be followed by \ref
 * jitc_stream_sync() if managed memory is subsequently accessed on the CPU.
 *
 * When both source & target are of type \ref AllocType::Device, and if the
 * current device (\ref jitc_device_set()) does not match the device associated
 * with the allocation, a peer-to-peer migration is performed.
 *
 * Note: Migrations involving AllocType::Host are currently not supported.
 */
extern JITC_EXPORT void jitc_var_migrate(uint32_t idx, enum AllocType type);

/// Indicate that evaluation of the given variable causes side effects
extern JITC_EXPORT void jitc_var_mark_side_effect(uint32_t index);

/// Mark variable as dirty, e.g. because of pending scatter operations
extern JITC_EXPORT void jitc_var_mark_dirty(uint32_t index);

/// Return a human-readable summary of registered variables
extern JITC_EXPORT const char *jitc_whos();

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
extern JITC_EXPORT void jitc_set_parallel_dispatch(int enable);

/// Return whether or not parallel dispatch is enabled. Returns \c 0 or \c 1.
extern JITC_EXPORT int jitc_parallel_dispatch();

/// Evaluate all computation that is queued on the current stream
extern JITC_EXPORT void jitc_eval();

/// Call jitc_eval() only if the variable 'index' requires evaluation
extern JITC_EXPORT void jitc_eval_var(uint32_t index);

#if defined(__cplusplus)
}
#endif
