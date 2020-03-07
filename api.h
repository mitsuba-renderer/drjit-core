#pragma once

#include <stdlib.h>
#include <stdint.h>
#define ENOKI_EXPORT

// ====================================================================
//         Initialization, device enumeration, and management
// ====================================================================

/**
 * \brief Initialize core data structures of the JIT compiler
 *
 * Must be called before using any of the remaining API. The function does
 * nothing when the JIT compiler is already initialized.
 */
extern ENOKI_EXPORT void jitc_init();

/**
 * \brief Launch an ansynchronous thread that will execute jit_init() and
 * return immediately
 *
 * On machines with several GPUs, \ref jit_init() can easily consume a second
 * or so just to initialize CUDA. This function is convenient alternative to
 * reduce latencies, e.g., when importing a Python module that uses the JIT
 * compiler.
 */
extern ENOKI_EXPORT void jitc_init_async();

/// Release all resources used by the JIT compiler, and report reference leaks.
extern ENOKI_EXPORT void jitc_shutdown();

/// Return the log level for messages
extern ENOKI_EXPORT uint32_t jitc_get_log_level();

/// Set the minimum log level for messages (0: error, 1: warning, 2: info, 3: debug, 4: trace)
extern ENOKI_EXPORT void jitc_set_log_level(uint32_t log_level);

/// Return the number of target devices (excluding the "host"/CPU)
extern ENOKI_EXPORT uint32_t jitc_device_count();

/// Set the currently active device & stream
extern ENOKI_EXPORT void jitc_device_set(uint32_t device, uint32_t stream);

/// Wait for all computation on the current stream to finish
extern ENOKI_EXPORT void jitc_stream_sync();

/// Wait for all computation on the current device to finish
extern ENOKI_EXPORT void jitc_device_sync();

// ====================================================================
//                         Memory allocation
// ====================================================================

enum class AllocType {
    /// Memory that is located on the host (i.e., the CPU)
    Host              = 0,

    /**
     * Memory on the host that is "pinned" and thus cannot be paged out.
     * Host-pinned memory is accessible (albeit slowly) from CUDA-capable GPUs
     * as part of the unified memory model, and it also can be a source or
     * destination of asynchronous host <-> device memcpy operations.
     */
    HostPinned        = 1,

    /// Memory that is located on a device (i.e., one of potentially several GPUs)
    Device            = 2,

    /// Memory that is mapped in the address space of both host & all GPU devices
    Managed           = 3,

    /// Like \c Managed, but more efficient when almost all accesses are reads
    ManagedReadMostly = 4,

    /// Number of AllocType entries
    Count             = 5
};


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
extern ENOKI_EXPORT void *jitc_malloc(AllocType type, size_t size)
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
extern ENOKI_EXPORT void jitc_free(void *ptr);

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
extern ENOKI_EXPORT void* jitc_malloc_migrate(void *ptr, AllocType type);

/// Release all unused memory to the GPU / OS
extern ENOKI_EXPORT void jitc_malloc_trim();

// ====================================================================
//                        Variable management
// ====================================================================
//
/// Register an existing variable with the JIT compiler
extern ENOKI_EXPORT uint32_t jitc_var_register(uint32_t type,
                                               void *ptr,
                                               size_t size,
                                               bool free);

/// Register pointer literal as a special variable within the JIT compiler
extern ENOKI_EXPORT uint32_t jitc_var_register_ptr(const void *ptr);

/// Copy a memory region onto the device and return its variable index
extern ENOKI_EXPORT uint32_t jitc_var_copy_to_device(uint32_t type,
                                                     const void *value,
                                                     size_t size);

/// Append a variable to the instruction trace (no operand)
extern ENOKI_EXPORT uint32_t jitc_trace_append(uint32_t type,
                                               const char *cmd);

/// Append a variable to the instruction trace (1 operand)
extern ENOKI_EXPORT uint32_t jitc_trace_append(uint32_t type,
                                               const char *cmd,
                                               uint32_t arg1);

/// Append a variable to the instruction trace (2 operands)
extern ENOKI_EXPORT uint32_t jitc_trace_append(uint32_t type,
                                               const char *cmd,
                                               uint32_t arg1,
                                               uint32_t arg2);

/// Append a variable to the instruction trace (3 operands)
extern ENOKI_EXPORT uint32_t jitc_trace_append(uint32_t type,
                                               const char *cmd,
                                               uint32_t arg1,
                                               uint32_t arg2,
                                               uint32_t arg3);

/// Increase the internal reference count of a given variable
extern ENOKI_EXPORT void jitc_inc_ref_int(uint32_t index);

/// Decrease the internal reference count of a given variable
extern ENOKI_EXPORT void jitc_dec_ref_int(uint32_t index);

/// Increase the external reference count of a given variable
extern ENOKI_EXPORT void jitc_inc_ref_ext(uint32_t index);

/// Decrease the external reference count of a given variable
extern ENOKI_EXPORT void jitc_dec_ref_ext(uint32_t index);

/// Query the pointer variable associated with a given variable
extern ENOKI_EXPORT void *jitc_var_ptr(uint32_t index);

/// Query the size of a given variable
extern ENOKI_EXPORT size_t jitc_var_size(uint32_t index);

/**
 * Set the size of a given variable (if possible, otherwise throw)
 *
 * In case the variable has already been evaluated and is a scalar, Enoki
 * can optionally perform a copy instead of failing if copy=true is specified.
 *
 * Returns the ID of the changed or new variable
 */
extern ENOKI_EXPORT uint32_t jitc_var_set_size(uint32_t index, size_t size,
                                               bool copy = false);

/// Assign a descriptive label to a given variable
extern ENOKI_EXPORT void jitc_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern ENOKI_EXPORT const char *jitc_var_label(uint32_t index);

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
extern ENOKI_EXPORT void jitc_var_migrate(uint32_t idx, AllocType type);

/// Indicate that evaluation of the given variable causes side effects
extern ENOKI_EXPORT void jitc_var_mark_side_effect(uint32_t index);

/// Mark variable as dirty, e.g. because of pending scatter operations
extern ENOKI_EXPORT void jitc_var_mark_dirty(uint32_t index);

/// Return a human-readable summary of registered variables
extern ENOKI_EXPORT const char *jitc_whos();

// Evaluate currently all queued operations
extern ENOKI_EXPORT void jitc_eval();
