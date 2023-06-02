#pragma once

#include <drjit-core/jit.h>
#include <vector>
#include "core.h"

#if defined(_MSC_VER)
#  define NB_TLS __declspec(thread)
#else
#  define NB_TLS __thread
#endif


/// Represents a single stream of a parallel communication
class ThreadState {
public:
    // ================== Abstract interface ==================

    /// Virtual destructor
    virtual ~ThreadState() = default;

    /// Allocate private or shared memory accessible on host+device
    virtual void* malloc(size_t size, bool shared) = 0;

    /// Associate the CPU memory address associated with an allocation
    virtual void *host_ptr(void *ptr) = 0;

    /**
     * Return an opaque set of flags characterizing the kernel being compiled
     * based on backend compilation settings. This isolates the kernel from
     * other variants of the same code compiled with *different* flags.
     */
    virtual uint64_t kernel_flags() const { return 0; }

    /**
     * Compile the kernel source (buf, buf_size) and with entry point 'name'
     * and store the result in 'kernel'.
     *
     * The function returns `true` if compilation could be avoided by means of
     * a secondary caching mechanism (e.g., OptiX)
     */
    virtual bool compile(const char *buf, size_t buf_size, const char *name,
                         Kernel &kernel) = 0;

    /// Enqueue a memory copy operation
    virtual void enqueue_memcpy(void *dst, const void *src, size_t size) = 0;

    /// Enqueue a host callback function
    virtual void enqueue_callback(void (*fn)(void *), void *payload) = 0;

    /// Wait for queued computation to finish
    virtual void sync() = 0;

public:
    // ================== Public members ==================

    /// Backend type
    JitBackend backend;

    /**
     * \brief Index into a corresponding entry of the `state.devices` list
     *
     * Don't confuse this ID with the CUDA device ID, which is stored
     * in the \ref Device data structure.
     *
     * For backends that don't support multiple devices (e.g., LLVM), this
     * value is set to -1.
     */
    int device = -1;

    /**
     * List of variables that are scheduled for evaluation (via
     * jitc_var_schedule()) that will take place at the next call to jitc_eval().
     */
    std::vector<uint32_t> scheduled;

    /**
     * List of special variables of type VarType::Void, whose evaluation will
     * cause side effects that modify other variables. They will be evaluated
     * at the next call to jitc_eval().
     */
    std::vector<uint32_t> side_effects;

    /// When recording loops or virtual function calls, side effects go here.
    std::vector<uint32_t> side_effects_recorded;

    /**
     * Stack of variable indices indicating the list of active SIMD lanes.
     * This is used to constrain the behavior of gather/scatter operations.
     */
    std::vector<uint32_t> mask_stack;

    /// Stack of variable name prefixes, mainly useful for GraphViz exports
    std::vector<char *> prefix_stack;

    /// Combined version of the elements of 'prefix_stack'
    char *prefix = nullptr;

    /// Identifier associated with the current basic block
    uint32_t scope = 0;

    /// Registry index of the 'self' pointer of the vcall being recorded
    uint32_t vcall_self_value = 0;

    /// .. and the JIT variable that it will be mapped to
    uint32_t vcall_self_index = 0;

};

/// State specific to threads
#if defined(DRJIT_ENABLE_LLVM)
extern NB_TLS ThreadState* thread_state_llvm;
#endif

#if defined(DRJIT_ENABLE_CUDA)
extern NB_TLS ThreadState* thread_state_cuda;
#endif

#if defined(DRJIT_ENABLE_METAL)
extern NB_TLS ThreadState* thread_state_metal;
#endif

extern ThreadState *jitc_init_thread_state(JitBackend backend);

inline ThreadState *thread_state(JitBackend backend) {
    ThreadState *result;

    switch (backend) {
#if defined(DRJIT_ENABLE_LLVM)
        case JitBackend::LLVM:
            result = thread_state_llvm;
            break;
#endif

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

        default:
            result = nullptr;
    }

    if (unlikely(!result))
        result = jitc_init_thread_state(backend);

    return result;
}

inline ThreadState *thread_state(uint32_t backend) {
    return thread_state((JitBackend) backend);
}

extern void jitc_set_flags(uint32_t flags);

extern uint32_t jitc_flags();
