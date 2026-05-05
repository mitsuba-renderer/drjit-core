// metal_device_lib.metal -- MSL helper library prepended to every
// Dr.Jit-generated kernel.
//
// This mirrors the role of LuisaCompute's ``metal_device_lib.metal``: it
// declares the small set of helper types, macros, and atomic shims used
// throughout generated kernels. Currently the file is intentionally tiny;
// Phase-2 work will fold in additional helpers (math, ray-tracing utilities,
// type wrappers, etc.) as needed.

#pragma once

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>

using namespace metal;

// ---------------------------------------------------------------------
// Buffer wrapper used to hand pointers + sizes around in a type-safe way.
// We don't rely on ``device pointer + size`` tuples in the generated
// kernels because Dr.Jit always knows the launch size up front. This
// type is provided primarily for Phase-3 utility kernels.
// ---------------------------------------------------------------------
template <typename T>
struct LCBuffer {
    device T *data;
    uint size;
};

// ---------------------------------------------------------------------
// Atomic helpers
//
// Metal's atomic API uses ``atomic<T>`` wrappers which are typically not
// in scope when we cast a plain ``device T*`` pointer. The helpers below
// provide a uniform interface that works for the integer types we care
// about. Float atomic-add is available on Metal 3.0 (Apple7+).
// ---------------------------------------------------------------------

inline int32_t atomic_add_i32(device int32_t *addr, int32_t value) {
    device atomic_int *a = (device atomic_int *) addr;
    return atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}

inline uint32_t atomic_add_u32(device uint32_t *addr, uint32_t value) {
    device atomic_uint *a = (device atomic_uint *) addr;
    return atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 300
inline float atomic_add_f32(device float *addr, float value) {
    device atomic_float *a = (device atomic_float *) addr;
    return atomic_fetch_add_explicit(a, value, memory_order_relaxed);
}
#endif

// ---------------------------------------------------------------------
// 64-bit atomic add via CAS loop -- Metal does not provide this natively.
// Used by the Phase-3 utility kernels for Int64/UInt64 reductions.
// ---------------------------------------------------------------------
inline ulong atomic_add_u64(device ulong *addr, ulong value) {
    device atomic_ulong *a = (device atomic_ulong *) addr;
    ulong cur = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        a, &cur, cur + value, memory_order_relaxed,
        memory_order_relaxed)) {
    }
    return cur;
}
