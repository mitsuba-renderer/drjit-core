/*
    src/llvm_gemm.h -- Tiled CPU GEMM kernels for the LLVM backend

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include <drjit-core/half.h>
#include <type_traits>

/*
  Tiled CPU GEMM for row-major matrices
  =====================================

  Computes ``C = op_A(A) @ op_B(B)`` of shape ``(M, N)``, with
  optional transposes selected by ``At`` / ``Bt``.

  BLIS / GotoBLAS [1, 2] vocabulary:

   - **Microtile** (``MR x NR``): output block one microkernel call
     produces. Its ``MR * NR`` accumulators live in vector registers
     across the full K reduction, so each microtile touches ``C``
     only once. ``NR`` spans one SIMD vector; ``MR = 6`` leaves
     register room for the K-loop operands.

   - **Packing**: copy a strip of ``A`` or ``B`` into a contiguous
     scratch buffer laid out in microkernel order. Packing removes
     strided source accesses and absorbs transposes. Trailing partial
     strips pad with zeros so the microkernel runs unchanged on every
     tile.

   - **K segment** (``KC``): depth of one K-reduction chunk. ``KC``
     keeps the packed strips of ``A`` and ``B`` cache-resident across
     the microtile sweep.

  Schedule. Walk ``K`` in ``KC`` segments. Per segment, parallel-pack
  ``B[:KC, :N]`` into a shared ``packed_B`` scratch, then dispatch one
  compute task per ``MR``-row slab in parallel. Each task packs its
  own ``MR x KC`` strip of ``A`` and sweeps every column slab against
  ``packed_B``, writing bulk tiles directly to ``C`` and edge tiles
  through a scratch buffer.

  [1] Goto & van de Geijn, "Anatomy of High-Performance Matrix
      Multiplication", ACM TOMS 34(3), 2008.
  [2] Van Zee & van de Geijn, "BLIS: A Framework for Rapidly
      Instantiating BLAS Functionality", ACM TOMS 41(3), 2015.
*/

template <typename T>
using GemmAcc = std::conditional_t<std::is_same_v<T, drjit::half>, float, T>;

// SIMD register width in bytes. Picks up the widest vector ISA the
// compiler targets; override with -DGEMM_SIMD_BYTES=N for testing.
#if defined(GEMM_SIMD_BYTES)
    // honour the caller's override
#elif defined(__AVX512F__)
    #define GEMM_SIMD_BYTES 64
#elif defined(__AVX2__) || defined(__AVX__)
    #define GEMM_SIMD_BYTES 32
#elif defined(__ARM_NEON) || defined(__SSE2__)
    #define GEMM_SIMD_BYTES 16
#else
    #define GEMM_SIMD_BYTES 16
#endif

// Microtile dimensions. ``NR = simd_width / sizeof(Acc)`` gives one
// SIMD register per row accumulator, so ``MR`` accumulators plus one
// broadcast and one B-load fit in the vector register file (16 ymm
// on AVX2, 32 zmm on AVX-512, 32 v-regs on NEON). ``MR = 6`` is the
// BLIS sweet spot across all three ISAs.
template <typename T> struct GemmTile {
    static constexpr uint32_t MR = 6;
    static constexpr uint32_t NR = GEMM_SIMD_BYTES / sizeof(GemmAcc<T>);
};

/// K-segment depth. Chosen so the per-task ``MR x KC`` pack of ``A``
/// stays L1-resident (``MR * KC * sizeof(Acc)`` = 24 KB for f32) and
/// the pack / compute ratio stays amortised.
static constexpr uint32_t GEMM_KC = 1024;

/// Maximum column-panel width (columns, rounded up to ``NR`` by the
/// launcher). Caps the shared ``packed_B`` scratch at
/// ``KC * GEMM_NC * sizeof(Acc)`` = 16 MiB for f32, which fits a
/// modern CPU's last-level cache. Typical ``N <= GEMM_NC`` runs as a
/// single panel; larger ``N`` splits across panels.
static constexpr uint32_t GEMM_NC = 4096;

// Reduce-dim slice of ``GemmBatch`` that the launcher hands to the
// block kernels. ``n_rdims == 0`` (or ``r_count == 1``) degenerates
// to a single iteration with A/B unchanged.
struct GemmReduce {
    uint32_t r_count;               // prod(extent[..])
    uint32_t n_rdims;               // number of reduce dims
    const uint32_t *extent;         // reduce extents [n_rdims]
    const uint32_t *a_stride;       // per-reduce-dim A strides [n_rdims]
    const uint32_t *b_stride;       // per-reduce-dim B strides [n_rdims]
};

/// Decode a flat reduce index ``rb`` (in ``[0, r.r_count)``) into
/// per-operand byte offsets ``a_off`` / ``b_off``, treating ``rb`` as
/// a mixed-radix number over ``r.extent[0..n_rdims)`` (innermost at
/// ``d = 0``). Each dim contributes ``idx[d] * a_stride[d]`` to
/// ``a_off`` (likewise for ``b_off``). Zero strides encode broadcast.
static inline void gemm_reduce_decode(const GemmReduce &r, uint32_t rb,
                                      size_t &a_off, size_t &b_off) {
    a_off = 0; b_off = 0;
    uint32_t z = rb;
    for (uint32_t d = 0; d < r.n_rdims; ++d) {
        uint32_t idx = z % r.extent[d];
        z /= r.extent[d];
        a_off += (size_t) idx * r.a_stride[d];
        b_off += (size_t) idx * r.b_stride[d];
    }
}

// Pack one MR-wide row strip of A (``kc_len`` K steps wide) into
// ``packed`` with layout ``[k][r]``. The first ``m_valid`` rows carry
// real data; we fill the rest with zeros so the microkernel runs
// unchanged on edge tiles.
template <typename T, bool At>
static inline void pack_a(const T *A, GemmAcc<T> *packed,
                          uint32_t M, uint32_t K,
                          uint32_t i_start, uint32_t kc0, uint32_t kc_len,
                          uint32_t m_valid) {
    using Acc = GemmAcc<T>;
    constexpr uint32_t MR = GemmTile<T>::MR;
    if constexpr (!At) {
        for (uint32_t r = 0; r < MR; ++r) {
            if (r < m_valid) {
                const T *row = A + (size_t) (i_start + r) * K + kc0;
                for (uint32_t k = 0; k < kc_len; ++k)
                    packed[k * MR + r] = (Acc) row[k];
            } else {
                for (uint32_t k = 0; k < kc_len; ++k)
                    packed[k * MR + r] = Acc(0);
            }
        }
    } else {
        for (uint32_t k = 0; k < kc_len; ++k) {
            const T *row = A + (size_t) (kc0 + k) * M + i_start;
            uint32_t r = 0;
            for (; r < m_valid; ++r)
                packed[k * MR + r] = (Acc) row[r];
            for (; r < MR; ++r)
                packed[k * MR + r] = Acc(0);
        }
    }
}

// Pack one NR-wide column strip of B (``kc_len`` K steps wide) into
// ``packed`` with layout ``[k][b]``. The first ``n_valid`` columns
// carry real data; we fill the rest with zeros.
template <typename T, bool Bt>
static inline void pack_b(const T *B, GemmAcc<T> *packed,
                          uint32_t N, uint32_t K,
                          uint32_t j_start, uint32_t kc0, uint32_t kc_len,
                          uint32_t n_valid) {
    using Acc = GemmAcc<T>;
    constexpr uint32_t NR = GemmTile<T>::NR;
    if constexpr (!Bt) {
        for (uint32_t k = 0; k < kc_len; ++k) {
            const T *row = B + (size_t) (kc0 + k) * N + j_start;
            uint32_t b = 0;
            for (; b < n_valid; ++b)
                packed[k * NR + b] = (Acc) row[b];
            for (; b < NR; ++b)
                packed[k * NR + b] = Acc(0);
        }
    } else {
        for (uint32_t b = 0; b < NR; ++b) {
            if (b < n_valid) {
                const T *row = B + (size_t) (j_start + b) * K + kc0;
                for (uint32_t k = 0; k < kc_len; ++k)
                    packed[k * NR + b] = (Acc) row[k];
            } else {
                for (uint32_t k = 0; k < kc_len; ++k)
                    packed[k * NR + b] = Acc(0);
            }
        }
    }
}

// MR x NR microkernel. ``first`` toggles zero-init vs RMW against
// ``C_tile``, so successive K segments and reduce iterations fold
// into the same output tile. The six accumulators expand into named
// ``NR``-wide arrays so a vectorizing compiler keeps them in
// registers across the full ``kc_len`` sweep; the inner ``b`` loop
// vectorizes into ``NR``-wide SIMD.
// JIT_NO_UBSAN: i32 reuses the u32 kernel body; wraparound mul/add is intended.
template <typename T> JIT_NO_UBSAN
static inline void gemm_micro(const GemmAcc<T> *packed_A,
                              const GemmAcc<T> *packed_B,
                              T *C_tile, uint32_t ldc,
                              uint32_t kc_len, bool first) {
    using Acc = GemmAcc<T>;
    constexpr uint32_t MR = GemmTile<T>::MR;
    constexpr uint32_t NR = GemmTile<T>::NR;
    static_assert(MR == 6, "gemm_micro: inner kernel is manually unrolled for MR=6.");

    Acc r0[NR], r1[NR], r2[NR], r3[NR], r4[NR], r5[NR];

    if (first) {
        for (uint32_t b = 0; b < NR; ++b) {
            r0[b] = Acc(0); r1[b] = Acc(0); r2[b] = Acc(0);
            r3[b] = Acc(0); r4[b] = Acc(0); r5[b] = Acc(0);
        }
    } else {
        for (uint32_t b = 0; b < NR; ++b) {
            r0[b] = (Acc) C_tile[0 * ldc + b];
            r1[b] = (Acc) C_tile[1 * ldc + b];
            r2[b] = (Acc) C_tile[2 * ldc + b];
            r3[b] = (Acc) C_tile[3 * ldc + b];
            r4[b] = (Acc) C_tile[4 * ldc + b];
            r5[b] = (Acc) C_tile[5 * ldc + b];
        }
    }

    for (uint32_t k = 0; k < kc_len; ++k) {
        Acc av0 = packed_A[k * MR + 0];
        Acc av1 = packed_A[k * MR + 1];
        Acc av2 = packed_A[k * MR + 2];
        Acc av3 = packed_A[k * MR + 3];
        Acc av4 = packed_A[k * MR + 4];
        Acc av5 = packed_A[k * MR + 5];
        for (uint32_t b = 0; b < NR; ++b) {
            Acc bv = packed_B[k * NR + b];
            r0[b] += av0 * bv;
            r1[b] += av1 * bv;
            r2[b] += av2 * bv;
            r3[b] += av3 * bv;
            r4[b] += av4 * bv;
            r5[b] += av5 * bv;
        }
    }

    for (uint32_t b = 0; b < NR; ++b) {
        C_tile[0 * ldc + b] = (T) r0[b];
        C_tile[1 * ldc + b] = (T) r1[b];
        C_tile[2 * ldc + b] = (T) r2[b];
        C_tile[3 * ldc + b] = (T) r3[b];
        C_tile[4 * ldc + b] = (T) r4[b];
        C_tile[5 * ldc + b] = (T) r5[b];
    }
}

// Sweep one MR-row slab across all column slabs in the panel
// ``[jc, jc + n_slabs * NR)``. Bulk tiles (full MR rows, full NR
// cols) write directly to ``C``; edge tiles accumulate through a
// small scratch buffer and copy back only the valid cells. Zero-
// padding in the packed buffers lets the microkernel run unchanged
// on every tile. A nonzero ``n_tail`` marks the last slab as partial
// with ``n_tail`` valid columns out of ``NR``.
// JIT_NO_UBSAN: i32 reuses the u32 kernel body; wraparound mul/add is intended.
template <typename T, bool At> JIT_NO_UBSAN
static inline void gemm_row_sweep(const T *A, const GemmAcc<T> *packed_B,
                                  T *C, uint32_t M, uint32_t N, uint32_t K,
                                  uint32_t i_abs, uint32_t jc,
                                  uint32_t pc, uint32_t kc_len,
                                  uint32_t n_slabs, uint32_t n_tail,
                                  bool first) {
    using Acc = GemmAcc<T>;
    constexpr uint32_t MR = GemmTile<T>::MR;
    constexpr uint32_t NR = GemmTile<T>::NR;
    constexpr uint32_t KC = GEMM_KC;

    uint32_t m_valid = std::min(MR, M - i_abs);

    alignas(64) Acc packed_A[KC * MR];
    pack_a<T, At>(A, packed_A, M, K, i_abs, pc, kc_len, m_valid);

    T *c_row = C + (size_t) i_abs * N + jc;
    for (uint32_t s = 0; s < n_slabs; ++s) {
        uint32_t n_valid = (n_tail > 0 && s == n_slabs - 1) ? n_tail : NR;
        const Acc *pb = packed_B + (size_t) s * KC * NR;
        T *c_tile = c_row + s * NR;

        if (m_valid == MR && n_valid == NR) {
            gemm_micro<T>(packed_A, pb, c_tile, N, kc_len, first);
        } else {
            alignas(64) T scratch[MR * NR];
            if (!first) {
                for (uint32_t r = 0; r < m_valid; ++r)
                    for (uint32_t b = 0; b < n_valid; ++b)
                        scratch[r * NR + b] = c_tile[r * N + b];
            }
            gemm_micro<T>(packed_A, pb, scratch, NR, kc_len, first);
            for (uint32_t r = 0; r < m_valid; ++r)
                for (uint32_t b = 0; b < n_valid; ++b)
                    c_tile[r * N + b] = scratch[r * NR + b];
        }
    }
}

// Type-erased dispatch table. ``gemm_dispatch`` builds one per
// (``type``, ``At``, ``Bt``) triple and hands it to the launcher in
// ``llvm_ts.cpp``. A null ``row_sweep`` signals an unsupported type.
using GemmPackBFn = void (*)(const void *B, void *packed,
                             uint32_t N, uint32_t K, uint32_t jr_abs,
                             uint32_t pc, uint32_t kc_len, uint32_t n_valid);

using GemmRowSweepFn = void (*)(const void *A, const void *packed_B, void *C,
                                uint32_t M, uint32_t N, uint32_t K,
                                uint32_t i_abs, uint32_t jc,
                                uint32_t pc, uint32_t kc_len,
                                uint32_t n_slabs, uint32_t n_tail, bool first);

struct GemmDispatch {
    uint32_t MR;        // microtile row count
    uint32_t NR;        // microtile col count
    uint32_t acc_size;  // sizeof(GemmAcc<T>) (4 or 8)

    GemmPackBFn    pack_b;     // pack one NR-wide slab of B (zero-padded at tail)
    GemmRowSweepFn row_sweep;  // pack MR rows of A + sweep all column slabs
};

template <typename T, bool Bt>
static void gemm_pack_b_trampoline(const void *B, void *packed,
                                   uint32_t N, uint32_t K, uint32_t jr_abs,
                                   uint32_t pc, uint32_t kc_len,
                                   uint32_t n_valid) {
    pack_b<T, Bt>((const T *) B, (GemmAcc<T> *) packed,
                  N, K, jr_abs, pc, kc_len, n_valid);
}

template <typename T, bool At>
static void gemm_row_sweep_trampoline(const void *A, const void *packed_B,
                                      void *C, uint32_t M, uint32_t N,
                                      uint32_t K, uint32_t i_abs,
                                      uint32_t jc, uint32_t pc,
                                      uint32_t kc_len, uint32_t n_slabs,
                                      uint32_t n_tail, bool first) {
    gemm_row_sweep<T, At>((const T *) A, (const GemmAcc<T> *) packed_B,
                          (T *) C, M, N, K, i_abs, jc, pc, kc_len,
                          n_slabs, n_tail, first);
}

template <typename T, bool At, bool Bt>
static GemmDispatch gemm_dispatch_fill() {
    GemmDispatch d{};
    d.MR        = GemmTile<T>::MR;
    d.NR        = GemmTile<T>::NR;
    d.acc_size  = (uint32_t) sizeof(GemmAcc<T>);
    d.pack_b    = &gemm_pack_b_trampoline<T, Bt>;
    d.row_sweep = &gemm_row_sweep_trampoline<T, At>;
    return d;
}

// Pick a dispatch table for one of the three handled ``(At, Bt)``
// combos. The Python caller rewrites ``At == Bt == true`` via the
// ``A^T @ B^T = (B @ A)^T`` identity, so it never reaches here.
template <typename T>
static GemmDispatch gemm_dispatch_pick(bool At, bool Bt) {
    if (!At && !Bt) return gemm_dispatch_fill<T, false, false>();
    if ( At && !Bt) return gemm_dispatch_fill<T, true,  false>();
    return gemm_dispatch_fill<T, false, true>();
}

// Int32 and UInt32 share a kernel (identical under 2's-complement mul/add).
static GemmDispatch gemm_dispatch(VarType vt, bool At, bool Bt) {
    switch (vt) {
        case VarType::Float16: return gemm_dispatch_pick<drjit::half>(At, Bt);
        case VarType::Float32: return gemm_dispatch_pick<float      >(At, Bt);
        case VarType::Float64: return gemm_dispatch_pick<double     >(At, Bt);
        case VarType::Int32:
        case VarType::UInt32:  return gemm_dispatch_pick<uint32_t   >(At, Bt);
        default: return GemmDispatch{};
    }
}
