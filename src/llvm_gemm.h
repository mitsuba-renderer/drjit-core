/*
    src/llvm_gemm.h -- Tiled CPU GEMM kernels for the LLVM backend

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include <drjit-core/half.h>
#include <cstring>
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

// Microtile dimensions. ``NR = NR_VECS * simd_width / sizeof(Acc)``
// packs ``NR_VECS`` SIMD registers along the column axis per row,
// for ``MR * NR_VECS`` independent accumulator chains per tile.
//
// ``MR = 6`` is fixed. ``NR_VECS`` is then picked so that:
//   (1) Live state -- ``MR * NR_VECS`` accumulators, ``NR_VECS``
//       B-loads, one A broadcast -- fits the vector register file.
//   (2) ``MR * NR_VECS`` chains cover the FMA pipeline depth
//       (issue width x latency) to keep the pipes busy.
//
//     ISA         SIMD   lanes f32   vec regs   FMA pipes
//     AVX-512      512        16         32         2
//     AVX / AVX2   256         8         16         2
//     NEON         128         4         32       2 to 4
//
//  - AVX-512: ``NR_VECS = 1``. One 512-bit FMA covers two 256-bit
//    lanes of work, so ``MR = 6`` zmm chains behave like 12 ymm
//    chains — comfortably above the 2 pipes x ~4-cycle latency
//    (~8 chains) needed to saturate the FMAs. Live state is
//    6 + 1 + 1 = 8 zmm registers in 32.
//  - AVX / AVX2: ``NR_VECS = 2``. ``MR * NR_VECS = 12`` ymm chains
//    fully cover the 2-pipe, ~4-cycle FMA pipeline; live state is
//    12 + 2 + 1 = 15 ymm registers, fitting the 16-register file
//    with one spare. ``NR_VECS = 1`` underfills the pipeline on
//    this ISA and leaves half the register file unused.
//  - NEON: ``NR_VECS = 4``. Sized for 4-pipe implementations like
//    Apple M-series (pipeline ~12 deep): ``6 * 4 = 24`` chains,
//    live state is 24 + 4 = 28 of 32 v-regs. Some ARM reference
//    cores have only 2 FMA pipes, where this is wider than strictly
//    needed but still fits.
template <typename T> struct GemmTile {
    static constexpr uint32_t NR_VECS =
#if defined(__AVX512F__)
        1;
#elif defined(__AVX2__) || defined(__AVX__)
        2;
#else
        4;
#endif
    static constexpr uint32_t MR = 6;
    static constexpr uint32_t NR =
        NR_VECS * (GEMM_SIMD_BYTES / sizeof(GemmAcc<T>));
};

// Fixed-width SIMD vector type used by ``gemm_micro``. The GCC/Clang
// path uses vector extensions so arithmetic lowers directly to SIMD
// FMAs. The MSVC fallback uses a POD struct with elementwise operators
// and defers codegen to the compiler's auto-vectorizer.
#if defined(__GNUC__) || defined(__clang__)
// Per-scalar specialisations so ``vector_size`` is applied at a
// non-dependent declaration -- GCC silently strips the attribute otherwise.
template <typename T, uint32_t V> struct GemmVecImpl;

#define DRJIT_DECLARE_GEMM_VEC(T_)                                             \
    template <uint32_t V> struct GemmVecImpl<T_, V> {                          \
        typedef T_ type __attribute__((vector_size(V * sizeof(T_))));          \
    };
DRJIT_DECLARE_GEMM_VEC(float)
DRJIT_DECLARE_GEMM_VEC(double)
DRJIT_DECLARE_GEMM_VEC(uint32_t)
#undef DRJIT_DECLARE_GEMM_VEC

template <typename T, uint32_t V>
using GemmVec = typename GemmVecImpl<T, V>::type;
#else
template <typename T, uint32_t V> struct GemmVec {
    T d[V];
    T  operator[](uint32_t i) const { return d[i]; }
    T &operator[](uint32_t i)       { return d[i]; }
};
template <typename T, uint32_t V>
static inline GemmVec<T, V> operator+(const GemmVec<T, V> &a,
                                      const GemmVec<T, V> &b) {
    GemmVec<T, V> r;
    for (uint32_t i = 0; i < V; ++i) r.d[i] = a.d[i] + b.d[i];
    return r;
}
template <typename T, uint32_t V>
static inline GemmVec<T, V> operator*(T s, const GemmVec<T, V> &v) {
    GemmVec<T, V> r;
    for (uint32_t i = 0; i < V; ++i) r.d[i] = s * v.d[i];
    return r;
}
#endif

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
// into the same output tile. Per k-step we load ``NR_VECS`` B-vectors
// and one broadcast A-scalar per row, yielding ``MR * NR_VECS`` vector
// FMAs that feed an ``MR x NR_VECS`` tile of accumulator registers.
// JIT_NO_UBSAN: i32 reuses the u32 kernel body; wraparound mul/add is intended.
template <typename T> JIT_NO_UBSAN
static inline void gemm_micro(const GemmAcc<T> *packed_A,
                              const GemmAcc<T> *packed_B,
                              T *C_tile, uint32_t ldc,
                              uint32_t kc_len, bool first) {
    using Acc = GemmAcc<T>;
    constexpr uint32_t MR = GemmTile<T>::MR;
    constexpr uint32_t NR = GemmTile<T>::NR;
    constexpr uint32_t NR_VECS = GemmTile<T>::NR_VECS;
    constexpr uint32_t V = NR / NR_VECS;
    static_assert(NR == NR_VECS * V, "NR must be a multiple of the SIMD lane count.");

    using Vec = GemmVec<Acc, V>;

    Vec r[MR][NR_VECS];

    if (first) {
        for (uint32_t m = 0; m < MR; ++m)
            for (uint32_t v = 0; v < NR_VECS; ++v)
                r[m][v] = Vec{};
    } else if constexpr (std::is_same_v<T, Acc>) {
        for (uint32_t m = 0; m < MR; ++m)
            for (uint32_t v = 0; v < NR_VECS; ++v)
                std::memcpy(&r[m][v], &C_tile[m * ldc + v * V], sizeof(Vec));
    } else {
        // Narrower-storage path (e.g. T=half, Acc=float): widen per-lane.
        for (uint32_t m = 0; m < MR; ++m)
            for (uint32_t v = 0; v < NR_VECS; ++v)
                for (uint32_t i = 0; i < V; ++i)
                    r[m][v][i] = (Acc) C_tile[m * ldc + v * V + i];
    }

    for (uint32_t k = 0; k < kc_len; ++k) {
        Vec bv[NR_VECS];
        for (uint32_t v = 0; v < NR_VECS; ++v)
            std::memcpy(&bv[v], &packed_B[k * NR + v * V], sizeof(Vec));
        for (uint32_t m = 0; m < MR; ++m) {
            Acc av = packed_A[k * MR + m];
            for (uint32_t v = 0; v < NR_VECS; ++v)
                r[m][v] = r[m][v] + av * bv[v];
        }
    }

    if constexpr (std::is_same_v<T, Acc>) {
        for (uint32_t m = 0; m < MR; ++m)
            for (uint32_t v = 0; v < NR_VECS; ++v)
                std::memcpy(&C_tile[m * ldc + v * V], &r[m][v], sizeof(Vec));
    } else {
        for (uint32_t m = 0; m < MR; ++m)
            for (uint32_t v = 0; v < NR_VECS; ++v)
                for (uint32_t i = 0; i < V; ++i)
                    C_tile[m * ldc + v * V + i] = (T) r[m][v][i];
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
