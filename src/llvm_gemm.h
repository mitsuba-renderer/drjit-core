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

  Computes ``C = op_A(A) @ op_B(B)`` of shape ``(M, N)``. ``At`` / ``Bt``
  select whether ``A`` / ``B`` are transposed.

  Schedule. This is the standard two-level blocked GEMM from the GotoBLAS
  / BLIS literature [1, 2]. Three levels of blocking are at play:

   - **Outer block** (``MB x NB``): the output matrix is partitioned into
     ``MB x NB`` blocks; each block runs as an independent nanothread task.

   - **K segment** (``KC``): within a block, the K dimension is walked in
     ``KC``-wide segments. Per segment we copy the relevant ``(MB x KC)``
     strip of ``A`` and ``(KC x NB)`` strip of ``B`` into contiguous stack
     buffers (``pack_a`` / ``pack_b``). The copy also absorbs the
     ``At`` / ``Bt`` transpose, so the downstream compute kernel reads the
     packed strips with a single, transpose-agnostic access pattern.

   - **Microtile** (``MR x NR``): within a block/segment we sweep a grid
     of ``MR x NR`` microtiles. Each one calls ``gemm_micro``, which holds
     the accumulator block live in registers across the full ``KC`` sweep
     and only reads/writes ``C`` at segment boundaries.

  Edge handling. Partial ``M`` / ``N`` strips at a block boundary use a
  *shifted* microtile: see ``gemm_shifted_tile`` for a detailed discussion.
  A scalar fallback (``gemm_edge``) runs only in the degenerate case where
  an entire block is smaller than ``MR`` / ``NR`` (e.g. tiny inputs).

  [1] Goto & van de Geijn, "Anatomy of High-Performance Matrix
      Multiplication", ACM TOMS 34(3), 2008.
  [2] Van Zee & van de Geijn, "BLIS: A Framework for Rapidly
      Instantiating BLAS Functionality", ACM TOMS 41(3), 2015.
*/

template <typename T>
using GemmAcc = std::conditional_t<std::is_same_v<T, drjit::half>, float, T>;

// Microtile dimensions: each call into the compute microkernel produces
// an ``MR x NR`` sub-block of ``C``. ``MR = 6`` is paired with an
// ``NR``-wide SIMD row so that the six row accumulators, plus one
// broadcast register for the shared A lane, fit comfortably in the
// vector register file on typical targets.
template <typename T> struct GemmTile {
    static constexpr uint32_t MR = 6;
    static constexpr uint32_t NR = sizeof(T) == 8 ? 8 : 16;
};

/// Outer block height: number of rows of ``C`` processed by one task.
/// 60 = 10 * MR, i.e. ten microtile rows per block.
static constexpr uint32_t GEMM_MB = 60;
/// Outer block width: number of columns of ``C`` processed by one task.
/// 64 spans 4 microtile cols at ``NR = 16`` (f16/f32/i32) or 8 microtile
/// cols at ``NR = 8`` (f64).
static constexpr uint32_t GEMM_NB = 64;
/// K-segment depth: sweep the K dimension in ``KC``-wide segments,
/// packing one strip of ``A`` and ``B`` per segment. Sized so the two
/// packed strips (``MB * KC`` + ``KC * NB`` accumulator-typed values)
/// comfortably fit into L1.
static constexpr uint32_t GEMM_KC = 128;

// Reduce-dim spec: forwards the reduce-batch portion of ``GemmBatch`` to
// the block/edge kernels. ``n_rdims == 0`` (or ``r_count == 1``) degenerates
// to a single iteration with A/B unchanged.
struct GemmReduce {
    uint32_t r_count;               // prod(extent[..])
    uint32_t n_rdims;               // number of reduce dims
    const uint32_t *extent;         // reduce extents [n_rdims]
    const uint32_t *a_stride;       // per-reduce-dim A strides [n_rdims]
    const uint32_t *b_stride;       // per-reduce-dim B strides [n_rdims]
};

/// Decode a flat reduce index ``rb`` (in ``[0, r.r_count)``) into
/// per-operand byte offsets ``a_off`` / ``b_off``. ``rb`` is interpreted
/// as a mixed-radix number over ``r.extent[0..n_rdims)`` (innermost at
/// ``d = 0``), and each dim contributes ``idx[d] * a_stride[d]`` to
/// ``a_off`` (likewise for ``b_off``). Strides of zero encode broadcast.
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

// Pack ``mr_rows`` MR-wide row strips of A, each ``kc_len`` K steps
// wide, into ``packed`` with layout ``[mr_row][k][r]`` of stride
// ``KC * MR`` per mr_row. Cold path; ``At`` selects the source
// stride pattern.
template <typename T, bool At>
static inline void pack_a(const T *A, GemmAcc<T> *packed,
                          uint32_t M, uint32_t K,
                          uint32_t i_start, uint32_t kc0, uint32_t kc_len,
                          uint32_t mr_rows) {
    using Acc = GemmAcc<T>;
    constexpr uint32_t MR = GemmTile<T>::MR;
    if constexpr (!At) {
        for (uint32_t mr = 0; mr < mr_rows; ++mr) {
            Acc *pout = packed + mr * GEMM_KC * MR;
            for (uint32_t r = 0; r < MR; ++r) {
                const T *row = A + (size_t) (i_start + mr * MR + r) * K + kc0;
                for (uint32_t k = 0; k < kc_len; ++k)
                    pout[k * MR + r] = (Acc) row[k];
            }
        }
    } else {
        for (uint32_t mr = 0; mr < mr_rows; ++mr) {
            Acc *pout = packed + mr * GEMM_KC * MR;
            for (uint32_t k = 0; k < kc_len; ++k) {
                const T *row = A + (size_t) (kc0 + k) * M + (i_start + mr * MR);
                for (uint32_t r = 0; r < MR; ++r)
                    pout[k * MR + r] = (Acc) row[r];
            }
        }
    }
}

// Pack ``nr_cols`` NR-wide column strips of B, each ``kc_len`` K steps
// wide, into ``packed`` with layout ``[nr_col][k][b]`` of stride
// ``KC * NR`` per nr_col.
template <typename T, bool Bt>
static inline void pack_b(const T *B, GemmAcc<T> *packed,
                          uint32_t N, uint32_t K,
                          uint32_t j_start, uint32_t kc0, uint32_t kc_len,
                          uint32_t nr_cols) {
    using Acc = GemmAcc<T>;
    constexpr uint32_t NR = GemmTile<T>::NR;
    if constexpr (!Bt) {
        for (uint32_t nc = 0; nc < nr_cols; ++nc) {
            Acc *pout = packed + nc * GEMM_KC * NR;
            for (uint32_t k = 0; k < kc_len; ++k) {
                const T *row = B + (size_t) (kc0 + k) * N + (j_start + nc * NR);
                for (uint32_t b = 0; b < NR; ++b)
                    pout[k * NR + b] = (Acc) row[b];
            }
        }
    } else {
        for (uint32_t nc = 0; nc < nr_cols; ++nc) {
            Acc *pout = packed + nc * GEMM_KC * NR;
            for (uint32_t b = 0; b < NR; ++b) {
                const T *row = B + (size_t) (j_start + nc * NR + b) * K + kc0;
                for (uint32_t k = 0; k < kc_len; ++k)
                    pout[k * NR + b] = (Acc) row[k];
            }
        }
    }
}

// MR x NR microkernel. ``first`` selects between zero-init and RMW
// against the current contents of ``C_tile`` (used to fold successive
// K segments and reduce iterations into the same output tile).
// The six accumulators expand into named ``NR``-wide arrays so that a
// vectorizing compiler can keep them in registers across the entire
// ``kc_len`` sweep; the ``b`` inner loop is the one that gets
// auto-vectorized into ``NR``-wide SIMD.
// JIT_NO_UBSAN: i32 reuses the u32 kernel body; wraparound mul/add is intended.
template <typename T>
static inline void gemm_micro(const GemmAcc<T> *packed_A,
                              const GemmAcc<T> *packed_B,
                              T *C_tile, uint32_t ldc,
                              uint32_t kc_len, bool first) JIT_NO_UBSAN {
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

// Scalar fallback for partial M/N edge strips. The reduce loop wraps the
// per-(i, j) dot product so the output write happens once per cell.
// JIT_NO_UBSAN: i32 reuses the u32 kernel body; wraparound mul/add is intended.
template <typename T>
static void gemm_edge(const T *A, const T *B, T *C,
                      uint32_t M, uint32_t N, uint32_t K,
                      uint32_t i_begin, uint32_t i_end,
                      uint32_t j_begin, uint32_t j_end,
                      bool At, bool Bt,
                      const GemmReduce &reduce) JIT_NO_UBSAN {
    using Acc = GemmAcc<T>;
    for (uint32_t i = i_begin; i < i_end; ++i) {
        for (uint32_t j = j_begin; j < j_end; ++j) {
            Acc sum = Acc(0);
            for (uint32_t rb = 0; rb < reduce.r_count; ++rb) {
                size_t a_off, b_off;
                gemm_reduce_decode(reduce, rb, a_off, b_off);
                const T *A_r = A + a_off, *B_r = B + b_off;
                for (uint32_t k = 0; k < K; ++k) {
                    Acc a = !At ? (Acc) A_r[i * K + k] : (Acc) A_r[k * M + i];
                    Acc b = !Bt ? (Acc) B_r[k * N + j] : (Acc) B_r[j * K + k];
                    sum += a * b;
                }
            }
            C[i * N + j] = (T) sum;
        }
    }
}

// Handle a partial M / N strip at the trailing edge of a block using a
// *shifted microtile*.
//
// When the block's ``M`` or ``N`` span is not a multiple of the microtile
// dimensions, there are a few leftover rows / columns (``m_tail < MR``
// and/or ``n_tail < NR``) that the bulk microtile sweep cannot cover. A
// naive edge path would fall back to a scalar loop for those cells, which
// is dramatically slower than the vectorized microkernel.
//
// Instead we run the *same* vectorized ``MR x NR`` microkernel, but
// positioned so its starting row / column is ``span - MR`` / ``span - NR``
// instead of ``m_bulk`` / ``n_bulk``. The microtile thus reads ``MR`` real
// rows (resp. ``NR`` real cols), which always fit because the block is at
// least ``MR x NR`` by the caller's precondition. The catch is that the
// first ``MR - m_tail`` rows (resp. ``NR - n_tail`` cols) overlap with
// microtiles the bulk sweep already wrote; to avoid double-writing those
// cells, the microkernel accumulates into a private scratch buffer and
// only the rows ``[r_skip, MR)`` / cols ``[b_skip, NR)`` are copied back
// to ``C``.
//
// Net effect: the edge strips run at full microkernel throughput, and the
// scalar fallback (``gemm_edge``) is reached only for genuinely tiny
// inputs (entire block below ``MR x NR``).
//
// JIT_NO_UBSAN: i32 reuses the u32 kernel body; wraparound mul/add is intended.
template <typename T>
static inline void gemm_shifted_tile(const T *A, const T *B, T *C,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t i_start, uint32_t j_start,
                                     uint32_t r_skip, uint32_t b_skip,
                                     bool At, bool Bt,
                                     const GemmReduce &reduce) JIT_NO_UBSAN {
    using Acc = GemmAcc<T>;
    constexpr uint32_t MR = GemmTile<T>::MR;
    constexpr uint32_t NR = GemmTile<T>::NR;
    constexpr uint32_t KC = GEMM_KC;

    alignas(64) Acc packed_A[KC * MR];
    alignas(64) Acc packed_B[KC * NR];
    alignas(64) T scratch[MR * NR];

    for (uint32_t rb = 0; rb < reduce.r_count; ++rb) {
        size_t ra_off, rb_off;
        gemm_reduce_decode(reduce, rb, ra_off, rb_off);
        const T *A_r = A + ra_off, *B_r = B + rb_off;

        for (uint32_t kc0 = 0; kc0 < K; kc0 += KC) {
            uint32_t kc_len = std::min(KC, K - kc0);

            if (At) pack_a<T, true >(A_r, packed_A, M, K, i_start, kc0, kc_len, 1);
            else    pack_a<T, false>(A_r, packed_A, M, K, i_start, kc0, kc_len, 1);
            if (Bt) pack_b<T, true >(B_r, packed_B, N, K, j_start, kc0, kc_len, 1);
            else    pack_b<T, false>(B_r, packed_B, N, K, j_start, kc0, kc_len, 1);

            bool first = (rb == 0 && kc0 == 0);
            gemm_micro<T>(packed_A, packed_B, scratch, NR, kc_len, first);
        }
    }

    for (uint32_t r = r_skip; r < MR; ++r)
        for (uint32_t b = b_skip; b < NR; ++b)
            C[(size_t) (i_start + r) * N + j_start + b] = scratch[r * NR + b];
}

// Compute one ``MB x NB`` output block spanning
// ``[m_start, m_end) x [n_start, n_end)``. Outer order: reduce dim,
// K segment, microtile sweep. The K-segment body packs ``A`` and ``B``
// into contiguous strips once and reuses them across the
// ``mr_rows x nr_cols`` microtile grid. ``first`` on the microkernel is
// true only for the first ``(reduce, K-segment)`` pair so the accumulator
// is zeroed instead of loaded from an undefined ``C``. Partial ``M`` /
// ``N`` strips at the block boundary are handled by ``gemm_shifted_tile``;
// ``gemm_edge`` is only reached when the entire block is below
// ``MR`` / ``NR`` (e.g. tiny input matrices).
// JIT_NO_UBSAN: i32 reuses the u32 kernel body; wraparound mul/add is intended.
template <typename T>
static void gemm_block(const T *A, const T *B, T *C,
                       uint32_t M, uint32_t N, uint32_t K,
                       uint32_t m_start, uint32_t m_end,
                       uint32_t n_start, uint32_t n_end,
                       bool At, bool Bt,
                       const GemmReduce &reduce) JIT_NO_UBSAN {
    using Acc = GemmAcc<T>;
    constexpr uint32_t MR = GemmTile<T>::MR, NR = GemmTile<T>::NR;
    constexpr uint32_t MB = GEMM_MB,         NB = GEMM_NB;
    constexpr uint32_t KC = GEMM_KC;

    uint32_t m_span = m_end - m_start, n_span = n_end - n_start;

    if (m_span < MR || n_span < NR) {
        gemm_edge<T>(A, B, C, M, N, K,
                     m_start, m_end, n_start, n_end, At, Bt, reduce);
        return;
    }

    uint32_t m_bulk = (m_span / MR) * MR, n_bulk = (n_span / NR) * NR;
    uint32_t mr_rows = m_bulk / MR;
    uint32_t nr_cols = n_bulk / NR;
    uint32_t m_tail = m_span - m_bulk;
    uint32_t n_tail = n_span - n_bulk;

    {
        alignas(64) Acc packed_A[MB * KC];
        alignas(64) Acc packed_B[KC * NB];

        for (uint32_t rb = 0; rb < reduce.r_count; ++rb) {
            size_t ra_off, rb_off;
            gemm_reduce_decode(reduce, rb, ra_off, rb_off);
            const T *A_r = A + ra_off, *B_r = B + rb_off;

            for (uint32_t kc0 = 0; kc0 < K; kc0 += KC) {
                uint32_t kc_len = std::min(KC, K - kc0);

                if (At) pack_a<T, true >(A_r, packed_A, M, K, m_start, kc0, kc_len, mr_rows);
                else    pack_a<T, false>(A_r, packed_A, M, K, m_start, kc0, kc_len, mr_rows);
                if (Bt) pack_b<T, true >(B_r, packed_B, N, K, n_start, kc0, kc_len, nr_cols);
                else    pack_b<T, false>(B_r, packed_B, N, K, n_start, kc0, kc_len, nr_cols);

                bool first = (rb == 0 && kc0 == 0);

                for (uint32_t mr = 0; mr < mr_rows; ++mr) {
                    const Acc *pa = packed_A + mr * KC * MR;
                    T *c_row = C + (size_t) (m_start + mr * MR) * N + n_start;
                    for (uint32_t nc = 0; nc < nr_cols; ++nc) {
                        const Acc *pb = packed_B + nc * KC * NR;
                        gemm_micro<T>(pa, pb, c_row + nc * NR, N, kc_len, first);
                    }
                }
            }
        }
    }

    if (m_tail > 0) {
        uint32_t i_start_s = m_start + m_span - MR;
        uint32_t r_skip = MR - m_tail;
        for (uint32_t nc = 0; nc < nr_cols; ++nc)
            gemm_shifted_tile<T>(A, B, C, M, N, K, i_start_s,
                                 n_start + nc * NR, r_skip, 0, At, Bt, reduce);
    }

    if (n_tail > 0) {
        uint32_t j_start_s = n_start + n_span - NR;
        uint32_t b_skip = NR - n_tail;
        for (uint32_t mr = 0; mr < mr_rows; ++mr)
            gemm_shifted_tile<T>(A, B, C, M, N, K, m_start + mr * MR,
                                 j_start_s, 0, b_skip, At, Bt, reduce);
    }

    if (m_tail > 0 && n_tail > 0)
        gemm_shifted_tile<T>(A, B, C, M, N, K,
                             m_start + m_span - MR, n_start + n_span - NR,
                             MR - m_tail, NR - n_tail, At, Bt, reduce);
}

// Type-erased entry point used by the launcher.
using GemmBlockFn = void (*)(const void *A, const void *B, void *C,
                             uint32_t M, uint32_t N, uint32_t K,
                             uint32_t m_start, uint32_t m_end,
                             uint32_t n_start, uint32_t n_end,
                             const GemmReduce *reduce);

// One trampoline per (T, At, Bt) to keep the function-pointer dispatch
// at the launcher, but the body is a thin forwarding call into the
// type-generic ``gemm_block<T>``.
template <typename T, bool At, bool Bt>
static void gemm_block_trampoline(const void *A, const void *B, void *C,
                                  uint32_t M, uint32_t N, uint32_t K,
                                  uint32_t m_start, uint32_t m_end,
                                  uint32_t n_start, uint32_t n_end,
                                  const GemmReduce *reduce) {
    gemm_block<T>((const T *) A, (const T *) B, (T *) C,
                  M, N, K, m_start, m_end, n_start, n_end, At, Bt, *reduce);
}

// Pick a trampoline for one of the three handled ``(At, Bt)`` combos.
// ``At == Bt == true`` is rewritten by the Python caller (via the
// ``A^T @ B^T = (B @ A)^T`` identity) and never reaches the launcher.
template <typename T>
static GemmBlockFn gemm_block_pick(bool At, bool Bt) {
    if (!At && !Bt) return &gemm_block_trampoline<T, false, false>;
    if ( At && !Bt) return &gemm_block_trampoline<T, true,  false>;
    return &gemm_block_trampoline<T, false, true>;
}

static GemmBlockFn gemm_block_dispatch(VarType vt, bool At, bool Bt) {
    // Int32 and UInt32 share a kernel (identical under 2's-complement mul/add).
    switch (vt) {
        case VarType::Float16: return gemm_block_pick<drjit::half>(At, Bt);
        case VarType::Float32: return gemm_block_pick<float      >(At, Bt);
        case VarType::Float64: return gemm_block_pick<double     >(At, Bt);
        case VarType::Int32:
        case VarType::UInt32:  return gemm_block_pick<uint32_t   >(At, Bt);
        default: return nullptr;
    }
}
