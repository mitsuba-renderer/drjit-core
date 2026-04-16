/*
    kernels/gemm.cuh -- Shared-memory tiled CUDA GEMM kernels

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

// Maximum number of batch dimensions supported by ``jit_batched_gemm``.
// Must match ``DRJIT_GEMM_MAX_BDIMS`` in ``include/drjit-core/jit.h``.
#define DRJIT_GEMM_MAX_BDIMS 6

// Device-side mirror of the host-side ``GemmBatch`` struct in
// ``include/drjit-core/jit.h``. Layout must match exactly.
struct GemmBatch {
    uint32_t n_bdims;
    uint32_t n_rdims;
    uint32_t extent[DRJIT_GEMM_MAX_BDIMS];
    uint32_t a_stride[DRJIT_GEMM_MAX_BDIMS];
    uint32_t b_stride[DRJIT_GEMM_MAX_BDIMS];
};

/*
  Tiled row-major GEMM: C = op_A(A) @ op_B(B), shape (M, N).

  A thread block computes one BM x BM output tile for BM in {8, 16, 32,
  64}, using an 8 x 8 thread grid (NT = 64, two warps) that walks K in
  steps of BK = 16. Each thread owns a TM x TM register tile with
  TM = BM / 8. Global loads are vectorized; half inputs accumulate in
  float, others in their native type.

  Transpose handling: ``At`` / ``Bt`` are runtime kernel arguments.
  ``gemm_load`` dispatches on them once per block into a pair of
  template specializations. ``At == Bt == true`` is rewritten by the
  Python caller via ``A^T @ B^T = (B @ A)^T`` and is not implemented here.

  Smem layout:
    sA[BM][BK+1]  -- +1 pad makes per-row bank offsets coprime with 32, so
                     scalar LDS in gemm_compute is bank-conflict-free.
    sB[BK][BN]    -- natural stride; a cyclic thread-to-column mapping in
                     gemm_compute spreads V-packet LDS across 32 banks.

  Bounds handling has two independent axes. M/N edges are captured by the
  loop-invariant ``interior`` flag, which specializes the K loop into a
  branch-free path and a per-lane checked path. The K tail runs as a
  single extra iteration with K-axis checks that zero OOB lanes.
*/

// Aligned vector copy; V * sizeof(T) must be one of {2, 4, 8, 16}.
template <uint32_t V, typename T>
DEVICE FINLINE void gemm_vec_load(const T *src, T *dst) {
    if constexpr (V * sizeof(T) == 16)
        *(uint4 *) dst = *(const uint4 *) src;
    else if constexpr (V * sizeof(T) == 8)
        *(uint2 *) dst = *(const uint2 *) src;
    else if constexpr (V * sizeof(T) == 4)
        *(uint32_t *) dst = *(const uint32_t *) src;
    else
        *(uint16_t *) dst = *(const uint16_t *) src;
}

// Cooperatively stream a ``BRows x BCols`` global slab into ``smem``. The
// NT threads share the work: each issues ``L_V = BRows*BCols/(NT*V)``
// vector loads of ``V`` contiguous inner-axis elements. ``CheckRow`` /
// ``CheckCol`` gate per-lane bounds checks; OOB lanes write ``T(0)``.
// If ``Transpose`` is set, the slab is transposed on smem deposit.
//
// The deposit is always scalar: vector smem writes would hit the
// misaligned addresses induced by the +1 row padding on ``sA``.
template <typename T, uint32_t BRows, uint32_t BCols, uint32_t NT,
          uint32_t SmemStride, uint32_t V,
          bool CheckRow, bool CheckCol, bool Transpose>
DEVICE FINLINE void gemm_load_slab(const T *src, T *smem, uint32_t tid,
                                   uint32_t r0, uint32_t c0,
                                   uint32_t R, uint32_t C) {
    constexpr uint32_t BC_V = BCols / V;
    constexpr uint32_t L_V  = BRows * BC_V / NT;

    for (uint32_t l = 0; l < L_V; ++l) {
        uint32_t flat  = tid + l * NT;
        uint32_t r_loc = flat / BC_V;
        uint32_t c_loc = (flat % BC_V) * V;
        uint32_t r_g   = r0 + r_loc;
        uint32_t c_g   = c0 + c_loc;

        T tmp[V];
        bool ok = (!CheckRow || r_g < R) && (!CheckCol || c_g + V <= C);
        if (ok) {
            gemm_vec_load<V>(src + r_g * C + c_g, tmp);
        } else {
            bool row_ok = !CheckRow || r_g < R;
            for (uint32_t v = 0; v < V; ++v) {
                bool in = row_ok && (!CheckCol || c_g + v < C);
                tmp[v] = in ? src[r_g * C + c_g + v] : T(0);
            }
        }

        if constexpr (!Transpose) {
            for (uint32_t v = 0; v < V; ++v)
                smem[r_loc * SmemStride + c_loc + v] = tmp[v];
        } else {
            for (uint32_t v = 0; v < V; ++v)
                smem[(c_loc + v) * SmemStride + r_loc] = tmp[v];
        }
    }
}

// Load one operand tile into ``smem``, absorbing its transpose flag. The
// compute stage always sees a ``(BR, BC)``-shaped smem tile; when the
// global operand is transposed, we load a ``(BC, BR)`` tile and
// transpose-deposit instead.
template <typename T, uint32_t BR, uint32_t BC, uint32_t NT,
          uint32_t SmemStride, uint32_t V, bool CheckR, bool CheckC>
DEVICE FINLINE void gemm_load(bool transposed,
                              const T *src, T *smem, uint32_t tid,
                              uint32_t r0, uint32_t c0,
                              uint32_t R, uint32_t C) {
    if (!transposed)
        gemm_load_slab<T, BR, BC, NT, SmemStride, V, CheckR, CheckC, false>(
            src, smem, tid, r0, c0, R, C);
    else
        gemm_load_slab<T, BC, BR, NT, SmemStride, V, CheckC, CheckR, true>(
            src, smem, tid, c0, r0, C, R);
}

// Accumulate one ``BK``-step into ``acc``. Values are promoted to ``Acc``
// on read so the FMA runs at accumulator precision (matters for half).
//
// sA reads are scalar and stay conflict-free via the +1 stride pad.
// sB reads use cyclic V-packet LDS: thread ``tx`` reads packet ``p``
// (``p = 0..TM/V-1``) from column ``p * NT_n*V + tx*V``, spreading the
// packets of ``NT_n`` threads across ``NT_n*V <= 32`` distinct bank
// groups. The same mapping is reused for the C store, where it also
// gives fully coalesced vector writes.
template <typename T, typename Acc, uint32_t BK, uint32_t SA_S, uint32_t SB_S,
          uint32_t BN, uint32_t TM, uint32_t V>
DEVICE FINLINE void gemm_compute(const T *sA, const T *sB,
                                 uint32_t tx, uint32_t ty, Acc acc[TM][TM]) {
    constexpr uint32_t NT_n    = BN / TM;
    constexpr uint32_t StripeN = NT_n * V;

    #pragma unroll
    for (uint32_t kk = 0; kk < BK; ++kk) {
        Acc a_reg[TM], b_reg[TM];

        #pragma unroll
        for (uint32_t i = 0; i < TM; ++i)
            a_reg[i] = (Acc) sA[(ty * TM + i) * SA_S + kk];

        #pragma unroll
        for (uint32_t jv = 0; jv < TM; jv += V) {
            const uint32_t n_col = (jv / V) * StripeN + tx * V;
            T tmp[V];
            gemm_vec_load<V>(&sB[kk * SB_S + n_col], tmp);

            #pragma unroll
            for (uint32_t v = 0; v < V; ++v)
                b_reg[jv + v] = (Acc) tmp[v];
        }

        #pragma unroll
        for (uint32_t i = 0; i < TM; ++i)
            #pragma unroll
            for (uint32_t j = 0; j < TM; ++j)
                acc[i][j] = acc[i][j] + a_reg[i] * b_reg[j];
    }
}

// -----------------------------------------------------------------------------

template <typename T, uint32_t BM>
DEVICE FINLINE void gemm_impl(const T *__restrict__ A,
                              const T *__restrict__ B,
                              T *__restrict__ C,
                              uint32_t M, uint32_t N, uint32_t K,
                              uint32_t At, uint32_t Bt,
                              GemmBatch batch) {
    using Acc = std::conditional_t<std::is_same<T, half>::value, float, T>;

    // Per-block grid offset. ``blockIdx.z`` decomposes over the grid dims
    // ``[0, n_bdims)``. Broadcast along a grid dim is encoded by
    // ``a_stride[d] = 0`` or ``b_stride[d] = 0``. Output C advances along
    // grid dims only, with implicit stride ``M * N`` per grid step.
    {
        uint32_t z = blockIdx.z;
        size_t a_off = 0, b_off = 0;
        for (uint32_t d = 0; d < batch.n_bdims; ++d) {
            uint32_t idx = z % batch.extent[d];
            z /= batch.extent[d];
            a_off += (size_t) idx * batch.a_stride[d];
            b_off += (size_t) idx * batch.b_stride[d];
        }
        A += a_off;
        B += b_off;
        C += (size_t) blockIdx.z * M * N;
    }

    // Base pointers after grid decode; the reduce loop rebases these each
    // iteration without perturbing C.
    const T *A_base = A, *B_base = B;

    // Reduce-dim decode: iterate r over the product of reduce extents and
    // sum all contributions into the same accumulator tile. ``n_rdims == 0``
    // collapses to a single iteration that leaves A/B at their grid bases.
    uint32_t r_count = 1;
    for (uint32_t d = 0; d < batch.n_rdims; ++d)
        r_count *= batch.extent[batch.n_bdims + d];

    constexpr uint32_t BN   = BM, BK = 16;
    constexpr uint32_t TM   = BM / 8;
    constexpr uint32_t NT   = (BM / TM) * (BN / TM);
    constexpr uint32_t Vmax = 16 / sizeof(T);
    constexpr uint32_t V    = Vmax < TM ? Vmax : TM;

    static_assert(NT == 64, "Kernel assumes an 8x8 thread grid (64 threads).");
    static_assert(BM == 8 || BM == 16 || BM == 32 || BM == 64,
                  "BM must be one of {8, 16, 32, 64}.");
    static_assert(BK % V == 0 && BM % V == 0,
                  "Tile dimensions must be divisible by the vector width.");
    static_assert((BM * BK) % (NT * V) == 0,
                  "Slab size must be divisible by NT*V (one load per thread).");

    constexpr uint32_t SA_S = BK + 1, SB_S = BN;
    __shared__ T sA[BM * SA_S];
    __shared__ T sB[BK * SB_S];

    const uint32_t tid = threadIdx.x;
    const uint32_t tx  = tid % (BN / TM), ty = tid / (BN / TM);
    const uint32_t m0  = blockIdx.y * BM, n0 = blockIdx.x * BN;

    Acc acc[TM][TM] = {};

    const bool interior   = (m0 + BM <= M) && (n0 + BN <= N);
    const uint32_t K_bulk = K - K % BK;

    for (uint32_t r = 0; r < r_count; ++r) {
        // Rebase A/B for this reduce-dim slice. The ``n_rdims <= 1``
        // cases are specialized so they avoid the per-block %/div pair.
        if (batch.n_rdims == 0) {
            // No-op: A_base, B_base unchanged.
        } else if (batch.n_rdims == 1) {
            uint32_t dd = batch.n_bdims;
            A = A_base + (size_t) r * batch.a_stride[dd];
            B = B_base + (size_t) r * batch.b_stride[dd];
        } else {
            size_t a_off = 0, b_off = 0;
            uint32_t rz = r;
            for (uint32_t d = 0; d < batch.n_rdims; ++d) {
                uint32_t dd  = batch.n_bdims + d;
                uint32_t idx = rz % batch.extent[dd];
                rz /= batch.extent[dd];
                a_off += (size_t) idx * batch.a_stride[dd];
                b_off += (size_t) idx * batch.b_stride[dd];
            }
            A = A_base + a_off;
            B = B_base + b_off;
        }

        // Bulk K loop, specialized on whether the output tile lies fully
        // inside the matrix bounds.
        if (interior) {
            for (uint32_t k0 = 0; k0 < K_bulk; k0 += BK) {
                gemm_load<T, BM, BK, NT, SA_S, V, false, false>(At, A, sA, tid, m0, k0, M, K);
                gemm_load<T, BK, BN, NT, SB_S, V, false, false>(Bt, B, sB, tid, k0, n0, K, N);
                __syncthreads();
                gemm_compute<T, Acc, BK, SA_S, SB_S, BN, TM, V>(sA, sB, tx, ty, acc);
                __syncthreads();
            }
        } else {
            for (uint32_t k0 = 0; k0 < K_bulk; k0 += BK) {
                gemm_load<T, BM, BK, NT, SA_S, V, true,  false>(At, A, sA, tid, m0, k0, M, K);
                gemm_load<T, BK, BN, NT, SB_S, V, false, true >(Bt, B, sB, tid, k0, n0, K, N);
                __syncthreads();
                gemm_compute<T, Acc, BK, SA_S, SB_S, BN, TM, V>(sA, sB, tx, ty, acc);
                __syncthreads();
            }
        }

        // K tail: at most one partial slab, always bounds-checked along K.
        if (K_bulk < K) {
            if (interior) {
                gemm_load<T, BM, BK, NT, SA_S, V, false, true >(At, A, sA, tid, m0, K_bulk, M, K);
                gemm_load<T, BK, BN, NT, SB_S, V, true,  false>(Bt, B, sB, tid, K_bulk, n0, K, N);
            } else {
                gemm_load<T, BM, BK, NT, SA_S, V, true,  true >(At, A, sA, tid, m0, K_bulk, M, K);
                gemm_load<T, BK, BN, NT, SB_S, V, true,  true >(Bt, B, sB, tid, K_bulk, n0, K, N);
            }
            __syncthreads();
            gemm_compute<T, Acc, BK, SA_S, SB_S, BN, TM, V>(sA, sB, tx, ty, acc);
            __syncthreads();
        }
    }

    // Write the register tile back to C (narrowing float -> half if
    // applicable). The cyclic column mapping matches gemm_compute's sB
    // LDS, giving coalesced V-wide vector stores.
    constexpr uint32_t NT_n    = BN / TM;
    constexpr uint32_t StripeN = NT_n * V;

    for (uint32_t i = 0; i < TM; ++i) {
        uint32_t m_g = m0 + ty * TM + i;
        if (!interior && m_g >= M)
            continue;
        for (uint32_t jv = 0; jv < TM; jv += V) {
            uint32_t n_g = n0 + (jv / V) * StripeN + tx * V;
            if (interior) {
                T out[V];
                for (uint32_t v = 0; v < V; ++v)
                    out[v] = (T) acc[i][jv + v];
                gemm_vec_load<V>(out, &C[m_g * N + n_g]);
            } else {
                for (uint32_t v = 0; v < V; ++v)
                    if (n_g + v < N)
                        C[m_g * N + n_g + v] = (T) acc[i][jv + v];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel instantiations: 4 types x 4 tile sizes = 16 kernels, named
// ``gemm_<type>_<BM>`` (e.g. ``gemm_f32_16``). The launcher routes
// int32 to the u32 kernel (same bit pattern under two's-complement
// mul/add).

#define GEMM_TILE(T_, TName, BM_)                                              \
    KERNEL void gemm_##TName##_##BM_(                                          \
        const T_ *A, const T_ *B, T_ *C,                                       \
        uint32_t M, uint32_t N, uint32_t K,                                    \
        uint32_t At, uint32_t Bt, GemmBatch batch) {                           \
        gemm_impl<T_, BM_>(A, B, C, M, N, K, At, Bt, batch);                   \
    }

#define GEMM_TYPE(T_, TName)                                                   \
    GEMM_TILE(T_, TName,  8)                                                   \
    GEMM_TILE(T_, TName, 16)                                                   \
    GEMM_TILE(T_, TName, 32)                                                   \
    GEMM_TILE(T_, TName, 64)

GEMM_TYPE(half,     f16)
GEMM_TYPE(float,    f32)
GEMM_TYPE(double,   f64)
GEMM_TYPE(uint32_t, u32)
