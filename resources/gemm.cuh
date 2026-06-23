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

  Transpose handling: ``At`` / ``Bt`` are compile-time template
  parameters. Three kernel variants are instantiated per (type, tile)
  pair: ``_nn``, ``_nt``, ``_tn``. The ``At == Bt == true`` case is
  rewritten by the Python caller via ``A^T @ B^T = (B @ A)^T`` and is
  not implemented here.

  Smem layout:
    sA[BM][BK+1]  -- +1 pad makes per-row bank offsets coprime with 32, so
                     scalar LDS in gemm_compute is bank-conflict-free.
    sB[BK][SB_S]  -- SB_S = BN + V for BM>=32 (BN is a multiple of 32
                     for those tiles, giving 4-way bank conflicts on
                     the transposed-deposit path; the +V pad cuts that
                     to 2-way while keeping V-element vector reads
                     aligned). For BM<=16, SB_S = BN already gives a
                     clean bank distribution -- the +V pad would
                     wrap-around into already-used banks. Read pattern
                     in gemm_compute is conflict-free in either case.

  Bounds handling: edge blocks pay for per-lane M/N bounds checks in the
  K-loop. The branch predictor handles interior blocks (where checks are
  always true) at near-zero cost. The K tail runs as a single extra
  iteration with K-axis checks that zero OOB lanes.

  Register budget: each kernel is annotated with ``__launch_bounds__``
  to target a specific register count derived from the formula
  TM^2 + 4*TM + 32 (accumulators + ILP staging + misc), giving:
    BM= 8 -> 37 regs (24 blocks/SM, hw max),
    BM=16 -> 44 regs (23 blocks/SM),
    BM=32 -> 64 regs (16 blocks/SM),
    BM=64 -> 128 regs (8 blocks/SM).
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

// Load one operand tile into ``smem``, absorbing the transpose flag at
// compile time. The compute stage always sees a ``(BR, BC)``-shaped
// smem tile; when the global operand is transposed, we load a
// ``(BC, BR)`` slab and transpose-deposit instead. ``if constexpr``
// prunes the dead path so only one ``gemm_load_slab`` instantiation
// gets emitted per call site.
template <typename T, uint32_t BR, uint32_t BC, uint32_t NT,
          uint32_t SmemStride, uint32_t V,
          bool CheckR, bool CheckC, bool Transpose>
DEVICE FINLINE void gemm_load(const T *src, T *smem, uint32_t tid,
                              uint32_t r0, uint32_t c0,
                              uint32_t R, uint32_t C) {
    if constexpr (!Transpose)
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
          uint32_t BN, uint32_t TM, uint32_t V, uint32_t KU>
DEVICE FINLINE void gemm_compute(const T *sA, const T *sB,
                                 uint32_t tx, uint32_t ty, Acc acc[TM][TM]) {
    constexpr uint32_t NT_n    = BN / TM;
    constexpr uint32_t StripeN = NT_n * V;

    #pragma unroll KU
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

template <typename T, uint32_t BM, bool At, bool Bt>
DEVICE FINLINE void gemm_impl(const T *__restrict__ A,
                              const T *__restrict__ B,
                              T *__restrict__ C,
                              uint32_t M, uint32_t N, uint32_t K,
                              const GemmBatch &batch) {
    using Acc = std::conditional_t<std::is_same<T, half>::value, float, T>;

    // Per-block grid offset. ``blockIdx.z`` decomposes over the grid dims
    // ``[0, n_bdims)``. Broadcast along a grid dim is encoded by
    // ``a_stride[d] = 0`` or ``b_stride[d] = 0``. Output C advances along
    // grid dims only, with implicit stride ``M * N`` per grid step.
    // ``#pragma unroll 1`` keeps the loop compact: NVCC would otherwise
    // unroll it over ``DRJIT_GEMM_MAX_BDIMS=6`` predicated iterations,
    // each with its own constant-memory loads.
    {
        uint32_t z = blockIdx.z;
        size_t a_off = 0, b_off = 0;
        #pragma unroll 1
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
    #pragma unroll 1
    for (uint32_t d = 0; d < batch.n_rdims; ++d)
        r_count *= batch.extent[batch.n_bdims + d];

    constexpr uint32_t BN   = BM, BK = 16;
    constexpr uint32_t TM   = BM / 8;
    constexpr uint32_t NT   = (BM / TM) * (BN / TM);
    constexpr uint32_t Vmax = 16 / sizeof(T);
    constexpr uint32_t V    = Vmax < TM ? Vmax : TM;

    // K-loop unroll in gemm_compute. float is compute-bound and needs full
    // ILP; half (LDS-bound) and double (FP64-bound) keep throughput at lower
    // factors, shrinking their PTX (the bulk of it). 2 beats 1 for f64.
    constexpr uint32_t KU = std::is_same<T, float>::value  ? BK
                          : std::is_same<T, double>::value ? 2
                                                           : 4;

    static_assert(NT == 64, "Kernel assumes an 8x8 thread grid (64 threads).");
    static_assert(BM == 8 || BM == 16 || BM == 32 || BM == 64,
                  "BM must be one of {8, 16, 32, 64}.");
    static_assert(BK % V == 0 && BM % V == 0,
                  "Tile dimensions must be divisible by the vector width.");
    static_assert((BM * BK) % (NT * V) == 0,
                  "Slab size must be divisible by NT*V (one load per thread).");

    // SB_S: +V pad for BM>=32 cuts transposed sB store conflicts from
    // 4-way to 2-way and is neutral for non-transposed. For BM<=16 the
    // unpadded BN gives a clean bank distribution, so adding the pad
    // would actually introduce wrap-around conflicts; keep it natural.
    constexpr uint32_t SA_S = BK + 1,
                       SB_S = (BM <= 16) ? BN : BN + V;
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
            #pragma unroll 1
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

        // Bulk K loop. Bounds checks always fire on the M/N axes; the
        // branch predictor folds them to a no-op for interior blocks.
        for (uint32_t k0 = 0; k0 < K_bulk; k0 += BK) {
            gemm_load<T, BM, BK, NT, SA_S, V, true, false, At>(A, sA, tid, m0, k0, M, K);
            gemm_load<T, BK, BN, NT, SB_S, V, false, true, Bt>(B, sB, tid, k0, n0, K, N);
            __syncthreads();
            gemm_compute<T, Acc, BK, SA_S, SB_S, BN, TM, V, KU>(sA, sB, tx, ty, acc);
            __syncthreads();
        }

        // K tail: at most one partial slab, always bounds-checked along K.
        if (K_bulk < K) {
            gemm_load<T, BM, BK, NT, SA_S, V, true, true, At>(A, sA, tid, m0, K_bulk, M, K);
            gemm_load<T, BK, BN, NT, SB_S, V, true, true, Bt>(B, sB, tid, K_bulk, n0, K, N);
            __syncthreads();
            gemm_compute<T, Acc, BK, SA_S, SB_S, BN, TM, V, KU>(sA, sB, tx, ty, acc);
            __syncthreads();
        }
    }

    // Write the register tile back to C (narrowing float -> half if
    // applicable). The cyclic column mapping matches gemm_compute's sB
    // LDS, giving coalesced V-wide vector stores. Interior blocks use
    // vector writes; edge blocks fall back to scalar with N-axis bounds.
    constexpr uint32_t NT_n    = BN / TM;
    constexpr uint32_t StripeN = NT_n * V;

    if (interior) {
        for (uint32_t i = 0; i < TM; ++i) {
            uint32_t m_g = m0 + ty * TM + i;
            for (uint32_t jv = 0; jv < TM; jv += V) {
                uint32_t n_g = n0 + (jv / V) * StripeN + tx * V;
                T out[V];
                for (uint32_t v = 0; v < V; ++v)
                    out[v] = (T) acc[i][jv + v];
                gemm_vec_load<V>(out, &C[m_g * N + n_g]);
            }
        }
    } else {
        for (uint32_t i = 0; i < TM; ++i) {
            uint32_t m_g = m0 + ty * TM + i;
            if (m_g >= M)
                continue;
            for (uint32_t jv = 0; jv < TM; jv += V) {
                uint32_t n_g = n0 + (jv / V) * StripeN + tx * V;
                for (uint32_t v = 0; v < V; ++v)
                    if (n_g + v < N)
                        C[m_g * N + n_g + v] = (T) acc[i][jv + v];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel instantiations: 3 types x 4 tile sizes x 3 transpose variants
// = 36 kernels, named ``gemm_<type>_<BM>_<tt>`` where ``<tt>`` is one of
// ``nn``, ``nt``, ``tn``. The ``tt`` case (At=Bt=true) is rewritten by
// the caller via ``A^T @ B^T = (B @ A)^T``.
//
// ``__launch_bounds__(64, GEMM_BLOCKS(BM))`` caps registers at the
// computed budget. With 65536 regs/SM and 64 threads/block, asking for
// N blocks/SM tells ptxas to target 65536/(64*N) = 1024/N registers per
// thread. Combined with ``__grid_constant__`` on the GemmBatch
// parameter (eliminates the local-memory copy of the batch struct),
// the resulting kernels hit:
//   TM^2 + 4*TM + 32 = 37/44/64/128 regs for BM = 8/16/32/64.

#define GEMM_REGS(BM)   ((BM/8)*(BM/8) + 4*(BM/8) + 32)
#define GEMM_BLOCKS(BM) (1024u / GEMM_REGS(BM))

#define GEMM_VARIANT(T_, TName, BM_, At_, Bt_, Suffix)                         \
    KERNEL __launch_bounds__(64, GEMM_BLOCKS(BM_))                             \
    void gemm_##TName##_##BM_##Suffix(                                         \
        const T_ *A, const T_ *B, T_ *C,                                       \
        uint32_t M, uint32_t N, uint32_t K,                                    \
        const __grid_constant__ GemmBatch batch) {                             \
        gemm_impl<T_, BM_, At_, Bt_>(A, B, C, M, N, K, batch);                 \
    }

#define GEMM_TILE(T_, TName, BM_)                                              \
    GEMM_VARIANT(T_, TName, BM_, false, false, _nn)                            \
    GEMM_VARIANT(T_, TName, BM_, false, true,  _nt)                            \
    GEMM_VARIANT(T_, TName, BM_, true,  false, _tn)

#define GEMM_TYPE(T_, TName)                                                   \
    GEMM_TILE(T_, TName,  8)                                                   \
    GEMM_TILE(T_, TName, 16)                                                   \
    GEMM_TILE(T_, TName, 32)                                                   \
    GEMM_TILE(T_, TName, 64)

GEMM_TYPE(half,   f16)
GEMM_TYPE(float,  f32)
GEMM_TYPE(double, f64)
