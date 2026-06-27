#include "internal.h"

#if defined(DRJIT_ENABLE_AMD)

struct AMDThreadState final : ThreadState {
    Task *launch(Kernel kernel, KernelKey &key, XXH128_hash_t hash,
                 uint32_t size, std::vector<void *> &kernel_params,
                 const std::vector<uint32_t> &kernel_param_ids,
                 KernelHistoryEntry *kernel_history_entry) override;

    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override;

    void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                      uint32_t block_size, const void *in, void *out) override;

    void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, bool exclusive, bool reverse,
                             const void *in, void *out) override;

    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override;

    void batched_gemm(VarType vt, bool At, bool Bt,
                      uint32_t M, uint32_t N, uint32_t K,
                      const GemmBatch *batch,
                      const void *A, const void *B, void *C) override;

    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override;

    uint32_t block_mkperm(const uint32_t *values, uint32_t size,
                          uint32_t block_size, uint32_t bucket_count,
                          uint32_t *perm, uint32_t *offsets) override;

    void memcpy(void *dst, const void *src, size_t size) override;

    void memcpy_async(void *dst, const void *src, size_t size) override;

    void poke(void *dst, const void *src, uint32_t size) override;

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override;

    void enqueue_host_func(void (*callback)(void *), void *payload) override;

    void barrier() override;

    void flush_deferred_free() override;

    void coop_vec_pack(uint32_t count, const void *in,
                       const MatrixDescr *in_d, void *out,
                       const MatrixDescr *out_d) override;
};

#endif
