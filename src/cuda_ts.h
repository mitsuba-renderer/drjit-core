#include "internal.h"
#include "log.h"

struct CUDAThreadState : ThreadState {
    Task *launch(Kernel kernel, KernelKey *key, XXH128_hash_t hash,
                 uint32_t size, std::vector<void *> *kernel_params,
                 const std::vector<uint32_t> *kernel_param_ids) override;

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override;

    /// Reduce elements within blocks
    void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                      uint32_t block_size, const void *in, void *out) override;

    /// Implements various kinds of prefix reductions
    void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, bool exclusive, bool reverse,
                             const void *in, void *out) override;

    /// Compute a dot product of two equal-sized arrays
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override;

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override;

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm,
                    uint32_t *offsets) override;

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override;

    /// Perform an assynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override;

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override;

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override;

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override;

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType, ReduceOp, void *, uint32_t,
                         uint32_t) override {
        jitc_raise("jitc_reduce_expanded(): unsupported by CUDAThreadState!");
    }

    /// Pack a set of matrices/vectors for use with the cooperative vector API
    void coop_vec_pack(uint32_t count, const void *in, const MatrixDescr *in_d,
                       void *out, const MatrixDescr *out_d) override;
};
