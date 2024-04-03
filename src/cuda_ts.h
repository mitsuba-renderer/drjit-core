#include "eval.h"
#include "internal.h"
#include "log.h"

struct CUDAThreadState: ThreadState{

    Task *launch(Kernel kernel, uint32_t size, 
                 std::vector<ScheduledVariable> &scheduled) override;

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override;

    /// Reduce the given array to a single value
    void reduce(VarType type, ReduceOp rtype, const void *ptr, uint32_t size,
                void *out) override;

    /// 'All' reduction for boolean arrays
    bool all(uint8_t *values, uint32_t size) override;

    /// 'Any' reduction for boolean arrays
    bool any(uint8_t *values, uint32_t size) override;

    /// Exclusive prefix sum
    void prefix_sum(VarType vt, bool exclusive, const void *in, uint32_t size,
                    void *out) override;

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

    /// Sum over elements within blocks
    void block_sum(enum VarType type, const void *in, void *out, uint32_t size,
                   uint32_t block_size) override;

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override;

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override;

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override;

    /// LLVM: reduce a variable that was previously expanded due to dr.ReduceOp.Expand
    void reduce_expanded(VarType, ReduceOp, void *, uint32_t,
                         uint32_t) override {
      jitc_raise("jitc_reduce_expanded(): unsupported by CUDAThreadState!");
    }

    ~CUDAThreadState(){}
};
