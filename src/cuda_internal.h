#include "internal.h"
#include "log.h"

struct CUDAThreadState: ThreadState{
    
    /// Fill a device memory region with constants of a given type
    void jitc_memset_async(void *ptr, uint32_t size,
                                   uint32_t isize, const void *src) override;
    
    /// Reduce the given array to a single value
    void jitc_reduce(VarType type, ReduceOp rtype, const void *ptr,
                             uint32_t size, void *out) override;
    
    /// 'All' reduction for boolean arrays
    bool jitc_all(uint8_t *values, uint32_t size) override;

    /// 'Any' reduction for boolean arrays
    bool jitc_any(uint8_t *values, uint32_t size) override;
    
    /// Exclusive prefix sum
    void jitc_prefix_sum(VarType vt, bool exclusive, const void *in,
                                 uint32_t size, void *out) override;
    
    /// Mask compression
    uint32_t jitc_compress(const uint8_t *in, uint32_t size,
                                   uint32_t *out) override;
    
    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t jitc_mkperm(const uint32_t *values, uint32_t size,
                                 uint32_t bucket_count, uint32_t *perm,
                                 uint32_t *offsets) override;

    /// Perform a synchronous copy operation
    void jitc_memcpy(void *dst, const void *src, size_t size) override;
    
    /// Perform an assynchronous copy operation
    void jitc_memcpy_async(void *dst, const void *src, size_t size) override;
    
    /// Replicate individual input elements to larger blocks
    void jitc_block_copy(enum VarType type, const void *in, void *out,
                                 uint32_t size, uint32_t block_size) override;

    /// Sum over elements within blocks
    void jitc_block_sum(enum VarType type, const void *in, void *out,
                                uint32_t size, uint32_t block_size) override;
    
    /// Asynchronously update a single element in memory
    void jitc_poke(void *dst, const void *src, uint32_t size) override;

    void jitc_aggregate(void *dst, AggregationEntry *agg,
                                uint32_t size) override;

    // Enqueue a function to be run on the host once backend computation is done
    void jitc_enqueue_host_func(void (*callback)(void *),
                                        void *payload) override;
    
    /// LLVM: reduce a variable that was previously expanded due to dr.ReduceOp.Expand
    void jitc_reduce_expanded(VarType vt, ReduceOp op, void *data, uint32_t exp, uint32_t size) override{
        jitc_raise("jitc_reduce_expanded(): unsupported by CUDAThreadState!");
    }

    ~CUDAThreadState(){}
};
