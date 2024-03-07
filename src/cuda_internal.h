#include "internal.h"

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

    ~CUDAThreadState(){}
};
