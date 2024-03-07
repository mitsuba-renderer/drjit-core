#include "internal.h"

struct CUDAThreadState: ThreadState{

    /// Perform a synchronous copy operation
    void jitc_memcpy(void *dst, const void *src, size_t size) override;

    ~CUDAThreadState(){}
};
