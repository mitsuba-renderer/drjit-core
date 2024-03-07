#include "internal.h"


struct LLVMThreadState: ThreadState{

    /// Perform a synchronous copy operation
    void jitc_memcpy(void *dst, const void *src, size_t size) override;
    
    ~LLVMThreadState(){}
};
