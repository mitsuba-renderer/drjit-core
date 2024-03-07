#include "llvm_internal.h"

void LLVMThreadState::jitc_memcpy(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
}
