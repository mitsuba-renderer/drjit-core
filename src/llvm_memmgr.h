#pragma once

#include "llvm_api.h"

/// Memory-manager state of one LLVM compiler instance (see llvm_orcv2.cpp).
/// Generated code and data sections accumulate in a bump-allocated buffer,
/// whose contents form the position-independent unit image.
struct LLVMMemMgrContext {
    /// Internal storage used by the memory manager
    uint8_t *data = nullptr;

    /// Size of the buffer backing 'data'
    size_t size = 0;

    /// Current position within 'data'
    size_t offset = 0;

    /// Was a global offset table (GOT) generated?
    bool got = false;
};

/// A self-contained LLVM compiler instance: parsing context, target machine,
/// LLJIT, and a private memory manager. Since LLVMContext is not thread-safe,
/// each in-flight unit compilation checks one out of a shared pool (see
/// llvm_orcv2.cpp); instances are created lazily and reused across kernels.
struct LLVMCompiler {
    /// Parsing context
    LLVMContextRef context = nullptr;

    /// Target machine driving the optimization pipeline
    LLVMTargetMachineRef tm = nullptr;

    /// LLJIT instance and its main dylib
    LLVMOrcLLJITRef lljit = nullptr;
    LLVMOrcJITDylibRef dylib = nullptr;

    /// Bump allocator receiving the generated code and data sections
    LLVMMemMgrContext memmgr;
};

/// Prepare the memory manager for IR of a given size
extern void jitc_llvm_memmgr_prepare(LLVMMemMgrContext &ctx, size_t size);

/// Release resources held by the given memory manager context
extern void jitc_llvm_memmgr_release(LLVMMemMgrContext &ctx);

/// -------------- LLVM C-API memory manager callbacks --------------
/// The 'opaque' argument is the LLVMMemMgrContext of the compiler instance,
/// threaded through the object-linking-layer creator (see llvm_orcv2.cpp).

extern uint8_t *jitc_llvm_memmgr_allocate(void *, uintptr_t, unsigned, unsigned, const char *);
extern uint8_t *jitc_llvm_memmgr_allocate_data(void *, uintptr_t, unsigned,
                                               unsigned, const char *, LLVMBool);
extern LLVMBool jitc_llvm_memmgr_finalize(void *, char **);
extern void jitc_llvm_memmgr_destroy(void *);
extern void* jitc_llvm_memmgr_create_context(void *);
extern void jitc_llvm_memmgr_notify_terminating(void *);
