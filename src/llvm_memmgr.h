#pragma once

#include "llvm_api.h"

/// Internal storage used by the memory manager
extern uint8_t *jitc_llvm_memmgr_data;

/// Current position within 'jitc_llvm_memmgr_data'
extern size_t jitc_llvm_memmgr_offset;

/// Was a global offset table (GOT) generated?
extern bool jitc_llvm_memmgr_got;

/// Prepare the LLVM compilation memory manager for IR of a given size
extern void jitc_llvm_memmgr_prepare(size_t size);

/// Release resources held by the LLVM compilation memory manager
extern void jitc_llvm_memmgr_shutdown();

/// -------------- LLVM C-API memory manager callbacks --------------

extern uint8_t *jitc_llvm_memmgr_allocate(void *, uintptr_t, unsigned, unsigned, const char *);
extern uint8_t *jitc_llvm_memmgr_allocate_data(void *, uintptr_t, unsigned,
                                               unsigned, const char *, LLVMBool);
extern LLVMBool jitc_llvm_memmgr_finalize(void *, char **);
extern void jitc_llvm_memmgr_destroy(void *);
extern void* jitc_llvm_memmgr_create_context(void *);
extern void jitc_llvm_memmgr_notify_terminating(void *);
