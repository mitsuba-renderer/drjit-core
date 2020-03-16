#pragma once

#include <enoki/jit.h>

using LLVMBool = int;
using LLVMDisasmContextRef = void *;
using LLVMExecutionEngineRef = void *;
using LLVMModuleRef = void *;
using LLVMMemoryBufferRef = void *;
using LLVMContextRef = void *;

// LLVM C API
extern size_t (*LLVMDisasmInstruction)(LLVMDisasmContextRef, uint8_t *, uint64_t,
                                uint64_t, char *, size_t);
extern char *(*LLVMPrintModuleToString)(LLVMModuleRef);
extern uint64_t (*LLVMGetFunctionAddress)(LLVMExecutionEngineRef, const char *);
extern LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool);
extern void (*LLVMAddModule)(LLVMExecutionEngineRef, LLVMModuleRef);
extern LLVMBool (*LLVMParseIRInContext)(LLVMContextRef, LLVMMemoryBufferRef,
                                 LLVMModuleRef *, char **);

// Enoki API
extern LLVMDisasmContextRef jit_llvm_disasm;
extern LLVMExecutionEngineRef jit_llvm_engine;
extern LLVMContextRef jit_llvm_context;
extern char *jit_llvm_target_cpu;

/// Try to load the LLVM backend
extern bool jit_llvm_init();

/// Fully unload LLVM
extern void jit_llvm_shutdown();
