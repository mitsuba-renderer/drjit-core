#pragma once

#  if defined(__aarch64__)
      #define LLVMInitializeDrJitTarget       LLVMInitializeAArch64Target
      #define LLVMInitializeDrJitTargetInfo   LLVMInitializeAArch64TargetInfo
      #define LLVMInitializeDrJitTargetMC     LLVMInitializeAArch64TargetMC
      #define LLVMInitializeDrJitAsmPrinter   LLVMInitializeAArch64AsmPrinter
      #define LLVMInitializeDrJitDisassembler LLVMInitializeAArch64Disassembler
#  else
      #define LLVMInitializeDrJitTarget       LLVMInitializeX86Target
      #define LLVMInitializeDrJitTargetInfo   LLVMInitializeX86TargetInfo
      #define LLVMInitializeDrJitTargetMC     LLVMInitializeX86TargetMC
      #define LLVMInitializeDrJitAsmPrinter   LLVMInitializeX86AsmPrinter
      #define LLVMInitializeDrJitDisassembler LLVMInitializeX86Disassembler
#  endif

#if !defined(DRJIT_DYNAMIC_LLVM)
#  include <llvm-c/Core.h>
#  include <llvm-c/ExecutionEngine.h>
#  include <llvm-c/Disassembler.h>
#  include <llvm-c/IRReader.h>
#  include <llvm-c/Analysis.h>
#  include <llvm-c/Transforms/Scalar.h>
#  include <llvm-c/LLJIT.h>
#  include <llvm-c/OrcEE.h>
#else
#  include <stdint.h>
#  include <stdlib.h>
#  define LLVMDisassembler_Option_PrintImmHex       2
#  define LLVMDisassembler_Option_AsmPrinterVariant 4
#  define LLVMReturnStatusAction 2
#  define LLVMCodeGenLevelAggressive 3
#  define LLVMRelocPIC 2
#  define LLVMCodeModelSmall 3

/// LLVM API
using LLVMBool = int;
using LLVMDisasmContextRef = void *;
using LLVMExecutionEngineRef = void *;
using LLVMModuleRef = void *;
using LLVMMemoryBufferRef = void *;
using LLVMContextRef = void *;
using LLVMPassManagerRef = void *;
using LLVMPassManagerBuilderRef = void *;
using LLVMMCJITMemoryManagerRef = void *;
using LLVMTargetMachineRef = void *;
using LLVMTargetRef = void *;
using LLVMCodeModel = int;
using LLVMRelocMode = int;
using LLVMCodeGenOptLevel = int;
using LLVMOrcThreadSafeContextRef = void*;
using LLVMOrcThreadSafeModuleRef = void*;
using LLVMOrcObjectLayerRef = void*;
using LLVMOrcExecutionSessionRef = void*;
using LLVMOrcJITTargetMachineBuilderRef = void*;
using LLVMOrcLLJITBuilderRef = void*;
using LLVMOrcLLJITRef = void*;
using LLVMErrorRef = void*;
using LLVMOrcJITDylibRef = void *;
using LLVMOrcExecutorAddress = uint64_t;

using LLVMMemoryManagerAllocateCodeSectionCallback =
    uint8_t *(*) (void *, uintptr_t, unsigned, unsigned, const char *);
using LLVMMemoryManagerAllocateDataSectionCallback =
    uint8_t *(*) (void *, uintptr_t, unsigned, unsigned, const char *,
                  LLVMBool);
using LLVMMemoryManagerFinalizeMemoryCallback = LLVMBool (*)(void *, char **);
using LLVMMemoryManagerDestroyCallback = void (*)(void *);
using LLVMMemoryManagerCreateContextCallback = void* (*)(void *);
using LLVMMemoryManagerNotifyTerminatingCallback = void(void *);
using LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction =
    LLVMOrcObjectLayerRef(void *, LLVMOrcExecutionSessionRef, const char *);

struct LLVMMCJITCompilerOptions {
  unsigned OptLevel;
  LLVMCodeModel CodeModel;
  LLVMBool NoFramePointerElim;
  LLVMBool EnableFastISel;
  void *MCJMM;
};

#if !defined(DRJIT_SYMBOL)
#  define DRJIT_SYMBOL(x) extern x;
#endif

DRJIT_SYMBOL(void (*LLVMLinkInMCJIT)());
DRJIT_SYMBOL(void (*LLVMInitializeDrJitAsmPrinter)());
DRJIT_SYMBOL(void (*LLVMInitializeDrJitDisassembler)());
DRJIT_SYMBOL(void (*LLVMInitializeDrJitTarget)());
DRJIT_SYMBOL(void (*LLVMInitializeDrJitTargetInfo)());
DRJIT_SYMBOL(void (*LLVMInitializeDrJitTargetMC)());
DRJIT_SYMBOL(char *(*LLVMCreateMessage)(const char *));
DRJIT_SYMBOL(void (*LLVMDisposeMessage)(char *));
DRJIT_SYMBOL(char *(*LLVMGetDefaultTargetTriple)());
DRJIT_SYMBOL(char *(*LLVMGetHostCPUName)());
DRJIT_SYMBOL(char *(*LLVMGetHostCPUFeatures)());
DRJIT_SYMBOL(LLVMContextRef (*LLVMGetGlobalContext)());
DRJIT_SYMBOL(LLVMDisasmContextRef (*LLVMCreateDisasm)(const char *, void *, int,
                                                void *, void *));
DRJIT_SYMBOL(void (*LLVMDisasmDispose)(LLVMDisasmContextRef));
DRJIT_SYMBOL(int (*LLVMSetDisasmOptions)(LLVMDisasmContextRef, uint64_t));
DRJIT_SYMBOL(void (*LLVMAddModule)(LLVMExecutionEngineRef, LLVMModuleRef));
DRJIT_SYMBOL(void (*LLVMDisposeModule)(LLVMModuleRef));
DRJIT_SYMBOL(LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool));
DRJIT_SYMBOL(LLVMBool (*LLVMParseIRInContext)(LLVMContextRef, LLVMMemoryBufferRef,
                                        LLVMModuleRef *, char **));
DRJIT_SYMBOL(char *(*LLVMPrintModuleToString)(LLVMModuleRef));
DRJIT_SYMBOL(uint64_t (*LLVMGetGlobalValueAddress)(LLVMExecutionEngineRef, const char *));
DRJIT_SYMBOL(LLVMBool (*LLVMRemoveModule)(LLVMExecutionEngineRef, LLVMModuleRef,
                                    LLVMModuleRef *, char **));
DRJIT_SYMBOL(size_t (*LLVMDisasmInstruction)(LLVMDisasmContextRef, uint8_t *,
                                       uint64_t, uint64_t, char *,
                                       size_t));
DRJIT_SYMBOL(LLVMPassManagerRef (*LLVMCreatePassManager)());
DRJIT_SYMBOL(void (*LLVMRunPassManager)(LLVMPassManagerRef, LLVMModuleRef));
DRJIT_SYMBOL(void (*LLVMDisposePassManager)(LLVMPassManagerRef));
DRJIT_SYMBOL(void (*LLVMAddLICMPass)(LLVMPassManagerRef));
DRJIT_SYMBOL(LLVMPassManagerBuilderRef (*LLVMPassManagerBuilderCreate)());
DRJIT_SYMBOL(void (*LLVMPassManagerBuilderSetOptLevel)(
    LLVMPassManagerBuilderRef, unsigned));
DRJIT_SYMBOL(void (*LLVMPassManagerBuilderPopulateModulePassManager)(
    LLVMPassManagerBuilderRef, LLVMPassManagerRef));
DRJIT_SYMBOL(void (*LLVMPassManagerBuilderDispose)(LLVMPassManagerBuilderRef));
DRJIT_SYMBOL(bool (*LLVMVerifyModule)(LLVMModuleRef, int action, char **msg));
DRJIT_SYMBOL(void (*LLVMGetVersion)(unsigned*, unsigned*, unsigned*));

// API for MCJIT interface
DRJIT_SYMBOL(LLVMModuleRef (*LLVMModuleCreateWithName)(const char *));
DRJIT_SYMBOL(LLVMTargetMachineRef (*LLVMGetExecutionEngineTargetMachine)(
    LLVMExecutionEngineRef));
DRJIT_SYMBOL(LLVMBool (*LLVMCreateMCJITCompilerForModule)(
    LLVMExecutionEngineRef *, LLVMModuleRef, LLVMMCJITCompilerOptions *, size_t,
    char **));
DRJIT_SYMBOL(LLVMMCJITMemoryManagerRef (*LLVMCreateSimpleMCJITMemoryManager)(
    void *, LLVMMemoryManagerAllocateCodeSectionCallback,
    LLVMMemoryManagerAllocateDataSectionCallback,
    LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback));
DRJIT_SYMBOL(void (*LLVMDisposeExecutionEngine)(LLVMExecutionEngineRef));
DRJIT_SYMBOL(uint64_t (*LLVMGetFunctionAddress)(LLVMExecutionEngineRef, const char *));

// API for ORCv2 interface
DRJIT_SYMBOL(LLVMBool (*LLVMGetTargetFromTriple)(const char *, LLVMTargetRef *,
                                                 char **));
DRJIT_SYMBOL(LLVMTargetMachineRef (*LLVMCreateTargetMachine)(
    LLVMTargetRef, const char *, const char *, const char *,
    LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel));
DRJIT_SYMBOL(LLVMOrcThreadSafeContextRef (*LLVMOrcCreateNewThreadSafeContext)());
DRJIT_SYMBOL(LLVMOrcThreadSafeModuleRef (*LLVMOrcCreateNewThreadSafeModule)(
    LLVMModuleRef, LLVMOrcThreadSafeContextRef));
DRJIT_SYMBOL(
    void (*LLVMOrcDisposeThreadSafeContext)(LLVMOrcThreadSafeContextRef));
DRJIT_SYMBOL(LLVMOrcJITTargetMachineBuilderRef (
    *LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine)(
    LLVMTargetMachineRef));
DRJIT_SYMBOL(LLVMOrcLLJITBuilderRef (*LLVMOrcCreateLLJITBuilder)());
DRJIT_SYMBOL(void (*LLVMOrcLLJITBuilderSetJITTargetMachineBuilder)(
    LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef));
DRJIT_SYMBOL(LLVMErrorRef (*LLVMOrcCreateLLJIT)(LLVMOrcLLJITRef *,
                                                LLVMOrcLLJITBuilderRef));
DRJIT_SYMBOL(char *(*LLVMGetErrorMessage)(LLVMErrorRef));
DRJIT_SYMBOL(LLVMErrorRef (*LLVMOrcLLJITAddLLVMIRModule)(
    LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMOrcThreadSafeModuleRef));
DRJIT_SYMBOL(LLVMErrorRef (*LLVMOrcLLJITLookup)(LLVMOrcLLJITRef,
                                                LLVMOrcExecutorAddress *,
                                                const char *));
DRJIT_SYMBOL(
    LLVMOrcJITDylibRef (*LLVMOrcLLJITGetMainJITDylib)(LLVMOrcLLJITRef));
DRJIT_SYMBOL(void (*LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator)(
    LLVMOrcLLJITBuilderRef,
    LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction, void *));
DRJIT_SYMBOL(LLVMOrcObjectLayerRef (
    *LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks)(
    LLVMOrcExecutionSessionRef, void *,
    LLVMMemoryManagerCreateContextCallback,
    LLVMMemoryManagerNotifyTerminatingCallback,
    LLVMMemoryManagerAllocateCodeSectionCallback,
    LLVMMemoryManagerAllocateDataSectionCallback,
    LLVMMemoryManagerFinalizeMemoryCallback,
    LLVMMemoryManagerDestroyCallback));
DRJIT_SYMBOL(LLVMErrorRef (*LLVMOrcDisposeLLJIT)(LLVMOrcLLJITRef));
DRJIT_SYMBOL(LLVMErrorRef (*LLVMOrcJITDylibClear)(LLVMOrcJITDylibRef));
#endif
