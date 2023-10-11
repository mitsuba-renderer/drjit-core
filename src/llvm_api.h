/*
    src/cuda_api.h -- Low-level interface to the LLVM C API

    Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

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
#  if LLVM_VERSION_MAJOR >= 15
#    include <llvm-c/Transforms/PassBuilder.h>
#  endif
#  if LLVM_VERSION_MAJOR < 17
#    include <llvm-c/Transforms/Scalar.h>
#  endif
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
using LLVMPassBuilderOptionsRef = void *;
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

#if !defined(DR_LLVM_SYM)
#  define DR_LLVM_SYM(x) extern x;
#endif

DR_LLVM_SYM(void (*LLVMLinkInMCJIT)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitAsmPrinter)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitDisassembler)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitTarget)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitTargetInfo)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitTargetMC)());
DR_LLVM_SYM(char *(*LLVMCreateMessage)(const char *) );
DR_LLVM_SYM(void (*LLVMDisposeMessage)(char *));
DR_LLVM_SYM(char *(*LLVMGetDefaultTargetTriple)());
DR_LLVM_SYM(char *(*LLVMGetHostCPUName)());
DR_LLVM_SYM(char *(*LLVMGetHostCPUFeatures)());
DR_LLVM_SYM(LLVMContextRef (*LLVMGetGlobalContext)());
DR_LLVM_SYM(LLVMDisasmContextRef (*LLVMCreateDisasm)(const char *, void *, int,
                                                     void *, void *));
DR_LLVM_SYM(void (*LLVMDisasmDispose)(LLVMDisasmContextRef));
DR_LLVM_SYM(int (*LLVMSetDisasmOptions)(LLVMDisasmContextRef, uint64_t));
DR_LLVM_SYM(void (*LLVMAddModule)(LLVMExecutionEngineRef, LLVMModuleRef));
DR_LLVM_SYM(void (*LLVMDisposeModule)(LLVMModuleRef));
DR_LLVM_SYM(LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool));
DR_LLVM_SYM(LLVMBool (*LLVMParseIRInContext)(LLVMContextRef,
                                             LLVMMemoryBufferRef,
                                             LLVMModuleRef *, char **));
DR_LLVM_SYM(char *(*LLVMPrintModuleToString)(LLVMModuleRef));
DR_LLVM_SYM(uint64_t (*LLVMGetGlobalValueAddress)(LLVMExecutionEngineRef,
                                                  const char *));
DR_LLVM_SYM(LLVMBool (*LLVMRemoveModule)(LLVMExecutionEngineRef, LLVMModuleRef,
                                         LLVMModuleRef *, char **));
DR_LLVM_SYM(size_t (*LLVMDisasmInstruction)(LLVMDisasmContextRef, uint8_t *,
                                            uint64_t, uint64_t, char *,
                                            size_t));
DR_LLVM_SYM(bool (*LLVMVerifyModule)(LLVMModuleRef, int action, char **msg));
DR_LLVM_SYM(void (*LLVMGetVersion)(unsigned *, unsigned *, unsigned *));
DR_LLVM_SYM(void (*LLVMDisposeTargetMachine)(LLVMTargetMachineRef));

// Legacy pass manager
DR_LLVM_SYM(LLVMPassManagerRef (*LLVMCreatePassManager)());
DR_LLVM_SYM(void (*LLVMRunPassManager)(LLVMPassManagerRef, LLVMModuleRef));
DR_LLVM_SYM(void (*LLVMDisposePassManager)(LLVMPassManagerRef));
DR_LLVM_SYM(void (*LLVMAddLICMPass)(LLVMPassManagerRef));

// New pass manager
DR_LLVM_SYM(LLVMPassBuilderOptionsRef (*LLVMCreatePassBuilderOptions)());
DR_LLVM_SYM(void (*LLVMPassBuilderOptionsSetLoopVectorization)(LLVMPassBuilderOptionsRef, LLVMBool));
DR_LLVM_SYM(void (*LLVMPassBuilderOptionsSetLoopUnrolling)(LLVMPassBuilderOptionsRef, LLVMBool));
DR_LLVM_SYM(void (*LLVMPassBuilderOptionsSetSLPVectorization)(LLVMPassBuilderOptionsRef, LLVMBool));
DR_LLVM_SYM(void (*LLVMDisposePassBuilderOptions)(LLVMPassBuilderOptionsRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMRunPasses)(LLVMModuleRef, const char *,
                                          LLVMTargetMachineRef,
                                          LLVMPassBuilderOptionsRef));

// API for MCJIT interface
DR_LLVM_SYM(LLVMModuleRef (*LLVMModuleCreateWithName)(const char *));
DR_LLVM_SYM(LLVMTargetMachineRef (*LLVMGetExecutionEngineTargetMachine)(
    LLVMExecutionEngineRef));
DR_LLVM_SYM(LLVMBool (*LLVMCreateMCJITCompilerForModule)(
    LLVMExecutionEngineRef *, LLVMModuleRef, LLVMMCJITCompilerOptions *, size_t,
    char **));
DR_LLVM_SYM(LLVMMCJITMemoryManagerRef (*LLVMCreateSimpleMCJITMemoryManager)(
    void *, LLVMMemoryManagerAllocateCodeSectionCallback,
    LLVMMemoryManagerAllocateDataSectionCallback,
    LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback));
DR_LLVM_SYM(void (*LLVMDisposeExecutionEngine)(LLVMExecutionEngineRef));
DR_LLVM_SYM(uint64_t (*LLVMGetFunctionAddress)(LLVMExecutionEngineRef,
                                               const char *));

// API for ORCv2 interface
DR_LLVM_SYM(LLVMBool (*LLVMGetTargetFromTriple)(const char *, LLVMTargetRef *,
                                                char **));
DR_LLVM_SYM(LLVMTargetMachineRef (*LLVMCreateTargetMachine)(
    LLVMTargetRef, const char *, const char *, const char *,
    LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel));
DR_LLVM_SYM(LLVMOrcThreadSafeContextRef (*LLVMOrcCreateNewThreadSafeContext)());
DR_LLVM_SYM(LLVMOrcThreadSafeModuleRef (*LLVMOrcCreateNewThreadSafeModule)(
    LLVMModuleRef, LLVMOrcThreadSafeContextRef));
DR_LLVM_SYM(
    void (*LLVMOrcDisposeThreadSafeContext)(LLVMOrcThreadSafeContextRef));
DR_LLVM_SYM(LLVMOrcJITTargetMachineBuilderRef (
    *LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine)(
    LLVMTargetMachineRef));
DR_LLVM_SYM(LLVMOrcLLJITBuilderRef (*LLVMOrcCreateLLJITBuilder)());
DR_LLVM_SYM(void (*LLVMOrcLLJITBuilderSetJITTargetMachineBuilder)(
    LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcCreateLLJIT)(LLVMOrcLLJITRef *,
                                               LLVMOrcLLJITBuilderRef));
DR_LLVM_SYM(char *(*LLVMGetErrorMessage)(LLVMErrorRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcLLJITAddLLVMIRModule)(
    LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMOrcThreadSafeModuleRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcLLJITLookup)(LLVMOrcLLJITRef,
                                               LLVMOrcExecutorAddress *,
                                               const char *));
DR_LLVM_SYM(LLVMOrcJITDylibRef (*LLVMOrcLLJITGetMainJITDylib)(LLVMOrcLLJITRef));
DR_LLVM_SYM(void (*LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator)(
    LLVMOrcLLJITBuilderRef,
    LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction, void *));
DR_LLVM_SYM(LLVMOrcObjectLayerRef (
    *LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks)(
    LLVMOrcExecutionSessionRef, void *, LLVMMemoryManagerCreateContextCallback,
    LLVMMemoryManagerNotifyTerminatingCallback,
    LLVMMemoryManagerAllocateCodeSectionCallback,
    LLVMMemoryManagerAllocateDataSectionCallback,
    LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcDisposeLLJIT)(LLVMOrcLLJITRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcJITDylibClear)(LLVMOrcJITDylibRef));
#endif
