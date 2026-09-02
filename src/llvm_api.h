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
#  else
      #define LLVMInitializeDrJitTarget       LLVMInitializeX86Target
      #define LLVMInitializeDrJitTargetInfo   LLVMInitializeX86TargetInfo
      #define LLVMInitializeDrJitTargetMC     LLVMInitializeX86TargetMC
      #define LLVMInitializeDrJitAsmPrinter   LLVMInitializeX86AsmPrinter
#  endif

#if !defined(DRJIT_DYNAMIC_LLVM)
#  include <llvm-c/Core.h>
#  include <llvm-c/ExecutionEngine.h>
#  include <llvm-c/IRReader.h>
#  include <llvm-c/Analysis.h>
#  if LLVM_VERSION_MAJOR >= 15
#    include <llvm-c/Transforms/PassBuilder.h>
#  endif
#  include <llvm-c/TargetMachine.h>
#  include <llvm-c/Error.h>
#  include <llvm-c/Orc.h>
#  include <llvm-c/LLJIT.h>
#else
#  include <stdint.h>
#  include <stdlib.h>
#  define LLVMReturnStatusAction 2
#  define LLVMCodeGenLevelAggressive 3
#  define LLVMRelocPIC 2
#  define LLVMCodeModelSmall 3
#  define LLVMObjectFile 1

/// LLVM API
using LLVMBool = int;
using LLVMModuleRef = void *;
using LLVMMemoryBufferRef = void *;
using LLVMContextRef = void *;
using LLVMPassBuilderOptionsRef = void *;
using LLVMTargetMachineRef = void *;
using LLVMTargetRef = void *;
using LLVMCodeModel = int;
using LLVMRelocMode = int;
using LLVMCodeGenOptLevel = int;
using LLVMCodeGenFileType = int;
using LLVMOrcJITTargetMachineBuilderRef = void *;
using LLVMOrcLLJITBuilderRef = void *;
using LLVMOrcLLJITRef = void *;
using LLVMErrorRef = void *;
using LLVMOrcJITDylibRef = void *;
using LLVMOrcResourceTrackerRef = void *;
using LLVMOrcDefinitionGeneratorRef = void *;
using LLVMOrcSymbolStringPoolEntryRef = void *;
using LLVMOrcExecutorAddress = uint64_t;
using LLVMOrcSymbolPredicate = int (*)(void *, LLVMOrcSymbolStringPoolEntryRef);

#if !defined(DR_LLVM_SYM)
#  define DR_LLVM_SYM(x) extern x;
#endif

DR_LLVM_SYM(void (*LLVMInitializeDrJitAsmPrinter)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitTarget)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitTargetInfo)());
DR_LLVM_SYM(void (*LLVMInitializeDrJitTargetMC)());
DR_LLVM_SYM(char *(*LLVMCreateMessage)(const char *) );
DR_LLVM_SYM(void (*LLVMDisposeMessage)(char *));
DR_LLVM_SYM(char *(*LLVMGetDefaultTargetTriple)());
DR_LLVM_SYM(char *(*LLVMGetHostCPUName)());
DR_LLVM_SYM(char *(*LLVMGetHostCPUFeatures)());
DR_LLVM_SYM(LLVMContextRef (*LLVMContextCreate)());
DR_LLVM_SYM(void (*LLVMContextDispose)(LLVMContextRef));
DR_LLVM_SYM(LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool));
DR_LLVM_SYM(LLVMBool (*LLVMParseIRInContext)(LLVMContextRef,
                                             LLVMMemoryBufferRef,
                                             LLVMModuleRef *, char **));
DR_LLVM_SYM(bool (*LLVMVerifyModule)(LLVMModuleRef, int action, char **msg));
DR_LLVM_SYM(void (*LLVMGetVersion)(unsigned *, unsigned *, unsigned *));
DR_LLVM_SYM(void (*LLVMDisposeTargetMachine)(LLVMTargetMachineRef));

// New pass manager
DR_LLVM_SYM(LLVMPassBuilderOptionsRef (*LLVMCreatePassBuilderOptions)());
DR_LLVM_SYM(void (*LLVMPassBuilderOptionsSetLoopVectorization)(LLVMPassBuilderOptionsRef, LLVMBool));
DR_LLVM_SYM(void (*LLVMPassBuilderOptionsSetLoopUnrolling)(LLVMPassBuilderOptionsRef, LLVMBool));
DR_LLVM_SYM(void (*LLVMPassBuilderOptionsSetSLPVectorization)(LLVMPassBuilderOptionsRef, LLVMBool));
DR_LLVM_SYM(void (*LLVMDisposePassBuilderOptions)(LLVMPassBuilderOptionsRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMRunPasses)(LLVMModuleRef, const char *,
                                          LLVMTargetMachineRef,
                                          LLVMPassBuilderOptionsRef));

// Code generation into relocatable object files
DR_LLVM_SYM(LLVMBool (*LLVMGetTargetFromTriple)(const char *, LLVMTargetRef *,
                                                char **));
DR_LLVM_SYM(LLVMTargetMachineRef (*LLVMCreateTargetMachine)(
    LLVMTargetRef, const char *, const char *, const char *,
    LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel));
DR_LLVM_SYM(LLVMBool (*LLVMTargetMachineEmitToMemoryBuffer)(
    LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, char **,
    LLVMMemoryBufferRef *));
DR_LLVM_SYM(void (*LLVMDisposeModule)(LLVMModuleRef));
DR_LLVM_SYM(const char *(*LLVMGetBufferStart)(LLVMMemoryBufferRef));
DR_LLVM_SYM(size_t (*LLVMGetBufferSize)(LLVMMemoryBufferRef));
DR_LLVM_SYM(void (*LLVMDisposeMemoryBuffer)(LLVMMemoryBufferRef));
DR_LLVM_SYM(LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRangeCopy)(
    const char *, size_t, const char *));
DR_LLVM_SYM(char *(*LLVMGetErrorMessage)(LLVMErrorRef));
DR_LLVM_SYM(void (*LLVMDisposeErrorMessage)(char *));

// ORCv2 linker
DR_LLVM_SYM(LLVMOrcJITTargetMachineBuilderRef (
    *LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine)(
    LLVMTargetMachineRef));
DR_LLVM_SYM(LLVMOrcLLJITBuilderRef (*LLVMOrcCreateLLJITBuilder)());
DR_LLVM_SYM(void (*LLVMOrcLLJITBuilderSetJITTargetMachineBuilder)(
    LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcCreateLLJIT)(LLVMOrcLLJITRef *,
                                               LLVMOrcLLJITBuilderRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcDisposeLLJIT)(LLVMOrcLLJITRef));
DR_LLVM_SYM(LLVMOrcJITDylibRef (*LLVMOrcLLJITGetMainJITDylib)(LLVMOrcLLJITRef));
DR_LLVM_SYM(char (*LLVMOrcLLJITGetGlobalPrefix)(LLVMOrcLLJITRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcLLJITAddObjectFileWithRT)(
    LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcLLJITLookup)(LLVMOrcLLJITRef,
                                               LLVMOrcExecutorAddress *,
                                               const char *));
DR_LLVM_SYM(LLVMOrcResourceTrackerRef (*LLVMOrcJITDylibCreateResourceTracker)(
    LLVMOrcJITDylibRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcResourceTrackerRemove)(
    LLVMOrcResourceTrackerRef));
DR_LLVM_SYM(void (*LLVMOrcReleaseResourceTracker)(LLVMOrcResourceTrackerRef));
DR_LLVM_SYM(LLVMErrorRef (*LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess)(
    LLVMOrcDefinitionGeneratorRef *, char, LLVMOrcSymbolPredicate, void *));
DR_LLVM_SYM(void (*LLVMOrcJITDylibAddGenerator)(LLVMOrcJITDylibRef,
                                                LLVMOrcDefinitionGeneratorRef));
DR_LLVM_SYM(const char *(*LLVMOrcSymbolStringPoolEntryStr)(
    LLVMOrcSymbolStringPoolEntryRef));
#endif
