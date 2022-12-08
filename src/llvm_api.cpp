/*
    src/llvm_api.cpp -- Dynamic interface to LLVM via the C API bindings

    Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#define DRJIT_SYMBOL(...) __VA_ARGS__ = nullptr;

#include "llvm_api.h"
#include "internal.h"
#include "log.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#define EXPAND

#define LOAD_IMPL(api, name)                                                   \
    name = decltype(name)(dlsym(handle, #name));                               \
    jitc_llvm_has_##api &= name != nullptr

#define EVAL(x) x
#define LOAD(api, name) EVAL(LOAD_IMPL(api, name))

#define CLEAR(name) name = nullptr;

#if !defined(DRJIT_DYNAMIC_LLVM)
/// If we link against a specific LLVM version, there is nothing to do
bool jitc_llvm_api_init() { return LLVM_VERSION_MAJOR >= 8; }
void jitc_llvm_api_shutdown() {}
bool jitc_llvm_api_has_core() { return true; }
bool jitc_llvm_api_has_mcjit() { return true; }
bool jitc_llvm_api_has_orcv2() { return true; }
uint32_t jitc_llvm_version_major = LLVM_VERSION_MAJOR;
uint32_t jitc_llvm_version_minor = LLVM_VERSION_MINOR;
uint32_t jitc_llvm_version_patch = LLVM_VERSION_PATCH;

#else

/// Otherwise, resolve LLVM symbols dynamically
static void *jitc_llvm_handle = nullptr;
static bool jitc_llvm_has_core = false;
static bool jitc_llvm_has_version = false;
static bool jitc_llvm_has_mcjit = false;
static bool jitc_llvm_has_orcv2 = false;

uint32_t jitc_llvm_version_major = 0;
uint32_t jitc_llvm_version_minor = 0;
uint32_t jitc_llvm_version_patch = 0;

bool jitc_llvm_api_init() {
    if (!jitc_llvm_handle) {
#if defined(_WIN32)
        const char *llvm_fname = "LLVM-C.dll", *llvm_glob = nullptr;
#elif defined(__linux__)
        const char *llvm_fname = "libLLVM.so",
                   *llvm_glob = "/usr/lib/x86_64-linux-gnu/libLLVM*.so.*";
#elif defined(__APPLE__) && defined(__x86_64__)
        const char *llvm_fname = "libLLVM.dylib",
                   *llvm_glob = "/usr/local/Cellar/llvm/*/lib/libLLVM.dylib";
#elif defined(__APPLE__) && defined(__aarch64__)
        const char *llvm_fname = "libLLVM.dylib",
                   *llvm_glob = "/opt/homebrew/Cellar/llvm/*/lib/libLLVM.dylib";
#endif

#if !defined(_WIN32)
        // Don't dlopen libLLVM.so if it was loaded by another library
        if (dlsym(RTLD_NEXT, "LLVMDisposeMessage"))
            jitc_llvm_handle = RTLD_NEXT;
#endif

        if (!jitc_llvm_handle) {
            jitc_llvm_handle =
                jitc_find_library(llvm_fname, llvm_glob, "DRJIT_LIBLLVM_PATH");

            if (!jitc_llvm_handle) // LLVM library cannot be loaded, give up
                return false;
        }
    }

    jitc_llvm_has_core = true;
    jitc_llvm_has_version = true;
    jitc_llvm_has_mcjit = true;
    jitc_llvm_has_orcv2 = true;
    jitc_llvm_version_major = 0;
    jitc_llvm_version_minor = 0;
    jitc_llvm_version_patch = 0;

    void *handle = jitc_llvm_handle;
    LOAD(core, LLVMLinkInMCJIT);
    LOAD(core, LLVMInitializeDrJitAsmPrinter);
    LOAD(core, LLVMInitializeDrJitDisassembler);
    LOAD(core, LLVMInitializeDrJitTarget);
    LOAD(core, LLVMInitializeDrJitTargetInfo);
    LOAD(core, LLVMInitializeDrJitTargetMC);
    LOAD(core, LLVMCreateMessage);
    LOAD(core, LLVMDisposeMessage);
    LOAD(core, LLVMGetDefaultTargetTriple);
    LOAD(core, LLVMGetHostCPUName);
    LOAD(core, LLVMGetHostCPUFeatures);
    LOAD(core, LLVMGetGlobalContext);
    LOAD(core, LLVMCreateDisasm);
    LOAD(core, LLVMDisasmDispose);
    LOAD(core, LLVMSetDisasmOptions);
    LOAD(core, LLVMAddModule);
    LOAD(core, LLVMDisposeModule);
    LOAD(core, LLVMCreateMemoryBufferWithMemoryRange);
    LOAD(core, LLVMParseIRInContext);
    LOAD(core, LLVMPrintModuleToString);
    LOAD(core, LLVMGetGlobalValueAddress);
    LOAD(core, LLVMRemoveModule);
    LOAD(core, LLVMDisasmInstruction);
    LOAD(core, LLVMCreatePassManager);
    LOAD(core, LLVMRunPassManager);
    LOAD(core, LLVMDisposePassManager);
    LOAD(core, LLVMAddLICMPass);
    LOAD(core, LLVMPassManagerBuilderCreate);
    LOAD(core, LLVMPassManagerBuilderSetOptLevel);
    LOAD(core, LLVMPassManagerBuilderPopulateModulePassManager);
    LOAD(core, LLVMPassManagerBuilderDispose);
    LOAD(core, LLVMVerifyModule);

    LOAD(version, LLVMGetVersion);

    LOAD(mcjit, LLVMModuleCreateWithName);
    LOAD(mcjit, LLVMGetExecutionEngineTargetMachine);
    LOAD(mcjit, LLVMCreateMCJITCompilerForModule);
    LOAD(mcjit, LLVMCreateSimpleMCJITMemoryManager);
    LOAD(mcjit, LLVMDisposeExecutionEngine);
    LOAD(mcjit, LLVMGetFunctionAddress);

    LOAD(orcv2, LLVMCreateTargetMachine);
    LOAD(orcv2, LLVMGetTargetFromTriple);
    LOAD(orcv2, LLVMOrcCreateNewThreadSafeContext);
    LOAD(orcv2, LLVMOrcDisposeThreadSafeContext);
    LOAD(orcv2, LLVMOrcCreateNewThreadSafeModule);
    LOAD(orcv2, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine);
    LOAD(orcv2, LLVMOrcCreateLLJITBuilder);
    LOAD(orcv2, LLVMOrcLLJITBuilderSetJITTargetMachineBuilder);
    LOAD(orcv2, LLVMOrcCreateLLJIT);
    LOAD(orcv2, LLVMGetErrorMessage);
    LOAD(orcv2, LLVMOrcLLJITAddLLVMIRModule);
    LOAD(orcv2, LLVMOrcLLJITLookup);
    LOAD(orcv2, LLVMOrcLLJITGetMainJITDylib);
    LOAD(orcv2, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator);
    LOAD(orcv2, LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks);
    LOAD(orcv2, LLVMOrcDisposeLLJIT);
    LOAD(orcv2, LLVMOrcJITDylibClear);

    // It's tricky to find the LLVM version ..
    if (jitc_llvm_has_version) {
        unsigned major, minor, patch;
        LLVMGetVersion(&major, &minor, &patch);
        jitc_llvm_version_major = major;
        jitc_llvm_version_minor = minor;
        jitc_llvm_version_patch = patch;
    } else {
        auto get_version_string = (const char *(*) ()) dlsym(
            handle, "_ZN4llvm16LTOCodeGenerator16getVersionStringEv");

        if (get_version_string) {
            const char *version_string = get_version_string();
            if (sscanf(version_string, "LLVM version %u.%u.%u",
                       &jitc_llvm_version_major, &jitc_llvm_version_minor,
                       &jitc_llvm_version_patch) != 3) {
                jitc_log(
                    Warn,
                    "jit_llvm_init(): could not parse LLVM version string \"%s\".",
                    version_string);
            }
        }
    }

    if (jitc_llvm_version_major == 0) {
        // Assume some generic LLVM version :-(
        jitc_llvm_version_major = 10;
        jitc_llvm_version_minor = 0;
        jitc_llvm_version_patch = 0;
    }

    return jitc_llvm_version_major >= 8;
}

void jitc_llvm_api_shutdown() {
    if (!jitc_llvm_handle)
        return;

    CLEAR(LLVMLinkInMCJIT);
    CLEAR(LLVMInitializeDrJitAsmPrinter);
    CLEAR(LLVMInitializeDrJitDisassembler);
    CLEAR(LLVMInitializeDrJitTarget);
    CLEAR(LLVMInitializeDrJitTargetInfo);
    CLEAR(LLVMInitializeDrJitTargetMC);
    CLEAR(LLVMCreateMessage);
    CLEAR(LLVMDisposeMessage);
    CLEAR(LLVMGetDefaultTargetTriple);
    CLEAR(LLVMGetHostCPUName);
    CLEAR(LLVMGetHostCPUFeatures);
    CLEAR(LLVMGetGlobalContext);
    CLEAR(LLVMCreateDisasm);
    CLEAR(LLVMDisasmDispose);
    CLEAR(LLVMSetDisasmOptions);
    CLEAR(LLVMAddModule);
    CLEAR(LLVMDisposeModule);
    CLEAR(LLVMCreateMemoryBufferWithMemoryRange);
    CLEAR(LLVMParseIRInContext);
    CLEAR(LLVMPrintModuleToString);
    CLEAR(LLVMGetGlobalValueAddress);
    CLEAR(LLVMRemoveModule);
    CLEAR(LLVMDisasmInstruction);
    CLEAR(LLVMCreatePassManager);
    CLEAR(LLVMRunPassManager);
    CLEAR(LLVMDisposePassManager);
    CLEAR(LLVMAddLICMPass);
    CLEAR(LLVMPassManagerBuilderCreate);
    CLEAR(LLVMPassManagerBuilderSetOptLevel);
    CLEAR(LLVMPassManagerBuilderPopulateModulePassManager);
    CLEAR(LLVMPassManagerBuilderDispose);
    CLEAR(LLVMVerifyModule);

    // Version
    CLEAR(LLVMGetVersion);

    // MCJIT
    CLEAR(LLVMModuleCreateWithName);
    CLEAR(LLVMGetExecutionEngineTargetMachine);
    CLEAR(LLVMCreateMCJITCompilerForModule);
    CLEAR(LLVMCreateSimpleMCJITMemoryManager);
    CLEAR(LLVMDisposeExecutionEngine);
    CLEAR(LLVMGetFunctionAddress);

    // ORCv2
    CLEAR(LLVMGetTargetFromTriple);
    CLEAR(LLVMCreateTargetMachine);
    CLEAR(LLVMOrcCreateNewThreadSafeContext);
    CLEAR(LLVMOrcDisposeThreadSafeContext);
    CLEAR(LLVMOrcCreateNewThreadSafeModule);
    CLEAR(LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine);
    CLEAR(LLVMOrcCreateLLJITBuilder);
    CLEAR(LLVMOrcLLJITBuilderSetJITTargetMachineBuilder);
    CLEAR(LLVMOrcCreateLLJIT);
    CLEAR(LLVMGetErrorMessage);
    CLEAR(LLVMOrcLLJITAddLLVMIRModule);
    CLEAR(LLVMOrcLLJITLookup);
    CLEAR(LLVMOrcLLJITGetMainJITDylib);
    CLEAR(LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator);
    CLEAR(LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks);
    CLEAR(LLVMOrcDisposeLLJIT);
    CLEAR(LLVMOrcJITDylibClear);

#if !defined(_WIN32)
    if (jitc_llvm_handle != RTLD_NEXT)
        dlclose(jitc_llvm_handle);
#else
    FreeLibrary((HMODULE) jitc_llvm_handle);
#endif

    jitc_llvm_handle = nullptr;
    jitc_llvm_has_core = false;
    jitc_llvm_has_version = false;
    jitc_llvm_has_mcjit = false;
    jitc_llvm_has_orcv2 = false;
    jitc_llvm_version_major = 0;
    jitc_llvm_version_minor = 0;
    jitc_llvm_version_patch = 0;
}

bool jitc_llvm_api_has_core() { return jitc_llvm_has_core; }
bool jitc_llvm_api_has_mcjit() { return jitc_llvm_has_mcjit; }
bool jitc_llvm_api_has_orcv2() { return jitc_llvm_has_orcv2; }

#endif
