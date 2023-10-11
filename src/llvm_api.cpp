/*
    src/llvm_api.cpp -- Dynamic interface to LLVM via the C API bindings

    Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#define DR_LLVM_SYM(...) __VA_ARGS__ = nullptr;

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
bool jitc_llvm_api_has_orcv2() { return LLVM_VERSION_MAJOR >= 16; }
bool jitc_llvm_api_has_pb_legacy() { return LLVM_VERSION_MAJOR < 17; }
bool jitc_llvm_api_has_pb_new() { return LLVM_VERSION_MAJOR >= 15; }
int jitc_llvm_version_major = LLVM_VERSION_MAJOR;
int jitc_llvm_version_minor = LLVM_VERSION_MINOR;
int jitc_llvm_version_patch = LLVM_VERSION_PATCH;

#else

/// Otherwise, resolve LLVM symbols dynamically
static void *jitc_llvm_handle = nullptr;
static bool jitc_llvm_has_core = false;
static bool jitc_llvm_has_version = false;
static bool jitc_llvm_has_mcjit = false;
static bool jitc_llvm_has_orcv2 = false;
static bool jitc_llvm_has_pb_legacy = false;
static bool jitc_llvm_has_pb_new = false;

int jitc_llvm_version_major = -1;
int jitc_llvm_version_minor = -1;
int jitc_llvm_version_patch = -1;

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
    jitc_llvm_has_pb_legacy = true;
    jitc_llvm_has_pb_new = true;
    jitc_llvm_version_major = -1;
    jitc_llvm_version_minor = -1;
    jitc_llvm_version_patch = -1;

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
    LOAD(core, LLVMVerifyModule);
    LOAD(core, LLVMDisposeTargetMachine);

    LOAD(version, LLVMGetVersion);

    LOAD(pb_legacy, LLVMCreatePassManager);
    LOAD(pb_legacy, LLVMRunPassManager);
    LOAD(pb_legacy, LLVMDisposePassManager);
    LOAD(pb_legacy, LLVMAddLICMPass);

    LOAD(pb_new, LLVMCreatePassBuilderOptions);
    LOAD(pb_new, LLVMPassBuilderOptionsSetLoopVectorization);
    LOAD(pb_new, LLVMPassBuilderOptionsSetLoopUnrolling);
    LOAD(pb_new, LLVMPassBuilderOptionsSetSLPVectorization);
    LOAD(pb_new, LLVMDisposePassBuilderOptions);
    LOAD(pb_new, LLVMRunPasses);

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

    /*
       Dr.Jit needs to know the LLVM version number to emit the right set of
       intrinsics. Unfortunately, it's tricky to find the exact version.

       LLVM 16+ exposes 'LLVMGetVersion()', which solves the problem. On older
       versions, we can try to call an obscure function from the link-time
       optimization code generator to extract the version.

       If that doesn't work, we try to roughly get it right by searching the
       shared library for telltale symbols that are only available in some LLVM
       versions. This is good enough for determining the major version, which
       is what matters.
    */

    if (jitc_llvm_has_version) {
        unsigned major, minor, patch;
        LLVMGetVersion(&major, &minor, &patch);
        jitc_llvm_version_major = (int) major;
        jitc_llvm_version_minor = (int) minor;
        jitc_llvm_version_patch = (int) patch;
    } else {
        auto get_version_string = (const char *(*) ()) dlsym(
            handle, "_ZN4llvm16LTOCodeGenerator16getVersionStringEv");

        if (get_version_string) {
            const char *version_string = get_version_string();
            if (sscanf(version_string, "LLVM version %i.%i.%i",
                       &jitc_llvm_version_major, &jitc_llvm_version_minor,
                       &jitc_llvm_version_patch) != 3) {
                jitc_log(
                    Warn,
                    "jit_llvm_init(): could not parse LLVM version string \"%s\".",
                    version_string);
                jitc_llvm_version_major = -1;
                jitc_llvm_version_minor = -1;
                jitc_llvm_version_patch = -1;
            }
        }
    }

    // Try to at least guess the LLVM major version based on what symbols are available
    if (jitc_llvm_version_major == -1) {
        struct Symbol {
            int major;
            const char *name;
        };

        Symbol symbols[] = {
            { 8,  "LLVMDisposeErrorMessage" },
            { 9,  "LLVMCreateBinary" },
            { 10, "LLVMBuildFreeze" },
            { 12, "LLVMIsPoison" },
            { 13, "LLVMRunPasses" },
            { 14, "LLVMAddMetadataToInst" },
            { 15, "LLVMDeleteInstruction" }
        };

        for (Symbol s : symbols) {
            if (dlsym(handle, s.name))
                jitc_llvm_version_major = s.major;
        }

        if (jitc_llvm_version_major < 8) {
            jitc_log(Warn, "jit_llvm_init(): the detected LLVM version was too "
                           "old, at least 8.0.0 is needed.");
            return false;
        }
    }

    return true;
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
    CLEAR(LLVMVerifyModule);
    CLEAR(LLVMDisposeTargetMachine);

    // Version
    CLEAR(LLVMGetVersion);

    // Legacy pass manager
    CLEAR(LLVMCreatePassManager);
    CLEAR(LLVMRunPassManager);
    CLEAR(LLVMDisposePassManager);
    CLEAR(LLVMAddLICMPass);

    // New pass manager
    CLEAR(LLVMCreatePassBuilderOptions);
    CLEAR(LLVMPassBuilderOptionsSetLoopVectorization);
    CLEAR(LLVMPassBuilderOptionsSetLoopUnrolling);
    CLEAR(LLVMPassBuilderOptionsSetSLPVectorization);
    CLEAR(LLVMDisposePassBuilderOptions);
    CLEAR(LLVMRunPasses);

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
    jitc_llvm_has_pb_legacy = false;
    jitc_llvm_has_pb_new = false;
    jitc_llvm_version_major = -1;
    jitc_llvm_version_minor = -1;
    jitc_llvm_version_patch = -1;
}

bool jitc_llvm_api_has_core() { return jitc_llvm_has_core; }
bool jitc_llvm_api_has_mcjit() { return jitc_llvm_has_mcjit; }
bool jitc_llvm_api_has_orcv2() { return jitc_llvm_has_orcv2; }
bool jitc_llvm_api_has_pb_legacy() { return jitc_llvm_has_pb_legacy; }
bool jitc_llvm_api_has_pb_new() { return jitc_llvm_has_pb_new; }

#endif
