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
bool jitc_llvm_api_init() { return LLVM_VERSION_MAJOR >= 11; }
void jitc_llvm_api_shutdown() {}
bool jitc_llvm_api_has_core() { return true; }
bool jitc_llvm_api_has_orcv2() { return LLVM_VERSION_MAJOR >= 16; }
bool jitc_llvm_api_has_pb_new() { return LLVM_VERSION_MAJOR >= 15; }
int jitc_llvm_version_major = LLVM_VERSION_MAJOR;
int jitc_llvm_version_minor = LLVM_VERSION_MINOR;
int jitc_llvm_version_patch = LLVM_VERSION_PATCH;

#else

/// Otherwise, resolve LLVM symbols dynamically
static void *jitc_llvm_handle = nullptr;
static bool jitc_llvm_has_core = false;
static bool jitc_llvm_has_version = false;
static bool jitc_llvm_has_orcv2 = false;
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
                   *llvm_glob  = "/usr/lib/{x86_64-linux-gnu,aarch64-linux-gnu,wsl/lib}/libLLVM*.so.*";
#elif defined(__APPLE__) && defined(__x86_64__)
        const char *llvm_fname = "libLLVM.dylib",
                   *llvm_glob  = "/usr/local/Cellar/llvm/*/lib/libLLVM.dylib";
#elif defined(__APPLE__) && defined(__aarch64__)
        const char *llvm_fname = "libLLVM.dylib",
                   *llvm_glob  = "/opt/homebrew/Cellar/llvm/*/lib/libLLVM.dylib";
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
    jitc_llvm_has_orcv2 = true;
    jitc_llvm_has_pb_new = true;
    jitc_llvm_version_major = -1;
    jitc_llvm_version_minor = -1;
    jitc_llvm_version_patch = -1;

    void *handle = jitc_llvm_handle;
    LOAD(core, LLVMInitializeDrJitAsmPrinter);
    LOAD(core, LLVMInitializeDrJitTarget);
    LOAD(core, LLVMInitializeDrJitTargetInfo);
    LOAD(core, LLVMInitializeDrJitTargetMC);
    LOAD(core, LLVMCreateMessage);
    LOAD(core, LLVMDisposeMessage);
    LOAD(core, LLVMGetDefaultTargetTriple);
    LOAD(core, LLVMGetHostCPUName);
    LOAD(core, LLVMGetHostCPUFeatures);
    LOAD(core, LLVMContextCreate);
    LOAD(core, LLVMContextDispose);
    LOAD(core, LLVMCreateMemoryBufferWithMemoryRange);
    LOAD(core, LLVMParseIRInContext);
    LOAD(core, LLVMVerifyModule);
    LOAD(core, LLVMDisposeTargetMachine);

    LOAD(version, LLVMGetVersion);

    LOAD(pb_new, LLVMCreatePassBuilderOptions);
    LOAD(pb_new, LLVMPassBuilderOptionsSetLoopVectorization);
    LOAD(pb_new, LLVMPassBuilderOptionsSetLoopUnrolling);
    LOAD(pb_new, LLVMPassBuilderOptionsSetSLPVectorization);
    LOAD(pb_new, LLVMDisposePassBuilderOptions);
    LOAD(pb_new, LLVMRunPasses);

    LOAD(orcv2, LLVMCreateTargetMachine);
    LOAD(orcv2, LLVMGetTargetFromTriple);
    LOAD(orcv2, LLVMTargetMachineEmitToMemoryBuffer);
    LOAD(orcv2, LLVMDisposeModule);
    LOAD(orcv2, LLVMGetBufferStart);
    LOAD(orcv2, LLVMGetBufferSize);
    LOAD(orcv2, LLVMDisposeMemoryBuffer);
    LOAD(orcv2, LLVMCreateMemoryBufferWithMemoryRangeCopy);
    LOAD(orcv2, LLVMGetErrorMessage);
    LOAD(orcv2, LLVMDisposeErrorMessage);
    LOAD(orcv2, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine);
    LOAD(orcv2, LLVMOrcCreateLLJITBuilder);
    LOAD(orcv2, LLVMOrcLLJITBuilderSetJITTargetMachineBuilder);
    LOAD(orcv2, LLVMOrcCreateLLJIT);
    LOAD(orcv2, LLVMOrcDisposeLLJIT);
    LOAD(orcv2, LLVMOrcLLJITGetMainJITDylib);
    LOAD(orcv2, LLVMOrcLLJITGetGlobalPrefix);
    LOAD(orcv2, LLVMOrcLLJITAddObjectFileWithRT);
    LOAD(orcv2, LLVMOrcLLJITLookup);
    LOAD(orcv2, LLVMOrcJITDylibCreateResourceTracker);
    LOAD(orcv2, LLVMOrcResourceTrackerRemove);
    LOAD(orcv2, LLVMOrcReleaseResourceTracker);
    LOAD(orcv2, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess);
    LOAD(orcv2, LLVMOrcJITDylibAddGenerator);
    LOAD(orcv2, LLVMOrcSymbolStringPoolEntryStr);

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
    }

    return true;
}

void jitc_llvm_api_shutdown() {
    if (!jitc_llvm_handle)
        return;

    CLEAR(LLVMInitializeDrJitAsmPrinter);
    CLEAR(LLVMInitializeDrJitTarget);
    CLEAR(LLVMInitializeDrJitTargetInfo);
    CLEAR(LLVMInitializeDrJitTargetMC);
    CLEAR(LLVMCreateMessage);
    CLEAR(LLVMDisposeMessage);
    CLEAR(LLVMGetDefaultTargetTriple);
    CLEAR(LLVMGetHostCPUName);
    CLEAR(LLVMGetHostCPUFeatures);
    CLEAR(LLVMContextCreate);
    CLEAR(LLVMContextDispose);
    CLEAR(LLVMCreateMemoryBufferWithMemoryRange);
    CLEAR(LLVMParseIRInContext);
    CLEAR(LLVMVerifyModule);
    CLEAR(LLVMDisposeTargetMachine);

    // Version
    CLEAR(LLVMGetVersion);

    // New pass manager
    CLEAR(LLVMCreatePassBuilderOptions);
    CLEAR(LLVMPassBuilderOptionsSetLoopVectorization);
    CLEAR(LLVMPassBuilderOptionsSetLoopUnrolling);
    CLEAR(LLVMPassBuilderOptionsSetSLPVectorization);
    CLEAR(LLVMDisposePassBuilderOptions);
    CLEAR(LLVMRunPasses);

    // ORCv2
    CLEAR(LLVMGetTargetFromTriple);
    CLEAR(LLVMCreateTargetMachine);
    CLEAR(LLVMTargetMachineEmitToMemoryBuffer);
    CLEAR(LLVMDisposeModule);
    CLEAR(LLVMGetBufferStart);
    CLEAR(LLVMGetBufferSize);
    CLEAR(LLVMDisposeMemoryBuffer);
    CLEAR(LLVMCreateMemoryBufferWithMemoryRangeCopy);
    CLEAR(LLVMGetErrorMessage);
    CLEAR(LLVMDisposeErrorMessage);
    CLEAR(LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine);
    CLEAR(LLVMOrcCreateLLJITBuilder);
    CLEAR(LLVMOrcLLJITBuilderSetJITTargetMachineBuilder);
    CLEAR(LLVMOrcCreateLLJIT);
    CLEAR(LLVMOrcDisposeLLJIT);
    CLEAR(LLVMOrcLLJITGetMainJITDylib);
    CLEAR(LLVMOrcLLJITGetGlobalPrefix);
    CLEAR(LLVMOrcLLJITAddObjectFileWithRT);
    CLEAR(LLVMOrcLLJITLookup);
    CLEAR(LLVMOrcJITDylibCreateResourceTracker);
    CLEAR(LLVMOrcResourceTrackerRemove);
    CLEAR(LLVMOrcReleaseResourceTracker);
    CLEAR(LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess);
    CLEAR(LLVMOrcJITDylibAddGenerator);
    CLEAR(LLVMOrcSymbolStringPoolEntryStr);

#if !defined(_WIN32)
    if (jitc_llvm_handle != RTLD_NEXT)
        dlclose(jitc_llvm_handle);
#else
    FreeLibrary((HMODULE) jitc_llvm_handle);
#endif

    jitc_llvm_handle = nullptr;
    jitc_llvm_has_core = false;
    jitc_llvm_has_version = false;
    jitc_llvm_has_orcv2 = false;
    jitc_llvm_has_pb_new = false;
    jitc_llvm_version_major = -1;
    jitc_llvm_version_minor = -1;
    jitc_llvm_version_patch = -1;
}

bool jitc_llvm_api_has_core() { return jitc_llvm_has_core; }
bool jitc_llvm_api_has_orcv2() { return jitc_llvm_has_orcv2; }
bool jitc_llvm_api_has_pb_new() { return jitc_llvm_has_pb_new; }

#endif
