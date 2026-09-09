/*
    src/llvm_orcv2.cpp -- ORCv2-based compilation and linking of LLVM units

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "llvm_api.h"
#include "llvm_orcv2.h"
#include "llvm.h"
#include "unit.h"
#include "log.h"
#include <mutex>
#include <string>
#include <vector>

static LLVMTargetRef jitc_llvm_target_ref = nullptr;

/// Maintain a separate linker for release and debug modes
static LLVMOrcLLJITRef jitc_llvm_lljit[2] = { };
static bool jitc_llvm_debug_tried = false;

/// Idle compiler instances
static std::vector<LLVMCompiler *> compiler_pool;
static std::mutex compiler_mutex;

static std::string jitc_llvm_error_str(LLVMErrorRef err) {
    char *msg = LLVMGetErrorMessage(err);
    std::string result(msg);
    LLVMDisposeErrorMessage(msg);
    return result;
}

/// Called for every symbol that a unit references but does not define.
/// These are library calls (e.g. 'memcpy') that LLVM emitted for operations
/// it could not lower inline. They resolve against the running process.
static int jitc_llvm_symbol_filter(void *, LLVMOrcSymbolStringPoolEntryRef sym) {
    jitc_log(Debug, "jit_llvm_link(): resolving external symbol \"%s\" from "
                    "the running process.", LLVMOrcSymbolStringPoolEntryStr(sym));
    return 1;
}

/// JITLink is LLJIT's default linker on ELF and MachO. On Windows, LLJIT
/// defaults to RuntimeDyld, and JITLink must be requested explicitly through
/// C API entry points that exist in LLVM 22 and newer.
bool jitc_llvm_jitlink = false;

LLVMCodeModel jitc_llvm_code_model() {
#if defined(_WIN32)
    // RuntimeDyld's section memory manager may place an object's code and
    // constant pool more than 2 GiB apart and then silently truncates their
    // RIP-relative displacements. The large code model addresses constants
    // absolutely, which sidesteps the problem. JITLink allocates each object
    // as one contiguous region and does not need this workaround.
    if (!jitc_llvm_jitlink)
        return LLVMCodeModelLarge;
#endif
    return LLVMCodeModelSmall;
}

static LLVMTargetMachineRef jitc_llvm_tm_create() {
    return LLVMCreateTargetMachine(
        jitc_llvm_target_ref, jitc_llvm_target_triple, jitc_llvm_target_cpu,
        jitc_llvm_target_features, LLVMCodeGenLevelAggressive, LLVMRelocPIC,
        jitc_llvm_code_model());
}

static bool jitc_llvm_linker_init(bool debug) {
    LLVMOrcLLJITRef &lljit = jitc_llvm_lljit[debug];
    char *err_str = nullptr;
    if (LLVMGetTargetFromTriple(jitc_llvm_target_triple, &jitc_llvm_target_ref,
                                &err_str)) {
        jitc_log(Warn,
                 "jitc_llvm_init(): could not obtain target, ORCv2 "
                 "initialization failed: %s", err_str);
        LLVMDisposeMessage(err_str);
        return false;
    }

    LLVMOrcLLJITBuilderRef lljit_builder = LLVMOrcCreateLLJITBuilder();

#if !defined(_WIN32)
    jitc_llvm_jitlink = true;
#elif !defined(__aarch64__) && \
      (defined(DRJIT_DYNAMIC_LLVM) || LLVM_VERSION_MAJOR >= 22)
    // LLJIT defaults to RuntimeDyld on COFF. Request JITLink if the C API
    // provides it (LLVM 22+). JITLink has no COFF backend for AArch64.
    jitc_llvm_jitlink = jitc_llvm_api_has_jitlink();
    if (jitc_llvm_jitlink) {
        auto create = [](void *, LLVMOrcExecutionSessionRef es, const char *) {
            LLVMOrcObjectLayerRef layer = nullptr;
            LLVMErrorRef err =
                LLVMOrcCreateObjectLinkingLayerWithInProcessMemoryManager(&layer, es);
            if (err)
                jitc_fail("jit_llvm_init(): could not create the JITLink "
                          "linking layer: %s", jitc_llvm_error_str(err).c_str());
            return layer;
        };
        LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(lljit_builder, create,
                                                        nullptr);
    }
#endif

    // The LLJIT's target machine only determines the object format
    LLVMOrcJITTargetMachineBuilderRef machine_builder =
        LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(jitc_llvm_tm_create());
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(lljit_builder, machine_builder);

    LLVMErrorRef err = LLVMOrcCreateLLJIT(&lljit, lljit_builder);
    if (err) {
        jitc_log(Warn, "jitc_llvm_init(): could not create LLJIT: %s",
                 jitc_llvm_error_str(err).c_str());
        return false;
    }
    // The generator runs ahead of LLJIT's own process symbol resolution, so
    // that the filter sees every external symbol
    LLVMOrcDefinitionGeneratorRef generator = nullptr;
    err = LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
        &generator, LLVMOrcLLJITGetGlobalPrefix(lljit),
        jitc_llvm_symbol_filter, nullptr);
    if (!err) {
        LLVMOrcJITDylibAddGenerator(LLVMOrcLLJITGetMainJITDylib(lljit), generator);

        if (debug)
            err = LLVMOrcLLJITEnableDebugSupport(lljit);
    }
    if (err) {
        jitc_log(debug ? Debug : Warn,
                 "jitc_llvm_init(): could not initialize %slinker: %s",
                 debug ? "debug " : "", jitc_llvm_error_str(err).c_str());
        LLVMErrorRef dispose_err = LLVMOrcDisposeLLJIT(lljit);
        if (dispose_err)
            jitc_llvm_error_str(dispose_err);
        lljit = nullptr;
        return false;
    }

    return true;
}

bool jitc_llvm_orcv2_init() {
    return jitc_llvm_linker_init(false);
}

bool jitc_llvm_debug_init() {
    // Called during assembly, with state.lock and state.eval_lock held.
    if (!jitc_llvm_debug_tried) {
        jitc_llvm_debug_tried = true;
        if (jitc_llvm_api_has_orcdbg())
            jitc_llvm_linker_init(true);
    }
    return jitc_llvm_lljit[1] != nullptr;
}

void jitc_llvm_orcv2_shutdown() {
    jitc_llvm_compiler_pool_clear();

    for (LLVMOrcLLJITRef &lljit : jitc_llvm_lljit) {
        if (!lljit)
            continue;
        LLVMErrorRef err = LLVMOrcDisposeLLJIT(lljit);
        if (err)
            jitc_fail("jit_llvm_orcv2_shutdown(): could not dispose LLJIT: %s",
                      jitc_llvm_error_str(err).c_str());
        lljit = nullptr;
    }

    jitc_llvm_target_ref = nullptr;
    jitc_llvm_jitlink = false;
    jitc_llvm_debug_tried = false;
}

// ============================================================================
//  Compiler instance pool
// ============================================================================

LLVMCompiler *jitc_llvm_compiler_acquire() {
    {
        std::lock_guard<std::mutex> guard(compiler_mutex);
        if (!compiler_pool.empty()) {
            LLVMCompiler *c = compiler_pool.back();
            compiler_pool.pop_back();
            return c;
        }
    }

    LLVMCompiler *c = new LLVMCompiler();
    c->context = LLVMContextCreate();
    c->tm = jitc_llvm_tm_create();
    return c;
}

void jitc_llvm_compiler_release(LLVMCompiler *c) {
    std::lock_guard<std::mutex> guard(compiler_mutex);
    compiler_pool.push_back(c);
}

void jitc_llvm_compiler_pool_clear() {
    std::lock_guard<std::mutex> guard(compiler_mutex);
    for (LLVMCompiler *c : compiler_pool) {
        LLVMDisposeTargetMachine(c->tm);
        LLVMContextDispose(c->context);
        delete c;
    }
    compiler_pool.clear();
}

// ============================================================================
//  Linking
// ============================================================================

/// Mutex to serialize linking in debug mode. LLDB (when attached) generates
/// warnings when linking in parallel.
static std::mutex jitc_llvm_link_mutex;

void jitc_llvm_link(const char *symbol, const uint8_t *object, size_t size,
                    const char *source, bool debug, UnitArtifact &artifact) {
    std::unique_lock<std::mutex> guard(jitc_llvm_link_mutex, std::defer_lock);
    if (debug)
        guard.lock();
    LLVMOrcLLJITRef lljit = jitc_llvm_lljit[debug];

    // The linker takes ownership of this copy of the object
    LLVMMemoryBufferRef buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
        (const char *) object, size, symbol);
    LLVMOrcResourceTrackerRef rt =
        LLVMOrcJITDylibCreateResourceTracker(LLVMOrcLLJITGetMainJITDylib(lljit));

    // The lookup triggers the actual link
    LLVMOrcExecutorAddress address = 0;
    LLVMErrorRef err = LLVMOrcLLJITAddObjectFileWithRT(lljit, rt, buf);
    if (!err)
        err = LLVMOrcLLJITLookup(lljit, &address, symbol);
    if (err)
        jitc_fail("jit_llvm_link(): could not link unit \"%s\": %s\n\n"
                  "For reference, the LLVM IR of the unit follows:\n\n%s",
                  symbol, jitc_llvm_error_str(err).c_str(), source);

    artifact.ptr[0] = rt;
    artifact.ptr[1] = debug ? lljit : nullptr;
    artifact.value = address;
    artifact.size = (uint32_t) size;
}

void jitc_llvm_unlink(UnitArtifact &artifact) {
    std::unique_lock<std::mutex> guard(jitc_llvm_link_mutex, std::defer_lock);
    if (artifact.ptr[1])
        guard.lock();
    LLVMOrcResourceTrackerRef rt = (LLVMOrcResourceTrackerRef) artifact.ptr[0];
    LLVMErrorRef err = LLVMOrcResourceTrackerRemove(rt);
    if (err)
        jitc_fail("jit_llvm_unlink(): could not remove unit: %s",
                  jitc_llvm_error_str(err).c_str());
    LLVMOrcReleaseResourceTracker(rt);
}
