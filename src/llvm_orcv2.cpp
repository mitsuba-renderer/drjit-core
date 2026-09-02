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

/// The process-wide linker and the dylib holding every linked unit
static LLVMOrcLLJITRef jitc_llvm_lljit = nullptr;
static LLVMOrcJITDylibRef jitc_llvm_dylib = nullptr;

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

static LLVMTargetMachineRef jitc_llvm_tm_create() {
    return LLVMCreateTargetMachine(
        jitc_llvm_target_ref, jitc_llvm_target_triple, jitc_llvm_target_cpu,
        jitc_llvm_target_features, LLVMCodeGenLevelAggressive, LLVMRelocPIC,
        LLVMCodeModelSmall);
}

bool jitc_llvm_orcv2_init() {
    char *err_str = nullptr;
    if (LLVMGetTargetFromTriple(jitc_llvm_target_triple, &jitc_llvm_target_ref,
                                &err_str)) {
        jitc_log(Warn,
                 "jitc_llvm_init(): could not obtain target, ORCv2 "
                 "initialization failed: %s", err_str);
        LLVMDisposeMessage(err_str);
        return false;
    }

    // The LLJIT's target machine only determines the object format
    LLVMOrcJITTargetMachineBuilderRef machine_builder =
        LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(jitc_llvm_tm_create());
    LLVMOrcLLJITBuilderRef lljit_builder = LLVMOrcCreateLLJITBuilder();
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(lljit_builder, machine_builder);

    LLVMErrorRef err = LLVMOrcCreateLLJIT(&jitc_llvm_lljit, lljit_builder);
    if (err) {
        jitc_log(Warn, "jitc_llvm_init(): could not create LLJIT: %s",
                 jitc_llvm_error_str(err).c_str());
        return false;
    }
    jitc_llvm_dylib = LLVMOrcLLJITGetMainJITDylib(jitc_llvm_lljit);

    // The generator runs ahead of LLJIT's own process symbol resolution, so
    // that the filter sees every external symbol
    LLVMOrcDefinitionGeneratorRef generator = nullptr;
    err = LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
        &generator, LLVMOrcLLJITGetGlobalPrefix(jitc_llvm_lljit),
        jitc_llvm_symbol_filter, nullptr);
    if (err) {
        jitc_log(Warn, "jitc_llvm_init(): could not create symbol generator: %s",
                 jitc_llvm_error_str(err).c_str());
        return false;
    }
    LLVMOrcJITDylibAddGenerator(jitc_llvm_dylib, generator);

    return true;
}

void jitc_llvm_orcv2_shutdown() {
    jitc_llvm_compiler_pool_clear();

    if (jitc_llvm_lljit) {
        LLVMErrorRef err = LLVMOrcDisposeLLJIT(jitc_llvm_lljit);
        if (err)
            jitc_fail("jit_llvm_orcv2_shutdown(): could not dispose LLJIT: %s",
                      jitc_llvm_error_str(err).c_str());
    }

    jitc_llvm_lljit = nullptr;
    jitc_llvm_dylib = nullptr;
    jitc_llvm_target_ref = nullptr;
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

void jitc_llvm_link(const char *symbol, const uint8_t *object, size_t size,
                    const char *source, UnitArtifact &artifact) {
    // The linker takes ownership of this copy of the object
    LLVMMemoryBufferRef buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
        (const char *) object, size, symbol);
    LLVMOrcResourceTrackerRef rt =
        LLVMOrcJITDylibCreateResourceTracker(jitc_llvm_dylib);

    // The lookup triggers the actual link
    LLVMOrcExecutorAddress address = 0;
    LLVMErrorRef err = LLVMOrcLLJITAddObjectFileWithRT(jitc_llvm_lljit, rt, buf);
    if (!err)
        err = LLVMOrcLLJITLookup(jitc_llvm_lljit, &address, symbol);
    if (err)
        jitc_fail("jit_llvm_link(): could not link unit \"%s\": %s\n\n"
                  "For reference, the LLVM IR of the unit follows:\n\n%s",
                  symbol, jitc_llvm_error_str(err).c_str(), source);

    artifact.ptr[0] = rt;
    artifact.ptr[1] = nullptr;
    artifact.value = address;
    artifact.size = (uint32_t) size;
}

void jitc_llvm_unlink(UnitArtifact &artifact) {
    LLVMOrcResourceTrackerRef rt = (LLVMOrcResourceTrackerRef) artifact.ptr[0];
    LLVMErrorRef err = LLVMOrcResourceTrackerRemove(rt);
    if (err)
        jitc_fail("jit_llvm_unlink(): could not remove unit: %s",
                  jitc_llvm_error_str(err).c_str());
    LLVMOrcReleaseResourceTracker(rt);
}
