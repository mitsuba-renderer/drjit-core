/*
    src/llvm_orcv2.cpp -- Pool of ORCv2-based LLVM compiler instances

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "llvm_api.h"
#include "llvm_memmgr.h"
#include "llvm.h"
#include "log.h"
#include "eval.h"
#include <mutex>

static LLVMTargetRef jitc_llvm_target_ref = nullptr;

static std::vector<LLVMCompiler *> compiler_pool; // idle instances
static std::vector<LLVMCompiler *> compiler_all;  // every created instance
static std::mutex compiler_mutex;

static LLVMOrcObjectLayerRef oll_creator(void *ctx,
                                         LLVMOrcExecutionSessionRef es,
                                         const char *) {
#if defined(LLVM_VERSION_MAJOR) && LLVM_VERSION_MAJOR < 16
    (void) ctx; (void) es;
    jitc_fail("OrcV2 interface is not usable in LLVM versions < 16");
#else
    return LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(
        es, ctx, // forwarded to jitc_llvm_memmgr_create_context
        jitc_llvm_memmgr_create_context,
        jitc_llvm_memmgr_notify_terminating,
        jitc_llvm_memmgr_allocate,
        jitc_llvm_memmgr_allocate_data,
        jitc_llvm_memmgr_finalize,
        jitc_llvm_memmgr_destroy
    );
#endif
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

    return true;
}

static void jitc_llvm_compiler_destroy(LLVMCompiler *c) {
    if (c->lljit) {
        LLVMErrorRef err = LLVMOrcDisposeLLJIT(c->lljit);
        if (err)
            jitc_fail("jit_llvm_orcv2_shutdown(): could not dispose LLJIT: %s",
                      LLVMGetErrorMessage(err));
    }
    if (c->tm)
        LLVMDisposeTargetMachine(c->tm);
    if (c->context)
        LLVMContextDispose(c->context);
    jitc_llvm_memmgr_release(c->memmgr);
    delete c;
}

void jitc_llvm_orcv2_shutdown() {
    std::lock_guard<std::mutex> guard(compiler_mutex);
    for (LLVMCompiler *c : compiler_all)
        jitc_llvm_compiler_destroy(c);
    compiler_all.clear();
    compiler_pool.clear();
    jitc_llvm_target_ref = nullptr;
}

static LLVMCompiler *jitc_llvm_compiler_create() {
    LLVMCompiler *c = new LLVMCompiler();

    c->context = LLVMContextCreate();

    // Create the target machine twice: the LLJIT machine builder consumes
    // its copy, while 'c->tm' drives the optimization pipeline
    LLVMTargetMachineRef tm = nullptr;
    for (int i = 0; i < 2; ++i) {
        tm = LLVMCreateTargetMachine(
            jitc_llvm_target_ref, jitc_llvm_target_triple, jitc_llvm_target_cpu,
            jitc_llvm_target_features, LLVMCodeGenLevelAggressive, LLVMRelocPIC,
            LLVMCodeModelSmall);
        if (i == 0)
            c->tm = tm;
    }

    LLVMOrcJITTargetMachineBuilderRef machine_builder =
        LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);

    LLVMOrcLLJITBuilderRef lljit_builder = LLVMOrcCreateLLJITBuilder();

    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(lljit_builder,
                                                  machine_builder);

    LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(lljit_builder, oll_creator,
                                                    (void *) &c->memmgr);

    LLVMErrorRef err = LLVMOrcCreateLLJIT(&c->lljit, lljit_builder);
    if (err)
        jitc_fail("jit_llvm_compile(): could not create LLJIT: %s",
                  LLVMGetErrorMessage(err));

    c->dylib = LLVMOrcLLJITGetMainJITDylib(c->lljit);

    return c;
}

LLVMCompiler *jitc_llvm_compiler_acquire() {
    {
        std::lock_guard<std::mutex> guard(compiler_mutex);
        if (!compiler_pool.empty()) {
            LLVMCompiler *c = compiler_pool.back();
            compiler_pool.pop_back();
            return c;
        }
    }

    LLVMCompiler *c = jitc_llvm_compiler_create();

    std::lock_guard<std::mutex> guard(compiler_mutex);
    compiler_all.push_back(c);
    return c;
}

void jitc_llvm_compiler_release(LLVMCompiler *c) {
    std::lock_guard<std::mutex> guard(compiler_mutex);
    compiler_pool.push_back(c);
}
