#include "llvm_api.h"
#include "llvm_memmgr.h"
#include "llvm.h"
#include "log.h"
#include "eval.h"

static LLVMOrcLLJITRef jitc_llvm_lljit = nullptr;
static LLVMOrcJITDylibRef jitc_llvm_lljit_dylib = nullptr;
extern LLVMTargetMachineRef jitc_llvm_tm;

LLVMOrcObjectLayerRef oll_creator(void *, LLVMOrcExecutionSessionRef es, const char *) {
#if defined(LLVM_VERSION_MAJOR) && LLVM_VERSION_MAJOR < 16
    (void) es;
    jitc_fail("OrcV2 interface is not usable in LLVM versions < 16");
#else
    return LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(
        es, nullptr,
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
    if (jitc_llvm_lljit)
        return true;

    LLVMTargetRef target_ref;
    char *err_str = nullptr;
    if (LLVMGetTargetFromTriple(jitc_llvm_target_triple, &target_ref, &err_str)) {
        LLVMDisposeMessage(err_str);
        jitc_log(Warn,
                 "jitc_llvm_init(): could not obtain target, ORCv2 "
                 "initialization failed: %s", err_str);
        return false;
    }

    LLVMTargetMachineRef tm;
    for (int i = 0; i < 2; ++i) {
        // Create twice -- once for pass manager, once for LLJIT
        tm = LLVMCreateTargetMachine(
            target_ref, jitc_llvm_target_triple, jitc_llvm_target_cpu,
            jitc_llvm_target_features, LLVMCodeGenLevelAggressive, LLVMRelocPIC,
            LLVMCodeModelSmall);
        if (i == 0)
            jitc_llvm_tm = tm;
    }

    LLVMOrcJITTargetMachineBuilderRef machine_builder =
        LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);

    LLVMOrcLLJITBuilderRef lljit_builder = LLVMOrcCreateLLJITBuilder();

    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(lljit_builder,
                                                  machine_builder);

    LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(lljit_builder, oll_creator,
                                                    (void *) NULL);

    LLVMErrorRef err = LLVMOrcCreateLLJIT(&jitc_llvm_lljit, lljit_builder);
    if (err)
        jitc_fail("jit_llvm_compile(): could not create LLJIT: %s",
                  LLVMGetErrorMessage(err));

    jitc_llvm_lljit_dylib = LLVMOrcLLJITGetMainJITDylib(jitc_llvm_lljit);

    return true;
}

void jitc_llvm_orcv2_shutdown() {
    if (!jitc_llvm_lljit)
        return;

    LLVMErrorRef err = LLVMOrcDisposeLLJIT(jitc_llvm_lljit);
    if (err)
        jitc_fail("jit_llvm_orcv2_shutdown(): could not dispose LLJIT: %s",
                  LLVMGetErrorMessage(err));
    LLVMDisposeTargetMachine(jitc_llvm_tm);

    jitc_llvm_lljit = nullptr;
    jitc_llvm_lljit_dylib = nullptr;
    jitc_llvm_tm = nullptr;
}

void jitc_llvm_orcv2_compile(void *llvm_module,
                             std::vector<uint8_t*> &symbols) {
    LLVMErrorRef err = LLVMOrcJITDylibClear(jitc_llvm_lljit_dylib);
    if (err)
        jitc_fail("jit_llvm_compile(): could not clear dylib: %s",
                  LLVMGetErrorMessage(err));

    LLVMOrcThreadSafeContextRef ts_ctx = LLVMOrcCreateNewThreadSafeContext();
    LLVMOrcThreadSafeModuleRef ts_mod =
        LLVMOrcCreateNewThreadSafeModule((LLVMModuleRef) llvm_module, ts_ctx);
    LLVMOrcDisposeThreadSafeContext(ts_ctx);

    err = LLVMOrcLLJITAddLLVMIRModule(jitc_llvm_lljit, jitc_llvm_lljit_dylib,
                                      ts_mod);

    if (err)
        jitc_fail("jit_llvm_compile(): could not add module: %s",
                  LLVMGetErrorMessage(err));

    auto resolve = [&](const char *name) -> uint8_t * {
        LLVMOrcExecutorAddress p;
        LLVMErrorRef err = LLVMOrcLLJITLookup(jitc_llvm_lljit, &p, name);
        if (err)
            jitc_fail("jit_llvm_compile(): could not resolve symbol: %s",
                      LLVMGetErrorMessage(err));
        return (uint8_t *) p;
    };

    size_t symbol_pos = 0;
    symbols[symbol_pos++] = resolve(kernel_name);

    /// Does the kernel perform virtual function calls via @callables?
    if (callable_count_unique) {
        symbols[symbol_pos++] = resolve("callables");

        for (auto const &kv: globals_map) {
            if (!kv.first.callable)
                continue;

            char name_buf[38];
            snprintf(name_buf, sizeof(name_buf), "func_%016llx%016llx",
                     (unsigned long long) kv.first.hash.high64,
                     (unsigned long long) kv.first.hash.low64);
            symbols[symbol_pos++] = resolve(name_buf);
        }
    }
}
