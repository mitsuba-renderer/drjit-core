#include "llvm_memmgr.h"
#include "eval.h"
#include "internal.h"
#include "log.h"

static uint32_t jitc_llvm_patch_loc = 0;
static LLVMExecutionEngineRef m_jitc_llvm_engine = nullptr;
extern LLVMTargetMachineRef jitc_llvm_tm;

/// Create a MCJIT compilation engine configured for use with Dr.Jit
LLVMExecutionEngineRef jitc_llvm_engine_create(LLVMModuleRef mod_) {
    LLVMMCJITCompilerOptions options;
    options.OptLevel = LLVMCodeGenLevelAggressive;
    options.CodeModel = LLVMCodeModelSmall;
    options.NoFramePointerElim = false;
    options.EnableFastISel = false;
    options.MCJMM = LLVMCreateSimpleMCJITMemoryManager(
        nullptr,
        jitc_llvm_memmgr_allocate,
        jitc_llvm_memmgr_allocate_data,
        jitc_llvm_memmgr_finalize,
        jitc_llvm_memmgr_destroy);

    LLVMModuleRef mod = mod_;
    if (mod == nullptr)
        mod = LLVMModuleCreateWithName("drjit");

    LLVMExecutionEngineRef engine = nullptr;
    char *error = nullptr;
    if (LLVMCreateMCJITCompilerForModule(&engine, mod, &options,
                                         sizeof(options), &error)) {
        jitc_log(Warn, "jit_llvm_engine_create(): could not create MCJIT: %s", error);
        return nullptr;
    }

    jitc_llvm_tm = LLVMGetExecutionEngineTargetMachine(engine);

    if (jitc_llvm_patch_loc) {
        uint32_t *base = (uint32_t *) LLVMGetExecutionEngineTargetMachine(engine);
        base[jitc_llvm_patch_loc] = 1 /* Reloc::Model::PIC_ */;
    }

    return engine;
}

bool jitc_llvm_mcjit_init() {
#if defined(DRJIT_DYNAMIC_LLVM) && !defined(__aarch64__)
    m_jitc_llvm_engine = jitc_llvm_engine_create(nullptr);
    if (!m_jitc_llvm_engine)
        return false;

    /**
       The following is horrible, but it works and was without alternative.

       LLVM MCJIT uses a 'static' relocation model by default. But that's not
       good for us: we want a 'pic' relocation model that allows for *relocatable
       code*. This is important because generated kernels can be reloaded from
       the cache, at which point they may end up anywhere in memory.

       The LLVM C API is quite nice, but it's missing a function to adjust this
       crucial aspect. While we could in principle use C++ the API, this is totally
       infeasible without a hard dependency on the LLVM headers and being being
       permanently tied to a specific version.

       The code below simply hot-patches the TargetMachine* instance, which
       contains the following 3 fields (around byte offset 568 on my machine)

          Reloc::Model RM = Reloc::Static;
          CodeModel::Model CMModel = CodeModel::Small;
          CodeGenOpt::Level OptLevel = CodeGenOpt::Default;

       Which are all 32 bit integers. This interface has been stable since
       ancient times (LLVM 4), and the good thing is that we know the values of
       these fields and can therefore use them as a 12-byte key to find the
       precise byte offset and them overwrite the 'RM' field.
    */

    uint32_t *base =
        (uint32_t *) LLVMGetExecutionEngineTargetMachine(m_jitc_llvm_engine);
    jitc_llvm_patch_loc = 142 - 16;

    int key[3] = { 0, 1, 3 };
    bool found = false;
    for (int i = 0; i < 30; ++i) {
        if (memcmp(base + jitc_llvm_patch_loc, key, sizeof(uint32_t) * 3) == 0) {
            found = true;
            break;
        }
        jitc_llvm_patch_loc += 1;
    }

    if (!found) {
        jitc_log(Warn, "jit_llvm_init(): could not hot-patch TargetMachine "
                       "relocation model!");
        return false;
    }
#endif

    return true;
}

void jitc_llvm_mcjit_shutdown() {
    if (m_jitc_llvm_engine) {
        LLVMDisposeExecutionEngine(m_jitc_llvm_engine);
        m_jitc_llvm_engine = nullptr;
    }
    jitc_llvm_patch_loc = 0;
    jitc_llvm_tm = nullptr;
}

void jitc_llvm_mcjit_compile(void *llvm_module,
                             std::vector<uint8_t*> &symbols) {
    if (m_jitc_llvm_engine)
        LLVMDisposeExecutionEngine(m_jitc_llvm_engine);

    m_jitc_llvm_engine = jitc_llvm_engine_create((LLVMModuleRef) llvm_module);

    auto resolve = [&](const char *name) -> uint8_t * {
        uint8_t *p = (uint8_t *) LLVMGetFunctionAddress(m_jitc_llvm_engine, name);
        if (unlikely(!p))
            jitc_fail("jit_llvm_compile(): internal error: could not resolve "
                      "symbol \"%s\"!\n", name);
        return p;
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
