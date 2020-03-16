#include "llvm_api.h"
#include "internal.h"
#include "log.h"
#include <glob.h>
#include <dlfcn.h>

struct LLVMMCJITCompilerOptions {
  unsigned OptLevel;
  int CodeModel;
  LLVMBool NoFramePointerElim;
  LLVMBool EnableFastISel;
  void *MCJMM;
};

/// LLVM API
static void (*LLVMLinkInMCJIT)() = nullptr;
static void (*LLVMInitializeX86AsmPrinter)() = nullptr;
static void (*LLVMInitializeX86Disassembler)() = nullptr;
static void (*LLVMInitializeX86Target)() = nullptr;
static void (*LLVMInitializeX86TargetInfo)() = nullptr;
static void (*LLVMInitializeX86TargetMC)() = nullptr;
static void (*LLVMDisposeMessage)(char *) = nullptr;
static char *(*LLVMGetDefaultTargetTriple)() = nullptr;
static char *(*LLVMGetHostCPUName)() = nullptr;
static LLVMContextRef (*LLVMGetGlobalContext)() = nullptr;
static LLVMDisasmContextRef (*LLVMCreateDisasm)(const char *, void *, int, void *, void *) = nullptr;
static void (*LLVMDisasmDispose)(LLVMDisasmContextRef) = nullptr;
static int (*LLVMSetDisasmOptions)(LLVMDisasmContextRef, uint64_t) = nullptr;
static LLVMModuleRef (*LLVMModuleCreateWithName)(const char *) = nullptr;
static void (*LLVMInitializeMCJITCompilerOptions)(LLVMMCJITCompilerOptions *,
                                                  size_t) = nullptr;
static LLVMBool (*LLVMCreateMCJITCompilerForModule)(LLVMExecutionEngineRef *,
                                                    LLVMModuleRef,
                                                    LLVMMCJITCompilerOptions *,
                                                    size_t, char **) = nullptr;
static void (*LLVMDisposeExecutionEngine)(LLVMExecutionEngineRef) = nullptr;
void (*LLVMAddModule)(LLVMExecutionEngineRef, LLVMModuleRef) = nullptr;
static void (*LLVMDisposeModule)(LLVMModuleRef) = nullptr;
LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool) = nullptr;
LLVMBool (*LLVMParseIRInContext)(LLVMContextRef, LLVMMemoryBufferRef,
                                 LLVMModuleRef *, char **) = nullptr;
char *(*LLVMPrintModuleToString)(LLVMModuleRef) = nullptr;
uint64_t (*LLVMGetFunctionAddress)(LLVMExecutionEngineRef, const char *);
static LLVMBool (*LLVMRemoveModule)(LLVMExecutionEngineRef, LLVMModuleRef,
                                    LLVMModuleRef *, char **) = nullptr;
size_t (*LLVMDisasmInstruction)(LLVMDisasmContextRef, uint8_t *, uint64_t,
                                uint64_t, char *, size_t) = nullptr;

#define LLVMDisassembler_Option_PrintImmHex       2
#define LLVMDisassembler_Option_AsmPrinterVariant 4

void *jit_llvm_handle = nullptr;
LLVMDisasmContextRef jit_llvm_disasm = nullptr;
LLVMExecutionEngineRef jit_llvm_engine = nullptr;
LLVMContextRef jit_llvm_context = nullptr;
char *jit_llvm_target_cpu = nullptr;

bool jit_llvm_init_attempted = false;
bool jit_llvm_init_success = false;

#define LOAD(name)                                                             \
    symbol = #name;                                                            \
    name = decltype(name)(dlsym(jit_llvm_handle, symbol));                    \
    if (!name)                                                                 \
        break;                                                                 \
    symbol = nullptr

#define Z(x) x = nullptr

bool jit_llvm_init() {
    if (jit_llvm_init_attempted)
        return jit_llvm_init_success;
    jit_llvm_init_attempted = true;

#if defined(__linux__)
    jit_llvm_handle = dlopen("libLLVM.so", RTLD_LAZY);
#elif defined(__APPLE__)
    jit_llvm_handle = dlopen("libLLVM.dylib", RTLD_LAZY);

    if (!jit_llvm_handle) {
        glob_t g;
        if (glob("/usr/local/Cellar/llvm/*/lib/libLLVM.dylib", 0, nullptr, &g) == 0) {
            for (size_t i = 0; i < g.gl_pathc; ++i) {
                jit_llvm_handle = dlopen(g.gl_pathv[i], RTLD_LAZY);
                if (jit_llvm_handle)
                    break;
            }
            globfree(&g);
        }
    }

#endif

    if (!jit_llvm_handle) {
        jit_log(Warn, "jit_llvm_init(): libLLVM.so/.dylib not found -- "
                      "disabling LLVM backend!");
        return false;
    }

    const char *symbol = nullptr;
    do {
        LOAD(LLVMLinkInMCJIT);
        LOAD(LLVMInitializeX86Target);
        LOAD(LLVMInitializeX86TargetInfo);
        LOAD(LLVMInitializeX86TargetMC);
        LOAD(LLVMInitializeX86AsmPrinter);
        LOAD(LLVMInitializeX86Disassembler);
        LOAD(LLVMGetGlobalContext);
        LOAD(LLVMGetDefaultTargetTriple);
        LOAD(LLVMGetHostCPUName);
        LOAD(LLVMDisposeMessage);
        LOAD(LLVMCreateDisasm);
        LOAD(LLVMDisasmDispose);
        LOAD(LLVMSetDisasmOptions);
        LOAD(LLVMModuleCreateWithName);
        LOAD(LLVMInitializeMCJITCompilerOptions);
        LOAD(LLVMCreateMCJITCompilerForModule);
        LOAD(LLVMDisposeExecutionEngine);
        LOAD(LLVMAddModule);
        LOAD(LLVMDisposeModule);
        LOAD(LLVMCreateMemoryBufferWithMemoryRange);
        LOAD(LLVMParseIRInContext);
        LOAD(LLVMPrintModuleToString);
        LOAD(LLVMGetFunctionAddress);
        LOAD(LLVMRemoveModule);
        LOAD(LLVMDisasmInstruction);
    } while (false);

    if (symbol) {
        jit_log(Warn,
                "jit_llvm_init(): could not find symbol \"%s\" -- disabling "
                "LLVM backend!", symbol);
        return false;
    }

    LLVMLinkInMCJIT();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86Disassembler();

    jit_llvm_context = LLVMGetGlobalContext();
    if (!jit_llvm_context) {
        jit_log(Warn, "jit_llvm_init(): could not obtain context!");
        return false;
    }

    char* triple = LLVMGetDefaultTargetTriple();
    jit_llvm_disasm = LLVMCreateDisasm(triple, nullptr, 0, nullptr, nullptr);

    if (!jit_llvm_disasm) {
        jit_log(Warn, "jit_llvm_init(): could not create a disassembler!");
        LLVMDisposeMessage(triple);
        return false;
    }

    if (LLVMSetDisasmOptions(jit_llvm_disasm,
                             LLVMDisassembler_Option_PrintImmHex |
                             LLVMDisassembler_Option_AsmPrinterVariant) == 0) {
        jit_log(Warn, "jit_llvm_init(): could not configure disassembler!");
        LLVMDisasmDispose(jit_llvm_disasm);
        LLVMDisposeMessage(triple);
        return false;
    }

    LLVMMCJITCompilerOptions options;
    LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
    options.OptLevel = 3;
    options.NoFramePointerElim = false;
    options.EnableFastISel = false;

    LLVMModuleRef enoki_module = LLVMModuleCreateWithName("enoki");
    char *error = nullptr;
    if (LLVMCreateMCJITCompilerForModule(&jit_llvm_engine, enoki_module, &options, sizeof(options), &error)) {
        jit_log(Warn, "jit_llvm_init(): could not create MCJIT: %s", error);
        LLVMDisposeModule(enoki_module);
        LLVMDisasmDispose(jit_llvm_disasm);
        LLVMDisposeMessage(triple);
        return -1;
    }

    jit_llvm_target_cpu = LLVMGetHostCPUName();
    jit_log(Info, "jit_llvm_init(): found %s, cpu=%s", triple, jit_llvm_target_cpu);

    LLVMDisposeMessage(triple);

    jit_llvm_init_success = true;
    return true;
}

void jit_llvm_shutdown() {
    if (!jit_llvm_init_success)
        return;

    jit_log(Info, "jit_llvm_shutdown()");

    LLVMDisasmDispose(jit_llvm_disasm);
    LLVMDisposeExecutionEngine(jit_llvm_engine);
    LLVMDisposeMessage(jit_llvm_target_cpu);
    dlclose(jit_llvm_handle);

    jit_llvm_engine = nullptr;
    jit_llvm_disasm = nullptr;
    jit_llvm_context = nullptr;
    jit_llvm_target_cpu = nullptr;
    jit_llvm_handle = nullptr;

    Z(LLVMLinkInMCJIT); Z(LLVMInitializeX86Target);
    Z(LLVMInitializeX86TargetInfo); Z(LLVMInitializeX86TargetMC);
    Z(LLVMInitializeX86AsmPrinter); Z(LLVMInitializeX86Disassembler);
    Z(LLVMGetGlobalContext); Z(LLVMGetDefaultTargetTriple);
    Z(LLVMGetHostCPUName); Z(LLVMDisposeMessage); Z(LLVMCreateDisasm);
    Z(LLVMDisasmDispose); Z(LLVMSetDisasmOptions); Z(LLVMModuleCreateWithName);
    Z(LLVMInitializeMCJITCompilerOptions); Z(LLVMCreateMCJITCompilerForModule);
    Z(LLVMDisposeExecutionEngine); Z(LLVMAddModule); Z(LLVMDisposeModule);
    Z(LLVMCreateMemoryBufferWithMemoryRange); Z(LLVMParseIRInContext);
    Z(LLVMPrintModuleToString); Z(LLVMGetFunctionAddress); Z(LLVMRemoveModule);
    Z(LLVMDisasmInstruction);

    jit_llvm_init_success = false;
    jit_llvm_init_attempted = false;
}
