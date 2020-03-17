#include "llvm_api.h"
#include "internal.h"
#include "log.h"
#include <glob.h>
#include <dlfcn.h>

/// LLVM API
using LLVMBool = int;
using LLVMDisasmContextRef = void *;
using LLVMExecutionEngineRef = void *;
using LLVMModuleRef = void *;
using LLVMMemoryBufferRef = void *;
using LLVMContextRef = void *;
using LLVMMCJITMemoryManagerRef = void *;

struct LLVMMCJITCompilerOptions {
  unsigned OptLevel;
  int CodeModel;
  LLVMBool NoFramePointerElim;
  LLVMBool EnableFastISel;
  void *MCJMM;
};

static void (*LLVMLinkInMCJIT)() = nullptr;
static void (*LLVMInitializeX86AsmPrinter)() = nullptr;
static void (*LLVMInitializeX86Disassembler)() = nullptr;
static void (*LLVMInitializeX86Target)() = nullptr;
static void (*LLVMInitializeX86TargetInfo)() = nullptr;
static void (*LLVMInitializeX86TargetMC)() = nullptr;
static void (*LLVMDisposeMessage)(char *) = nullptr;
static char *(*LLVMGetDefaultTargetTriple)() = nullptr;
static char *(*LLVMGetHostCPUName)() = nullptr;
static char *(*LLVMGetHostCPUFeatures)() = nullptr;
static LLVMContextRef (*LLVMGetGlobalContext)() = nullptr;
static LLVMDisasmContextRef (*LLVMCreateDisasm)(const char *, void *, int,
                                                void *, void *) = nullptr;
static void (*LLVMDisasmDispose)(LLVMDisasmContextRef) = nullptr;
static int (*LLVMSetDisasmOptions)(LLVMDisasmContextRef, uint64_t) = nullptr;
static LLVMModuleRef (*LLVMModuleCreateWithName)(const char *) = nullptr;
static LLVMBool (*LLVMCreateMCJITCompilerForModule)(LLVMExecutionEngineRef *,
                                                    LLVMModuleRef,
                                                    LLVMMCJITCompilerOptions *,
                                                    size_t, char **) = nullptr;
static LLVMMCJITMemoryManagerRef (*LLVMCreateSimpleMCJITMemoryManager)(
    void *, void *, void *, void *, void *) = nullptr;
static void (*LLVMDisposeExecutionEngine)(LLVMExecutionEngineRef) = nullptr;
static void (*LLVMAddModule)(LLVMExecutionEngineRef, LLVMModuleRef) = nullptr;
static void (*LLVMDisposeModule)(LLVMModuleRef) = nullptr;
static LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool) = nullptr;
static LLVMBool (*LLVMParseIRInContext)(LLVMContextRef, LLVMMemoryBufferRef,
                                        LLVMModuleRef *, char **) = nullptr;
static char *(*LLVMPrintModuleToString)(LLVMModuleRef) = nullptr;
static uint64_t (*LLVMGetFunctionAddress)(LLVMExecutionEngineRef, const char *);
static LLVMBool (*LLVMRemoveModule)(LLVMExecutionEngineRef, LLVMModuleRef,
                                    LLVMModuleRef *, char **) = nullptr;
static size_t (*LLVMDisasmInstruction)(LLVMDisasmContextRef, uint8_t *,
                                       uint64_t, uint64_t, char *,
                                       size_t) = nullptr;

#define LLVMDisassembler_Option_PrintImmHex       2
#define LLVMDisassembler_Option_AsmPrinterVariant 4
#define LLVMCodeModelSmall 3

/// Enoki API
static void *jit_llvm_handle                  = nullptr;
static LLVMDisasmContextRef jit_llvm_disasm   = nullptr;
static LLVMExecutionEngineRef jit_llvm_engine = nullptr;
static LLVMContextRef jit_llvm_context        = nullptr;
char *jit_llvm_target_cpu                     = nullptr;
char *jit_llvm_target_features                = nullptr;
int   jit_llvm_vector_width                   = 0;


static bool     jit_llvm_init_attempted = false;
static bool     jit_llvm_init_success = false;
static uint8_t *jit_llvm_mem_start  = nullptr;
static size_t   jit_llvm_mem_size   = 0;
static size_t   jit_llvm_mem_offset = 0;

extern "C" {

static uint8_t *jit_llvm_mem_allocate(void * /* opaque */, uintptr_t size,
                                      unsigned align, unsigned /* id */,
                                      const char *name) {
    jit_trace("jit_llvm_mem_allocate(section=%s, size=%lu);", name, size);

    if (jit_llvm_mem_offset == 0 && strstr(name, "text") == nullptr)
        jit_fail(
            "jit_llvm_mem_allocate_code(): started with unknown section \"%s\"",
            name);

    if (align == 0)
        align = 16;

    size_t offset_align = (jit_llvm_mem_offset + (align - 1)) / align * align;

    if (offset_align + size > jit_llvm_mem_size) {
        size_t alloc_size =
            std::max(jit_llvm_mem_size * 2, offset_align + size);
        void *ptr = nullptr;
        int rv = posix_memalign(&ptr, 64, alloc_size);
        if (unlikely(rv))
            jit_fail("jit_llvm_mem_allocate_code(): out of memory!");
        if (jit_llvm_mem_start) {
            memcpy(ptr, jit_llvm_mem_start, jit_llvm_mem_offset);
            free(jit_llvm_mem_start);
        }
        jit_llvm_mem_start = (uint8_t *) ptr;
        jit_llvm_mem_size = alloc_size;
    }

    // Zero-fill padding region for alignment
    memset(jit_llvm_mem_start + jit_llvm_mem_offset, 0,
           offset_align - jit_llvm_mem_offset);

    uint8_t *target = jit_llvm_mem_start + offset_align;
    jit_llvm_mem_offset = offset_align + size;

    return target;
}

static uint8_t *jit_llvm_mem_allocate_data(void *opaque, uintptr_t size,
                                           unsigned align, unsigned id,
                                           const char *name,
                                           LLVMBool /* read_only */) {
    return jit_llvm_mem_allocate(opaque, size, align, id, name);
}

static LLVMBool jit_llvm_mem_finalize(void * /* opaque */, char ** /* err */) {
    jit_trace("jit_llvm_mem_finalize();");
    return 0;
}

static void jit_llvm_mem_destroy(void * /* opaque */) {
    jit_trace("jit_llvm_mem_destroy();");
    free(jit_llvm_mem_start);
    jit_llvm_mem_start  = nullptr;
    jit_llvm_mem_size   = 0;
    jit_llvm_mem_offset = 0;
}
};

void jit_llvm_compile(const char *buffer, size_t buffer_size) {
    jit_llvm_mem_offset = 0;

    LLVMMemoryBufferRef buf = LLVMCreateMemoryBufferWithMemoryRange(
        buffer, buffer_size, "enoki_kernel", 0);
    if (unlikely(!buf))
        jit_fail("jit_run_compile(): could not create memory buffer!");

    // 'buf' is consumed by this function.
    LLVMModuleRef module = nullptr;
    char *error = nullptr;
    LLVMParseIRInContext(jit_llvm_context, buf, &module, &error);
    if (unlikely(error))
        jit_fail("jit_llvm_compile(): could not parse IR: %s.\n", error);

    if (false) {
        char *llvm_ir = LLVMPrintModuleToString(module);
        jit_trace("jit_llvm_compile(): Parsed LLVM IR:\n%s", llvm_ir);
        LLVMDisposeMessage(llvm_ir);
    }

    LLVMAddModule(jit_llvm_engine, module);

    uint8_t *ptr = (uint8_t *) LLVMGetFunctionAddress(jit_llvm_engine, "enoki_kernel");
    // if (unlikely(ptr != jit_llvm_mem_start))
    //     jit_fail(
    //         "jit_llvm_compile(): internal error: address mismatch: %p vs %p.\n",
    //         ptr, jit_llvm_mem_start);

    char ins_buf[256];
    do {
        size_t cur_offset = ptr - jit_llvm_mem_start;
        if (cur_offset >= jit_llvm_mem_offset)
            break;
        size_t size = LLVMDisasmInstruction(
            jit_llvm_disasm, ptr, jit_llvm_mem_offset - cur_offset,
            (uintptr_t) ptr, ins_buf, sizeof(ins_buf));
        if (size == 0)
            break;
        char *start = ins_buf;
        while (*start == ' ' || *start == '\t')
            ++start;
        jit_trace("0x%08llx   %s", cur_offset, start);
        ptr += size;
    } while (true);

    LLVMRemoveModule(jit_llvm_engine, module, &module, &error);
    if (unlikely(error))
        jit_fail("jit_llvm_compile(): could remove module: %s.\n", error);
}

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
        LOAD(LLVMGetHostCPUFeatures);
        LOAD(LLVMDisposeMessage);
        LOAD(LLVMCreateDisasm);
        LOAD(LLVMDisasmDispose);
        LOAD(LLVMSetDisasmOptions);
        LOAD(LLVMModuleCreateWithName);
        LOAD(LLVMCreateMCJITCompilerForModule);
        LOAD(LLVMCreateSimpleMCJITMemoryManager);
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
    options.OptLevel = 3;
    options.CodeModel = LLVMCodeModelSmall;
    options.NoFramePointerElim = false;
    options.EnableFastISel = false;
    options.MCJMM = LLVMCreateSimpleMCJITMemoryManager(
        nullptr,
        (void *) jit_llvm_mem_allocate,
        (void *) jit_llvm_mem_allocate_data,
        (void *) jit_llvm_mem_finalize,
        (void *) jit_llvm_mem_destroy);

    LLVMModuleRef enoki_module = LLVMModuleCreateWithName("enoki");
    char *error = nullptr;
    if (LLVMCreateMCJITCompilerForModule(&jit_llvm_engine, enoki_module,
                                         &options, sizeof(options), &error)) {
        jit_log(Warn, "jit_llvm_init(): could not create MCJIT: %s", error);
        LLVMDisposeModule(enoki_module);
        LLVMDisasmDispose(jit_llvm_disasm);
        LLVMDisposeMessage(triple);
        return -1;
    }

    jit_llvm_target_cpu = LLVMGetHostCPUName();
    jit_llvm_target_features = LLVMGetHostCPUFeatures();
    jit_llvm_vector_width = 1;

    if (strstr(jit_llvm_target_features, "+sse4.2"))
        jit_llvm_vector_width = 4;
    if (strstr(jit_llvm_target_features, "+avx"))
        jit_llvm_vector_width = 8;
    if (strstr(jit_llvm_target_features, "+avx512f"))
        jit_llvm_vector_width = 16;

    jit_log(Info, "jit_llvm_init(): found %s, cpu=%s, width=%i", triple,
            jit_llvm_target_cpu, jit_llvm_vector_width);

    LLVMDisposeMessage(triple);

    jit_llvm_init_success = jit_llvm_vector_width > 1;

    if (!jit_llvm_init_success) {
        jit_log(Warn, "jit_llvm_init(): no suitable vector ISA found, shutting "
                      "down LLVM backend..");
        jit_llvm_shutdown();
    }

    return jit_llvm_init_success;
}

void jit_llvm_shutdown() {
    if (!jit_llvm_init_success)
        return;

    jit_log(Info, "jit_llvm_shutdown()");

    LLVMDisasmDispose(jit_llvm_disasm);
    LLVMDisposeExecutionEngine(jit_llvm_engine);
    LLVMDisposeMessage(jit_llvm_target_cpu);
    LLVMDisposeMessage(jit_llvm_target_features);
    dlclose(jit_llvm_handle);

    jit_llvm_engine = nullptr;
    jit_llvm_disasm = nullptr;
    jit_llvm_context = nullptr;
    jit_llvm_target_cpu = nullptr;
    jit_llvm_target_features = nullptr;
    jit_llvm_handle = nullptr;
    jit_llvm_vector_width = 0;

    Z(LLVMLinkInMCJIT); Z(LLVMInitializeX86Target);
    Z(LLVMInitializeX86TargetInfo); Z(LLVMInitializeX86TargetMC);
    Z(LLVMInitializeX86AsmPrinter); Z(LLVMInitializeX86Disassembler);
    Z(LLVMGetGlobalContext); Z(LLVMGetDefaultTargetTriple);
    Z(LLVMGetHostCPUName); Z(LLVMGetHostCPUFeatures); Z(LLVMDisposeMessage);
    Z(LLVMCreateDisasm); Z(LLVMDisasmDispose); Z(LLVMSetDisasmOptions);
    Z(LLVMModuleCreateWithName); Z(LLVMCreateMCJITCompilerForModule);
    Z(LLVMCreateSimpleMCJITMemoryManager); Z(LLVMDisposeExecutionEngine);
    Z(LLVMAddModule); Z(LLVMDisposeModule);
    Z(LLVMCreateMemoryBufferWithMemoryRange); Z(LLVMParseIRInContext);
    Z(LLVMPrintModuleToString); Z(LLVMGetFunctionAddress); Z(LLVMRemoveModule);
    Z(LLVMDisasmInstruction);

    jit_llvm_init_success = false;
    jit_llvm_init_attempted = false;
}
