#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>

extern "C" void jitc_llvm_set_coff_object_layer_flags(void *layer) {
    /*
       COFF generated constants such as __real@... and __ymm@... are not
       covered by the MaterializationResponsibility symbols that Dr.Jit passes
       to ORCv2. LLJIT's default layer enables these flags on COFF, but the
       C API helper used by Dr.Jit does not expose setters for them. Keep
       Dr.Jit's custom memory manager and set the flags on the C++ layer.
    */
    auto *rt_dyld_layer =
        static_cast<llvm::orc::RTDyldObjectLinkingLayer *>(
            reinterpret_cast<llvm::orc::ObjectLayer *>(layer));
    rt_dyld_layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
    rt_dyld_layer->setAutoClaimResponsibilityForObjectSymbols(true);
}
