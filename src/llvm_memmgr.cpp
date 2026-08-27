#include "llvm_memmgr.h"
#include "log.h"
#include <cstring>
#include <cstdlib>

uint8_t *jitc_llvm_memmgr_allocate(void *opaque, uintptr_t size,
                                   unsigned align, unsigned /* id */,
                                   const char *name) {
    LLVMMemMgrContext &ctx = *(LLVMMemMgrContext *) opaque;

    if (align == 0)
        align = 16;

    jitc_trace("jit_llvm_memmgr_allocate(section=%s, size=%zu, align=%u)", name,
               size, (uint32_t) align);

    /* It's bad news if LLVM decides to create a global offset table entry.
       This usually means that a compiler intrinsic didn't resolve to a machine
       instruction, and a function call to an external library was generated
       along with a relocation, which we don't support. */
    if (strncmp(name, ".got", 4) == 0)
        ctx.got = true;

    size_t offset_align = (ctx.offset + (align - 1)) / align * align,
           offset_new = offset_align + size;

    if (offset_new > ctx.size) {
        ctx.offset = offset_new;
        return nullptr;
    }

    // Zero-fill the padding and the allocation itself. LLVM does not write
    // every allocated byte, and leftover heap contents would otherwise leak
    // into the unit image, which lands in the disk cache.
    memset(ctx.data + ctx.offset, 0, offset_new - ctx.offset);

    ctx.offset = offset_new;
    return ctx.data + offset_align;
}

uint8_t *jitc_llvm_memmgr_allocate_data(void *opaque, uintptr_t size,
                                        unsigned align, unsigned id,
                                        const char *name,
                                        LLVMBool /* read_only */) {
    return jitc_llvm_memmgr_allocate(opaque, size, align, id, name);
}

LLVMBool jitc_llvm_memmgr_finalize(void * /* opaque */, char ** /* err */) {
    return 0;
}

void jitc_llvm_memmgr_destroy(void * /* opaque */) { }

void jitc_llvm_memmgr_prepare(LLVMMemMgrContext &ctx, size_t size) {
    // Central assumption: LLVM text IR is much larger than the resulting generated code.
    size_t target_size = size * 10;

    if (ctx.size <= target_size) {
#if !defined(_WIN32)
        free(ctx.data);
        if (posix_memalign((void **) &ctx.data, 4096, target_size))
            jitc_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", target_size);
#else
        _aligned_free(ctx.data);
        ctx.data = (uint8_t *) _aligned_malloc(target_size, 4096);
        if (!ctx.data)
            jitc_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", target_size);
#endif
        ctx.size = target_size;
    }

    ctx.offset = 0;
    ctx.got = false;
}

void jitc_llvm_memmgr_release(LLVMMemMgrContext &ctx) {
#if !defined(_WIN32)
    free(ctx.data);
#else
    _aligned_free(ctx.data);
#endif

    ctx.data = nullptr;
    ctx.size = 0;
    ctx.offset = 0;
    ctx.got = false;
}

/// The context passed at layer creation is the compiler's LLVMMemMgrContext
void* jitc_llvm_memmgr_create_context(void *ctx) { return ctx; }

void jitc_llvm_memmgr_notify_terminating(void *) { }
