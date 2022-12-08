#include "llvm_memmgr.h"
#include "log.h"
#include <cstring>

/// Internal storage used by the memory manager
uint8_t *jitc_llvm_memmgr_data = nullptr;

/// Was a global offset table (GOT) generated?
bool jitc_llvm_memmgr_got = false;

/// Current position within 'jitc_llvm_memmgr_data'
size_t jitc_llvm_memmgr_offset = 0;

/// Size of the buffer backing 'jitc_llvm_memmgr_data'
static size_t jitc_llvm_memmgr_size = 0;

uint8_t *jitc_llvm_memmgr_allocate(void * /* opaque */, uintptr_t size,
                                   unsigned align, unsigned /* id */,
                                   const char *name) {
    if (align == 0)
        align = 16;

    jitc_trace("jit_llvm_memmgr_allocate(section=%s, size=%zu, align=%u)", name,
               size, (uint32_t) align);

    /* It's bad news if LLVM decides to create a global offset table entry.
       This usually means that a compiler intrinsic didn't resolve to a machine
       instruction, and a function call to an external library was generated
       along with a relocation, which we don't support. */
    if (strncmp(name, ".got", 4) == 0)
        jitc_llvm_memmgr_got = true;

    size_t offset_align = (jitc_llvm_memmgr_offset + (align - 1)) / align * align;

    // Zero-fill including padding region
    memset(jitc_llvm_memmgr_data + jitc_llvm_memmgr_offset, 0,
           offset_align - jitc_llvm_memmgr_offset);

    jitc_llvm_memmgr_offset = offset_align + size;

    if (jitc_llvm_memmgr_offset > jitc_llvm_memmgr_size)
        return nullptr;

    return jitc_llvm_memmgr_data + offset_align;
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


void jitc_llvm_memmgr_prepare(size_t size) {
    // Central assumption: LLVM text IR is much larger than the resulting generated code.
    size_t target_size = size * 10;

    if (jitc_llvm_memmgr_size <= target_size) {
#if !defined(_WIN32)
        free(jitc_llvm_memmgr_data);
        if (posix_memalign((void **) &jitc_llvm_memmgr_data, 4096, target_size))
            jitc_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", target_size);
#else
        _aligned_free(jitc_llvm_memmgr_data);
        jitc_llvm_memmgr_data = (uint8_t *) _aligned_malloc(target_size, 4096);
        if (!jitc_llvm_memmgr_data)
            jitc_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", target_size);
#endif
        jitc_llvm_memmgr_size = target_size;
    }

    jitc_llvm_memmgr_offset = 0;
}

void jitc_llvm_memmgr_shutdown() {
#if !defined(_WIN32)
    free(jitc_llvm_memmgr_data);
#else
    _aligned_free(jitc_llvm_memmgr_data);
#endif

    jitc_llvm_memmgr_data = nullptr;
    jitc_llvm_memmgr_size = 0;
    jitc_llvm_memmgr_offset = 0;
    jitc_llvm_memmgr_got = false;
}

void* jitc_llvm_memmgr_create_context(void *) { return nullptr; }

void jitc_llvm_memmgr_notify_terminating(void *) { }
