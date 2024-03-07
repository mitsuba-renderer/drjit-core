#include "llvm_internal.h"
#include "log.h"

/// Helper function: enqueue parallel CPU task (synchronous or asynchronous)
template <typename Func>
void jitc_submit_cpu(KernelType type, Func &&func, uint32_t width,
                     uint32_t size = 1) {

    struct Payload { Func f; };
    Payload payload{ std::forward<Func>(func) };

    static_assert(std::is_trivially_copyable<Payload>::value &&
                  std::is_trivially_destructible<Payload>::value, "Internal error!");

    Task *new_task = task_submit_dep(
        nullptr, &jitc_task, 1, size,
        [](uint32_t index, void *payload) { ((Payload *) payload)->f(index); },
        &payload, sizeof(Payload), nullptr, 0);

    if (unlikely(jit_flag(JitFlag::LaunchBlocking))) {
        unlock_guard guard(state.lock);
        task_wait(new_task);
    }

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        KernelHistoryEntry entry = {};
        entry.backend = JitBackend::LLVM;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        task_retain(new_task);
        entry.task = new_task;
        state.kernel_history.append(entry);
    }

    task_release(jitc_task);
    jitc_task = new_task;
}

void LLVMThreadState::jitc_memset_async(void *ptr, uint32_t size_,
                                        uint32_t isize, const void *src){
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("LLVMThreadState::jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jitc_trace("LLVMThreadState::jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
              (uintptr_t) ptr, isize, size_);

    if (size_ == 0)
        return;

    size_t size = size_;

    // Try to convert into ordinary memset if possible
    uint64_t zero = 0;
    if (memcmp(src, &zero, isize) == 0) {
        size *= isize;
        isize = 1;
    }


    // LLVM Specific
    uint8_t src8[8] { };
    memcpy(&src8, src, isize);

    jitc_submit_cpu(KernelType::Other,
        [ptr, src8, size, isize](uint32_t) {
            switch (isize) {
                case 1:
                    memset(ptr, src8[0], size);
                    break;

                case 2: {
                        uint16_t value = ((uint16_t *) src8)[0],
                                *p = (uint16_t *) ptr;
                        for (uint32_t i = 0; i < size; ++i)
                            p[i] = value;
                    }
                    break;

                case 4: {
                        uint32_t value = ((uint32_t *) src8)[0],
                                *p = (uint32_t *) ptr;
                        for (uint32_t i = 0; i < size; ++i)
                            p[i] = value;
                    }
                    break;

                case 8: {
                        uint64_t value = ((uint64_t *) src8)[0],
                                *p = (uint64_t *) ptr;
                        for (uint32_t i = 0; i < size; ++i)
                            p[i] = value;
                    }
                    break;
            }
        },

        (uint32_t) size
    );
}

void LLVMThreadState::jitc_memcpy(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
}
