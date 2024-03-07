#include "cuda_internal.h"
#include "log.h"

void jitc_submit_gpu2(KernelType type, CUfunction kernel, uint32_t block_count,
                     uint32_t thread_count, uint32_t shared_mem_bytes,
                     CUstream stream, void **args, void **extra,
                     uint32_t width) {

    KernelHistoryEntry entry = {};

    uint32_t flags = jit_flags();

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        cuda_check(cuEventCreate((CUevent *) &entry.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &entry.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) entry.event_start, stream));
    }

    cuda_check(cuLaunchKernel(kernel, block_count, 1, 1, thread_count, 1, 1,
                              shared_mem_bytes, stream, args, extra));

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        cuda_check(cuStreamSynchronize(stream));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        entry.backend = JitBackend::CUDA;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        cuda_check(cuEventRecord((CUevent) entry.event_end, stream));

        state.kernel_history.append(entry);
    }
}

/// Fill a device memory region with constants of a given type
void CUDAThreadState::jitc_memset_async(void *ptr, uint32_t size_,
                                        uint32_t isize, const void *src){
    
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("CUDAThreadState::jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jitc_trace("CUDAThreadState::jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
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
    
    // CUDA Specific
    scoped_set_context guard(this->context);
    switch (isize) {
        case 1:
            cuda_check(cuMemsetD8Async((CUdeviceptr) ptr,
                                       ((uint8_t *) src)[0], size,
                                       this->stream));
            break;

        case 2:
            cuda_check(cuMemsetD16Async((CUdeviceptr) ptr,
                                        ((uint16_t *) src)[0], size,
                                        this->stream));
            break;

        case 4:
            cuda_check(cuMemsetD32Async((CUdeviceptr) ptr,
                                        ((uint32_t *) src)[0], size,
                                        this->stream));
            break;

        case 8: {
                const Device &device = state.devices[this->device];
                uint32_t block_count, thread_count;
                device.get_launch_config(&block_count, &thread_count, size_);
                void *args[] = { &ptr, &size_, (void *) src };
                CUfunction kernel = jitc_cuda_fill_64[device.id];
                jitc_submit_gpu2(KernelType::Other, kernel, block_count,
                                thread_count, 0, this->stream, args, nullptr,
                                size_);
            }
            break;
    }
}

void CUDAThreadState::jitc_memcpy(void *dst, const void *src, size_t size) {
    scoped_set_context guard_2(this->context);
    cuda_check(cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, size));
}
