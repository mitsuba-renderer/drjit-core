#include "internal.h"
#include "util.h"
#include "var.h"
#include "log.h"

const char *reduction_name[(int) ReductionType::Count] = { "add", "mul", "min",
                                                           "max", "and", "or" };

/// Fill a device memory region with constants of a given type
void jit_fill(VarType type, void *ptr, size_t size, const void *src) {
    jit_trace("jit_fill(" PTR ", type=%s, size=%zu)", ptr,
              var_type_name[(int) type], size);

    Stream *stream = active_stream;
    if (stream) {
        switch (var_type_size[(int) type]) {
            case 1:
                cuda_check(cuMemsetD8Async(ptr, ((uint8_t *) src)[0], size, stream->handle));
                break;

            case 2:
                cuda_check(cuMemsetD16Async(ptr, ((uint16_t *) src)[0], size, stream->handle));
                break;

            case 4:
                cuda_check(cuMemsetD32Async(ptr, ((uint32_t *) src)[0], size, stream->handle));
                break;

            case 8: {
                    int num_sm = state.devices[stream->device].num_sm * 4;
                    void *args[] = { &ptr, &size, (void *) src };
                    cuda_check(cuLaunchKernel(kernel_fill_64, num_sm, 1, 1, 1024,
                                              1, 1, 0, stream->handle, args, nullptr));
                }
                break;

            default:
                jit_raise("jit_fill(): unknown type!");
        }
    } else {
        unlock_guard guard(state.mutex);
        switch (var_type_size[(int) type]) {
            case 1: {
                    uint8_t value = ((uint8_t *) src)[0],
                            *p    = (uint8_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 2: {
                    uint16_t value = ((uint16_t *) src)[0],
                             *p    = (uint16_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 4: {
                    uint32_t value = ((uint32_t *) src)[0],
                             *p    = (uint32_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            case 8: {
                    uint64_t value = ((uint64_t *) src)[0],
                             *p    = (uint64_t *) ptr;
                    for (size_t i = 0; i < size; ++i)
                        p[i] = value;
                }
                break;

            default:
                jit_raise("jit_fill(): unknown type!");
        }
    }
}

void jit_reduce(VarType type, ReductionType rtype, const void *ptr, size_t size,
                void *out) {
    jit_trace("jit_reduce(" PTR ", type=%s, rtype=%s, size=%zu)", ptr,
              var_type_name[(int) type], reduction_name[(int) rtype], size);

    size_t type_size = var_type_size[(int) type];
    Stream *stream = active_stream;

    if (stream) {
        CUfunction func = kernel_reductions[(int) rtype][(int) type];
        if (!func)
            jit_raise(
                "jit_reduce(): no existing kernel for type=%s, rtype=%s!",
                var_type_name[(int) type], reduction_name[(int) rtype]);

        uint32_t num_blocks = state.devices[stream->device].num_sm * 4,
                 num_threads = 1024,
                 shared_size = num_threads * type_size;

        if (size > 1024) {
            // First reduction
            void *args_1[] = { &ptr, &size, &out };
            cuda_check(cuLaunchKernel(func, num_blocks, 1, 1, num_threads, 1, 1,
                                      shared_size, stream->handle, args_1,
                                      nullptr));

            // Second reduction
            size = num_blocks;
            void *args_2[] = { &out, &size, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, num_threads, 1, 1,
                                      shared_size, stream->handle, args_2,
                                      nullptr));
        } else {
            /// This is a small array, do everything in just one reduction.
            void *args[] = { &ptr, &size, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, num_threads, 1, 1,
                                      shared_size, stream->handle, args,
                                      nullptr));
        }
    }
}
