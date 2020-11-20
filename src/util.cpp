/*
    src/util.cpp -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <enoki-jit/util.h>
#include <condition_variable>
#include "internal.h"
#include "util.h"
#include "var.h"
#include "eval.h"
#include "registry.h"
#include "log.h"
#include "profiler.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

const char *reduction_name[(int) ReductionType::Count] = { "sum", "mul", "min",
                                                           "max", "and", "or" };

/// Helper function: enqueue a single CPU task (synchronous or asynchronous)
template <typename Input>
void jit_submit_cpu(Stream *stream, void (*func)(size_t, void *), Input &input,
                     size_t size = 1, bool release_prev = true) {
    if (stream->parallel_dispatch) {
        Task *new_task = pool_task_submit(
            nullptr, &stream->task, 1, size,
            func, &input, sizeof(Input));

        if (release_prev)
            pool_task_release(stream->task);

        stream->task = new_task;
    } else {
        unlock_guard guard(state.mutex);
        for (size_t i = 0; i < size; ++i)
            func(i, &input);
    }
}

/// Fill a device memory region with constants of a given type
void jit_memset_async(void *ptr, uint32_t size_, uint32_t isize, const void *src) {
    Stream *stream = active_stream;

    if (unlikely(!stream))
        jit_raise("jit_memset_async(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");
    else if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jit_raise("jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jit_trace("jit_memset_async(" ENOKI_PTR ", isize=%u, size=%u)",
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

    if (stream->cuda) {
        scoped_set_context guard(stream->context);
        switch (isize) {
            case 1:
                cuda_check(cuMemsetD8Async((CUdeviceptr) ptr,
                                           ((uint8_t *) src)[0], size,
                                           stream->handle));
                break;

            case 2:
                cuda_check(cuMemsetD16Async((CUdeviceptr) ptr,
                                            ((uint16_t *) src)[0], size,
                                            stream->handle));
                break;

            case 4:
                cuda_check(cuMemsetD32Async((CUdeviceptr) ptr,
                                            ((uint32_t *) src)[0], size,
                                            stream->handle));
                break;

            case 8: {
                    const Device &device = state.devices[stream->device];
                    uint32_t block_count, thread_count;
                    device.get_launch_config(&block_count, &thread_count, size_);
                    void *args[] = { &ptr, &size_, (void *) src };
                    CUfunction kernel = jit_cuda_fill_64[device.id];
                    cuda_check(cuLaunchKernel(kernel, block_count, 1, 1,
                                              thread_count, 1, 1, 0,
                                              stream->handle, args, nullptr));
                }
                break;
        }
    } else {
        struct Input {
            void *ptr;
            size_t size;
            uint32_t isize;
            uint8_t src[8];
        };

        Input input;
        input.ptr = ptr;
        input.size = size;
        input.isize = isize;
        memcpy(input.src, src, isize);

        auto func = [](size_t, void *input_) {
            Input input = *((Input *) input_);
            switch (input.isize) {
                case 1:
                    memset(input.ptr, input.src[0], input.size);
                    break;

                case 2: {
                        uint16_t value = ((uint16_t *) input.src)[0],
                                 *p    = (uint16_t *) input.ptr;
                        for (uint32_t i = 0; i < input.size; ++i)
                            p[i] = value;
                    }
                    break;

                case 4: {
                        uint32_t value = ((uint32_t *) input.src)[0],
                                 *p    = (uint32_t *) input.ptr;
                        for (uint32_t i = 0; i < input.size; ++i)
                            p[i] = value;
                    }
                    break;

                case 8: {
                        uint64_t value = ((uint64_t *) input.src)[0],
                                 *p    = (uint64_t *) input.ptr;
                        for (uint32_t i = 0; i < input.size; ++i)
                            p[i] = value;
                    }
                    break;
            }
        };

        jit_submit_cpu(stream, func, input);
    }
}

/// Perform a synchronous copy operation
void jit_memcpy(void *dst, const void *src, size_t size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_memcpy(): you must invoke jitc_set_device() to choose a "
                  "target device before calling this function.");

    // Temporarily release the lock while copying
    unlock_guard guard(state.mutex);
    if  (stream->cuda) {
        scoped_set_context guard(stream->context);
        cuda_check(cuStreamSynchronize(stream->handle));
        cuda_check(cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, size));
    } else {
        jit_sync_stream(stream);
        memcpy(dst, src, size);
    }
}

/// Perform an asynchronous copy operation
void jit_memcpy_async(void *dst, const void *src, size_t size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_memcpy_async(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");

    if  (stream->cuda) {
        scoped_set_context guard(stream->context);
        cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, size,
                                 stream->handle));
    } else {
        struct Input {
            void *dst;
            const void *src;
            size_t size;
        };

        Input input { dst, src, size };

        jit_submit_cpu(
            stream,
            [](size_t, void *input_) {
                Input input = *((Input *) input_);
                memcpy(input.dst, input.src, input.size);
            },
            input);
    }
}

using Reduction = void (*) (const void *ptr, uint32_t start, uint32_t end, void *out);

template <typename Value>
static Reduction jit_reduce_create(ReductionType rtype) {
    using UInt = enoki::uint_with_size_t<Value>;

    switch (rtype) {
        case ReductionType::Add:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result += ptr[i];
                *((Value *) out) = result;
            };

        case ReductionType::Mul:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = 1;
                for (uint32_t i = start; i != end; ++i)
                    result *= ptr[i];
                *((Value *) out) = result;
            };

        case ReductionType::Max:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::min()
                                   : -std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::max(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReductionType::Min:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::max()
                                   :  std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::max(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReductionType::Or:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result |= ptr[i];
                *((UInt *) out) = result;
            };

        case ReductionType::And:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = (UInt) -1;
                for (uint32_t i = start; i != end; ++i)
                    result &= ptr[i];
                *((UInt *) out) = result;
            };

        default: jit_raise("jit_reduce_create(): unsupported reduction type!");
            return nullptr;
    }
}

static Reduction jit_reduce_create(VarType type, ReductionType rtype) {
    switch (type) {
        case VarType::Int8:    return jit_reduce_create<int8_t  >(rtype);
        case VarType::UInt8:   return jit_reduce_create<uint8_t >(rtype);
        case VarType::Int16:   return jit_reduce_create<int16_t >(rtype);
        case VarType::UInt16:  return jit_reduce_create<uint16_t>(rtype);
        case VarType::Int32:   return jit_reduce_create<int32_t >(rtype);
        case VarType::UInt32:  return jit_reduce_create<uint32_t>(rtype);
        case VarType::Int64:   return jit_reduce_create<int64_t >(rtype);
        case VarType::UInt64:  return jit_reduce_create<uint64_t>(rtype);
        case VarType::Float32: return jit_reduce_create<float   >(rtype);
        case VarType::Float64: return jit_reduce_create<double  >(rtype);
        default: jit_raise("jit_reduce_create(): unsupported data type!");
            return nullptr;
    }
}

void jit_reduce(VarType type, ReductionType rtype, const void *ptr, uint32_t size,
                void *out) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_reduce(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");

    jit_log(Debug, "jit_reduce(" ENOKI_PTR ", type=%s, rtype=%s, size=%u)",
            (uintptr_t) ptr, var_type_name[(int) type],
            reduction_name[(int) rtype], size);

    uint32_t type_size = var_type_size[(int) type];

    if (stream->cuda) {
        scoped_set_context guard(stream->context);
        const Device &device = state.devices[stream->device];
        CUfunction func = jit_cuda_reductions[(int) rtype][(int) type][device.id];
        if (!func)
            jit_raise("jit_reduce(): no existing kernel for type=%s, rtype=%s!",
                      var_type_name[(int) type], reduction_name[(int) rtype]);

        uint32_t thread_count = 1024,
                 shared_size = thread_count * type_size,
                 block_count;

        device.get_launch_config(&block_count, nullptr, size, thread_count);

        if (size <= 1024) {
            /// This is a small array, do everything in just one reduction.
            void *args[] = { &ptr, &size, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, thread_count, 1, 1,
                                      shared_size, stream->handle, args,
                                      nullptr));
        } else {
            void *temp = jit_malloc(AllocType::Device, size_t(block_count) * type_size);

            // First reduction
            void *args_1[] = { &ptr, &size, &temp };
            cuda_check(cuLaunchKernel(func, block_count, 1, 1, thread_count, 1, 1,
                                      shared_size, stream->handle, args_1,
                                      nullptr));

            // Second reduction
            void *args_2[] = { &temp, &block_count, &out };
            cuda_check(cuLaunchKernel(func, 1, 1, 1, thread_count, 1, 1,
                                      shared_size, stream->handle, args_2,
                                      nullptr));

            jit_free(temp);
        }
    } else {
        uint32_t block_size = size, blocks = 1;
        if (stream->parallel_dispatch) {
            block_size = ENOKI_POOL_BLOCK_SIZE;
            blocks     = (size + block_size - 1) / block_size;
        }

        struct Input {
            uint32_t block_size;
            uint32_t size;
            uint32_t isize;
            Reduction reduction;
            const void *source;
            void *target;
        };

        Input input;
        input.block_size = block_size;
        input.size = size;
        input.isize = var_type_size[(int) type];
        input.reduction = jit_reduce_create(type, rtype);

        input.source = ptr;
        if (blocks <= 1)
            input.target = out;
        else
            input.target = jit_malloc(AllocType::HostAsync, blocks * input.isize);

        jit_submit_cpu(
            stream,
            [](size_t index, void *input_) {
                Input input = *(Input *) input_;
                input.reduction(
                    input.source,
                    index * input.block_size,
                    std::min(((uint32_t) index + 1) * input.block_size, input.size),
                    (uint8_t *) input.target + index * input.isize
                );
            },
            input, std::max(1u, blocks)
        );

        if (blocks > 1) {
            jit_reduce(type, rtype, input.target, blocks, out);
            jit_free(input.target);
        }
    }
}

/// 'All' reduction for boolean arrays
uint8_t jit_all(uint8_t *values, uint32_t size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_all(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jit_log(Debug, "jit_all(" ENOKI_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = true;
        jit_memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    uint8_t result;
    if (stream->cuda) {
        uint8_t *out = (uint8_t *) jit_malloc(AllocType::HostPinned, 4);
        jit_reduce(VarType::UInt32, ReductionType::And, values, reduced_size, out);
        jit_sync_stream();
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
        jit_free(out);
    } else {
        uint8_t out[4];
        jit_reduce(VarType::UInt32, ReductionType::And, values, reduced_size, out);
        jit_sync_stream();
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
    }

    return result;
}

/// 'Any' reduction for boolean arrays
uint8_t jit_any(uint8_t *values, uint32_t size) {
    Stream *stream = active_stream;
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jit_log(Debug, "jit_any(" ENOKI_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = false;
        jit_memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    uint8_t result;
    if (stream->cuda) {
        uint8_t *out = (uint8_t *) jit_malloc(AllocType::HostPinned, 4);
        jit_reduce(VarType::UInt32, ReductionType::Or, values, reduced_size, out);
        jit_sync_stream();
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
        jit_free(out);
    } else {
        uint8_t out[4];
        jit_reduce(VarType::UInt32, ReductionType::Or, values, reduced_size, out);
        jit_sync_stream();
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
    }

    return result;
}

/// Exclusive prefix sum
void jit_scan_u32(const uint32_t *in, uint32_t size, uint32_t *out) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_scan_u32(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");

    if (stream->cuda) {
        const Device &device = state.devices[stream->device];
        scoped_set_context guard(stream->context);

        if (size == 0) {
            return;
        } else if (size == 1) {
            cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, sizeof(uint32_t),
                                       stream->handle));
        } else if (size <= 4096) {
            /// Kernel for small arrays
            uint32_t items_per_thread = 4,
                     thread_count     = round_pow2((size + items_per_thread - 1)
                                                    / items_per_thread),
                     shared_size      = thread_count * 2 * sizeof(uint32_t);

            jit_log(Debug,
                    "jit_scan(" ENOKI_PTR " -> " ENOKI_PTR
                    ", size=%u, type=small, threads=%u, shared=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, thread_count,
                    shared_size);

            void *args[] = { &in, &out, &size };
            cuda_check(cuLaunchKernel(jit_cuda_scan_small_u32[device.id], 1, 1, 1,
                                      thread_count, 1, 1, shared_size,
                                      stream->handle, args, nullptr));
        } else {
            /// Kernel for large arrays
            uint32_t items_per_thread = 16,
                     thread_count     = 128,
                     items_per_block  = items_per_thread * thread_count,
                     block_count      = (size + items_per_block - 1) / items_per_block,
                     shared_size      = items_per_block * sizeof(uint32_t),
                     scratch_items    = block_count + 32;

            jit_log(Debug,
                    "jit_scan(" ENOKI_PTR " -> " ENOKI_PTR
                    ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                    "scratch=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, block_count,
                    thread_count, shared_size, scratch_items * 4);

            uint64_t *scratch = (uint64_t *) jit_malloc(
                AllocType::Device, scratch_items * sizeof(uint64_t));

            /// Initialize scratch space and padding
            uint32_t block_count_init, thread_count_init;
            device.get_launch_config(&block_count_init, &thread_count_init,
                                     scratch_items);

            void *args[] = { &scratch, &scratch_items };
            cuda_check(cuLaunchKernel(jit_cuda_scan_large_u32_init[device.id],
                                      block_count_init, 1, 1, thread_count_init, 1, 1, 0,
                                      stream->handle, args, nullptr));

            scratch += 32; // move beyond padding area
            void *args_2[] = { &in, &out, &scratch };
            cuda_check(cuLaunchKernel(jit_cuda_scan_large_u32[device.id], block_count, 1, 1,
                                      thread_count, 1, 1, shared_size,
                                      stream->handle, args_2, nullptr));
            scratch -= 32;

            jit_free(scratch);
        }
    } else {
        uint32_t block_size = size, blocks = 1;
        if (stream->parallel_dispatch) {
            block_size = ENOKI_POOL_BLOCK_SIZE;
            blocks     = (size + block_size - 1) / block_size;
        }

        jit_log(Debug,
                "jit_scan(" ENOKI_PTR " -> " ENOKI_PTR
                ", size=%u, block_size=%u, blocks=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

        struct Input {
            uint32_t block_size;
            uint32_t size;
            uint32_t isize;
            const uint32_t *in;
            uint32_t *out;
            uint32_t *scratch;
        };

        Input input;
        input.block_size = block_size;
        input.size = size;
        input.in = in;
        input.out = out;
        input.scratch = nullptr;

        if (blocks > 1) {
            input.scratch = (uint32_t *) jit_malloc(AllocType::HostAsync,
                                                    blocks * sizeof(uint32_t));

            jit_submit_cpu(
                stream,
                [](size_t index, void *input_) {
                    Input input = *(Input *) input_;
                    uint32_t start = index * input.block_size,
                             end = std::min(start + input.block_size, input.size);

                    uint32_t accum = 0;
                    for (uint32_t i = start; i != end; ++i)
                        accum += input.in[i];

                    input.scratch[index] = accum;
                },

                input, blocks
            );

            jit_scan_u32(input.scratch, blocks, input.scratch);
        }

        jit_submit_cpu(
            stream,
            [](size_t index, void *input_) {
                Input input = *(Input *) input_;
                uint32_t start = index * input.block_size,
                         end = std::min(start + input.block_size, input.size);

                uint32_t accum = 0;
                if (input.scratch)
                    accum = input.scratch[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t value = input.in[i];
                    input.out[i] = accum;
                    accum += value;
                }
            },

            input, blocks
        );

        if (blocks > 1)
            jit_free(input.scratch);
    }
}

/// Mask compression
uint32_t jit_compress(const uint8_t *in, uint32_t size, uint32_t *out) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_compress(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");

    if (size == 0)
        return 0;

    if (stream->cuda) {
        const Device &device = state.devices[stream->device];
        scoped_set_context guard(stream->context);

        uint32_t *count_out = (uint32_t *) jit_malloc(
            AllocType::HostPinned, sizeof(uint32_t));

        if (size <= 4096) {
            /// Kernel for small arrays
            uint32_t items_per_thread = 4,
                     thread_count     = round_pow2((size + items_per_thread - 1)
                                                    / items_per_thread),
                     shared_size      = thread_count * 2 * sizeof(uint32_t),
                     trailer          = thread_count * items_per_thread - size;

            jit_log(Debug,
                    "jit_compress(" ENOKI_PTR " -> " ENOKI_PTR
                    ", size=%u, type=small, threads=%u, shared=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, thread_count,
                    shared_size);

            if (trailer > 0)
                cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                           stream->handle));

            void *args[] = { &in, &out, &size, &count_out };
            cuda_check(cuLaunchKernel(jit_cuda_compress_small[device.id], 1, 1, 1,
                                      thread_count, 1, 1, shared_size,
                                      stream->handle, args, nullptr));
        } else {
            /// Kernel for large arrays
            uint32_t items_per_thread = 16,
                     thread_count     = 128,
                     items_per_block  = items_per_thread * thread_count,
                     block_count      = (size + items_per_block - 1) / items_per_block,
                     shared_size      = items_per_block * sizeof(uint32_t),
                     scratch_items    = block_count + 32,
                     trailer          = items_per_block * block_count - size;

            jit_log(Debug,
                    "jit_compress(" ENOKI_PTR " -> " ENOKI_PTR
                    ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                    "scratch=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, block_count,
                    thread_count, shared_size, scratch_items * 4);

            uint64_t *scratch = (uint64_t *) jit_malloc(
                AllocType::Device, scratch_items * sizeof(uint64_t));

            /// Initialize scratch space and padding
            uint32_t block_count_init, thread_count_init;
            device.get_launch_config(&block_count_init, &thread_count_init,
                                     scratch_items);

            void *args[] = { &scratch, &scratch_items };
            cuda_check(cuLaunchKernel(jit_cuda_scan_large_u32_init[device.id],
                                      block_count_init, 1, 1, thread_count_init, 1, 1, 0,
                                      stream->handle, args, nullptr));

            if (trailer > 0)
                cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                           stream->handle));

            scratch += 32; // move beyond padding area
            void *args_2[] = { &in, &out, &scratch, &count_out };
            cuda_check(cuLaunchKernel(jit_cuda_compress_large[device.id], block_count, 1, 1,
                                      thread_count, 1, 1, shared_size,
                                      stream->handle, args_2, nullptr));
            scratch -= 32;

            jit_free(scratch);
        }
        jit_sync_stream();
        uint32_t count_out_v = *count_out;
        jit_free(count_out);
        return count_out_v;
    } else {
        uint32_t block_size = size, blocks = 1;
        if (stream->parallel_dispatch) {
            block_size = ENOKI_POOL_BLOCK_SIZE;
            blocks     = (size + block_size - 1) / block_size;
        }

        uint32_t count_out = 0;

        jit_log(Debug,
                "jit_compress(" ENOKI_PTR " -> " ENOKI_PTR
                ", size=%u, block_size=%u, blocks=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

        struct Input {
            uint32_t block_size;
            uint32_t size;
            const uint8_t *in;
            uint32_t *out;
            uint32_t *scratch;
            uint32_t *count_out;
        };

        Input input{ block_size, size, in, out, nullptr, &count_out };

        if (blocks > 1) {
            input.scratch = (uint32_t *) jit_malloc(AllocType::HostAsync,
                                                    blocks * sizeof(uint32_t));

            jit_submit_cpu(
                stream,
                [](size_t index, void *input_) {
                    Input input = *(Input *) input_;
                    uint32_t start = index * input.block_size,
                             end = std::min(start + input.block_size, input.size);

                    uint32_t accum = 0;
                    for (uint32_t i = start; i != end; ++i)
                        accum += (uint32_t) input.in[i];

                    input.scratch[index] = accum;
                },

                input, blocks
            );

            jit_scan_u32(input.scratch, blocks, input.scratch);
        }

        jit_submit_cpu(
            stream,
            [](size_t index, void *input_) {
                Input input = *(Input *) input_;
                uint32_t start = index * input.block_size,
                         end = std::min(start + input.block_size, input.size);

                uint32_t accum = 0;
                if (input.scratch)
                    accum = input.scratch[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t value = (uint32_t) input.in[i];
                    if (value)
                        input.out[accum] = i;
                    accum += value;
                }

                if (end == input.size)
                    *input.count_out = accum;
            },

            input, blocks
        );

        if (blocks > 1)
            jit_free(input.scratch);

        jit_sync_stream();
        return count_out;
    }
}

static void cuda_transpose(Stream *stream, const uint32_t *in, uint32_t *out,
                           uint32_t rows, uint32_t cols) {
    const Device &device = state.devices[stream->device];

    uint16_t blocks_x = (cols + 15u) / 16u,
             blocks_y = (rows + 15u) / 16u;

    scoped_set_context guard(stream->context);
    jit_log(Debug,
            "jit_transpose(" ENOKI_PTR " -> " ENOKI_PTR
            ", rows=%u, cols=%u, blocks=%ux%u)",
            (uintptr_t) in, (uintptr_t) out, rows, cols, blocks_x, blocks_y);

    void *args[] = { &in, &out, &rows, &cols };
    cuda_check(cuLaunchKernel(
        jit_cuda_transpose[device.id], blocks_x, blocks_y, 1, 16, 16, 1,
        16 * 17 * sizeof(uint32_t), stream->handle, args, nullptr));
}

static ProfilerRegion profiler_region_mkperm("jit_mkperm");
static ProfilerRegion profiler_region_mkperm_phase_1("jit_mkperm_phase_1");
static ProfilerRegion profiler_region_mkperm_phase_2("jit_mkperm_phase_2");

/// Compute a permutation to reorder an integer array into a sorted configuration
uint32_t jit_mkperm(const uint32_t *ptr, uint32_t size, uint32_t bucket_count,
                    uint32_t *perm, uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jit_fail("jit_mkperm(): bucket_count cannot be zero!");

    ProfilerPhase profiler(profiler_region_mkperm);
    Stream *stream = active_stream;

    if (stream->cuda) {
        scoped_set_context guard(stream->context);
        const Device &device = state.devices[stream->device];

        // Don't use more than 1 block/SM due to shared memory requirement
        const uint32_t warp_size = 32;
        uint32_t block_count, thread_count;
        device.get_launch_config(&block_count, &thread_count, size, 1024, 1);

        // Always launch full warps (the kernel impl. assumes that this is the case)
        uint32_t warp_count = (thread_count + warp_size - 1) / warp_size;
        thread_count = warp_count * warp_size;

        uint32_t bucket_size_1   = bucket_count * sizeof(uint32_t),
                 bucket_size_all = bucket_size_1 * block_count;

        /* If there is a sufficient amount of shared memory, atomically accumulate into a
           shared memory buffer. Otherwise, use global memory, which is much slower. */
        uint32_t shared_size = 0;
        const char *variant = nullptr;
        CUfunction phase_1 = nullptr, phase_4 = nullptr;
        bool initialize_buckets = false;

        if (bucket_size_1 * warp_count <= device.shared_memory_bytes) {
            /* "Tiny" variant, which uses shared memory atomics to produce a stable
               permutation. Handles up to 512 buckets with 64KiB of shared memory. */

            phase_1 = jit_cuda_mkperm_phase_1_tiny[device.id];
            phase_4 = jit_cuda_mkperm_phase_4_tiny[device.id];
            shared_size = bucket_size_1 * warp_count;
            bucket_size_all *= warp_count;
            variant = "tiny";
        } else if (bucket_size_1 <= device.shared_memory_bytes) {
            /* "Small" variant, which uses shared memory atomics and handles up to
               16K buckets with 64KiB of shared memory. The permutation can be
               somewhat unstable due to scheduling variations when performing atomic
               operations (although some effort is made to keep it stable within
               each group of 32 elements by performing an intra-warp reduction.) */

            phase_1 = jit_cuda_mkperm_phase_1_small[device.id];
            phase_4 = jit_cuda_mkperm_phase_4_small[device.id];
            shared_size = bucket_size_1;
            variant = "small";
        } else {
            /* "Large" variant, which uses global memory atomics and handles
               arbitrarily many elements (though this is somewhat slower than the
               previous two shared memory variants). The permutation can be somewhat
               unstable due to scheduling variations when performing atomic
               operations (although some effort is made to keep it stable within
               each group of 32 elements by performing an intra-warp reduction.)
               Buckets must be zero-initialized explicitly. */

            phase_1 = jit_cuda_mkperm_phase_1_large[device.id];
            phase_4 = jit_cuda_mkperm_phase_4_large[device.id];
            variant = "large";
            initialize_buckets = true;
        }

        bool needs_transpose = bucket_size_1 != bucket_size_all;
        uint32_t *buckets_1, *buckets_2, *counter = nullptr;
        buckets_1 = buckets_2 =
            (uint32_t *) jit_malloc(AllocType::Device, bucket_size_all);

        // Scratch space for matrix transpose operation
        if (needs_transpose)
            buckets_2 = (uint32_t *) jit_malloc(AllocType::Device, bucket_size_all);

        if (offsets) {
            counter = (uint32_t *) jit_malloc(AllocType::Device, sizeof(uint32_t)),
            cuda_check(cuMemsetD8Async((CUdeviceptr) counter, 0, sizeof(uint32_t),
                                       stream->handle));
        }

        if (initialize_buckets)
            cuda_check(cuMemsetD8Async((CUdeviceptr) buckets_1, 0,
                                       bucket_size_all, stream->handle));

        /* Determine the amount of work to be done per block, and ensure that it is
           divisible by the warp size (the kernel implementation assumes this.) */
        uint32_t size_per_block = (size + block_count - 1) / block_count;
        size_per_block = (size_per_block + warp_size - 1) / warp_size * warp_size;

        jit_log(Debug,
                "jit_mkperm(" ENOKI_PTR
                ", size=%u, bucket_count=%u, block_count=%u, thread_count=%u, "
                "size_per_block=%u, variant=%s, shared_size=%u)",
                (uintptr_t) ptr, size, bucket_count, block_count, thread_count,
                size_per_block, variant, shared_size);

        // Phase 1: Count the number of occurrences per block
        void *args_1[] = { &ptr, &buckets_1, &size, &size_per_block,
                           &bucket_count };

        cuda_check(cuLaunchKernel(phase_1, block_count, 1, 1, thread_count, 1, 1,
                                  shared_size, stream->handle, args_1, nullptr));

        // Phase 2: exclusive prefix sum over transposed buckets
        if (needs_transpose)
            cuda_transpose(stream, buckets_1, buckets_2,
                           bucket_size_all / bucket_size_1, bucket_count);

        jit_scan_u32(buckets_2, bucket_size_all / sizeof(uint32_t), buckets_2);

        if (needs_transpose)
            cuda_transpose(stream, buckets_2, buckets_1, bucket_count,
                           bucket_size_all / bucket_size_1);

        // Phase 3: collect non-empty buckets (optional)
        if (likely(offsets)) {
            uint32_t block_count_3, thread_count_3;
            device.get_launch_config(&block_count_3, &thread_count_3,
                                     bucket_count * block_count);

            // Round up to a multiple of the thread count
            uint32_t bucket_count_rounded =
                (bucket_count + thread_count_3 - 1) / thread_count_3 * thread_count_3;

            void *args_3[] = { &buckets_1, &bucket_count, &bucket_count_rounded,
                               &size,      &counter,      &offsets };

            cuda_check(cuLaunchKernel(jit_cuda_mkperm_phase_3[device.id],
                                      block_count_3, 1, 1, thread_count_3, 1, 1,
                                      sizeof(uint32_t) * thread_count_3,
                                      stream->handle, args_3, nullptr));

            cuda_check(cuMemcpyAsync((CUdeviceptr) (offsets + 4 * size_t(bucket_count)),
                                     (CUdeviceptr) counter, sizeof(uint32_t),
                                     stream->handle));

            cuda_check(cuEventRecord(stream->event, stream->handle));
        }

        // Phase 4: write out permutation based on bucket counts
        void *args_4[] = { &ptr, &buckets_1, &perm, &size, &size_per_block,
                           &bucket_count };
        cuda_check(cuLaunchKernel(phase_4, block_count, 1, 1, thread_count, 1, 1,
                                  shared_size, stream->handle, args_4, nullptr));

        if (likely(offsets)) {
            unlock_guard guard(state.mutex);
            cuda_check(cuEventSynchronize(stream->event));
        }

        jit_free(buckets_1);
        if (needs_transpose)
            jit_free(buckets_2);
        jit_free(counter);

        return offsets ? offsets[4 * bucket_count] : 0u;
    } else { // if (!stream->cuda)
        uint32_t blocks = 1, block_size = size;

        if (stream->parallel_dispatch) {
            // Try to spread out uniformly over cores
            blocks = pool_size() * 4;
            block_size = (size + blocks - 1) / blocks;

            // But don't make the blocks too small
            block_size = std::max((uint32_t) ENOKI_POOL_BLOCK_SIZE, block_size);

            // Finally re-adjust block size
            blocks = (size + block_size - 1) / block_size;
        }

        jit_log(Debug,
                "jit_mkperm(" ENOKI_PTR
                ", size=%u, bucket_count=%u, block_size=%u, blocks=%u)",
                (uintptr_t) ptr, size, bucket_count, block_size, blocks);

        uint32_t **buckets =
            (uint32_t **) jit_malloc(AllocType::Host, sizeof(uint32_t *) * blocks);

        for (uint32_t i = 0; i < blocks; ++i)
            buckets[i] = (uint32_t *) jit_malloc(
                AllocType::Host,
                sizeof(uint32_t) * (size_t) bucket_count);

        struct Input {
            uint32_t size;
            uint32_t blocks;
            uint32_t block_size;
            uint32_t bucket_count;
            const uint32_t *ptr;
            uint32_t **buckets;
            uint32_t *perm;
            uint32_t *offsets;
            uint32_t *unique_count;
        };

        uint32_t unique_count = 0;
        Input input{ size, blocks,  block_size, bucket_count,
                     ptr,  buckets, perm, offsets, &unique_count };

        // Phase 1
        jit_submit_cpu(
            stream,
            [](size_t index, void *input_) {
                ProfilerPhase profiler(profiler_region_mkperm_phase_1);
                Input input = *(Input *) input_;

                uint32_t start = index * input.block_size,
                         end = std::min(start + input.block_size, input.size);

                uint32_t *buckets_local = input.buckets[index];
                memset(buckets_local, 0,
                       sizeof(uint32_t) * (size_t) input.bucket_count);

                for (uint32_t i = start; i != end; ++i)
                    buckets_local[input.ptr[i]]++;
            },

            input, blocks
        );

        // Local accumulation step
        jit_submit_cpu(
            stream,
            [](size_t, void *input_) {
                Input input = *(Input *) input_;

                uint32_t sum = 0, unique_count = 0;
                for (uint32_t i = 0; i < input.bucket_count; ++i) {
                    uint32_t sum_local = 0;
                    for (uint32_t j = 0; j < input.blocks; ++j) {
                        uint32_t value = input.buckets[j][i];
                        input.buckets[j][i] = sum + sum_local;
                        sum_local += value;
                    }
                    if (sum_local > 0) {
                        if (input.offsets) {
                            input.offsets[unique_count*4] = i;
                            input.offsets[unique_count*4 + 1] = sum;
                            input.offsets[unique_count*4 + 2] = sum_local;
                            input.offsets[unique_count*4 + 3] = 0;
                        }
                        unique_count++;
                        sum += sum_local;
                    }
                }

                *input.unique_count = unique_count;
            },
            input, 1
        );

        Task *local_task = stream->task;

        // Phase 2
        jit_submit_cpu(
            stream,
            [](size_t index, void *input_) {
                ProfilerPhase profiler(profiler_region_mkperm_phase_2);
                Input input = *(Input *) input_;

                uint32_t start = index * input.block_size,
                         end = std::min(start + input.block_size, input.size);

                uint32_t *buckets_local = input.buckets[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t index = buckets_local[input.ptr[i]]++;
                    input.perm[index] = i;
                }
            },

            input, blocks, false
        );

        // Free memory (happens asynchronously after the above stmt.)
        for (uint32_t i = 0; i < blocks; ++i)
            jit_free(buckets[i]);
        jit_free(buckets);

        if (stream->parallel_dispatch)
            pool_task_wait_and_release(local_task);

        return unique_count;
    }
}

// Compute a permutation to reorder an array of registered pointers
VCallBucket *jit_vcall(const char *domain, uint32_t index,
                       uint32_t *bucket_count_out) {
    auto it = state.extra.find(index);
    if (it != state.extra.end()) {
        auto &v = it.value();
        if (v.vcall_bucket_count) {
            *bucket_count_out = v.vcall_bucket_count;
            return v.vcall_buckets;
        }
    }

    uint32_t bucket_count = jit_registry_get_max(domain) + 1;
    if (unlikely(bucket_count == 1))
        jit_raise("jit_vcall(): no instances registered for domain \"%s\"\n!", domain);


    jit_var_eval(index);
    Variable *v = jit_var(index);
    const void *ptr = v->data;
    uint32_t size = v->size;

    jit_log(Debug, "jit_vcall(%u, domain=\"%s\")", index, domain);

    size_t perm_size    = (size_t) size * (size_t) sizeof(uint32_t),
           offsets_size = (size_t(bucket_count) * 4 + 1) * sizeof(uint32_t);

    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_vcall(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");

    bool cuda = stream->cuda;
    uint32_t *offsets = (uint32_t *) jit_malloc(
        cuda ? AllocType::HostPinned : AllocType::Host, offsets_size);
    uint32_t *perm = (uint32_t *) jit_malloc(
        cuda ? AllocType::Device : AllocType::HostAsync, perm_size);

    // Compute permutation
    uint32_t unique_count = jit_mkperm((const uint32_t *) ptr, size,
                                       bucket_count, perm, offsets),
             unique_count_out = unique_count;

    // Register permutation variable with JIT backend and transfer ownership
    uint32_t perm_var = jit_var_map_mem(VarType::UInt32, cuda, perm, size, 1);

    Variable v2;
    v2.type = (uint32_t) VarType::UInt32;
    v2.dep[0] = perm_var;
    v2.retain_data = true;
    v2.tsize = 1;
    v2.cuda = cuda;
    v2.unaligned = 1;

    uint32_t *offsets_out = offsets;

    for (uint32_t i = 0; i < unique_count; ++i) {
        uint32_t bucket_id     = offsets[i * 4 + 0],
                 bucket_offset = offsets[i * 4 + 1],
                 bucket_size   = offsets[i * 4 + 2];

        /// Crete variable for permutation subrange
        v2.data = perm + bucket_offset;
        v2.size = bucket_size;

        uint32_t index;
        Variable *vo;
        std::tie(index, vo) = jit_var_new(v2);

        jit_var_inc_ref_int(perm_var);
        jit_var_inc_ref_ext(index, vo);

        void *ptr = jit_registry_get_ptr(domain, bucket_id);
        memcpy(offsets_out, &ptr, sizeof(void *));
        memcpy(offsets_out + 2, &index, sizeof(uint32_t));
        offsets_out += 4;

        jit_trace("jit_vcall(): registered variable %u: bucket %u (" ENOKI_PTR
                  ") of size %u.", index, bucket_id,
                  (uintptr_t) ptr, bucket_size);
    }

    jit_var_dec_ref_ext(perm_var);

    *bucket_count_out = unique_count_out;

    v = jit_var(index);
    v->has_extra = true;
    Extra &extra = state.extra[index];
    extra.vcall_bucket_count = unique_count_out;
    extra.vcall_buckets = (VCallBucket *) offsets;
    return extra.vcall_buckets;
}

using BlockOp = void (*) (const void *ptr, void *out, uint32_t start, uint32_t end, uint32_t block_size);

template <typename Value> static BlockOp jit_block_copy_create() {
    return [](const void *in_, void *out_, uint32_t start, uint32_t end, uint32_t block_size) {
        const Value *in = (const Value *) in_ + start;
        Value *out = (Value *) out_ + start * block_size;
        for (uint32_t i = start; i != end; ++i) {
            Value value = *in++;
            for (uint32_t j = 0; j != block_size; ++j)
                *out++ = value;
        }
    };
}

template <typename Value> static BlockOp jit_block_sum_create() {
    return [](const void *in_, void *out_, uint32_t start, uint32_t end, uint32_t block_size) {
        const Value *in = (const Value *) in_ + start * block_size;
        Value *out = (Value *) out_ + start;
        for (uint32_t i = start; i != end; ++i) {
            Value sum = 0;
            for (uint32_t j = 0; j != block_size; ++j)
                sum += *in++;
            *out++ = sum;
        }
    };
}

static BlockOp jit_block_copy_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jit_block_copy_create<uint8_t >();
        case VarType::UInt16:  return jit_block_copy_create<uint16_t>();
        case VarType::UInt32:  return jit_block_copy_create<uint32_t>();
        case VarType::UInt64:  return jit_block_copy_create<uint64_t>();
        case VarType::Float32: return jit_block_copy_create<float   >();
        case VarType::Float64: return jit_block_copy_create<double  >();
        default: jit_raise("jit_block_copy_create(): unsupported data type!");
            return nullptr;
    }
}

static BlockOp jit_block_sum_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jit_block_sum_create<uint8_t >();
        case VarType::UInt16:  return jit_block_sum_create<uint16_t>();
        case VarType::UInt32:  return jit_block_sum_create<uint32_t>();
        case VarType::UInt64:  return jit_block_sum_create<uint64_t>();
        case VarType::Float32: return jit_block_sum_create<float   >();
        case VarType::Float64: return jit_block_sum_create<double  >();
        default: jit_raise("jit_block_sum_create(): unsupported data type!");
            return nullptr;
    }
}

static VarType make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt16;
        case VarType::Int64: return VarType::UInt16;
        default: return type;
    }
}

/// Replicate individual input elements to larger blocks
void jit_block_copy(enum VarType type, const void *in, void *out, uint32_t size,
                    uint32_t block_size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_block_copy(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");
    else if (block_size == 0)
        jit_raise("jit_block_copy(): block_size cannot be zero!");

    jit_log(Debug,
            "jit_block_copy(" ENOKI_PTR " -> " ENOKI_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            var_type_name[(int) type], block_size, size);

    if (block_size == 1) {
        uint32_t type_size = var_type_size[(int) type];
        jit_memcpy_async(out, in, size * type_size);
        return;
    }

    type = make_int_type_unsigned(type);

    if (stream->cuda) {
        scoped_set_context guard(stream->context);
        const Device &device = state.devices[stream->device];
        size *= block_size;

        CUfunction func = jit_cuda_block_copy[(int) type][device.id];
        if (!func)
            jit_raise("jit_block_copy(): no existing kernel for type=%s!",
                      var_type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        cuda_check(cuLaunchKernel(func, block_count, 1, 1, thread_count, 1, 1,
                                  0, stream->handle, args, nullptr));
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (stream->parallel_dispatch) {
            work_unit_size = ENOKI_POOL_BLOCK_SIZE;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        struct Input {
            const void *in;
            void *out;
            uint32_t size;
            uint32_t block_size;
            uint32_t work_unit_size;
            BlockOp op;
        };

        Input input{ in, out, size, block_size, work_unit_size,
                     jit_block_copy_create(type) };

        jit_submit_cpu(
            stream,
            [](size_t index, void *input_) {
                Input input = *(Input *) input_;
                uint32_t start = index * input.work_unit_size,
                         end = std::min(start + input.work_unit_size, input.size);

                input.op(input.in, input.out, start, end, input.block_size);
            },

            input, work_units
        );
    }
}

/// Sum over elements within blocks
void jit_block_sum(enum VarType type, const void *in, void *out, uint32_t size,
                   uint32_t block_size) {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_raise("jit_block_sum(): you must invoke jitc_set_device() to "
                  "choose a target device before calling this function.");
    else if (block_size == 0)
        jit_raise("jit_block_sum(): block_size cannot be zero!");

    jit_log(Debug,
            "jit_block_sum(" ENOKI_PTR " -> " ENOKI_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            var_type_name[(int) type], block_size, size);

    if (block_size == 1) {
        uint32_t type_size = var_type_size[(int) type];
        jit_memcpy_async(out, in, size * type_size);
        return;
    }

    type = make_int_type_unsigned(type);

    if (stream->cuda) {
        scoped_set_context guard(stream->context);
        const Device &device = state.devices[stream->device];
        size *= block_size;

        CUfunction func = jit_cuda_block_sum[(int) type][device.id];
        if (!func)
            jit_raise("jit_block_sum(): no existing kernel for type=%s!",
                      var_type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        cuda_check(cuLaunchKernel(func, block_count, 1, 1, thread_count, 1, 1,
                                  0, stream->handle, args, nullptr));
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (stream->parallel_dispatch) {
            work_unit_size = ENOKI_POOL_BLOCK_SIZE;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        struct Input {
            const void *in;
            void *out;
            uint32_t size;
            uint32_t block_size;
            uint32_t work_unit_size;
            BlockOp op;
        };

        Input input{ in, out, size, block_size, work_unit_size,
                     jit_block_sum_create(type) };

        jit_submit_cpu(
            stream,
            [](size_t index, void *input_) {
                Input input = *(Input *) input_;
                uint32_t start = index * input.work_unit_size,
                         end = std::min(start + input.work_unit_size, input.size);

                input.op(input.in, input.out, start, end, input.block_size);
            },

            input, work_units
        );
    }
}

/// Asynchronously update a single element in memory
void jit_poke(void *dst, const void *src, uint32_t size) {
    Stream *stream = active_stream;

    jit_log(Debug, "jit_poke(" ENOKI_PTR ", size=%u)", (uintptr_t) dst, size);

    VarType type;
    switch (size) {
        case 1: type = VarType::UInt8; break;
        case 2: type = VarType::UInt16; break;
        case 4: type = VarType::UInt32; break;
        case 8: type = VarType::UInt64; break;
        default:
            jit_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");
    }

    if (stream->cuda) {
        scoped_set_context guard(stream->context);
        const Device &device = state.devices[stream->device];
        CUfunction func = jit_cuda_poke[(int) type][device.id];
        void *args[] = { &dst, (void *) src };
        cuda_check(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1,
                                  0, stream->handle, args, nullptr));
    } else {
        struct Input {
            void *dst;
            uint8_t src[8];
            uint32_t size;
        };
        Input input;
        input.dst = dst;
        input.size = size;
        memcpy(&input.src, src, size);

        jit_submit_cpu(
            stream,
            [](size_t, void *input_) {
                Input input = *(Input *) input_;
                memcpy(input.dst, &input.src, input.size);
            },
            input);
    }
}
