/*
    src/util.cpp -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <condition_variable>
#include "internal.h"
#include "util.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "call.h"
#include "profile.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

const char *reduction_name[(int) ReduceOp::Count] = { "none", "sum", "mul",
                                                      "min", "max", "and", "or" };

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

void jitc_submit_gpu(KernelType type, CUfunction kernel, uint32_t block_count,
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
void jitc_memset_async(JitBackend backend, void *ptr, uint32_t size_,
                       uint32_t isize, const void *src) {
    ThreadState *ts = thread_state(backend);
    ts->jitc_memset_async(ptr, size_, isize, src);
}

/// Perform a synchronous copy operation
void jitc_memcpy(JitBackend backend, void *dst, const void *src, size_t size) {
    ThreadState *ts = thread_state(backend);

    // Temporarily release the lock while copying
    jitc_sync_thread(ts);
    ts->jitc_memcpy(dst, src, size);
}

/// Perform an asynchronous copy operation
void jitc_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, size,
                                 ts->stream));
    } else {
        jitc_submit_cpu(
            KernelType::Other,
            [dst, src, size](uint32_t) {
                memcpy(dst, src, size);
            },

            (uint32_t) size
        );
    }
}

void jitc_reduce(JitBackend backend, VarType type, ReduceOp op, const void *ptr,
                uint32_t size, void *out) {
    ThreadState *ts = thread_state(backend);

    ts->jitc_reduce(type, op, ptr, size, out);
}

/// 'All' reduction for boolean arrays
bool jitc_all(JitBackend backend, uint8_t *values, uint32_t size) {
    ThreadState *ts = thread_state(backend);

    return ts->jitc_all(values, size);
}

/// 'Any' reduction for boolean arrays
bool jitc_any(JitBackend backend, uint8_t *values, uint32_t size) {
    ThreadState *ts = thread_state(backend);

    return ts->jitc_any(values, size);
}

/// Exclusive prefix sum
void jitc_prefix_sum(JitBackend backend, VarType vt, bool exclusive,
               const void *in, uint32_t size, void *out) {
    ThreadState *ts = thread_state(backend);

    ts->jitc_prefix_sum(vt, exclusive, in, size, out);
}

/// Mask compression
uint32_t jitc_compress(JitBackend backend, const uint8_t *in, uint32_t size, uint32_t *out) {
    ThreadState *ts = thread_state(backend);

    return ts->jitc_compress(in, size, out);
}

static ProfilerRegion profiler_region_mkperm("jit_mkperm");

/// Compute a permutation to reorder an integer array into a sorted configuration
uint32_t jitc_mkperm(JitBackend backend, const uint32_t *ptr, uint32_t size,
                     uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {
    
    ProfilerPhase profiler(profiler_region_mkperm);
    ThreadState *ts = thread_state(backend);

    return ts->jitc_mkperm(ptr, size, bucket_count, perm, offsets);
}

using BlockOp = void (*) (const void *ptr, void *out, uint32_t start, uint32_t end, uint32_t block_size);

template <typename Value> static BlockOp jitc_block_copy_create() {
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

template <typename Value> static BlockOp jitc_block_sum_create() {
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

static BlockOp jitc_block_copy_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jitc_block_copy_create<uint8_t >();
        case VarType::UInt16:  return jitc_block_copy_create<uint16_t>();
        case VarType::UInt32:  return jitc_block_copy_create<uint32_t>();
        case VarType::UInt64:  return jitc_block_copy_create<uint64_t>();
        case VarType::Float32: return jitc_block_copy_create<float   >();
        case VarType::Float64: return jitc_block_copy_create<double  >();
        default: jitc_raise("jit_block_copy_create(): unsupported data type!");
    }
}

static BlockOp jitc_block_sum_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jitc_block_sum_create<uint8_t >();
        case VarType::UInt16:  return jitc_block_sum_create<uint16_t>();
        case VarType::UInt32:  return jitc_block_sum_create<uint32_t>();
        case VarType::UInt64:  return jitc_block_sum_create<uint64_t>();
        case VarType::Float32: return jitc_block_sum_create<float   >();
        case VarType::Float64: return jitc_block_sum_create<double  >();
        default: jitc_raise("jit_block_sum_create(): unsupported data type!");
    }
}

static VarType make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

/// Replicate individual input elements to larger blocks
void jitc_block_copy(JitBackend backend, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    if (block_size == 0)
        jitc_raise("jit_block_copy(): block_size cannot be zero!");

    jitc_log(Debug,
            "jit_block_copy(" DRJIT_PTR " -> " DRJIT_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            type_name[(int) type], block_size, size);

    if (block_size == 1) {
        uint32_t tsize = type_size[(int) type];
        jitc_memcpy_async(backend, out, in, size * tsize);
        return;
    }

    type = make_int_type_unsigned(type);

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        size *= block_size;

        CUfunction func = jitc_cuda_block_copy[(int) type][device.id];
        if (!func)
            jitc_raise("jit_block_copy(): no existing kernel for type=%s!",
                      type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, size);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = jitc_llvm_block_size;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        BlockOp op = jitc_block_copy_create(type);

        jitc_submit_cpu(
            KernelType::Other,
            [in, out, op, work_unit_size, size, block_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                op(in, out, start, end, block_size);
            },

            size, work_units
        );
    }
}

/// Sum over elements within blocks
void jitc_block_sum(JitBackend backend, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    if (block_size == 0)
        jitc_raise("jit_block_sum(): block_size cannot be zero!");

    jitc_log(Debug,
            "jit_block_sum(" DRJIT_PTR " -> " DRJIT_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            type_name[(int) type], block_size, size);

    uint32_t tsize = type_size[(int) type];
    size_t out_size = size * tsize;

    if (block_size == 1) {
        jitc_memcpy_async(backend, out, in, out_size);
        return;
    }

    type = make_int_type_unsigned(type);

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];

        size *= block_size;

        CUfunction func = jitc_cuda_block_sum[(int) type][device.id];
        if (!func)
            jitc_raise("jit_block_sum(): no existing kernel for type=%s!",
                      type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, out_size, ts->stream));
        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, size);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = jitc_llvm_block_size;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        BlockOp op = jitc_block_sum_create(type);

        jitc_submit_cpu(
            KernelType::Other,
            [in, out, op, work_unit_size, size, block_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                op(in, out, start, end, block_size);
            },

            size, work_units
        );
    }
}

/// Asynchronously update a single element in memory
void jitc_poke(JitBackend backend, void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    VarType type;
    switch (size) {
        case 1: type = VarType::UInt8; break;
        case 2: type = VarType::UInt16; break;
        case 4: type = VarType::UInt32; break;
        case 8: type = VarType::UInt64; break;
        default:
            jitc_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");
    }

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_poke[(int) type][device.id];
        void *args[] = { &dst, (void *) src };
        jitc_submit_gpu(KernelType::Other, func, 1, 1, 0,
                        ts->stream, args, nullptr, 1);
    } else {
        uint8_t src8[8] { };
        memcpy(&src8, src, size);

        jitc_submit_cpu(
            KernelType::Other,
            [src8, size, dst](uint32_t) {
                memcpy(dst, &src8, size);
            },

            size
        );
    }
}

void jitc_aggregate(JitBackend backend, void *dst_, AggregationEntry *agg,
                    uint32_t size) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_aggregate[device.id];
        void *args[] = { &dst_, &agg, &size };

        uint32_t block_count, thread_count;
        device.get_launch_config(&block_count, &thread_count, size);

        jitc_log(InfoSym,
                 "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, blocks=%u, threads=%u)",
                 (uintptr_t) agg, (uintptr_t) dst_, size, block_count,
                 thread_count);

        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, 1);

        jitc_free(agg);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = jitc_llvm_block_size;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        jitc_log(InfoSym,
                 "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, work_units=%u)",
                 (uintptr_t) agg, (uintptr_t) dst_, size, work_units);

        jitc_submit_cpu(
            KernelType::Other,
            [dst_, agg, size, work_unit_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                for (uint32_t i = start; i != end; ++i) {
                    AggregationEntry e = agg[i];

                    const void *src = e.src;
                    void *dst = (uint8_t *) dst_ + e.offset;

                    switch (e.size) {
                        case  1: *(uint8_t *)  dst =  (uint8_t)  (uintptr_t) src; break;
                        case  2: *(uint16_t *) dst =  (uint16_t) (uintptr_t) src; break;
                        case  4: *(uint32_t *) dst =  (uint32_t) (uintptr_t) src; break;
                        case  8: *(uint64_t *) dst =  (uint64_t) (uintptr_t) src; break;
                        case -1: *(uint8_t *)  dst = *(uint8_t *)  src; break;
                        case -2: *(uint16_t *) dst = *(uint16_t *) src; break;
                        case -4: *(uint32_t *) dst = *(uint32_t *) src; break;
                        case -8: *(uint64_t *) dst = *(uint64_t *) src; break;
                    }
                }
            },
            size, work_units);

        jitc_submit_cpu(
            KernelType::Other, [agg](uint32_t) { free(agg); }, 1, 1);
    }
}

void jitc_enqueue_host_func(JitBackend backend, void (*callback)(void *),
                            void *payload) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        cuda_check(cuLaunchHostFunc(ts->stream, callback, payload));
    } else {
        if (!jitc_task) {
            unlock_guard guard(state.lock);
            callback(payload);
        } else {
            jitc_submit_cpu(
                KernelType::Other, [payload, callback](uint32_t) { callback(payload); }, 1, 1);
        }
    }
}

using ReduceExpanded = void (*) (void *ptr, uint32_t start, uint32_t end, uint32_t exp, uint32_t size);

template <typename Value, typename Op>
static void jitc_reduce_expanded_impl(void *ptr_, uint32_t start, uint32_t end,
                                 uint32_t exp, uint32_t size) {
    Value *ptr = (Value *) ptr_;
    Op op;

    const uint32_t block = 128;

    uint32_t i = start;
    for (; i + block <= end; i += block)
        for (uint32_t j = 1; j < exp; ++j)
            for (uint32_t k = 0; k < block; ++k)
                ptr[i + k] = op(ptr[i + k], ptr[i + k + j * size]);

    for (; i < end; i += 1)
        for (uint32_t j = 1; j < exp; ++j)
            ptr[i] = op(ptr[i], ptr[i + j * size]);
}

template <typename Value>
static ReduceExpanded jitc_reduce_expanded_create(ReduceOp op) {
    using UInt = uint_with_size_t<Value>;

    struct Add { Value operator()(Value a, Value b) JIT_NO_UBSAN { return a + b; }};
    struct Mul { Value operator()(Value a, Value b) JIT_NO_UBSAN { return a * b; }};
    struct Min { Value operator()(Value a, Value b) { return std::min(a, b); }};
    struct Max { Value operator()(Value a, Value b) { return std::max(a, b); }};
    struct And {
        Value operator()(Value a, Value b) {
            if constexpr (std::is_integral_v<Value>)
                return a & b;
            else
                return 0;
        }
    };
    struct Or {
        Value operator()(Value a, Value b) {
            if constexpr (std::is_integral_v<Value>)
                return a | b;
            else
                return 0;
        }
    };

    switch (op) {
        case ReduceOp::Add: return jitc_reduce_expanded_impl<Value, Add>;
        case ReduceOp::Mul: return jitc_reduce_expanded_impl<Value, Mul>;
        case ReduceOp::Max: return jitc_reduce_expanded_impl<Value, Max>;
        case ReduceOp::Min: return jitc_reduce_expanded_impl<Value, Min>;
        case ReduceOp::And: return jitc_reduce_expanded_impl<Value, And>;
        case ReduceOp::Or: return jitc_reduce_expanded_impl<Value, Or>;

        default: jitc_raise("jit_reduce_expanded_create(): unsupported reduction type!");
    }
}


static ReduceExpanded jitc_reduce_expanded_create(VarType type, ReduceOp op) {
    using half = drjit::half;
    switch (type) {
        case VarType::Int32:   return jitc_reduce_expanded_create<int32_t >(op);
        case VarType::UInt32:  return jitc_reduce_expanded_create<uint32_t>(op);
        case VarType::Int64:   return jitc_reduce_expanded_create<int64_t >(op);
        case VarType::UInt64:  return jitc_reduce_expanded_create<uint64_t>(op);
        case VarType::Float16: return jitc_reduce_expanded_create<half    >(op);
        case VarType::Float32: return jitc_reduce_expanded_create<float   >(op);
        case VarType::Float64: return jitc_reduce_expanded_create<double  >(op);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

void jitc_reduce_expanded(VarType vt, ReduceOp op, void *ptr, uint32_t exp, uint32_t size) {
    jitc_log(Debug, "jit_reduce_expanded(" DRJIT_PTR ", type=%s, op=%s, expfactor=%u, size=%u)",
            (uintptr_t) ptr, type_name[(int) vt],
            reduction_name[(int) op], exp, size);

    ReduceExpanded kernel = jitc_reduce_expanded_create(vt, op);

    uint32_t block_size = size, blocks = 1;
    if (pool_size() > 1) {
        block_size = jitc_llvm_block_size;
        blocks     = (size + block_size - 1) / block_size;
    }

    jitc_submit_cpu(
        KernelType::Reduce,
        [ptr, block_size, exp, size, kernel](uint32_t index) {
            kernel(ptr, index * block_size,
                   std::min((index + 1) * block_size, size), exp, size);
        },

        size, std::max(1u, blocks));
}
