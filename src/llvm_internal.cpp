#include "llvm_internal.h"
#include "log.h"
#include "var.h"
#include "common.h"
#include "profile.h"

const static char *reduction_name[(int) ReduceOp::Count] = { "none", "sum", "mul",
                                                      "min", "max", "and", "or" };

using Reduction = void (*) (const void *ptr, uint32_t start, uint32_t end, void *out);

template <typename Value>
static Reduction jitc_reduce_create(ReduceOp op) {
    using UInt = uint_with_size_t<Value>;

    switch (op) {
        case ReduceOp::Add:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
                const Value *ptr = (const Value *) ptr_;
                Value result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result += ptr[i];
                *((Value *) out) = result;
            };

        case ReduceOp::Mul:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
                const Value *ptr = (const Value *) ptr_;
                Value result = 1;
                for (uint32_t i = start; i != end; ++i)
                    result *= ptr[i];
                *((Value *) out) = result;
            };

        case ReduceOp::Max:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::min()
                                   : -std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::max(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReduceOp::Min:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::max()
                                   :  std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::min(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReduceOp::Or:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result |= ptr[i];
                *((UInt *) out) = result;
            };

        case ReduceOp::And:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = (UInt) -1;
                for (uint32_t i = start; i != end; ++i)
                    result &= ptr[i];
                *((UInt *) out) = result;
            };

        default: jitc_raise("jit_reduce_create(): unsupported reduction type!");
    }
}

static Reduction jitc_reduce_create(VarType type, ReduceOp op) {
    using half = drjit::half;
    switch (type) {
        case VarType::Int8:    return jitc_reduce_create<int8_t  >(op);
        case VarType::UInt8:   return jitc_reduce_create<uint8_t >(op);
        case VarType::Int16:   return jitc_reduce_create<int16_t >(op);
        case VarType::UInt16:  return jitc_reduce_create<uint16_t>(op);
        case VarType::Int32:   return jitc_reduce_create<int32_t >(op);
        case VarType::UInt32:  return jitc_reduce_create<uint32_t>(op);
        case VarType::Int64:   return jitc_reduce_create<int64_t >(op);
        case VarType::UInt64:  return jitc_reduce_create<uint64_t>(op);
        case VarType::Float16: return jitc_reduce_create<half    >(op);
        case VarType::Float32: return jitc_reduce_create<float   >(op);
        case VarType::Float64: return jitc_reduce_create<double  >(op);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

/// Helper function: enqueue parallel CPU task (synchronous or asynchronous)
template <typename Func>
static void jitc_submit_cpu(KernelType type, Func &&func, uint32_t width,
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

void LLVMThreadState::jitc_reduce(VarType type, ReduceOp op, const void *ptr,
                                  uint32_t size, void *out) {
    
    jitc_log(Debug, "jit_reduce(" DRJIT_PTR ", type=%s, op=%s, size=%u)",
             (uintptr_t) ptr, type_name[(int) type],
             reduction_name[(int) op], size);

    uint32_t tsize = type_size[(int) type];

    // LLVM specific
    uint32_t block_size = size, blocks = 1;
    if (pool_size() > 1) {
        block_size = jitc_llvm_block_size;
        blocks     = (size + block_size - 1) / block_size;
    }

    void *target = out;
    if (blocks > 1)
        target = jitc_malloc(AllocType::HostAsync, blocks * tsize);

    Reduction reduction = jitc_reduce_create(type, op);
    jitc_submit_cpu(
        KernelType::Reduce,
        [block_size, size, tsize, ptr, reduction, target](uint32_t index) {
            reduction(ptr, index * block_size,
                      std::min((index + 1) * block_size, size),
                      (uint8_t *) target + index * tsize);
        },

        size,
        std::max(1u, blocks));

    if (blocks > 1) {
        this->jitc_reduce(type, op, target, blocks, out);
        jitc_free(target);
    }
}

bool LLVMThreadState::jitc_all(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */
    
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_all(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = true;
        this->jitc_memset_async(values + size, trailing, sizeof(bool), &filler);
    }
    
    // LLVM specific
    bool result;
    
    uint8_t out[4];
    this->jitc_reduce(VarType::UInt32, ReduceOp::And, values, reduced_size, out);
    jitc_sync_thread();
    result = (out[0] & out[1] & out[2] & out[3]) != 0;

    return result;
}

bool LLVMThreadState::jitc_any(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */
    
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_any(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = false;
        this->jitc_memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    // LLVM specific
    bool result;
    
    uint8_t out[4];
    this->jitc_reduce(VarType::UInt32, ReduceOp::Or, values,
                reduced_size, out);
    jitc_sync_thread();
    result = (out[0] | out[1] | out[2] | out[3]) != 0;

    return result;
}

template <typename T>
static void sum_reduce_1(uint32_t start, uint32_t end, const void *in_, uint32_t index,
                  void *scratch) {
    const T *in = (const T *) in_;
    T accum = T(0);
    for (uint32_t i = start; i != end; ++i)
        accum += in[i];
    ((T*) scratch)[index] = accum;
}

template <typename T>
static void sum_reduce_2(uint32_t start, uint32_t end, const void *in_, void *out_,
                  uint32_t index, const void *scratch, bool exclusive) {
    const T *in = (const T *) in_;
    T *out = (T *) out_;

    T accum;
    if (scratch)
        accum = ((const T *) scratch)[index];
    else
        accum = T(0);

    if (exclusive) {
        for (uint32_t i = start; i != end; ++i) {
            T value = in[i];
            out[i] = accum;
            accum += value;
        }
    } else {
        for (uint32_t i = start; i != end; ++i) {
            T value = in[i];
            accum += value;
            out[i] = accum;
        }
    }
}

static void sum_reduce_1(VarType vt, uint32_t start, uint32_t end, const void *in, uint32_t index, void *scratch) {
    switch (vt) {
        case VarType::UInt32:  sum_reduce_1<uint32_t>(start, end, in, index, scratch); break;
        case VarType::UInt64:  sum_reduce_1<uint64_t>(start, end, in, index, scratch); break;
        case VarType::Float32: sum_reduce_1<float>   (start, end, in, index, scratch); break;
        case VarType::Float64: sum_reduce_1<double>  (start, end, in, index, scratch); break;
        default:
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);
    }
}

static void sum_reduce_2(VarType vt, uint32_t start, uint32_t end, const void *in, void *out, uint32_t index, const void *scratch, bool exclusive) {
    switch (vt) {
        case VarType::UInt32:  sum_reduce_2<uint32_t>(start, end, in, out, index, scratch, exclusive); break;
        case VarType::UInt64:  sum_reduce_2<uint64_t>(start, end, in, out, index, scratch, exclusive); break;
        case VarType::Float32: sum_reduce_2<float>   (start, end, in, out, index, scratch, exclusive); break;
        case VarType::Float64: sum_reduce_2<double>  (start, end, in, out, index, scratch, exclusive); break;
        default:
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);
    }
}

void LLVMThreadState::jitc_prefix_sum(VarType vt, bool exclusive,
                                      const void *in, uint32_t size,
                                      void *out) {
    if (size == 0)
        return;
    if (vt == VarType::Int32)
        vt = VarType::UInt32;

    const uint32_t isize = type_size[(int) vt];

    // LLVM specific
    uint32_t block_size = size, blocks = 1;
    if (pool_size() > 1) {
        block_size = jitc_llvm_block_size;
        blocks     = (size + block_size - 1) / block_size;
    }

    jitc_log(Debug,
            "jit_prefix_sum(" DRJIT_PTR " -> " DRJIT_PTR
            ", size=%u, block_size=%u, blocks=%u)",
            (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

    void *scratch = nullptr;

    if (blocks > 1) {
        scratch = (void *) jitc_malloc(AllocType::HostAsync, blocks * isize);

        jitc_submit_cpu(
            KernelType::Other,
            [block_size, size, in, vt, scratch](uint32_t index) {
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                sum_reduce_1(vt, start, end, in, index, scratch);
            },
            size, blocks);

        this->jitc_prefix_sum(vt, true, scratch, blocks, scratch);
    }

    jitc_submit_cpu(
        KernelType::Other,
        [block_size, size, in, out, vt, scratch, exclusive](uint32_t index) {
            uint32_t start = index * block_size,
                     end = std::min(start + block_size, size);

            sum_reduce_2(vt, start, end, in, out, index, scratch, exclusive);
        },
        size, blocks
    );

    jitc_free(scratch);
}

uint32_t LLVMThreadState::jitc_compress(const uint8_t *in, uint32_t size,
                                        uint32_t *out) {
    if (size == 0)
        return 0;

    // LLVM specific
    uint32_t block_size = size, blocks = 1;
    if (pool_size() > 1) {
        block_size = jitc_llvm_block_size;
        blocks     = (size + block_size - 1) / block_size;
    }

    uint32_t count_out = 0;

    jitc_log(Debug,
            "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
            ", size=%u, block_size=%u, blocks=%u)",
            (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

    uint32_t *scratch = nullptr;

    if (blocks > 1) {
        scratch = (uint32_t *) jitc_malloc(AllocType::HostAsync,
                                           blocks * sizeof(uint32_t));

        jitc_submit_cpu(
            KernelType::Other,
            [block_size, size, in, scratch](uint32_t index) {
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                uint32_t accum = 0;
                for (uint32_t i = start; i != end; ++i)
                    accum += (uint32_t) in[i];

                scratch[index] = accum;
            },

            size, blocks
        );

        this->jitc_prefix_sum(VarType::UInt32, true, scratch, blocks, scratch);
    }

    jitc_submit_cpu(
        KernelType::Other,
        [block_size, size, scratch, in, out, &count_out](uint32_t index) {
            uint32_t start = index * block_size,
                     end = std::min(start + block_size, size);

            uint32_t accum = 0;
            if (scratch)
                accum = scratch[index];

            for (uint32_t i = start; i != end; ++i) {
                uint32_t value = (uint32_t) in[i];
                if (value)
                    out[accum] = i;
                accum += value;
            }

            if (end == size)
                count_out = accum;
        },

        size, blocks
    );

    jitc_free(scratch);
    jitc_sync_thread();

    return count_out;
}

static ProfilerRegion profiler_region_mkperm_phase_1("jit_mkperm_phase_1");
static ProfilerRegion profiler_region_mkperm_phase_2("jit_mkperm_phase_2");

uint32_t LLVMThreadState::jitc_mkperm(const uint32_t *ptr, uint32_t size,
                                      uint32_t bucket_count, uint32_t *perm,
                                      uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jitc_fail("jit_mkperm(): bucket_count cannot be zero!");

    // LLVM specific
    uint32_t blocks = 1, block_size = size, pool_size = ::pool_size();

    if (pool_size > 1) {
        // Try to spread out uniformly over cores
        blocks = pool_size * 4;
        block_size = (size + blocks - 1) / blocks;

        // But don't make the blocks too small
        block_size = std::max(jitc_llvm_block_size, block_size);

        // Finally re-adjust block count given the selected block size
        blocks = (size + block_size - 1) / block_size;
    }

    jitc_log(Debug,
            "jit_mkperm(" DRJIT_PTR
            ", size=%u, bucket_count=%u, block_size=%u, blocks=%u)",
            (uintptr_t) ptr, size, bucket_count, block_size, blocks);

    uint32_t **buckets =
        (uint32_t **) jitc_malloc(AllocType::HostAsync, sizeof(uint32_t *) * blocks);

    uint32_t unique_count = 0;

    // Phase 1
    jitc_submit_cpu(
        KernelType::CallReduce,
        [block_size, size, buckets, bucket_count, ptr](uint32_t index) {
            ProfilerPhase profiler(profiler_region_mkperm_phase_1);
            uint32_t start = index * block_size,
                     end = std::min(start + block_size, size);

            size_t bsize = sizeof(uint32_t) * (size_t) bucket_count;
            uint32_t *buckets_local = (uint32_t *) malloc_check(bsize);
            memset(buckets_local, 0, bsize);

             for (uint32_t i = start; i != end; ++i)
                 buckets_local[ptr[i]]++;

             buckets[index] = buckets_local;
        },

        size, blocks
    );

    // Local accumulation step
    jitc_submit_cpu(
        KernelType::CallReduce,
        [bucket_count, blocks, buckets, offsets, &unique_count](uint32_t) {
            uint32_t sum = 0, unique_count_local = 0;
            for (uint32_t i = 0; i < bucket_count; ++i) {
                uint32_t sum_local = 0;
                for (uint32_t j = 0; j < blocks; ++j) {
                    uint32_t value = buckets[j][i];
                    buckets[j][i] = sum + sum_local;
                    sum_local += value;
                }
                if (sum_local > 0) {
                    if (offsets) {
                        offsets[unique_count_local*4] = i;
                        offsets[unique_count_local*4 + 1] = sum;
                        offsets[unique_count_local*4 + 2] = sum_local;
                        offsets[unique_count_local*4 + 3] = 0;
                    }
                    unique_count_local++;
                    sum += sum_local;
                }
            }

            unique_count = unique_count_local;
        },

        size
    );

    Task *local_task = jitc_task;
    task_retain(local_task);

    // Phase 2
    jitc_submit_cpu(
        KernelType::CallReduce,
        [block_size, size, buckets, perm, ptr](uint32_t index) {
            ProfilerPhase profiler(profiler_region_mkperm_phase_2);

            uint32_t start = index * block_size,
                     end = std::min(start + block_size, size);

            uint32_t *buckets_local = buckets[index];

            for (uint32_t i = start; i != end; ++i) {
                uint32_t idx = buckets_local[ptr[i]]++;
                perm[idx] = i;
            }

            free(buckets_local);
        },

        size, blocks
    );

    // Free memory (happens asynchronously after the above stmt.)
    jitc_free(buckets);

    unlock_guard guard(state.lock);
    task_wait_and_release(local_task);
    return unique_count;
}

void LLVMThreadState::jitc_memcpy(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
}
