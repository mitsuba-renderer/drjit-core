#include "llvm_ts.h"
#include "log.h"
#include "var.h"
#include "common.h"
#include "profile.h"
#include "util.h"

using Reduction = void (*) (const void *ptr, uint32_t start, uint32_t end, void *out);
using BlockReduction = void (*) (const void *ptr, uint32_t start, uint32_t end, uint32_t block_size, void *out);

using Reduction2 = void (*) (const void *ptr_1, const void *ptr_2, uint32_t start, uint32_t end, void *out);

template <typename Ts>
static Reduction reduce_create(ReduceOp op) {
    using UInt = uint_with_size_t<Ts>;
    using Tv = std::conditional_t<
        std::is_same_v<Ts, drjit::half>,
        float,
        Ts
    >;

    switch (op) {
        case ReduceOp::Add:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
                const Ts *ptr = (const Ts *) ptr_;
                Tv result = Tv(0);
                for (uint32_t i = start; i != end; ++i)
                    result += (Tv) ptr[i];
                *((Ts *) out) = Ts(result);
            };

        case ReduceOp::Mul:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
                const Ts *ptr = (const Ts *) ptr_;
                Tv result = Tv(1);
                for (uint32_t i = start; i != end; ++i)
                    result *= Tv(ptr[i]);
                *((Ts *) out) = Ts(result);
            };

        case ReduceOp::Max:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Ts *ptr = (const Ts *) ptr_;
                Ts result;

                if constexpr (std::is_integral<Ts>::value)
                    result = std::numeric_limits<Ts>::min();
                else // the next line generates a warning if not performed in an 'if constexpr' block
                    result = -std::numeric_limits<Ts>::infinity();

                for (uint32_t i = start; i != end; ++i)
                    result = std::max(result, ptr[i]);

                *((Ts *) out) = result;
            };

        case ReduceOp::Min:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Ts *ptr = (const Ts *) ptr_;
                Ts result = std::is_integral<Ts>::value
                                   ?  std::numeric_limits<Ts>::max()
                                   :  std::numeric_limits<Ts>::infinity();

                for (uint32_t i = start; i != end; ++i)
                    result = std::min(result, ptr[i]);

                *((Ts *) out) = result;
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

template <typename Ts> static BlockReduction block_reduce_create(ReduceOp op) {
    using UInt = uint_with_size_t<Ts>;
    using Tv = std::conditional_t<
        std::is_same_v<Ts, drjit::half>,
        float,
        Ts
    >;

    switch (op) {
        case ReduceOp::Add:
            return [](const void *in_, uint32_t start, uint32_t end, uint32_t block_size, void *out_) {
                const Ts *in = (const Ts *) in_ + start * block_size;
                Ts *out = (Ts *) out_ + start;
                for (uint32_t i = start; i != end; ++i) {
                    Tv result = Tv(0);
                    for (uint32_t j = 0; j != block_size; ++j)
                        result += Tv(*in++);
                    *out++ = Ts(result);
                }
            };

        case ReduceOp::Mul:
            return [](const void *in_, uint32_t start, uint32_t end, uint32_t block_size, void *out_) {
                const Ts *in = (const Ts *) in_ + start * block_size;
                Ts *out = (Ts *) out_ + start;
                for (uint32_t i = start; i != end; ++i) {
                    Tv result = Tv(1);
                    for (uint32_t j = 0; j != block_size; ++j)
                        result *= Tv(*in++);
                    *out++ = Ts(result);
                }
            };

        case ReduceOp::Max:
            return [](const void *in_, uint32_t start, uint32_t end, uint32_t block_size, void *out_) {
                const Ts *in = (const Ts *) in_ + start * block_size;
                Ts *out = (Ts *) out_ + start;
                for (uint32_t i = start; i != end; ++i) {
                    Ts result;

                    if constexpr (std::is_integral<Ts>::value)
                        result = std::numeric_limits<Ts>::min();
                    else // the next line generates a warning if not performed in an 'if constexpr' block
                        result = -std::numeric_limits<Ts>::infinity();

                    for (uint32_t j = 0; j != block_size; ++j)
                        result = std::max(result, *in++);

                    *out++ = result;
                }
            };

        case ReduceOp::Min:
            return [](const void *in_, uint32_t start, uint32_t end, uint32_t block_size, void *out_) {
                const Ts *in = (const Ts *) in_ + start * block_size;
                Ts *out = (Ts *) out_ + start;
                for (uint32_t i = start; i != end; ++i) {
                    Ts result = std::is_integral<Ts>::value
                                       ?  std::numeric_limits<Ts>::max()
                                       :  std::numeric_limits<Ts>::infinity();

                    for (uint32_t j = 0; j != block_size; ++j)
                        result = std::min(result, *in++);

                    *out++ = result;
                }
            };

        case ReduceOp::Or:
            return [](const void *in_, uint32_t start, uint32_t end, uint32_t block_size, void *out_) {
                const UInt *in = (const UInt *) in_ + start * block_size;
                UInt *out = (UInt *) out_ + start;

                for (uint32_t i = start; i != end; ++i) {
                    UInt result(0);

                    for (uint32_t j = 0; j != block_size; ++j)
                        result |= *in++;

                    *out++ = result;
                }
            };

        case ReduceOp::And:
            return [](const void *in_, uint32_t start, uint32_t end, uint32_t block_size, void *out_) {
                const UInt *in = (const UInt *) in_ + start * block_size;
                UInt *out = (UInt *) out_ + start;

                for (uint32_t i = start; i != end; ++i) {
                    UInt result(-1);

                    for (uint32_t j = 0; j != block_size; ++j)
                        result &= *in++;

                    *out++ = result;
                }
            };

        default: jitc_raise("jit_block_reduce_create(): unsupported reduction type!");
    }
}

static Reduction jitc_reduce_create(VarType type, ReduceOp op) {
    using half = drjit::half;
    switch (type) {
        case VarType::Int32:   return reduce_create<int32_t >(op);
        case VarType::UInt32:  return reduce_create<uint32_t>(op);
        case VarType::Int64:   return reduce_create<int64_t >(op);
        case VarType::UInt64:  return reduce_create<uint64_t>(op);
        case VarType::Float16: return reduce_create<half    >(op);
        case VarType::Float32: return reduce_create<float   >(op);
        case VarType::Float64: return reduce_create<double  >(op);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

static BlockReduction jitc_block_reduce_create(VarType type, ReduceOp op) {
    using half = drjit::half;
    switch (type) {
        case VarType::Int32:   return block_reduce_create<int32_t >(op);
        case VarType::UInt32:  return block_reduce_create<uint32_t>(op);
        case VarType::Int64:   return block_reduce_create<int64_t >(op);
        case VarType::UInt64:  return block_reduce_create<uint64_t>(op);
        case VarType::Float16: return block_reduce_create<half    >(op);
        case VarType::Float32: return block_reduce_create<float   >(op);
        case VarType::Float64: return block_reduce_create<double  >(op);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

template <typename Value>
static Reduction2 reduce_dot_create() {
    return [](const void *ptr_1_, const void *ptr_2_,
              uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
        const Value *ptr_1 = (const Value *) ptr_1_;
        const Value *ptr_2 = (const Value *) ptr_2_;
        Value result = 0;
        for (uint32_t i = start; i != end; ++i) {
            result = std::fma(ptr_1[i], ptr_2[i], result);
        }
        *((Value *) out) = result;
    };
}


static Reduction2 jitc_reduce_dot_create(VarType type) {
    using half = drjit::half;
    switch (type) {
        case VarType::Float16: return reduce_dot_create<half  >();
        case VarType::Float32: return reduce_dot_create<float >();
        case VarType::Float64: return reduce_dot_create<double>();
        default: jitc_raise("jit_reduce_dot_create(): unsupported data type!");
    }
}


/// Helper function: enqueue parallel CPU task (synchronous or asynchronous)
template <typename Func>
static void submit_cpu(KernelType type, Func &&func, uint32_t width,
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

/// Temporary scratch space for scheduled tasks
static std::vector<Task *> scheduled_tasks;

void LLVMThreadState::barrier(){
    if (scheduled_tasks.size() == 1) {
        task_release(jitc_task);
        jitc_task = scheduled_tasks[0];
    } else {
        jitc_assert(!scheduled_tasks.empty(),
                    "jit_eval(): no tasks generated!");

        // Insert a barrier task
        Task *new_task = task_submit_dep(nullptr, scheduled_tasks.data(),
                                         (uint32_t)scheduled_tasks.size());
        task_release(jitc_task);
        for (Task *t : scheduled_tasks)
            task_release(t);
        jitc_task = new_task;
    }
    scheduled_tasks.clear();
}

Task *LLVMThreadState::launch(Kernel kernel, uint32_t size,
                              std::vector<void *> *kernel_params,
                              const std::vector<uint32_t> *) {
    Task *ret_task = nullptr;

    uint32_t packet_size = jitc_llvm_vector_width,
             desired_block_size = jitc_llvm_block_size,
             packets = (size + packet_size - 1) / packet_size,
             cores = pool_size();

    // We really don't know how much computation this kernel performs, and
    // it's important to benefit from parallelism. The following heuristic
    // therefore errs on the side of making too many parallel work units
    // ("blocks").
    //
    // The code below implements the following strategy with 3 "regimes":
    //
    // 1. If there is very little work to be done, 1 block == 1 SIMD packet.
    //    As the amount of work increases, add more blocks until we have
    //    enough of them to give one to each processor.
    //
    // 2. As the amount of work further increases, do more work within each
    //    block until a maximum is reached (16K elements per block by default)
    //
    // 3. Now, keep the block size fixed and instead more blocks. This permits
    //    better load balancing in case some of the blocks terminate early.

    uint32_t blocks, block_size;
    if (cores <= 1) {
        blocks = 1;
        block_size = size;
    } else if (packets <= cores) {
        blocks = packets;
        block_size = packet_size;
    } else if (size <= desired_block_size * cores * 2) {
        blocks = cores;
        block_size = (packets + blocks - 1) / blocks * packet_size;
    } else {
        block_size = desired_block_size;
        blocks = (size + block_size - 1) / block_size;
    }

    auto callback = [](uint32_t index, void *ptr) {
        void **params = (void **) ptr;
        LLVMKernelFunction kernel = (LLVMKernelFunction) params[0];
        uint32_t size       = (uint32_t) (uintptr_t) params[1],
                 block_size = (uint32_t) ((uintptr_t) params[1] >> 32),
                 start      = index * block_size,
                 thread_id  = pool_thread_id(),
                 end        = std::min(start + block_size, size);

        if (start >= end)
            return;

#if defined(DRJIT_ENABLE_ITTNOTIFY)
        // Signal start of kernel
        __itt_task_begin(drjit_domain, __itt_null, __itt_null,
                         (__itt_string_handle *) params[2]);
#endif
        // Perform the main computation
        kernel(start, end, thread_id, params);

#if defined(DRJIT_ENABLE_ITTNOTIFY)
        // Signal termination of kernel
        __itt_task_end(drjit_domain);
#endif
    };

    (*kernel_params)[0] = (void *) kernel.llvm.reloc[0];
    (*kernel_params)[1] = (void *) ((((uintptr_t) block_size) << 32) +
                                 (uintptr_t) size);

#if defined(DRJIT_ENABLE_ITTNOTIFY)
    kernel_params[2] = kernel.llvm.itt;
#endif

    jitc_trace("jit_run(): launching %u %u-wide packet%s in %u block%s of size %u ..",
               packets, packet_size, packets == 1 ? "" : "s", blocks,
               blocks == 1 ? "" : "s", block_size);

    ret_task = task_submit_dep(
        nullptr, &jitc_task, 1, blocks,
        callback, kernel_params->data(),
        (uint32_t) (kernel_params->size() * sizeof(void *)),
        nullptr
    );

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        task_wait(ret_task);

    scheduled_tasks.push_back(ret_task);
    return ret_task;
}

void LLVMThreadState::memset_async(void *ptr, uint32_t size_, uint32_t isize,
                                   const void *src){
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
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
    std::memcpy(&src8, src, isize);

    submit_cpu(KernelType::Other,
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

void LLVMThreadState::reduce(VarType type, ReduceOp op, const void *ptr,
                             uint32_t size, void *out) {

    uint32_t tsize = type_size[(int) type];

    // LLVM specific
    uint32_t block_size = size, blocks = 1;
    if (pool_size() > 1) {
        block_size = jitc_llvm_block_size;
        blocks     = (size + block_size - 1) / block_size;
    }

    jitc_log(Debug, "jit_reduce(" DRJIT_PTR ", type=%s, op=%s, size=%u, block_size=%u, blocks=%u)",
             (uintptr_t) ptr, type_name[(int) type], red_name[(int) op], size, block_size, blocks);

    void *target = out;
    if (blocks > 1)
        target = jitc_malloc(AllocType::HostAsync, blocks * tsize);

    Reduction reduction = jitc_reduce_create(type, op);
    submit_cpu(
        KernelType::Reduce,
        [block_size, size, tsize, ptr, reduction, target](uint32_t index) {
            reduction(ptr, index * block_size,
                      std::min((index + 1) * block_size, size),
                      (uint8_t *) target + index * tsize);
        },

        size,
        std::max(1u, blocks));

    if (blocks > 1) {
        reduce(type, op, target, blocks, out);
        jitc_free(target);
    }
}

void LLVMThreadState::block_reduce(VarType type, ReduceOp op, const void *in,
                                   uint32_t size, uint32_t block_size, void *out) {
    if (block_size == 0)
        jitc_raise("jit_block_sum(): block_size cannot be zero!");

    uint32_t tsize = type_size[(int) type];

    if (block_size == 1) {
        memcpy_async(out, in, size * tsize);
        return;
    }

    uint32_t reduced = size / block_size,
             work_unit_size = reduced,
             work_units = 1;

    if (pool_size() > 1) {
        work_unit_size = (jitc_llvm_block_size + block_size - 1) / block_size;
        work_units     = (reduced + work_unit_size - 1) / work_unit_size;
    }

    jitc_log(Debug,
            "jit_block_sum(" DRJIT_PTR " -> " DRJIT_PTR
            ", type=%s, op=%s, size=%u, block_size=%u, work_units=%u, work_unit_size=%u)",
            (uintptr_t) in, (uintptr_t) out, type_name[(int) type], red_name[(int) op],
            size, block_size, work_units, work_unit_size);

    BlockReduction red = jitc_block_reduce_create(type, op);

    submit_cpu(
        KernelType::Other,
        [in, out, red, work_unit_size, reduced, block_size](uint32_t index) {
            uint32_t start = index * work_unit_size,
                     end = std::min(start + work_unit_size, reduced);
            red(in, start, end, block_size, out);
        },

        reduced, work_units
    );
}

void LLVMThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    // LLVM specific
    uint8_t src8[8] { };
    std::memcpy(&src8, src, size);

    submit_cpu(
        KernelType::Other,
        [src8, size, dst](uint32_t) {
            std::memcpy(dst, &src8, size);
        },

        size
    );
}


void LLVMThreadState::reduce_dot(VarType type, const void *ptr_1,
                                 const void *ptr_2,
                                 uint32_t size, void *out) {

    jitc_log(Debug, "jit_reduce_dot(" DRJIT_PTR ", " DRJIT_PTR ", type=%s, size=%u)",
             (uintptr_t) ptr_1, (uintptr_t) ptr_2, type_name[(int) type],
             size);

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

    Reduction2 reduction = jitc_reduce_dot_create(type);
    submit_cpu(
        KernelType::Reduce,
        [block_size, size, tsize, ptr_1, ptr_2, reduction, target](uint32_t index) {
            reduction(ptr_1, ptr_2, index * block_size,
                      std::min((index + 1) * block_size, size),
                      (uint8_t *) target + index * tsize);
        },

        size,
        std::max(1u, blocks));

    if (blocks > 1) {
        reduce(type, ReduceOp::Add, target, blocks, out);
        jitc_free(target);
    }
}

bool LLVMThreadState::all(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_all(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = true;
        this->memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    // LLVM specific
    bool result;

    uint8_t out[4];
    reduce(VarType::UInt32, ReduceOp::And, values, reduced_size, out);
    jitc_sync_thread();
    result = (out[0] & out[1] & out[2] & out[3]) != 0;

    return result;
}

bool LLVMThreadState::any(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_any(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = false;
        this->memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    // LLVM specific
    bool result;

    uint8_t out[4];
    reduce(VarType::UInt32, ReduceOp::Or, values, reduced_size, out);
    jitc_sync_thread();
    result = (out[0] | out[1] | out[2] | out[3]) != 0;

    return result;
}

template <typename T>
static void sum_reduce_1(uint32_t start, uint32_t end, const void *in_,
                         uint32_t index, void *scratch) {
    const T *in = (const T *) in_;
    T accum = T(0);
    for (uint32_t i = start; i != end; ++i)
        accum += in[i];
    ((T*) scratch)[index] = accum;
}

template <typename T>
static void sum_reduce_2(uint32_t start, uint32_t end, const void *in_,
                         void *out_, uint32_t index, const void *scratch,
                         bool exclusive) {
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

static void sum_reduce_1(VarType vt, uint32_t start, uint32_t end,
                         const void *in, uint32_t index, void *scratch) {
    switch (vt) {
        case VarType::UInt32:  sum_reduce_1<uint32_t>(start, end, in, index, scratch); break;
        case VarType::UInt64:  sum_reduce_1<uint64_t>(start, end, in, index, scratch); break;
        case VarType::Float32: sum_reduce_1<float>   (start, end, in, index, scratch); break;
        case VarType::Float64: sum_reduce_1<double>  (start, end, in, index, scratch); break;
        default:
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);
    }
}

static void sum_reduce_2(VarType vt, uint32_t start, uint32_t end,
                         const void *in, void *out, uint32_t index,
                         const void *scratch, bool exclusive) {
    switch (vt) {
        case VarType::UInt32:  sum_reduce_2<uint32_t>(start, end, in, out, index, scratch, exclusive); break;
        case VarType::UInt64:  sum_reduce_2<uint64_t>(start, end, in, out, index, scratch, exclusive); break;
        case VarType::Float32: sum_reduce_2<float>   (start, end, in, out, index, scratch, exclusive); break;
        case VarType::Float64: sum_reduce_2<double>  (start, end, in, out, index, scratch, exclusive); break;
        default:
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);
    }
}

void LLVMThreadState::prefix_sum(VarType vt, bool exclusive, const void *in,
                                 uint32_t size, void *out) {
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

        submit_cpu(
            KernelType::Other,
            [block_size, size, in, vt, scratch](uint32_t index) {
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                sum_reduce_1(vt, start, end, in, index, scratch);
            },
            size, blocks);

        this->prefix_sum(vt, true, scratch, blocks, scratch);
    }

    submit_cpu(
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

uint32_t LLVMThreadState::compress(const uint8_t *in, uint32_t size,
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

        submit_cpu(
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

        this->prefix_sum(VarType::UInt32, true, scratch, blocks, scratch);
    }

    submit_cpu(
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

uint32_t LLVMThreadState::mkperm(const uint32_t *ptr, uint32_t size,
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
    submit_cpu(
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
    submit_cpu(
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
    submit_cpu(
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

void LLVMThreadState::memcpy(void *dst, const void *src, size_t size) {
    std::memcpy(dst, src, size);
}

void LLVMThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    submit_cpu(
        KernelType::Other,
        [dst, src, size](uint32_t) {
            std::memcpy(dst, src, size);
        },

        (uint32_t) size
    );
}

void LLVMThreadState::aggregate(void *dst_, AggregationEntry *agg,
                                uint32_t size) {
    uint32_t work_unit_size = size, work_units = 1;
    if (pool_size() > 1) {
        work_unit_size = jitc_llvm_block_size;
        work_units     = (size + work_unit_size - 1) / work_unit_size;
    }

    jitc_log(InfoSym,
             "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
             ", size=%u, work_units=%u)",
             (uintptr_t) agg, (uintptr_t) dst_, size, work_units);

    submit_cpu(
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

    submit_cpu(
        KernelType::Other, [agg](uint32_t) { free(agg); }, 1, 1);
}

void LLVMThreadState::enqueue_host_func(void (*callback)(void *),
                                        void *payload) {
    if (!jitc_task) {
        unlock_guard guard(state.lock);
        callback(payload);
    } else {
        submit_cpu(
            KernelType::Other, [payload, callback](uint32_t) { callback(payload); }, 1, 1);
    }
}

using ReduceExpanded = void (*) (void *ptr, uint32_t start, uint32_t end, uint32_t exp, uint32_t size);

template <typename Value, typename Op>
static void reduce_expanded_impl(void *ptr_, uint32_t start, uint32_t end,
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
static ReduceExpanded reduce_expanded_create(ReduceOp op) {
    using UInt = uint_with_size_t<Value>;

    struct Add { Value operator()(Value a, Value b) JIT_NO_UBSAN { return a + b; }};
    struct Mul { Value operator()(Value a, Value b) JIT_NO_UBSAN { return a * b; }};
    struct Min { Value operator()(Value a, Value b) { return std::min(a, b); }};
    struct Max { Value operator()(Value a, Value b) { return std::max(a, b); }};
    struct And {
        Value operator()(Value a, Value b) {
            (void) a; (void) b;
            if constexpr (std::is_integral_v<Value>)
                return a & b;
            else
                return 0;
        }
    };
    struct Or {
        Value operator()(Value a, Value b) {
            (void) a; (void) b;
            if constexpr (std::is_integral_v<Value>)
                return a | b;
            else
                return 0;
        }
    };

    switch (op) {
        case ReduceOp::Add: return reduce_expanded_impl<Value, Add>;
        case ReduceOp::Mul: return reduce_expanded_impl<Value, Mul>;
        case ReduceOp::Max: return reduce_expanded_impl<Value, Max>;
        case ReduceOp::Min: return reduce_expanded_impl<Value, Min>;
        case ReduceOp::And: return reduce_expanded_impl<Value, And>;
        case ReduceOp::Or: return reduce_expanded_impl<Value, Or>;

        default: jitc_raise("jit_reduce_expanded_create(): unsupported reduction type!");
    }
}

static ReduceExpanded reduce_expanded_create(VarType type, ReduceOp op) {
    using half = drjit::half;
    switch (type) {
        case VarType::Int32:   return reduce_expanded_create<int32_t >(op);
        case VarType::UInt32:  return reduce_expanded_create<uint32_t>(op);
        case VarType::Int64:   return reduce_expanded_create<int64_t >(op);
        case VarType::UInt64:  return reduce_expanded_create<uint64_t>(op);
        case VarType::Float16: return reduce_expanded_create<half    >(op);
        case VarType::Float32: return reduce_expanded_create<float   >(op);
        case VarType::Float64: return reduce_expanded_create<double  >(op);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

void LLVMThreadState::reduce_expanded(VarType vt, ReduceOp op, void *ptr,
                                      uint32_t exp, uint32_t size) {
    jitc_log(Debug, "jit_reduce_expanded(" DRJIT_PTR ", type=%s, op=%s, expfactor=%u, size=%u)",
            (uintptr_t) ptr, type_name[(int) vt],
            red_name[(int) op], exp, size);

    ReduceExpanded kernel = reduce_expanded_create(vt, op);

    uint32_t block_size = size, blocks = 1;
    if (pool_size() > 1) {
        block_size = jitc_llvm_block_size;
        blocks     = (size + block_size - 1) / block_size;
    }

    submit_cpu(
        KernelType::Reduce,
        [ptr, block_size, exp, size, kernel](uint32_t index) {
            kernel(ptr, index * block_size,
                   std::min((index + 1) * block_size, size), exp, size);
        },

        size, std::max(1u, blocks));
}
