#include "llvm_ts.h"
#include "llvm.h"
#include "log.h"
#include "var.h"
#include "common.h"
#include "profile.h"
#include "util.h"
#include "llvm_red.h"

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

void LLVMThreadState::barrier() {
    // All tasks in 'scheduled_tasks' have 'jitc_task' (if present) as parent.
    // To create a barrier, we release 'jitc_task' and initialize it with a new
    // dummy task that has all members of 'scheduled_tasks' as parents. This
    // allows groups of kernel launches to run in parallel, while serializing
    // subsequent groups.

    if (scheduled_tasks.size() == 1) {
        task_release(jitc_task);
        jitc_task = scheduled_tasks[0];
    } else {
        jitc_assert(!scheduled_tasks.empty(),
                    "jit_eval(): no tasks generated!");

        // Insert a barrier task
        Task *new_task = task_submit_dep(nullptr, scheduled_tasks.data(),
                                         (uint32_t) scheduled_tasks.size());
        task_release(jitc_task);
        for (Task *t : scheduled_tasks)
            task_release(t);
        jitc_task = new_task;
    }
    scheduled_tasks.clear();
}

Task *
LLVMThreadState::launch(Kernel kernel, KernelKey * /*key*/,
                        XXH128_hash_t /*hash*/, uint32_t size,
                        std::vector<void *> *kernel_params,
                        const std::vector<uint32_t> * /*kernel_param_ids*/) {
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
    // 3. Now, keep the block size fixed and instead add more blocks. This
    //    improves load balancing if some of the blocks terminate early.

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

void LLVMThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                   uint32_t block_size, const void *in,
                                   void *out) {
    uint32_t tsize = type_size[(int) vt];
    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise(
            "jit_block_reduce(): invalid block size (size=%u, block_size=%u)!",
            size, block_size);
    } else if (block_size == 1) {
        memcpy_async(out, in, size * (size_t) tsize);
        return;
    }

    // Get the size of the thread pool to inform the decisions below
    uint32_t workers = pool_size();

    // How many blocks are there to reduce?
    // Note: 'size' is not necessary divisible by 'block_size'.
    uint32_t block_count = ceil_div(size, block_size);

    // Ideally, parallelize over 'ideal_size'-length regions
    uint32_t ideal_size = jitc_llvm_block_size;

    // The kernel can split each block into smaller chunks to help with this
    uint32_t chunk_size = block_size;
    if (chunk_size > 2 * ideal_size && block_count < 2 * workers)
        chunk_size = ideal_size;

    // How many chunks in total need to be processed?
    uint32_t chunks_per_block = ceil_div(block_size, chunk_size),
             chunk_count = block_count * chunks_per_block;

    // Split chunks into work units for the parallelization layer
    uint32_t work_units, work_unit_size;
    if (workers > 1) {
        // Do multiple chunks per work unit if they are smaller than 'ideal_size'
        work_unit_size = std::min(ceil_div(ideal_size, chunk_size), chunk_count);
        work_units     = ceil_div(chunk_count, work_unit_size);
    } else {
        work_unit_size = chunk_count;
        work_units = 1;
    }

    BlockReduction red = create_block_reduction(vt, op);

    void *buf = out;
    if (chunks_per_block > 1) {
        buf = jitc_malloc(AllocType::HostAsync, tsize * chunk_count);

        jitc_log(Debug,
                 "jit_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
                 ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u): "
                 "launching %u work unit%s, each processing %u chunk%s (%u "
                 "chunks/block). ",
                 (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
                 red_name[(int) op], size, block_size, block_count, work_units,
                 work_units > 1 ? "s" : "", work_unit_size,
                 work_unit_size > 1 ? "" : "", chunks_per_block);
    } else {
        jitc_log(Debug,
                 "jit_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
                 ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u): "
                 "launching %u work unit%s, each processing %u block%s.",
                 (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
                 red_name[(int) op], size, block_size, block_count, work_units,
                 work_units > 1 ? "s" : "", work_unit_size,
                 work_unit_size > 1 ? "s" : "");
    }

    submit_cpu(
        KernelType::Reduce,
        [red, work_unit_size, size, block_size, chunk_size, chunk_count, chunks_per_block, in, buf](uint32_t index) {
            red(index, work_unit_size, size, block_size, chunk_size, chunk_count, chunks_per_block, in, buf);
        },
        size, work_units
    );

    if (chunks_per_block > 1) {
        // Multiple chunks per block, we require an additional reduction
        block_reduce(vt, op, chunk_count, chunks_per_block, buf, out);
        jitc_free(buf);
    }
}

void LLVMThreadState::block_prefix_reduce(VarType vt, ReduceOp op,
                                          uint32_t size, uint32_t block_size,
                                          bool exclusive, bool reverse,
                                          const void *in, void *out) {
    uint32_t tsize = type_size[(int) vt];
    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise(
            "jit_block_prefix_reduce(): invalid block size (size=%u, block_size=%u)!",
            size, block_size);
    } else if (block_size == 1) {
        uint64_t z = 0;
        if (exclusive)
            memset_async(out, size, tsize, &z);
        else if (in != out)
            memcpy_async(out, in, size * tsize);
        return;
    }

    // Get the size of the thread pool to inform the decisions below
    uint32_t workers = pool_size();

    // How many blocks are there to reduce?
    // Note: 'size' is not necessary divisible by 'block_size'.
    uint32_t block_count = ceil_div(size, block_size);

    // Ideally, parallelize over 'ideal_size'-length regions
    uint32_t ideal_size = jitc_llvm_block_size;

    // The kernel can split each block into smaller chunks to help with this
    uint32_t chunk_size = block_size;
    if (chunk_size > 2 * ideal_size && block_count < 2 * workers)
        chunk_size = ideal_size;

    // How many chunks in total need to be processed?
    uint32_t chunks_per_block = ceil_div(block_size, chunk_size),
             chunk_count = block_count * chunks_per_block;

    // Split chunks into work units for the parallelization layer
    uint32_t work_units, work_unit_size;
    if (workers > 1) {
        // Do multiple chunks per work unit if they are smaller than 'ideal_size'
        work_unit_size = std::min(ceil_div(ideal_size, chunk_size), chunk_count);
        work_units     = ceil_div(chunk_count, work_unit_size);
    } else {
        work_unit_size = chunk_count;
        work_units = 1;
    }

    BlockReduction red_1 = create_block_reduction(vt, op);
    BlockPrefixReduction red_2 = create_block_prefix_reduction(vt, op);

    void *scratch = nullptr;
    if (chunks_per_block > 1) {
        scratch = jitc_malloc(AllocType::HostAsync, tsize * chunk_count);

        // Launch a block_reduce within block_prefix_reduce. We need to access to its
        // intermediate state, which is why the kernel is called directly here
        jitc_log(Debug,
                 "jit_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
                 ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u): "
                 "launching %u work unit%s, each processing %u chunk%s (%u "
                 "chunks/block). ",
                 (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
                 red_name[(int) op], size, block_size, block_count, work_units,
                 work_units > 1 ? "s" : "", work_unit_size,
                 work_unit_size > 1 ? "" : "", chunks_per_block);

        submit_cpu(
            KernelType::Reduce,
            [red_1, work_unit_size, size, block_size, chunk_size, chunk_count,
             chunks_per_block, in, scratch](uint32_t index) {
                red_1(index, work_unit_size, size, block_size, chunk_size,
                      chunk_count, chunks_per_block, in, scratch);
            },
            size, work_units);

        block_prefix_reduce(vt, op, chunk_count, chunks_per_block, true,
                            reverse, scratch, scratch);
    }

    jitc_log(Debug,
             "jit_block_prefix_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u, "
             "exclusive=%i, reverse=%i): launching %u work unit%s, each "
             "processing %u block%s.",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, block_count, exclusive,
             reverse, work_units, work_units > 1 ? "s" : "", work_unit_size,
             work_unit_size > 1 ? "s" : "");

    submit_cpu(
        KernelType::Reduce,
        [red_2, work_unit_size, size, block_size, chunk_size, chunk_count,
         chunks_per_block, exclusive, reverse, in, scratch,
         out](uint32_t index) {
            red_2(index, work_unit_size, size, block_size, chunk_size,
                  chunk_count, chunks_per_block, exclusive, reverse, in,
                  scratch, out);
        },
        size, work_units);

    if (chunks_per_block > 1)
        jitc_free(scratch);
}

void LLVMThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

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
        block_reduce(type, ReduceOp::Add, blocks, blocks, target, out);
        jitc_free(target);
    }
}

uint32_t LLVMThreadState::compress(const uint8_t *in, uint32_t size,
                                   uint32_t *out) {
    if (size == 0)
        return 0;

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

        block_prefix_reduce(VarType::UInt32, ReduceOp::Add, blocks,
                            blocks, true, false, scratch, scratch);
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
