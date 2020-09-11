/*
    src/tbb.cpp -- Parallelization via LLVM (optional)

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(ENOKI_JIT_ENABLE_TBB)

#include "tbb.h"
#include "internal.h"
#include "llvm_api.h"
#include "var.h"
#include "profiler.h"
#include <deque>
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#define __TBB_show_deprecation_message_task_H 1
#include <tbb/task.h>

extern std::vector<std::pair<uint32_t, uint32_t>> jit_llvm_scatter_add_variables;
tsl::robin_map<uint32_t, std::vector<void *>*> tbb_scatter_add;

/**
 * Task base class, provides the ability to fetch the next job from the
 * task queue or stop parallel execution.
 */
struct EnokiTaskBase : public tbb::task {
public:
    EnokiTaskBase(Stream *stream) : m_stream(stream) { }
    task *execute() {
        std::lock_guard<std::mutex> guard(m_stream->tbb_task_queue_mutex);
        m_stream->tbb_task_queue.pop_front();
        if (m_stream->tbb_task_queue.empty())
            return nullptr;
        set_ref_count(1);
        return m_stream->tbb_task_queue.front();
    }
private:
    Stream *m_stream;
};

/**
 * Launch of one or more kernels that can execute in parallel. Recursively
 * expands into many \ref EnokiKernelTaskRange instances.
 */
struct EnokiKernelTask : public EnokiTaskBase {
public:
    EnokiKernelTask(Stream *stream) : EnokiTaskBase(stream) { }

    tbb::task *execute() {
        if (m_task_count == 0)
            return EnokiTaskBase::execute();

        recycle_as_safe_continuation();
        set_ref_count(m_task_count + 1);
        task &first_task = m_tasks.pop_front();
        spawn(m_tasks);
        m_task_count = 0;
        return &first_task;
    }

    bool empty() {
        return m_task_count == 0;
    }

    void append(tbb::task *task) {
        m_tasks.push_back(*task);
        m_task_count++;
    }

private:
    tbb::task_list m_tasks;
    uint32_t m_task_count = 0;
};

/// Task that executes a kernel over a given range of inputs
struct EnokiKernelTaskRange : public tbb::task {
public:
    EnokiKernelTaskRange(LLVMKernelFunction kernel, uint32_t start,
                         uint32_t end, const std::shared_ptr<void *> &args,
                         const void *itt)
        : m_kernel(kernel), m_start(start), m_end(end), m_args(args) {
#if defined(ENOKI_ENABLE_ITTNOTIFY)
        m_itt = itt;
#else
        (void) itt;
#endif
    }

    tbb::task *execute() {
        // Signal kernel being executed via ITT (optional)
#if defined(ENOKI_ENABLE_ITTNOTIFY)
        __itt_task_begin(enoki_domain, __itt_null, __itt_null,
                         (__itt_string_handle *) m_itt);
#endif

        // Perform the main computation
        m_kernel(m_start, m_end, m_args.get());

        // Signal termination of kernel
#if defined(ENOKI_ENABLE_ITTNOTIFY)
        __itt_task_end(enoki_domain);
#endif
        m_args.reset();

        return nullptr;
    }

private:
    LLVMKernelFunction m_kernel;
    uint32_t m_start, m_end;
    std::shared_ptr<void *> m_args;
#if defined(ENOKI_ENABLE_ITTNOTIFY)
    const void *m_itt;
#endif
};

/// Task that executes a function asynchronously
struct EnokiFuncTask : public EnokiTaskBase {
public:
    EnokiFuncTask(Stream *stream, void (*func)(void *), void *buf,
                  size_t extra_size)
        : EnokiTaskBase(stream), m_func(func) {
        if (extra_size > sizeof(m_extra))
            jit_fail("EnokiFuncTask: function requires too much extra memory!");
        memcpy(m_extra, buf, extra_size);
    }

    tbb::task *execute() {
        m_func(m_extra);
        return EnokiTaskBase::execute();
    }

private:
    void (*m_func)(void *);
    uint8_t m_extra[48];
};

/// Initialize TBB's task scheduler for a given Enoki stream
void tbb_stream_init(Stream *stream) {
    stream->tbb_task_root = new (tbb::task::allocate_root()) tbb::empty_task();
    stream->tbb_task_root->increment_ref_count();
}

/// Shut down TBB's task scheduler for a given Enoki stream
void tbb_stream_shutdown(Stream *stream) {
    stream->tbb_task_root->wait_for_all();
    stream->tbb_task_root->destroy(*stream->tbb_task_root);
}

/// Wait for all TBB tasks to finish
void tbb_stream_sync(Stream *stream) {
    stream->tbb_task_root->wait_for_all();
    stream->tbb_task_root->increment_ref_count();
}

/// Append a kernel execution, but do not submit it to the queue yet
void tbb_stream_enqueue_kernel(Stream *stream, LLVMKernelFunction kernel,
                               uint32_t start, uint32_t stop, uint32_t argc,
                               void **argv, bool parallel_dispatch,
                               const void *itt) {
    size_t size          = stop - start,
           tasks_desired = jit_llvm_thread_count * 4,
           grain_size    = 4096;

    // Scatter-add operations increase the cost of having many tasks
    if (!jit_llvm_scatter_add_variables.empty())
        tasks_desired = jit_llvm_thread_count;

    // Items to be processed per task (must be >= grain size, round up to packet size)
    size_t items_per_task =
        std::max((size + tasks_desired - 1u) / tasks_desired, grain_size);

    items_per_task = (items_per_task + jit_llvm_vector_width - 1) /
                     jit_llvm_vector_width * jit_llvm_vector_width;

    // Given that, how many tasks should we launch (might be smaller than 'tasks_desired')
    uint32_t task_count = (uint32_t) ((size + items_per_task - 1u) / items_per_task);

    if (unlikely(!parallel_dispatch)) {
        task_count = 1;
        items_per_task = size;
    }

    if (!stream->tbb_kernel_task)
        stream->tbb_kernel_task = new (stream->tbb_task_root->allocate_child())
            EnokiKernelTask(stream);

    if (jit_llvm_scatter_add_variables.empty()) {
        /// Easy case: the kernel doesn't contain any 'scatter_add' instruction
        std::shared_ptr<void*> args(new void*[argc], std::default_delete<void*[]>());
        memcpy(args.get(), argv, sizeof(void*) * argc);

        for (uint32_t i = 0; i < task_count; ++i) {
            uint32_t j = task_count - 1 - i;
            tbb::task *task = new (stream->tbb_kernel_task->allocate_child())
                EnokiKernelTaskRange(kernel,
                                     (uint32_t) (start + j * items_per_task),
                                     (uint32_t) (start + std::min(size, (j + 1) * items_per_task)),
                                     args, itt);

            ((EnokiKernelTask *) stream->tbb_kernel_task)->append(task);
        }
    } else {
        /// Hard case: create a separate output array per thread, accumulate afterwards
        for (uint32_t i = 0; i < task_count; ++i) {
            std::shared_ptr<void*> args(new void*[argc], std::default_delete<void*[]>());
            memcpy(args.get(), argv, sizeof(void*) * argc);

            for (auto kv : jit_llvm_scatter_add_variables) {
                const Variable *v = jit_var(kv.second);
                size_t size = (size_t) v->size * var_type_size[v->type];
                void *ptr = jit_malloc(AllocType::HostAsync, size);
                auto vec = tbb_scatter_add[kv.second];
                if (!vec)
                    vec = tbb_scatter_add[kv.second] = new std::vector<void*>();
                vec->push_back(ptr);
                args.get()[kv.first] = ptr;
            }

            uint32_t j = task_count - 1 - i;
            tbb::task *task = new (stream->tbb_kernel_task->allocate_child())
                EnokiKernelTaskRange(kernel,
                                     (uint32_t) (start + j * items_per_task),
                                     (uint32_t) (start + std::min(size, (j + 1) * items_per_task)),
                                     args, itt);

            ((EnokiKernelTask *) stream->tbb_kernel_task)->append(task);
        }
    }

    if (unlikely(!parallel_dispatch))
        tbb_stream_submit_kernel(stream);
}

static void enqueue(Stream *stream, tbb::task *task) {
    stream->tbb_task_root->increment_ref_count();
    stream->tbb_task_queue.push_back(task);

    if (stream->tbb_task_queue.size() == 1)
        tbb::task::spawn(*task);
}

/// Scatter-add special handling (to be called just before a kernel launch)
void tbb_scatter_add_pre(Stream *stream) {
    for (auto &kv : tbb_scatter_add) {
        const Variable *v = jit_var(kv.first);
        struct Inputs {
            size_t size;
            std::vector<void *> *ptrs;
        };
        Inputs inputs { (size_t) v->size * var_type_size[v->type], kv.second };

        auto func = [](void *inputs_) {
            Inputs inputs = *(Inputs *) inputs_;
            tbb::parallel_for(
                tbb::blocked_range<uint32_t>(0, (uint32_t) inputs.ptrs->size(), 1),
                [&](const tbb::blocked_range<uint32_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i)
                        memset((*inputs.ptrs)[i], 0, inputs.size);
                },
                tbb::simple_partitioner()
            );
        };
        enqueue(stream, new (stream->tbb_task_root->allocate_child())
                            EnokiFuncTask(stream, func, &inputs, sizeof(Inputs)));
    }
}

/// Scatter-add special handling (to be called just after a kernel launch)
void tbb_scatter_add_post(Stream *stream) {
    for (auto &kv : tbb_scatter_add) {
        const Variable *v = jit_var(kv.first);
        struct Inputs {
            void *out;
            uint32_t size;
            VarType type;
            std::vector<void *> *ptrs;
            Stream *stream;
        };
        Inputs inputs { v->data, v->size, (VarType) v->type, kv.second, stream };

        auto func = [](void *inputs_) {
            Inputs inputs = *(Inputs *) inputs_;
            tbb::parallel_for(
                tbb::blocked_range<uint32_t>(0, inputs.size, 4096),
                [&](const tbb::blocked_range<uint32_t> &range) {
                    uint32_t ptr_size = (uint32_t) inputs.ptrs->size();
                    void **ptr_val = inputs.ptrs->data();

                    switch (inputs.type) {
                        case VarType::Int32:
                        case VarType::UInt32:
                            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                                for (uint32_t j = 0; j < ptr_size; ++j)
                                    ((uint32_t *) inputs.out)[i] += ((const uint32_t *) ptr_val[j])[i];
                            }
                            break;

                        case VarType::Int64:
                        case VarType::UInt64:
                            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                                for (uint32_t j = 0; j < ptr_size; ++j)
                                    ((uint64_t *) inputs.out)[i] += ((const uint64_t *) ptr_val[j])[i];
                            }
                            break;

                        case VarType::Float32:
                            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                                for (uint32_t j = 0; j < ptr_size; ++j)
                                    ((float *) inputs.out)[i] += ((const float *) ptr_val[j])[i];
                            }
                            break;

                        case VarType::Float64:
                            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                                for (uint32_t j = 0; j < ptr_size; ++j)
                                    ((double *) inputs.out)[i] += ((const double *) ptr_val[j])[i];
                            }
                            break;

                        default:
                            jitc_fail("tbb_scatter_add_post(): variable type "
                                      "%s not handled!",
                                      var_type_name[(int) inputs.type]);
                    }
                }
            );
            std::swap(active_stream, inputs.stream);
            for (auto p : *inputs.ptrs)
                jitc_free(p);
            std::swap(active_stream, inputs.stream);
            delete inputs.ptrs;
        };
        enqueue(stream, new (stream->tbb_task_root->allocate_child())
                            EnokiFuncTask(stream, func, &inputs, sizeof(Inputs)));
    }
    tbb_scatter_add.clear();
}

/// Submit a set of kernel tasks to the TBB task scheduler
void tbb_stream_submit_kernel(Stream *stream) {
    std::lock_guard<std::mutex> guard(stream->tbb_task_queue_mutex);
    if (stream->tbb_kernel_task == nullptr ||
        ((EnokiKernelTask *) stream->tbb_kernel_task)->empty())
        return;

    if (unlikely(!tbb_scatter_add.empty()))
        tbb_scatter_add_pre(stream);

    enqueue(stream, stream->tbb_kernel_task);
    stream->tbb_kernel_task = nullptr;

    if (unlikely(!tbb_scatter_add.empty()))
        tbb_scatter_add_post(stream);
}

/// Enqueue a function for asynchronous execution
void tbb_stream_enqueue_func(Stream *stream, void (*func)(void *),
                             void *extra, size_t extra_size) {
    std::lock_guard<std::mutex> guard(stream->tbb_task_queue_mutex);
    enqueue(stream, new (stream->tbb_task_root->allocate_child())
                        EnokiFuncTask(stream, func, extra, extra_size));
}

#endif
