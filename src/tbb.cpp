/*
    src/tbb.cpp -- Parallelization via LLVM (optional)

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(ENOKI_TBB)

#include "tbb.h"
#include "internal.h"
#include "llvm_api.h"
#include <deque>
#include <mutex>
#include <tbb/tbb.h>

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
                         uint32_t end, const std::shared_ptr<void *> &args)
        : m_kernel(kernel), m_start(start), m_end(end), m_args(args) { }

    tbb::task *execute() {
        m_kernel(m_start, m_end, m_args.get());
        m_args.reset();
        return nullptr;
    }

private:
    LLVMKernelFunction m_kernel;
    uint32_t m_start, m_end;
    std::shared_ptr<void *> m_args;
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
    uint8_t m_extra[32];
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
                               void **argv) {
    size_t size          = stop - start,
           tasks_desired = jit_llvm_thread_count * 4,
           grain_size    = 4096;

    // Items to be processed per task (must be >= grain size, round up to packet size)
    size_t items_per_task =
        std::max((size + tasks_desired - 1u) / tasks_desired, grain_size);

    items_per_task = (items_per_task + jit_llvm_vector_width - 1) /
                     jit_llvm_vector_width * jit_llvm_vector_width;

    // Given that, how many tasks should we launch (might be smaller than 'tasks_desired')
    size_t task_count = (size + items_per_task - 1u) / items_per_task;

    if (!stream->tbb_kernel_task)
        stream->tbb_kernel_task = new (stream->tbb_task_root->allocate_child())
            EnokiKernelTask(stream);

    std::shared_ptr<void*> args(new void*[argc], std::default_delete<void*[]>());
    memcpy(args.get(), argv, sizeof(void*) * argc);

    for (uint32_t i = 0; i < task_count; ++i) {
        uint32_t j = task_count - 1 - i;
        tbb::task *task = new (stream->tbb_kernel_task->allocate_child())
            EnokiKernelTaskRange(kernel,
                                 (uint32_t) (start + j * items_per_task),
                                 (uint32_t) (start + std::min(size, (j + 1) * items_per_task)),
                                 args);

        ((EnokiKernelTask *) stream->tbb_kernel_task)->append(task);
    }
}

static void enqueue(Stream *stream, tbb::task *task) {
    stream->tbb_task_root->increment_ref_count();
    stream->tbb_task_queue.push_back(task);

    if (stream->tbb_task_queue.size() == 1)
        tbb::task::spawn(*task);
}

/// Submit a set of kernel tasks to the TBB task scheduler
void tbb_stream_submit_kernel(Stream *stream) {
    std::lock_guard<std::mutex> guard(stream->tbb_task_queue_mutex);
    if (stream->tbb_kernel_task == nullptr ||
        ((EnokiKernelTask *) stream->tbb_kernel_task)->empty())
        return;

    enqueue(stream, stream->tbb_kernel_task);
    stream->tbb_kernel_task = nullptr;
}

/// Enqueue a function for asynchronous execution
void tbb_stream_enqueue_func(Stream *stream, void (*func)(void *),
                             void *extra, size_t extra_size) {
    std::lock_guard<std::mutex> guard(stream->tbb_task_queue_mutex);
    enqueue(stream, new (stream->tbb_task_root->allocate_child())
                        EnokiFuncTask(stream, func, extra, extra_size));
}

#endif
