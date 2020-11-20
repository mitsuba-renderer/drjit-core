/*
    enoki-jit/pool.h -- Simple thread pool with a task-based API

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>
#include <stddef.h>

#if defined(_MSC_VER)
#  if defined(ENOKI_THREAD_BUILD)
#    define EK_THREAD_EXPORT    __declspec(dllexport)
#  else
#    define EK_THREAD_EXPORT    __declspec(dllimport)
#  endif
#else
#  define EK_THREAD_EXPORT    __attribute__ ((visibility("default")))
#endif

#if defined(__cplusplus)
#  define EK_THREAD_DEF(x) = x
#else
#  define EK_THREAD_DEF(x)
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \brief Create a new thread pool with the specified number of threads
 *
 * A value of \c 0 will cause the implementation to auto-detect the number of cores.
 */
extern EK_THREAD_EXPORT struct Pool *pool_create(size_t size EK_THREAD_DEF(0));

/**
 * \brief Destroy the thread pool and discard remaining unfinished work.
 *
 * It is undefined behavior to destroy the thread pool while other threads
 * are waiting for the completion of scheduled work, e.g. via \ref pool_task_wait()
 * and \ref pool_task_wait_and_release().
 *
 * \param pool
 *     The thread pool to destroy. \c nullptr refers to the default pool.
 */
extern EK_THREAD_EXPORT void pool_destroy(struct Pool *pool EK_THREAD_DEF(0));

/**
 * \brief Return the number of threads that are part of the pool
 *
 * \param pool
 *     The thread pool to query. \c nullptr refers to the default pool.
 */
extern EK_THREAD_EXPORT size_t pool_size(struct Pool *pool EK_THREAD_DEF(0));

/**
 * \brief Resize the thread pool to the given number of threads
 *
 * \param pool
 *     The thread pool to resize. \c nullptr refers to the default pool.
 */
extern EK_THREAD_EXPORT void pool_set_size(struct Pool *pool, size_t size);

/**
 * \brief Return a unique number identifying the current worker thread
 *
 * When called from a thread pool worker (e.g. while executing a parallel
 * task), this function returns a unique identifying number between 1 and the
 * pool's total thread count.
 *
 * The IDs of separate thread pools overlap. When the current thread is not a
 * thread pool worker, the function returns zero.
 */
extern EK_THREAD_EXPORT uint32_t pool_thread_id();

/*
 * \brief Submit a new task to a thread pool
 *
 * This function submits a new task consisting of \c size work units to the
 * thread pool \c pool.
 *
 * The \c pred and \c pred_count parameters can be used to specify predecessor
 * tasks that must be completed before execution of this task can commence. If
 * the task does not depend on any other tasks (e.g. <tt>pred_count == 0</tt>
 * and <tt>pred == nullptr</tt>), or when all of those other tasks have already
 * finished executing, then it will be immediately appended to the end of the
 * task queue. Otherwise, the task will be appended once all predecessor tasks
 * have finished executing.
 *
 * The task callback \c func will be invoked \c size times by the various
 * thread pool workers. Its first argument will range from 0 to \c size - 1,
 * and the second argument refers to a payload memory region specified via the
 * \c payload parameter.
 *
 * This payload is handled using one of two possible modes:
 *
 * <ol>
 *    <li>When <tt>size == 0</tt> or <tt>payload_deleter != nullptr</tt>, the
 *    value of \c payload is simply forwarded to \c func. In the latter case,
 *    <tt>payload_deleter(payload)</tt> is invoked following completion of the
 *    task, which can carry out additional cleanup operations if needed. In
 *    both cases, the memory region targeted by \c payload may be accessed
 *    asynchronously and must remain valid until the task is done.</li>
 *
 *    <li>Otherwise, the function will internally create a copy of the payload
 *    and free it following completion of the task. In this case, it is fine to
 *    delete the the memory region targeted by \c payload right after the
 *    function call.</li>
 * </ol>
 *
 * The function returns a task handle that can be used to schedule other
 * dependent tasks, and to wait for task completion if desired. This handle
 * must eventually be released using either \ref pool_task_release() or \ref
 * pool_task_wait_and_release(). A failure to do so will result in memory
 * leaks.
 *
 * Barriers and similar dependency relations can be encoded by via artificial
 * tasks using <tt>size == 1</tt> and <tt>func == nullptr<tt> along with a set
 * of predecessor tasks.
 *
 * \param pool
 *     The thread pool that should execute the specified task. \c nullptr
 *     refers to the default pool.
 *
 * \param pred
 *     List of predecessors of size \c pred_count.
 *     \c nullptr-valued elements are ignored
 *
 * \param pred
 *     Numer of predecessor tasks
 *
 * \param size
 *     Total number of work units, the callback \c func will be called this
 *     many times if provided. Must be greater than zero.
 *
 * \param func
 *     Callback function that will be invoked to perform the actual computation.
 *     If set to \c nullptr, the callback is ignored. This can be used to create
 *     artificial tasks that only encode dependencies.
 *
 * \param payload
 *     Optional payload that is passed to the function \c func
 *
 * \param payload_size
 *     When \c payload_deleter is equal to \c nullptr and when \c size is
 *     nonzero, a temporary copy of the payload will be made. This parameter is
 *     necessary to specify the payload size in that case.
 *
 * \param payload_deleter
 *     Optional callback that will be invoked to free the payload
 */
extern EK_THREAD_EXPORT
struct Task *pool_task_submit(Pool *pool,
                              Task **pred,
                              size_t pred_count,
                              size_t size EK_THREAD_DEF(1),
                              void (*func)(size_t, void *) EK_THREAD_DEF(0),
                              void *payload EK_THREAD_DEF(0),
                              size_t payload_size EK_THREAD_DEF(0),
                              void (*payload_deleter)(void *) EK_THREAD_DEF(0));

/*
 * \brief Release a task handle so that it can eventually be reused
 *
 * Releasing a task handle does not impact the tasks's execution, which could
 * be in one of three states: waiting, running, or complete. This operation is
 * important because it frees internal resources that would otherwise leak.
 *
 * Following a call to \ref pool_task_release(), the associated task can no
 * longer be used as a direct predecessor of other tasks, and it is no longer
 * possible to wait for its completion using an operation like \ref
 * pool_task_wait().
 *
 * \param pool
 *     The thread pool containing the task. \c nullptr refers to the default pool.
 *
 * \param task
 *     The task in question. When equal to \c nullptr, the operation is a no-op.
 */
extern EK_THREAD_EXPORT void pool_task_release(struct Task *task);

/*
 * \brief Wait for the completion of the specified task
 *
 * This function causes the calling thread to sleep until all work units of
 * 'task' have been completed.
 *
 * \param task
 *     The task in question. When equal to \c nullptr, the operation is a no-op.
 */
extern EK_THREAD_EXPORT void pool_task_wait(struct Task *task);

/*
 * \brief Wait for the completion of the specified task and release it
 *
 * This function is a more efficient combined version of \ref pool_task_wait()
 * followed by \ref pool_task_release().
 *
 * \param task
 *     The task in question. When equal to \c nullptr, the operation is a no-op.
 */
extern EK_THREAD_EXPORT void pool_task_wait_and_release(struct Task *task);

#if defined(__cplusplus)
}

#include <utility>

namespace enoki {
    template <typename Int> struct blocked_range {
    public:
        blocked_range(Int start, Int end, Int block_size = 1)
            : m_start(start), m_end(end), m_block_size(block_size) { }

        size_t blocks() const {
            return (size_t) ((m_end - m_start + m_block_size - 1) / m_block_size);
        }

        Int start() const { return m_start; }
        Int end() const { return m_end; }
        Int block_size() const { return m_block_size; }

    private:
        Int m_start;
        Int m_end;
        Int m_block_size;
    };


template <typename Int, typename Func>
void parallel_for(const blocked_range<Int> &range,
                  Func &&func,
                  Pool *pool = nullptr) {

    size_t blocks = range.blocks();

    if (blocks <= 1) {
        func(range);
    } else {
        struct Payload {
            Func *func_;
            Int start, end, block_size;
        };

        Payload payload { &func, range.start(), range.end(), range.block_size() };

        Task *task = pool_task_submit(
            pool, nullptr, 0, blocks,
            [](size_t i, void *p_) {
                Payload &p = *((Payload *) p_);
                uint32_t start = p.start + p.block_size * i,
                         end   = start + p.block_size;

                if (end > p.end)
                    end = p.end;

                (*p.func_)(blocked_range<Int>(start, end));
            }, &payload
        );

        pool_task_wait_and_release(task);
    }
}
}
#endif
