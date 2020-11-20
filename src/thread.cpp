/*
    src/pool.cpp -- Simple thread pool with task-based API

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <enoki-jit/thread.h>
#include <condition_variable>
#include <thread>
#include <vector>
#include <deque>

#if defined(_WIN32)
#  include <processthreadsapi.h>
#endif

#define pool_assert(x)                                                         \
    if (unlikely(!(x))) {                                                      \
        fprintf(stderr, "Assertion failed: " #x);                              \
        abort();                                                               \
    }

struct Worker;

/// Data structure describing a pool of workers
struct Pool {
    /// Mutex protecting all pool-related data structures
    std::mutex mutex;

    /// List of active queued tasks
    std::deque<Task*> tasks;

    /// Completed task records for later reuse
    std::vector<Task*> tasks_unused;

    /// Keeps track of the total number of Task instances created
    size_t tasks_total = 0;

    /// Used to send notifications when new work is added
    std::condition_variable cv_tasks_available;

    /// Used to send notifications when tasks complete
    std::condition_variable cv_tasks_completed;

    /// List of currently running worker threads
    std::vector<Worker *> workers;
};

/// A task consisting of one more work units
struct Task {
    /// Predecessors of this task that must still finish
    size_t wait;

    /// Total number of work units
    size_t size;

    /// Number of work units already started so far
    size_t started;

    /// Number of work units completed so far
    size_t completed;

    /// Callback of the work unit
    void (*func)(size_t, void *);

    /// Payload to be delivered to 'funch'
    void *payload;

    /// Custom deleter used to free 'payload'
    void (*payload_deleter)(void *);

    /// Pool that this tasks belongs to
    Pool *pool;

    /// Lists of tasks dependent on this one
    std::vector<Task *> children;

    /// Immediately recycle this task when the work is done?
    bool recycle_when_completed;

    /// Fixed-size payload storage region
    alignas(8) uint8_t payload_storage[256];

    void delete_payload() {
        if (!payload_deleter)
            ;
        else if (payload_deleter == free)
            free(payload);
        else
            payload_deleter(payload);
        payload_deleter = nullptr;
    }
};


#if defined(_MSC_VER)
  static __declspec(thread) uint32_t thread_id_tls = 0;
#else
  static __thread uint32_t thread_id_tls = 0;
#endif

struct Worker {
    Pool *pool;
    uint32_t id;
    bool shutdown;
    std::thread thread;

    Worker(Pool *pool, uint32_t id) : pool(pool), id(id), shutdown(false) {
        thread = std::thread(&Worker::run, this);
    }

    void run() {
        thread_id_tls = id;

        #if defined(_WIN32)
            wchar_t buf[24];
            _snwprintf(buf, sizeof(buf) / sizeof(wchar_t), L"Enoki worker %u", id);
            SetThreadDescription(GetCurrentThread(), buf);
        #else
            char buf[24];
            snprintf(buf, sizeof(buf), "Enoki worker %u", id);
            #if defined(__APPLE__)
                pthread_setname_np(buf);
            #else
                pthread_setname_np(pthread_self(), buf);
            #endif
        #endif

        lock_guard guard(pool->mutex);
        while (true) {
            // Wait until the queue is non-empty
            while (pool->tasks.empty() && !shutdown)
                pool->cv_tasks_available.wait(guard);

            if (shutdown)
                break;

            // Fetch the task at the top of the queue
            Task *task = pool->tasks.front();

            // Determine the current work unit index
            size_t index = task->started++;
            pool_assert(task->started <= task->size);

            // Last work unit from this task, remove from queue
            if (task->started == task->size)
                pool->tasks.pop_front();

            if (task->func) {
                // Release lock and execute work unit
                unlock_guard guard_2(pool->mutex);
                task->func(index, task->payload);
            }

            task->completed++;
            pool_assert(task->completed <= task->size);

            // Is this task finished?
            if (task->completed == task->size) {
                /// If so, potentially schedule dependent tasks
                if (!task->children.empty()) {
                    ssize_t notifications = 0;

                    for (Task *child : task->children) {
                        pool_assert(child->wait > 0);
                        if (--child->wait == 0) {
                            pool->tasks.push_back(child);
                            notifications += (ssize_t) child->size;
                        }
                    }

                    notifications -= 1; // Worker doesn't need to notify itself

                    if (notifications > 0) {
                        /// Also inform other workers
                        if (notifications < (ssize_t) pool->workers.size()) {
                            for (ssize_t i = 0; i < notifications; ++i)
                                pool->cv_tasks_available.notify_one();
                        } else {
                            pool->cv_tasks_available.notify_all();
                        }
                    }
                }

                task->delete_payload();

                if (task->recycle_when_completed)
                    pool->tasks_unused.push_back(task);

                pool->cv_tasks_completed.notify_all();
            }
        }
    }
};

static Pool *pool_default_inst = nullptr;
static std::mutex pool_default_lock;

uint32_t pool_thread_id() {
    return thread_id_tls;
}

Pool *pool_default() {
    lock_guard guard(pool_default_lock);

    if (unlikely(!pool_default_inst))
        pool_default_inst = pool_create();

    return pool_default_inst;
}

Pool *pool_create(size_t size) {
    Pool *pool = new Pool();
    if (size == 0)
        size = std::thread::hardware_concurrency();
    pool_set_size(pool, size);
    return pool;
}

void pool_destroy(Pool *pool) {
    if (pool) {
        pool_set_size(pool, 0);
        size_t incomplete = 0;

        for (size_t i = 0; i < pool->tasks.size(); ++i) {
            Task *task = pool->tasks[i];
            for (Task *child : task->children) {
                if (--child->wait == 0)
                    pool->tasks.push_back(child);
            }
            task->delete_payload();
            delete task;
            ++incomplete;
        }

        for (Task *task : pool->tasks_unused)
            delete task;

        size_t tasks = pool->tasks.size() +  pool->tasks_unused.size();
        if (tasks != pool->tasks_total)
            fprintf(stderr, "pool_destroy(): %zu/%zu tasks were leaked! "
                            "Did you forget to call pool_task_release()?\n",
                            pool->tasks_total - tasks, pool->tasks_total);

        if (incomplete > 0)
            fprintf(stderr, "pool_destroy(): %zu tasks were not completed!\n",
                    incomplete);

        delete pool;
    } else if (pool_default_inst) {
        pool_destroy(pool_default_inst);
        pool_default_inst = nullptr;
    }
}

size_t pool_size(Pool *pool) {
    if (!pool)
        pool = pool_default();

    return pool->workers.size();
}

void pool_set_size(Pool *pool, size_t size) {
    if (!pool)
        pool = pool_default();

    lock_guard guard(pool->mutex);

    ssize_t diff = (ssize_t) size - (ssize_t) pool->workers.size();
    if (diff > 0) {
        for (ssize_t i = 0; i < diff; ++i)
            pool->workers.push_back(
                new Worker(pool, (uint32_t) pool->workers.size() + 1));
    } else if (diff < 0) {
        std::vector<Worker *> workers;

        for (ssize_t i = diff; i != 0; ++i) {
            Worker *worker = pool->workers[pool->workers.size() + i];
            worker->shutdown = true;
            workers.push_back(worker);
        }

        pool->cv_tasks_available.notify_all();

        /* Wait for workers to quit */ {
            unlock_guard guard_2(pool->mutex);
            for (Worker *w : workers)
                w->thread.join();
        }

        for (ssize_t i = diff; i != 0; ++i) {
            delete pool->workers.back();
            pool->workers.pop_back();
        }
    }
}

void pool_task_wait(Task *task) {
    if (unlikely(task == nullptr))
        return;

    Pool *pool = task->pool;
    lock_guard guard(pool->mutex);
    while (task->completed != task->size)
        pool->cv_tasks_completed.wait(guard);
}

void pool_task_release(Task *task) {
    if (unlikely(task == nullptr))
        return;

    Pool *pool = task->pool;
    lock_guard guard(pool->mutex);
    pool_assert(!task->recycle_when_completed);

    if (task->completed == task->size)
        pool->tasks_unused.push_back(task);
    else
        task->recycle_when_completed = true;
}

void pool_task_wait_and_release(Task *task) {
    if (unlikely(task == nullptr))
        return;
    Pool *pool = task->pool;
    lock_guard guard(pool->mutex);

    while (task->completed != task->size)
        pool->cv_tasks_completed.wait(guard);

    pool_assert(!task->recycle_when_completed);
    pool->tasks_unused.push_back(task);
}

Task *pool_task_submit(Pool *pool, Task **pred, size_t pred_count,
                       size_t size, void (*func)(size_t, void *),
                       void *payload, size_t payload_size,
                       void (*payload_deleter)(void *)) {
    pool_assert(size > 0);

    if (!pool)
        pool = pool_default();

    lock_guard guard(pool->mutex);

    Task *task;
    if (!pool->tasks_unused.empty()) {
        task = pool->tasks_unused.back();
        pool->tasks_unused.pop_back();
    } else {
        task = new Task();
        pool->tasks_total++;
    }

    size_t wait = 0;
    for (size_t i = 0; i < pred_count; ++i) {
        Task *parent = pred[i];
        if (parent && parent->completed < parent->size) {
            ++wait;
            parent->children.push_back(task);
        }
    }

    task->wait = wait;
    task->size = size;
    task->started = 0;
    task->completed = 0;
    task->func = func;

    if (payload) {
        if (payload_deleter || payload_size == 0) {
            task->payload = payload;
            task->payload_deleter = payload_deleter;
        } else if (payload_size <= sizeof(Task::payload_storage)) {
            task->payload = task->payload_storage;
            memcpy(task->payload_storage, payload, payload_size);
            task->payload_deleter = nullptr;
        } else {
            /* Payload doesn't fit into temporary storage, and no
               custom deleter was provided. Make a temporary copy. */
            task->payload = malloc(payload_size);
            task->payload_deleter = free;
            pool_assert(task->payload != nullptr);
            memcpy(task->payload, payload, payload_size);
        }
    } else {
        task->payload = nullptr;
        task->payload_deleter = nullptr;
    }

    task->pool = pool;
    task->children.clear();
    task->recycle_when_completed = false;

    if (wait == 0) {
        pool->tasks.push_back(task);

        if (task->size < pool->workers.size()) {
            for (size_t i = 0; i < task->size; ++i)
                pool->cv_tasks_available.notify_one();
        } else {
            pool->cv_tasks_available.notify_all();
        }
    }

    return task;
}
