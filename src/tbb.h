/*
    src/tbb.h -- Parallelization via LLVM (optional)

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(ENOKI_JIT_ENABLE_TBB)

#include <stdlib.h>
#include <stdint.h>

struct Stream;
using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, void **ptr);

/// Initialize TBB's task scheduler for a given Enoki stream
extern void tbb_stream_init(Stream *stream);

/// Shut down TBB's task scheduler for a given Enoki stream
extern void tbb_stream_shutdown(Stream *stream);

/// Wait for all TBB tasks to finish
extern void tbb_stream_sync(Stream *stream);

/// Append a kernel execution, but do not submit it to the queue yet
extern void tbb_stream_enqueue_kernel(Stream *stream, LLVMKernelFunction kernel,
                                      uint32_t start, uint32_t stop,
                                      uint32_t argc, void **argv,
                                      bool parallel_dispatch);

/// Submit a set of kernel tassks to the TBB task scheduler
extern void tbb_stream_submit_kernel(Stream *stream);

/// Enqueue a function for asynchronous execution
extern void tbb_stream_enqueue_func(Stream *stream, void (*func)(void *),
                                    void *extra, size_t extra_size);

#endif
