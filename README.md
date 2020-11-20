<p align="center"><img src="https://github.com/mitsuba-renderer/enoki/raw/master/docs/enoki-logo.png" alt="Enoki logo" width="300"/></p>

# Enoki-JIT — CUDA & LLVM just-in-time compiler

| Continuous Integration |
|         :---:          |
|   [![rgl-ci][1]][2]    |

[1]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:EnokiJit_Build)/statusIcon.svg
[2]: https://rgl-ci.epfl.ch/buildConfiguration/EnokiJit_Build?guest=1


## Introduction

This project implements a lazy tracing just-in-time (JIT) compiler targeting
GPUs (via CUDA 10+ and [NVIDIA
PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) and
CPUs (via [LLVM 7+ IR](https://llvm.org/docs/LangRef.html)). *Lazy* refers its
behavior of capturing operations performed in C or C++, while attempting to
postpone the associated computation for as long as possible. Eventually, this
is no longer possible, at which point the system generates an efficient kernel
containing queued computation that is either evaluated on the CPU or GPU.

Enoki-JIT can be used just by itself, or as a component of the larger
[Enoki](https://github.com/mitsuba-renderer/enoki) library, which additionally
provides things like multidimensional arrays, automatic differentiation, and a
large library of mathematical functions.

This project has almost no dependencies: it can be compiled without CUDA or
LLVM actually being present on the system (it will attempt to find them at
runtime). The library is implemented in C++11 but exposes all functionality
through a C99-compatible interface.

## An example

Two header files
[enoki-jit/cuda.h](https://github.com/mitsuba-renderer/enoki-jit/blob/master/include/enoki-jit/cuda.h)
and
[enoki-jit/llvm.h](https://github.com/mitsuba-renderer/enoki-jit/blob/master/include/enoki-jit/llvm.h)
provide convenient C++ wrappers with operator operator overloading building on
the C-level API
([enoki-jit/jit.h](https://github.com/mitsuba-renderer/enoki-jit/blob/master/include/enoki-jit/jit.h)).
Here is an brief example on how these can be used:

```cpp
#include <enoki/cuda.h>

using Bool   = CUDAArray<bool>;
using Float  = CUDAArray<float>;
using UInt32 = CUDAArray<uint32_t>;

// [0, 0.01, 0.02, ..., 1]
Float x = linspace<Float>(0, 1, 101);

// [0, 2, 4, 8, .., 98]
UInt32 index = arange<UInt32>(50) * 2;

// Scatter/gather operations are available
Float y = gather(x, index);

/// Comparisons produce mask arrays
Bool mask = x < .5f;

// Ternary operator
Float z = select(mask, sqrt(x), 1.f / x);

printf("Value is = %s\n", z.str());
```

Running this program will trigger two kernel launches. The first generates the
``x`` array (size 100) when it is accessed by the ``gather()`` operation, and
the second generates ``z`` (size 50) when it is printed in the last line. Both
correspond to points during the execution where evaluation could no longer be
postponed.

Simply changing the first lines to

```cpp
#include <enoki/llvm.h>

using Bool   = LLVMArray<bool>;
using Float  = LLVMArray<float>;
using UInt32 = LLVMArray<uint32_t>;
```

switches to the functionally equivalent LLVM backend. By default, the LLVM
backend parallelizes execution via a built-in thread pool, enabling usage that
is very similar to the CUDA variant: a single thread issues computation that is
then processed in parallel by all cores of the system.

## Features

- Cross-platform: runs on Linux, macOS, and Windows.

- Kernels are cached and reused when the same computation is encountered again.
  Caching is done both in memory and on disk (``~/.enoki`` on Linux and macOS,
  ``~/AppData/Local/Temp/enoki`` on Windows).

- The internals of the JIT compiler heavily rely on hash table lookups (to keep
  track of variables) and string concatenation (to merge IR fragments into full
  kernels), and both of these steps are highly optimized. This means that the
  overhead of generating kernel IR code is minimal (only a few μs), with most
  time being spent either executing kernels or compiling from IR to machine
  code when a kernel is encountered for the first time.

- Supports parallel kernel execution on multiple devices (JITing from several
  CPU threads, or running kernels on multiple GPUs).

- The LLVM backend automatically targets the vector instruction sets supported
  by the host machine (e.g. AVX/AVX2, or AVX512 if available).

- The library provides an *asynchronous* memory allocator, which allocates and
  releases memory in the execution stream of a device that runs asynchronously
  with respect to the host CPU. Kernels frequently request and release large
  memory buffers, which both tend to be very costly operations. For this
  reason, memory allocations are also cached and reused.

- Provides a variety of parallel reductions for convenience. 
