cmake_minimum_required(VERSION 3.13...3.18)

project(enoki-jit
  DESCRIPTION
    "Enoki JIT compiler"
  LANGUAGES
    CXX C
)

include(ext/enoki-thread/cmake/EnokiDefaults.cmake)

add_subdirectory(ext/enoki-thread)

option(ENOKI_JIT_DYNAMIC_CUDA "Resolve CUDA dynamically at run time?" ON)
option(ENOKI_JIT_DYNAMIC_LLVM "Resolve LLVM dynamically at run time?" ON)
option(ENOKI_JIT_ENABLE_TESTS "Build Enoki test suite?" OFF)
set(ENOKI_NATIVE_FLAGS "-march=native" CACHE STRING "Compilation flags used to target the host processor architecture.")

if (ENOKI_JIT_DYNAMIC_CUDA)
  message(STATUS "Enoki-JIT: CUDA will be loaded dynamically at runtime.")
else()
  message(STATUS "Enoki-JIT: linking against CUDA shared libraries.")
endif()

if (ENOKI_JIT_DYNAMIC_LLVM)
  message(STATUS "Enoki-JIT: LLVM will be loaded dynamically at runtime.")
else()
  message(STATUS "Enoki-JIT: linking against LLVM shared libraries.")
endif()

add_library(
  enoki-jit SHARED
  include/enoki-jit/jit.h
  include/enoki-jit/traits.h
  include/enoki-jit/cuda.h
  include/enoki-jit/llvm.h

  src/internal.h
  src/alloc.h
  src/hash.h
  src/profiler.h
  src/log.h            src/log.cpp
  src/var.h            src/var.cpp
  src/eval.h           src/eval.cpp
  src/malloc.h         src/malloc.cpp
  src/registry.h       src/registry.cpp
  src/util.h           src/util.cpp
  src/cuda_api.h       src/cuda_api.cpp
  src/llvm_api.h       src/llvm_api.cpp
  src/io.h             src/io.cpp
  src/init.cpp
  src/api.cpp

  # LZ4 compression library & XXHash hash function
  ext/lz4/lz4.h ext/lz4/lz4.c
  ext/lz4/xxhash.h ext/lz4/xxh3.h ext/lz4/xxhash.c

  # Precompiled kernels in compressed PTX format
  kernels/kernels.h kernels/kernels.c
)

target_compile_features(enoki-jit PRIVATE cxx_std_11)
target_include_directories(enoki-jit PRIVATE
  include
  ext/enoki-thread/include
  ext/robin_map/include
  ext/lz4
)

target_compile_definitions(enoki-jit PRIVATE -DLZ4LIB_VISIBILITY=)

target_include_directories(enoki-jit
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


if (NOT MSVC)
  target_compile_options(enoki-jit PRIVATE -Wall -Wextra ${ENOKI_NATIVE_FLAGS})
  if (U_CMAKE_BUILD_TYPE MATCHES RELEASE OR U_CMAKE_BUILD_TYPE MATCHES MINSIZEREL)
    target_compile_options(enoki-jit PRIVATE -fvisibility=hidden)
  endif()
endif()

target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_BUILD=1)
target_link_libraries(enoki-jit PRIVATE enoki-thread)

if (ENOKI_JIT_DYNAMIC_CUDA)
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_DYNAMIC_CUDA=1)
else()
  find_package(CUDA REQUIRED)
  target_include_directories(enoki-jit PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(enoki-jit PRIVATE cuda)
endif()

if (ENOKI_JIT_DYNAMIC_LLVM)
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_DYNAMIC_LLVM=1)
else()
  find_package(LLVM REQUIRED)
  target_include_directories(enoki-jit PRIVATE ${LLVM_INCLUDE_DIRS})
  target_link_libraries(enoki-jit PRIVATE LLVM)
endif()

# Generate ITT notifications (e.g. for VTune) if part of a larger project
# that provides an 'ittnotify' target
if (TARGET ittnotify)
  target_link_libraries(enoki-jit PRIVATE ittnotify)
  target_include_directories(enoki-jit PRIVATE ${ITT_INCLUDE_DIRS})
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_ENABLE_ITTNOTIFY=1)
endif()

# Similar notifications for NVIDIA profilers
if (ENOKI_ENABLE_NVTX)
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_ENABLE_NVTX=1)
endif()

if (UNIX)
  target_link_libraries(enoki-jit PRIVATE dl pthread)
endif()

set_target_properties(enoki-jit PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE        TRUE)
set_target_properties(enoki-jit PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL     TRUE)
set_target_properties(enoki-jit PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)

if (ENOKI_JIT_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
