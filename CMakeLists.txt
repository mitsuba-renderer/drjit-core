# ----------------------------------------------------------
# Top-level Enoki-JIT CMake file, requires a recent version
# ----------------------------------------------------------
cmake_minimum_required (VERSION 3.13...3.18)

project(enoki-jit
  DESCRIPTION
    "Enoki-JIT"
  LANGUAGES
    CXX C
)

# ----------------------------------------------------------
#  Optional features available to users
# ----------------------------------------------------------

option(ENOKI_JIT_DYNAMIC_LLVM "Resolve LLVM dynamically at run time?" ON)
option(ENOKI_JIT_ENABLE_TESTS "Build Enoki test suite?" OFF)

if (NOT APPLE)
  option(ENOKI_JIT_DYNAMIC_CUDA "Resolve CUDA dynamically at run time?" ON)
  option(ENOKI_JIT_ENABLE_OPTIX "Allow the use of OptiX ray tracing calls in kernels?" ON)
endif()

# ----------------------------------------------------------
#  Check if submodules have been checked out, or fail early
# ----------------------------------------------------------

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/enoki-thread/ext/cmake-defaults")
  message(FATAL_ERROR "The Enoki-JIT dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
endif()

# ----------------------------------------------------------
#  Build defaults for projects by the Realistic Graphics Lab
# ----------------------------------------------------------

include(ext/enoki-thread/ext/cmake-defaults/CMakeLists.txt)

# ----------------------------------------------------------
#  Print a few messages explaining what will be compiled
# ----------------------------------------------------------

if (ENOKI_JIT_DYNAMIC_LLVM)
  message(STATUS "Enoki-JIT: LLVM will be loaded dynamically at runtime.")
else()
  message(STATUS "Enoki-JIT: linking against LLVM shared libraries.")
endif()

if (NOT APPLE)
  if (ENOKI_JIT_DYNAMIC_CUDA)
    message(STATUS "Enoki-JIT: CUDA will be loaded dynamically at runtime.")
  else()
    message(STATUS "Enoki-JIT: linking against CUDA shared libraries.")
  endif()
  if (ENOKI_JIT_ENABLE_OPTIX)
    message(STATUS "Enoki-JIT: OptiX support enabled.")
  else()
    message(STATUS "Enoki-JIT: OptiX support disabled.")
  endif()
endif()

# ----------------------------------------------------------
#  Build the Enoki-Thread library
# ----------------------------------------------------------

add_subdirectory(ext/enoki-thread)
mark_as_advanced(ENOKI_THREAD_ENABLE_TESTS)

# ----------------------------------------------------------
#  Build Enoki-JIT
# ----------------------------------------------------------

if (ENOKI_JIT_ENABLE_OPTIX)
  set(ENOKI_JIT_OPTIX_FILES src/optix_api.h src/optix_api.cpp)
endif()

add_library(
  enoki-jit SHARED
  include/enoki-jit/jit.h
  include/enoki-jit/traits.h
  include/enoki-jit/array.h

  src/common.h
  src/internal.h
  src/alloc.h
  src/hash.h
  src/profiler.h
  src/log.h           src/log.cpp
  src/var.h           src/var.cpp
  src/op.h            src/op.cpp
  src/malloc.h        src/malloc.cpp
  src/registry.h      src/registry.cpp
  src/util.h          src/util.cpp
  src/cuda_api.h      src/cuda_api.cpp
  src/cuda_tex.h      src/cuda_tex.cpp
  src/llvm_api.h      src/llvm_api.cpp
  src/io.h            src/io.cpp
  src/eval.h          src/eval.cpp
  src/vcall.h         src/vcall.cpp
  src/loop.h          src/loop.cpp
  src/printf.h        src/printf.cpp
  src/eval_cuda.cpp
  src/eval_llvm.cpp
  src/init.cpp
  src/api.cpp

  ${ENOKI_JIT_OPTIX_FILES}

  # LZ4 compression library & XXHash hash function
  ext/lz4/lz4.h ext/lz4/lz4.c
  ext/lz4/xxhash.h ext/lz4/xxh3.h ext/lz4/xxhash.c

  # Precompiled kernels in compressed PTX format
  kernels/kernels.h kernels/kernels.c
)

target_compile_features(enoki-jit PRIVATE cxx_std_14)
target_include_directories(enoki-jit PRIVATE
  ext/enoki-thread/include
  ext/robin_map/include
  ext/lz4
)

if (MSVC)
  # Conditional expression is constant (a few in robin_hash.h)
  target_compile_options(enoki-jit PRIVATE /wd4127)

  # Allow throwing exceptions even from "extern C" code
  target_compile_options(enoki-jit PUBLIC /EHs)
endif()

target_compile_definitions(enoki-jit PRIVATE -DLZ4LIB_VISIBILITY=)

target_include_directories(enoki-jit
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_BUILD=1)
target_link_libraries(enoki-jit PRIVATE enoki-thread)

if (ENOKI_JIT_DYNAMIC_CUDA OR APPLE)
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

if (ENOKI_JIT_ENABLE_OPTIX)
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_ENABLE_OPTIX=1)
  if (WIN32)
    target_link_libraries(enoki-jit PRIVATE Cfgmgr32)
  endif()
endif()

# Generate ITT notifications (e.g. for VTune) if part of a larger project
# that provides an 'ittnotify' target
if (TARGET ittnotify)
  target_link_libraries(enoki-jit PRIVATE ittnotify)
  target_include_directories(enoki-jit PRIVATE ${ITT_INCLUDE_DIRS})
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_ENABLE_ITTNOTIFY=1)
endif()

# Similar notifications for NVIDIA profilers
if (ENOKI_JIT_ENABLE_NVTX)
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_ENABLE_NVTX=1)
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
