project(enoki-jit)
cmake_minimum_required(VERSION 3.10.0)

option(ENOKI_JIT_ENABLE_TBB   "Use Intel Thread Building Blocks (TBB) with LLVM backend?" ON)
option(ENOKI_JIT_DYNAMIC_CUDA "Resolve CUDA dynamically at run time?" ON)
option(ENOKI_JIT_DYNAMIC_LLVM "Resolve LLVM dynamically at run time?" ON)
option(ENOKI_JIT_ENABLE_TESTS "Build Enoki test suite?" OFF)
set(ENOKI_NATIVE_FLAGS "-march=native" CACHE STRING "Compilation flags used to target the host processor architecture.")

# Set a default build configuration (Release)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if(CMAKE_SIZEOF_VOID_P STREQUAL "4")
  message(FATAL_ERROR "Enoki-JIT requires a 64 bit architecture!")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS NO)

include(CheckCXXSourceRuns)

macro(CHECK_CXX_COMPILER_AND_LINKER_FLAGS _RESULT _CXX_FLAGS _LINKER_FLAGS)
  set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_runs("#include <iostream>\nint main(int argc, char **argv) { std::cout << \"test\"; return 0; }" ${_RESULT})
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
endmacro()

# Prefer libc++ in conjunction with Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
  CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
  if (HAS_LIBCPP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
    message(STATUS "Enoki-JIT: using libc++.")
  else()
    CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP_AND_CPPABI "-stdlib=libc++" "-stdlib=libc++ -lc++abi")
    if (HAS_LIBCPP_AND_CPPABI)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      message(STATUS "Enoki-JIT: using libc++ and libc++abi.")
    else()
      message(FATAL_ERROR "When Clang is used to compile Enoki-JIT, libc++ must be available -- GCC's libstdc++ is not supported! (please insteall the libc++ development headers, provided e.g. by the packages 'libc++-dev' and 'libc++abi-dev' on Debian/Ubuntu).")
    endif()
  endif()
endif()

if (ENOKI_JIT_ENABLE_TBB)
  if(TARGET tbb)
    set(TBB_TARGET_FOUND TRUE)
    set(TBB_FOUND TRUE)
  else()
    # Try to detect TBB
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    find_package(TBB)
  endif()
endif()

if (ENOKI_JIT_ENABLE_TBB)
  if (TBB_FOUND)
    message(STATUS "Enoki-JIT: enabling TBB.")
  else()
    message(STATUS "Enoki-JIT: Intel Thread Building Blocks (TBB) could not be found! Computation involving LLVM arrays will run single-threaded.")
  endif()
else()
  message(STATUS "Enoki-JIT: *not* using Intel Thread Building Blocks (TBB).")
endif()

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

if (MSVC)
   add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -DNOMINMAX)
endif()

include_directories(
  include
  ext/robin_map/include
  ext/lz4
)

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
  src/tbb.h            src/tbb.cpp
  src/init.cpp
  src/api.cpp

  # LZ4 compression library & XXHash hash function
  ext/lz4/lz4.h ext/lz4/lz4.c
  ext/lz4/xxhash.h ext/lz4/xxh3.h ext/lz4/xxhash.c

  # Precompiled kernels in compressed PTX format
  kernels/kernels.h kernels/kernels.c
)

target_compile_definitions(enoki-jit PRIVATE -DLZ4LIB_VISIBILITY=)

if (NOT MSVC)
  target_compile_options(enoki-jit PRIVATE -Wall -Wextra ${ENOKI_NATIVE_FLAGS})
  if (U_CMAKE_BUILD_TYPE MATCHES RELEASE OR U_CMAKE_BUILD_TYPE MATCHES MINSIZEREL)
    target_compile_options(enoki-jit PRIVATE -fvisibility=hidden)
  endif()
endif()

target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_BUILD=1)

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

if (ENOKI_JIT_ENABLE_TBB AND TBB_FOUND)
  if (TBB_TARGET_FOUND)
    target_link_libraries(enoki-jit PRIVATE tbb)
  else()
    target_link_libraries(enoki-jit PRIVATE TBB::tbb)
  endif()
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_JIT_ENABLE_TBB=1)
endif()

if (TARGET ittnotify)
  # Generate ITT notifications (e.g. for VTune) if part of a larger project
  # that provides an 'ittnotify' target
  target_link_libraries(enoki-jit PRIVATE ittnotify)
  target_include_directories(enoki-jit PRIVATE ${ITT_INCLUDE_DIRS})
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_ENABLE_ITTNOTIFY=1)
endif()

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
