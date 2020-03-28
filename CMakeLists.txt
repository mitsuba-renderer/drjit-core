project("jit")
cmake_minimum_required(VERSION 3.9.0)

option(ENOKI_ENABLE_TESTS "Compile test suite?" ON)
option(ENOKI_DYNAMIC_CUDA "Resolve CUDA dynamically at run time?" ON)
option(ENOKI_DYNAMIC_LLVM "Resolve LLVM dynamically at run time?" ON)

# Set a default build configuration (Release)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

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

include_directories(
  include
  ext/robin_map/include
  ext/lz4
)

add_library(
  enoki-jit SHARED
  include/enoki/jit.h
  src/internal.h
  src/alloc.h
  src/hash.h
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
  ext/lz4/xxhash.h ext/lz4/xxhash.c

  # Precompiled kernels in compressed PTX format
  ptx/kernels.h ptx/kernels.c
)

# target_compile_options(enoki-jit PRIVATE -Wall -Wextra -march=native)
target_compile_options(enoki-jit PRIVATE -Wall -Wextra)
if (U_CMAKE_BUILD_TYPE MATCHES REL)
  target_compile_options(enoki-jit PRIVATE -fvisibility=hidden)
endif()

if (ENOKI_DYNAMIC_CUDA)
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_DYNAMIC_CUDA=1)
else()
  find_package(CUDA REQUIRED)
  target_include_directories(enoki-jit PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(enoki-jit PRIVATE cuda)
endif()

if (ENOKI_DYNAMIC_LLVM)
  target_compile_definitions(enoki-jit PRIVATE -DENOKI_DYNAMIC_LLVM=1)
else()
  find_package(LLVM REQUIRED)
  target_include_directories(enoki-jit PRIVATE ${LLVM_INCLUDE_DIRS})
  target_link_libraries(enoki-jit PRIVATE LLVM)
endif()

target_link_libraries(enoki-jit PRIVATE dl pthread)
set_target_properties(enoki-jit PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE        TRUE)
set_target_properties(enoki-jit PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL     TRUE)
set_target_properties(enoki-jit PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)

if (ENOKI_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
