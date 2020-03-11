project("jit")
cmake_minimum_required(VERSION 3.9.0)

option(ENOKI_ENABLE_TESTS "Compile test suite?" ON)

# Set a default build configuration (Release)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

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

if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|Intel)$")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif()

include_directories(
  include
  ext/robin_map/include
  ext/cub/cub
)

add_library(
  enoki-jit SHARED
  include/enoki/jit.h
  src/internal.h       
  src/log.h            src/log.cpp
  src/var.h            src/var.cpp
  src/eval.h           src/eval.cpp
  src/malloc.h         src/malloc.cpp
  src/hash.h           src/hash.cpp
  src/cuda.h           src/cuda.cpp
  src/cuda_api.h       src/cuda_api.cpp
  src/init.cpp
  src/api.cpp
)

set_property(TARGET enoki-jit PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_compile_options(enoki-jit PRIVATE -Wall -Wextra -march=native -fvisibility=hidden)
target_link_libraries(enoki-jit PRIVATE dl)

if (ENOKI_ENABLE_CUDA)
  # add_library(
  #   enoki-jit-cuda OBJECT
  #   src/kernels.cu
  # )
  # target_link_libraries(enoki-jit PRIVATE CUDA::cuda_driver CUDA::cudart enoki-jit-cuda)
  # target_compile_definitions(enoki-jit PRIVATE -DENOKI_CUDA=1)
  # target_compile_definitions(enoki-jit-cuda PRIVATE -DENOKI_CUDA=1)
#   enable_language(CUDA)
#   find_package(CUDAToolkit REQUIRED)
# set(ENOKI_CUDA_COMPUTE_CAPABILITY "61" CACHE
#     STRING "Compute capability as specified by https://developer.nvidia.com/cuda-gpus")

#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ENOKI_CUDA_COMPUTE_CAPABILITY},code=compute_${ENOKI_CUDA_COMPUTE_CAPABILITY} -cudart shared")
  # set_property(TARGET enoki-jit-cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if (ENOKI_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
