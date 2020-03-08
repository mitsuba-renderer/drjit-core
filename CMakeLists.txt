project("jit")
cmake_minimum_required(VERSION 3.17)

# Set a default build configuration (Release)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

option(ENOKI_ENABLE_CUDA "Enable CUDA" ON)

find_package(CUDAToolkit REQUIRED)

include_directories(robin-map/include)

add_executable(
  jit
  log.h    log.cpp
  api.h    api.cpp
  ssa.h    ssa.cpp
  eval.h   eval.cpp
  malloc.h malloc.cpp
  hash.h   hash.cpp
  jit.h    jit.cpp
  main.cpp
)

target_compile_features(jit PUBLIC cxx_std_17)
target_link_libraries(jit PRIVATE CUDA::cuda_driver CUDA::cudart pthread)

if (ENOKI_ENABLE_CUDA)
  target_compile_options(jit PRIVATE -DENOKI_CUDA=1 -Wall -Wextra -march=native)
endif()

