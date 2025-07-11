# ----------------------------------------------------------
# Top-level Dr.Jit-Core CMake file, requires a recent version
# ----------------------------------------------------------
cmake_minimum_required (VERSION 3.13...3.18)

project(drjit-core
  DESCRIPTION
    "Dr.Jit-Core"
  LANGUAGES
    CXX C
)

# ----------------------------------------------------------
#  Optional features available to users
# ----------------------------------------------------------

option(DRJIT_DYNAMIC_LLVM "Resolve LLVM dynamically at run time?" ON)
option(DRJIT_CORE_ENABLE_TESTS "Build Dr.Jit-Core test suite?" OFF)

if (NOT APPLE)
  option(DRJIT_DYNAMIC_CUDA "Resolve CUDA dynamically at run time?" ON)
  option(DRJIT_ENABLE_OPTIX "Allow the use of OptiX ray tracing calls in kernels?" ON)
endif()

if (NOT MSVC)
  option(DRJIT_SANITIZE_UBSAN   "Enable the address sanitizer (GCC, Clang)" OFF)
  option(DRJIT_SANITIZE_ASAN    "Enable the undefined behavior sanitizer (GCC, Clang)" OFF)
  option(DRJIT_SANITIZE_INTENSE "Add costly internal instrumentation to catch undefined behavior." OFF)
endif()

# ----------------------------------------------------------
#  Check if submodules have been checked out, or fail early
# ----------------------------------------------------------

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/nanothread/ext/cmake-defaults")
  message(FATAL_ERROR "The Dr.Jit-Core dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
endif()

# ----------------------------------------------------------
#  Build defaults for projects by the Realistic Graphics Lab
# ----------------------------------------------------------

include(ext/nanothread/ext/cmake-defaults/CMakeLists.txt)

# ----------------------------------------------------------
#  Print a few messages explaining what will be compiled
# ----------------------------------------------------------

if (DRJIT_DYNAMIC_LLVM)
  message(STATUS "Dr.Jit-Core: LLVM will be loaded dynamically at runtime.")
else()
  message(STATUS "Dr.Jit-Core: linking against LLVM shared libraries.")
endif()

if (NOT APPLE)
  if (DRJIT_DYNAMIC_CUDA)
    message(STATUS "Dr.Jit-Core: CUDA will be loaded dynamically at runtime.")
  else()
    message(STATUS "Dr.Jit-Core: linking against CUDA shared libraries.")
  endif()

  if (DRJIT_ENABLE_OPTIX)
    message(STATUS "Dr.Jit-Core: OptiX support enabled.")
  else()
    message(STATUS "Dr.Jit-Core: OptiX support disabled.")
  endif()
endif()

# ----------------------------------------------------------
#  Build the nanothread library
# ----------------------------------------------------------

add_subdirectory(ext/nanothread)
add_subdirectory(resources)
mark_as_advanced(NANOTHREAD_ENABLE_TESTS NANOTHREAD_STATIC)


# ----------------------------------------------------------
#  Build Dr.Jit-Core
# ----------------------------------------------------------

if (DRJIT_ENABLE_OPTIX)
  set(DRJIT_OPTIX_FILES
    src/optix.h
    src/optix_api.h
    src/optix_api.cpp
    src/optix_coop_vec.h
    src/optix_coop_vec.cpp
    src/optix_core.cpp)
endif()


add_library(
  drjit-core SHARED
  include/drjit-core/jit.h
  include/drjit-core/traits.h
  include/drjit-core/hash.h
  include/drjit-core/array.h
  include/drjit-core/texture.h
  include/drjit-core/half.h

  src/common.h
  src/internal.h
  src/alloc.h
  src/hash.h
  src/log.h           src/log.cpp
  src/strbuf.h        src/strbuf.cpp
  src/var.h           src/var.cpp
  src/array.h         src/array.cpp
  src/op.h            src/op.cpp
  src/malloc.h        src/malloc.cpp
  src/registry.h      src/registry.cpp
  src/util.h          src/util.cpp
  src/record_ts.h     src/record_ts.cpp
  src/coop_vec.h      src/coop_vec.cpp
  src/reorder.h       src/reorder.cpp

  # CUDA backend
  src/cuda_api.h
  src/cuda_api.cpp
  src/cuda_core.cpp
  src/cuda_tex.h
  src/cuda_tex.cpp
  src/cuda_eval.cpp
  src/cuda_ts.h
  src/cuda_ts.cpp
  src/cuda_scatter.h
  src/cuda_scatter.cpp
  src/cuda_packet.h
  src/cuda_packet.cpp
  src/cuda_array.h
  src/cuda_array.cpp

  # OptiX backend
  ${DRJIT_OPTIX_FILES}

  # LLVM backend
  src/llvm_api.h
  src/llvm_api.cpp
  src/llvm_memmgr.h
  src/llvm_memmgr.cpp
  src/llvm_core.cpp
  src/llvm_mcjit.cpp
  src/llvm_orcv2.cpp
  src/llvm_eval.h
  src/llvm_eval.cpp
  src/llvm_red.h
  src/llvm_ts.h
  src/llvm_ts.cpp
  src/llvm_scatter.h
  src/llvm_scatter.cpp
  src/llvm_packet.h
  src/llvm_packet.cpp
  src/llvm_array.h
  src/llvm_array.cpp
  src/llvm_coop_vec.h
  src/llvm_coop_vec.cpp

  src/io.h       src/io.cpp
  src/eval.h     src/eval.cpp
  src/call.h     src/call.cpp
  src/loop.h     src/loop.cpp
  src/cond.h     src/cond.cpp
  src/init.cpp
  src/api.cpp

  # Profiling
  src/nvtx_api.h src/nvtx_api.cpp
  src/profile.h src/profile.cpp

  # LZ4 compression library & XXHash hash function
  ext/lz4/lz4.h ext/lz4/lz4.c
  ext/lz4/xxhash.h ext/lz4/xxh3.h ext/lz4/xxhash.c

  # Precompiled kernels in compressed PTX format
  resources/kernels.h
  ${CMAKE_CURRENT_BINARY_DIR}/resources/kernels.c
)

target_include_directories(drjit-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(drjit-core PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_compile_features(drjit-core PUBLIC cxx_std_17)

target_include_directories(drjit-core PRIVATE
  ext/nanothread/include
  ext/robin_map/include
  ext/lz4
)

if (MSVC)
  # C24127 conditional expression is constant (a few in robin_hash.h)
  # C24324 structure was padded due to alignment specifier
  # C26800 use of a moved from object: 'object'. (false-positive when using structured bindings)
  target_compile_options(drjit-core PRIVATE /wd4127 /wd4324 /wd6800)

  # Allow throwing exceptions even from "extern C" code
  target_compile_options(drjit-core PUBLIC /EHs)

  # MSVC doesn't define this instruction set macro unlike all other compilers
  # so force enable it
  target_compile_definitions(drjit-core PUBLIC -D__F16C__)

  # Workaround to avoid a particular crash due to MS VC++ runtime library binary 
  # incompatibility. In general the runtime version needs to be at least as new 
  # as toolset version but this define provides one such "escape hatch" if that 
  # strictly isn't the case to combat against a commonly encountered reported crash
  target_compile_definitions(drjit-core PUBLIC -D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # GCC static analysis incorrectly warns about free() application to Variable::stmt
  target_compile_options(drjit-core PRIVATE -Wno-free-nonheap-object)
  target_link_options(drjit-core PRIVATE -Wno-free-nonheap-object)
endif()

target_compile_definitions(drjit-core PRIVATE -DLZ4LIB_VISIBILITY=)

target_include_directories(drjit-core
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(drjit-core PRIVATE -DDRJIT_BUILD=1)
target_link_libraries(drjit-core PRIVATE nanothread)

if (DRJIT_DYNAMIC_CUDA OR APPLE)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_DYNAMIC_CUDA=1)
else()
  find_package(CUDA REQUIRED)
  target_include_directories(drjit-core PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(drjit-core PRIVATE cuda)
endif()

if (DRJIT_DYNAMIC_LLVM)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_DYNAMIC_LLVM=1)
else()
  find_package(LLVM REQUIRED)
  target_include_directories(drjit-core PRIVATE ${LLVM_INCLUDE_DIRS})
  target_link_directories(drjit-core PRIVATE ${LLVM_LIB_DIRS})
  llvm_config(drjit-core USE_SHARED core support executionengine orcjit analysis native target mcdisassembler irreader scalaropts mcjit)
endif()

if (DRJIT_ENABLE_OPTIX)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_ENABLE_OPTIX=1)
  if (WIN32)
    target_link_libraries(drjit-core PRIVATE Cfgmgr32)
  endif()
endif()

if (UNIX)
  target_link_libraries(drjit-core PRIVATE dl pthread)
endif()

if (DRJIT_USE_STD_MUTEX)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_USE_STD_MUTEX)
endif()

if (DRJIT_SANITIZE_UBSAN)
  list(APPEND DRJIT_SANITIZE "undefined")
endif()

if (DRJIT_SANITIZE_ASAN)
  list(APPEND DRJIT_SANITIZE "address")
endif()

if (DRJIT_SANITIZE_INTENSE)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_SANITIZE_INTENSE)
endif()

if (DRJIT_SANITIZE)
  string (REPLACE ";" "," DRJIT_SANITIZE "${DRJIT_SANITIZE}")
  target_compile_options(drjit-core PUBLIC -fsanitize=${DRJIT_SANITIZE})
  target_link_options (drjit-core PUBLIC -fsanitize=${DRJIT_SANITIZE})
endif()

set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE        TRUE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL     TRUE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_DEBUG          FALSE)

if (DRJIT_CORE_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
