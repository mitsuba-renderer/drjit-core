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

option(DRJIT_USE_SYSTEM_NANOTHREAD "Use the system nanothread library?" OFF)
option(DRJIT_USE_SYSTEM_LZ4 "Use the system lz4 library?" OFF)
option(DRJIT_USE_SYSTEM_XXHASH "Use the system xxHash library?" OFF)
option(DRJIT_USE_SYSTEM_ROBIN_MAP "Use the system robin-map library?" OFF)

option(DRJIT_DYNAMIC_LLVM "Resolve LLVM dynamically at run time?" ON)
option(DRJIT_ENABLE_TESTS "Build Dr.Jit-Core test suite?" OFF)

if (NOT APPLE)
  option(DRJIT_DYNAMIC_CUDA "Resolve CUDA dynamically at run time?" ON)
  option(DRJIT_ENABLE_OPTIX "Allow the use of OptiX ray tracing calls in kernels?" ON)
  option(DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION "Enable debug and validation flags for OptiX" OFF)
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

message(STATUS "Dr.Jit-Core: using system nanothread: ${DRJIT_USE_SYSTEM_NANOTHREAD}")
message(STATUS "Dr.Jit-Core: using system lz4: ${DRJIT_USE_SYSTEM_LZ4}")
message(STATUS "Dr.Jit-Core: using system xxHash: ${DRJIT_USE_SYSTEM_XXHASH}")
message(STATUS "Dr.Jit-Core: using system robin-map: ${DRJIT_USE_SYSTEM_ROBIN_MAP}")

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

  if (DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION)
    message(STATUS "Dr.Jit-Core: OptiX debug and validation flags enabled.")
  else()
    message(STATUS "Dr.Jit-Core: OptiX debug and validation flags disabled.")
  endif()
endif()

# ----------------------------------------------------------
#  Find or build the dependencies
# ----------------------------------------------------------

if (DRJIT_USE_SYSTEM_NANOTHREAD)
  find_package(nanothread REQUIRED)
else()
  add_subdirectory(ext/nanothread)
  mark_as_advanced(NANOTHREAD_ENABLE_TESTS)
endif()

if (DRJIT_USE_SYSTEM_LZ4)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LZ4 REQUIRED liblz4 IMPORTED_TARGET)
  add_library(lz4 ALIAS PkgConfig::LZ4)
else()
  add_library(lz4 SHARED ext/lz4/lz4.c)
  # We only need to link against the lz4 library, not include its headers.
  # NOTE: It is important that we use a generator here to avoid including a relative path
  # in the INTERFACE_INCLUDE_DIRECTORIES of the lz4 target. This would cause the path to
  # be invalid when the library is used in another project.
  target_include_directories(lz4 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/lz4>)
endif()

if (DRJIT_USE_SYSTEM_XXHASH)
  find_package(xxHash REQUIRED)
  add_library(xxhash ALIAS xxHash::xxhash)
else()
  add_library(xxhash SHARED ext/lz4/xxhash.c)
  # We directly include the headers from the xxHash library, so we add the include directory
  # as a public include directory.
  # See note above for LZ4 about why we must use a generator expression for the path.
  target_include_directories(xxhash PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/lz4>)
endif()

if (DRJIT_USE_SYSTEM_ROBIN_MAP)
  find_package(tsl-robin-map REQUIRED)
else()
  add_subdirectory(ext/robin_map)
endif()

# ----------------------------------------------------------
#  Build Dr.Jit-Core
# ----------------------------------------------------------

if (DRJIT_ENABLE_OPTIX)
  set(DRJIT_OPTIX_FILES
    src/optix.h
    src/optix_api.h
    src/optix_api.cpp
    src/optix_core.cpp)
endif()

add_library(
  drjit-core SHARED
  include/drjit-core/jit.h
  include/drjit-core/traits.h
  include/drjit-core/array.h

  src/common.h
  src/internal.h
  src/alloc.h
  src/hash.h
  src/profiler.h
  src/log.h           src/log.cpp
  src/strbuf.h        src/strbuf.cpp
  src/var.h           src/var.cpp
  src/op.h            src/op.cpp
  src/malloc.h        src/malloc.cpp
  src/registry.h      src/registry.cpp
  src/util.h          src/util.cpp

  # CUDA backend
  src/cuda_api.h
  src/cuda_api.cpp
  src/cuda_core.cpp
  src/cuda_tex.h
  src/cuda_tex.cpp
  src/cuda_eval.cpp

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
  src/llvm_eval.cpp

  src/io.h            src/io.cpp
  src/eval.h          src/eval.cpp
  src/vcall.h         src/vcall.cpp
  src/loop.h          src/loop.cpp
  src/init.cpp
  src/api.cpp

  # Precompiled kernels in compressed PTX format
  resources/kernels.h resources/kernels.c
)

target_compile_features(drjit-core PRIVATE cxx_std_17)

# Link against the dependencies
target_link_libraries(drjit-core PRIVATE nanothread tsl::robin_map lz4 xxhash)

if (MSVC)
  # Conditional expression is constant (a few in robin_hash.h)
  target_compile_options(drjit-core PRIVATE /wd4127)

  # Allow throwing exceptions even from "extern C" code
  target_compile_options(drjit-core PUBLIC /EHs)
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

if (DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_ENABLE_OPTIX_DEBUG_VALIDATION)
endif()

# Generate ITT notifications (e.g. for VTune) if part of a larger project
# that provides an 'ittnotify' target
if (TARGET ittnotify)
  target_link_libraries(drjit-core PRIVATE ittnotify)
  target_include_directories(drjit-core PRIVATE ${ITT_INCLUDE_DIRS})
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_ENABLE_ITTNOTIFY=1)
endif()

# Similar notifications for NVIDIA profilers
if (DRJIT_ENABLE_NVTX)
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_ENABLE_NVTX=1)
endif()

if (UNIX)
  target_link_libraries(drjit-core PRIVATE dl pthread)
endif()

set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE        TRUE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL     TRUE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_DEBUG          FALSE)

if (DRJIT_ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

configure_file(drjit-core-config.cmake "${CMAKE_CURRENT_BINARY_DIR}/drjit-core/drjit-core-config.cmake")
install(
  TARGETS drjit-core
  EXPORT drjit-core-targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the dependencies if they are not system libraries or packaged with CMake.
foreach(_dep nanothread lz4 xxhash)
  string(TOUPPER ${_dep} _dep_upper)
  if (NOT DRJIT_USE_SYSTEM_${_dep_upper})
    install(
      TARGETS ${_dep}
      EXPORT drjit-core-targets
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  endif()
endforeach()

install(DIRECTORY include/drjit-core DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES drjit-core-config.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/drjit-core)
install(
  EXPORT drjit-core-targets
  FILE drjit-core-targets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/drjit-core)
