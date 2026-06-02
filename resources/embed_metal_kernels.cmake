# Precompile the Metal Shading Language source for the utility kernels

if (NOT INPUT)
  message(FATAL_ERROR "embed_metal_kernels.cmake: INPUT is not set")
endif()
if (NOT OUTPUT)
  message(FATAL_ERROR "embed_metal_kernels.cmake: OUTPUT is not set")
endif()

get_filename_component(OUT_DIR "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${OUT_DIR}")

set(AIR "${OUTPUT}.air")
set(LIB "${OUTPUT}.metallib")

execute_process(
  COMMAND xcrun -sdk macosx metal -std=metal3.0 -fno-fast-math
          -c "${INPUT}" -o "${AIR}"
  RESULT_VARIABLE METAL_RESULT
  ERROR_VARIABLE  METAL_ERROR)
if (NOT METAL_RESULT EQUAL 0)
  message(FATAL_ERROR
    "embed_metal_kernels.cmake: 'metal' compilation of ${INPUT} failed:\n${METAL_ERROR}")
endif()

# AIR -> metallib.
execute_process(
  COMMAND xcrun -sdk macosx metallib "${AIR}" -o "${LIB}"
  RESULT_VARIABLE METALLIB_RESULT
  ERROR_VARIABLE  METALLIB_ERROR)
if (NOT METALLIB_RESULT EQUAL 0)
  message(FATAL_ERROR
    "embed_metal_kernels.cmake: 'metallib' linking failed:\n${METALLIB_ERROR}")
endif()

# Read the archive back as a hex string and emit it as a byte array.
file(READ "${LIB}" LIB_HEX HEX)
string(LENGTH "${LIB_HEX}" LIB_HEX_LEN)
math(EXPR LIB_LEN "${LIB_HEX_LEN} / 2")

# Turn " deadbeef" into "0xde,0xad,0xbe,0xef," (16 bytes per line).
string(REGEX REPLACE "(..)" "0x\\1," LIB_BYTES "${LIB_HEX}")
string(REGEX REPLACE "((0x..,){16})" "\\1\n    " LIB_BYTES "${LIB_BYTES}")

set(CONTENT
"// Auto-generated from ${INPUT} by embed_metal_kernels.cmake.
// Do not edit — modify the .metal file and rebuild. The declarations live in
// src/metal.h.

#include <stddef.h>

const unsigned char metal_kernels_metallib[] = {
    ${LIB_BYTES}
};

const size_t metal_kernels_metallib_size = ${LIB_LEN};
")

file(WRITE "${OUTPUT}" "${CONTENT}")

# Clean up intermediates.
file(REMOVE "${AIR}" "${LIB}")
