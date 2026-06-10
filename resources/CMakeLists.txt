# ----------------------------------------------------------
#  Generate a C file containing a compressed kernel source
#  code prepared by the 'pack' utility
# ----------------------------------------------------------

set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/kernels.c)
set(GENERATE FALSE)
set(KERNEL_FILES kernels_dict kernels_75)

foreach(NAME ${KERNEL_FILES})
  if ("${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.lz4" IS_NEWER_THAN "${OUT_FILE}")
    set(GENERATE TRUE)
  endif()
endforeach()

if (GENERATE)
  # Create header for C file
  file(WRITE ${OUT_FILE} "#include \"resources/kernels.h\"\n\n")
  foreach(NAME ${KERNEL_FILES})
    message(STATUS "Processing \"resources/${NAME}.lz4\" ..")
    set(FILE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.lz4")
    # Read hex data from file
    file(READ ${FILE_NAME} FILE_DATA HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," FILE_DATA ${FILE_DATA})
    file(APPEND ${OUT_FILE} "const char ${NAME}[] = { ${FILE_DATA} };")
    # Rebuild when this file changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${FILE_NAME})
  endforeach()
endif()

if (DRJIT_ENABLE_METAL)
  set(METAL_OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.c)
  set(METAL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/metal_kernels.metallib)

  if ("${METAL_LIB}" IS_NEWER_THAN "${METAL_OUT_FILE}")
    message(STATUS "Processing \"resources/metal_kernels.metallib\" ..")
    file(READ "${METAL_LIB}" METAL_HEX HEX)
    string(LENGTH "${METAL_HEX}" METAL_HEX_LEN)
    math(EXPR METAL_LEN "${METAL_HEX_LEN} / 2")
    string(REGEX REPLACE "(..)" "0x\\1," METAL_BYTES "${METAL_HEX}")
    string(REGEX REPLACE "((0x..,){16})" "\\1\n    " METAL_BYTES "${METAL_BYTES}")
    file(WRITE ${METAL_OUT_FILE}
"// Auto-generated from resources/metal_kernels.metallib. The declarations live
// in src/metal.h.

#include <stddef.h>

const unsigned char metal_kernels_metallib[] = {
    ${METAL_BYTES}
};

const size_t metal_kernels_metallib_size = ${METAL_LEN};
")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${METAL_LIB})
  endif()
endif()
