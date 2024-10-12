# ----------------------------------------------------------
#  Generate a C file containing a compressed kernel source
#  code prepared by the 'pack' utility
# ----------------------------------------------------------

set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/kernels.c)
set(GENERATE FALSE)
set(KERNEL_FILES kernels_dict kernels_50 kernels_70)

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
