# Wrap the contents of ${INPUT} (the Metal Shading Language source for the
# precompiled utility kernels) in a C++ raw string literal and write the
# result to ${OUTPUT}. Used at build time to embed the .metal source so the
# runtime no longer needs to read it from disk.
#
# A raw string delimiter is chosen that does not appear in the input so the
# wrapping is unambiguous. We try `MSL`, then `MSL_`, etc.

if (NOT INPUT)
  message(FATAL_ERROR "embed_metal_kernels.cmake: INPUT is not set")
endif()
if (NOT OUTPUT)
  message(FATAL_ERROR "embed_metal_kernels.cmake: OUTPUT is not set")
endif()

file(READ "${INPUT}" SRC)

set(DELIM "MSL")
set(SUFFIX "")
while (SRC MATCHES "\\)${DELIM}\"")
  string(APPEND SUFFIX "_")
  set(DELIM "MSL${SUFFIX}")
endwhile()

set(CONTENT
"// Auto-generated from ${INPUT} by embed_metal_kernels.cmake.
// Do not edit — modify the .metal file and rebuild.

#pragma once

#if defined(DRJIT_ENABLE_METAL)

namespace drjit {

inline constexpr const char *metal_kernels_src = R\"${DELIM}(
${SRC})${DELIM}\";

} // namespace drjit

#endif
")

file(WRITE "${OUTPUT}" "${CONTENT}")
