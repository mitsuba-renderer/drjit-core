enable_testing()

set(TEST_FILES basics.cpp mem.cpp graphviz.cpp vcall.cpp loop.cpp reductions.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_SOURCE_DIR}>)

foreach (TEST_FILE ${TEST_FILES})
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp test.h test.cpp ${TEST_NAME}.cpp)
  target_link_libraries(test_${TEST_NAME} PRIVATE enoki-jit)
  target_compile_definitions(test_${TEST_NAME} PRIVATE -DTEST_NAME="${TEST_NAME}")
  target_compile_features(test_${TEST_NAME} PRIVATE cxx_std_11)

  if (ENOKI_JIT_ENABLE_OPTIX)
    target_compile_definitions(test_${TEST_NAME} PRIVATE -DENOKI_JIT_ENABLE_OPTIX=1)
  endif()
  add_test(
    NAME ${TEST_NAME}
    COMMAND test_${TEST_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endforeach()

add_custom_target(test COMMAND CTEST_OUTPUT_ON_FAILURE=1 ${CMAKE_CTEST_COMMAND})
set_property(TARGET test_vcall PROPERTY CXX_STANDARD 17)
set_property(TARGET test_loop PROPERTY CXX_STANDARD 17)

if (ENOKI_JIT_ENABLE_OPTIX)
 # target_sources(test_vcall PRIVATE optix_stubs.h optix_stubs.cpp)
 add_executable(triangle triangle.cpp optix_stubs.h optix_stubs.cpp)
 set_property(TARGET triangle PROPERTY CXX_STANDARD 17)
 # target_compile_definitions(test_vcall PRIVATE -DENOKI_JIT_ENABLE_OPTIX=1)
 target_link_libraries(triangle PRIVATE enoki-jit)
endif()
