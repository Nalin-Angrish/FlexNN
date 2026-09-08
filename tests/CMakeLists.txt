# FlexNN tests — GoogleTest harness
# Each PR adds its own test binary here. This file is intentionally minimal
# in PR-01 so the stack base builds with 0 tests.

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  # FetchContent already included at top-level for Eigen, but guard for standalone
  if(NOT COMMAND FetchContent_Declare)
    include(FetchContent)
  endif()
  # Prevent googletest's own tests from polluting ctest (BUILD_TESTING ON at top-level)
  set(_flexnn_save_BUILD_TESTING2 ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # For Windows: prevent overriding parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  set(BUILD_TESTING ${_flexnn_save_BUILD_TESTING2} CACHE BOOL "" FORCE)
  unset(_flexnn_save_BUILD_TESTING2)
  # After FetchContent, GTest::gtest etc. are available
endif()

include(GoogleTest)

# PR-02: activation tests
add_executable(activation_test activation_test.cpp)
target_link_libraries(activation_test PRIVATE FlexNN GTest::gtest_main)
gtest_discover_tests(activation_test)

# PR-03: dense tests
add_executable(dense_test dense_test.cpp)
target_link_libraries(dense_test PRIVATE FlexNN GTest::gtest_main)
gtest_discover_tests(dense_test)

# PR-04: layer variant wrapper
add_executable(layer_variant_test layer_variant_test.cpp)
target_link_libraries(layer_variant_test PRIVATE FlexNN GTest::gtest_main)
gtest_discover_tests(layer_variant_test)

# PR-05: Conv1D
add_executable(conv1d_test conv1d_test.cpp)
target_link_libraries(conv1d_test PRIVATE FlexNN GTest::gtest_main)
gtest_discover_tests(conv1d_test)

# PR-06: ModelIO
add_executable(export_import_test export_import_test.cpp)
target_link_libraries(export_import_test PRIVATE FlexNN GTest::gtest_main)
gtest_discover_tests(export_import_test)
