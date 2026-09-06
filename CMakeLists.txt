cmake_minimum_required(VERSION 3.14)
project(FlexNN CXX)

# -----------------------------------------------------------------------------
# Language standard — C++17 (required for std::variant, string_view, etc.)
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# Eigen3 — try system first, fallback to FetchContent for CI/hosts without apt
include(FetchContent) # needed for both Eigen and googletest
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found locally, fetching via FetchContent (3.4.0)")
  # Save BUILD_TESTING so Eigen's OFF doesn't clobber user's ON (e.g., -DFLEXNN_BUILD_TESTS=ON)
  set(_flexnn_save_BUILD_TESTING ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(Eigen3)
  # Eigen's CMake exports Eigen3::Eigen when fetched; if not, create interface (SYSTEM to avoid -Werror on Eigen headers)
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE ${eigen3_SOURCE_DIR})
  endif()
  # Restore BUILD_TESTING to previous value (if it was defined)
  if(DEFINED _flexnn_save_BUILD_TESTING)
    set(BUILD_TESTING ${_flexnn_save_BUILD_TESTING} CACHE BOOL "" FORCE)
  else()
    unset(BUILD_TESTING CACHE)
  endif()
  unset(_flexnn_save_BUILD_TESTING)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP ${OpenMP_CXX_VERSION}")
endif()

# -----------------------------------------------------------------------------
# Build type — default Release so -O3 -march=native applies (was CMAKE_CXX_FLAGS mutation)
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# -----------------------------------------------------------------------------
# Library — explicit source list (no GLOB, keeps upstack PR diffs clean)
# -----------------------------------------------------------------------------
set(LIB_SOURCES
  lib/FlexNN.cpp
  lib/Layer.cpp
  lib/Utility.cpp
  lib/activations/Activation.cpp
  lib/layers/Dense.cpp
  lib/layers/Conv1D.cpp
  lib/layers/BatchNorm1D.cpp
  lib/layers/Pool1D.cpp
  lib/layers/Layer.cpp
  lib/ModelIO.cpp
)

add_library(FlexNN ${LIB_SOURCES})

# Target-based includes/linking — no global include_directories()
target_include_directories(FlexNN PUBLIC include)
target_link_libraries(FlexNN PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

# Strict warnings — enforced on library only, not on external Eigen headers
target_compile_options(FlexNN PRIVATE -Wall -Wextra -Wpedantic -Werror -Wswitch-enum)
target_compile_features(FlexNN PUBLIC cxx_std_17)

# -----------------------------------------------------------------------------
# Executable — host example (MNIST)
# -----------------------------------------------------------------------------
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE FlexNN)

# -----------------------------------------------------------------------------
# Tests — opt-in via -DFLEXNN_BUILD_TESTS=ON (default ON for CI)
# -----------------------------------------------------------------------------
option(FLEXNN_BUILD_TESTS "Build FlexNN tests" ON)
if(FLEXNN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Doxygen — optional docs, output to build/docs to avoid clobbering docs/ LLD
# -----------------------------------------------------------------------------
find_package(Doxygen)
if(DOXYGEN_FOUND)
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  # Doxyfile has OUTPUT_DIRECTORY = docs (relative) which would clobber source docs/
  # when invoked via `doxygen Doxyfile` from source. Patch the configured copy to
  # emit under the build tree so `build/docs` is gitignored via `build/`.
  file(READ "${DOXYGEN_OUT}" _doxy_content)
  string(REPLACE "OUTPUT_DIRECTORY       = docs" "OUTPUT_DIRECTORY       = ${CMAKE_CURRENT_BINARY_DIR}/docs" _doxy_content "${_doxy_content}")
  file(WRITE "${DOXYGEN_OUT}" "${_doxy_content}")
  add_custom_target(doc_doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif()
