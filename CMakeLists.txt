cmake_minimum_required(VERSION 3.10)
project(FlexNN)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

# Include directories

# Add the library
set(LIB_SOURCES
    lib/FlexNN.cpp
    lib/Layer.cpp
    lib/Utility.cpp
)
add_library(FlexNN ${LIB_SOURCES})

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP!")
endif()

# Add the executable
add_executable(main src/main.cpp)

# Link the library to the executable
if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC FlexNN OpenMP::OpenMP_CXX Eigen3::Eigen)
else()
    message(STATUS "OpenMP not found, compiling without OpenMP support.")
    target_compile_definitions(main PRIVATE NO_OPENMP)
    target_link_libraries(main FlexNN Eigen3::Eigen)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
