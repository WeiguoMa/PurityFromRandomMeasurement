cmake_minimum_required(VERSION 3.28)
project(PurityFromShadow)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find Python3 interpreter and development components
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python 3 not found.")
endif()

message(STATUS "Using Python 3: ${Python3_EXECUTABLE}")

set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})

# Find OpenMP (optional, add only if needed for parallelization)
find_package(OpenMP)

# Add pybind11 and Eigen directories
add_subdirectory("./extern/pybind11")
add_subdirectory("./extern/eigen")
include_directories(${CMAKE_SOURCE_DIR}/extern/eigen)

# Add the pybind11 module and ensure ShadowState.cpp is compiled as part of the module
pybind11_add_module(RenyiEntropy src/cpp/RenyiEntropy.cpp src/cpp/ShadowState.cpp)
pybind11_add_module(fakeSampler_backend src/cpp/fake_sampler_binding.cpp)

#include_directories(/opt/homebrew/Cellar/quantum++/5.1/include/qpp)
#add_executable(standalone src/cpp/main.cpp)
