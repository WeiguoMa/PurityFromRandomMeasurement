cmake_minimum_required(VERSION 3.28)
project(PurityFromShadow)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python 3 not found.")
endif()

message(STATUS "Using Python 3: ${Python3_EXECUTABLE}")

set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})

add_subdirectory("./extern/pybind11")
add_subdirectory("./extern/eigen")
include_directories(${CMAKE_SOURCE_DIR}/extern/eigen)

pybind11_add_module(shadowState src/cpp/ShadowState.cpp)
add_executable(RenyiEntropy src/cpp/RenyiEntropy.cpp)

#include_directories(/opt/homebrew/Cellar/quantum++/5.1/include/qpp)
#add_executable(standalone src/cpp/main.cpp)
