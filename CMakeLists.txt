cmake_minimum_required(VERSION 3.28)
project(PurityFromRandomMeasurement)

# Set C++ standard and build options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

# Use different optimization flags based on the compiler
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")  # Clang optimization with OpenMP
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")  # GCC optimization with OpenMP
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")  # Generic optimization for other compilers
endif ()

# Output separator for better visibility
function(print_separator message)
    message(STATUS "------------------------------------------")
    message(STATUS "${message}")
    message(STATUS "------------------------------------------")
endfunction()

# Add option to enable or disable OpenMP, cache this variable
option(USE_OPENMP "Enable OpenMP support" OFF)
set(USE_OPENMP ${USE_OPENMP} CACHE BOOL "Enable or disable OpenMP support")

if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        print_separator("OpenMP Configuration")
        message(STATUS "OpenMP found: Enabling OpenMP support.")
        set(ENABLE_OPENMP TRUE CACHE BOOL "OpenMP is enabled")
    else ()
        message(FATAL_ERROR "OpenMP requested but not found.")
    endif ()
else ()
    print_separator("OpenMP disabled by user.")
    set(ENABLE_OPENMP FALSE CACHE BOOL "OpenMP is disabled")
endif ()

# Find Python3 interpreter and development components
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python 3 not found.")
else ()
    print_separator("Python 3 Configuration")
    message(STATUS "Python 3 Executable: ${Python3_EXECUTABLE}")
    message(STATUS "Python 3 Version: ${Python3_VERSION}")
endif ()

# Set Python version and paths
set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})

# Get the number of logical CPU cores
include(ProcessorCount)
ProcessorCount(CPU_CORES)
print_separator("System Configuration")
message(STATUS "Number of CPU cores detected: ${CPU_CORES}")

# Add pybind11 and Eigen directories
add_subdirectory("./extern/pybind11")
add_subdirectory("./extern/eigen")
include_directories(${CMAKE_SOURCE_DIR}/extern/eigen)

# Set the output directory for the modules
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/dependency)

# pybind11 modules
pybind11_add_module(RenyiEntropy_backend src/cpp/RenyiEntropy.cpp src/cpp/ShadowState.cpp)
if (ENABLE_OPENMP)
    message(STATUS "Linking RenyiEntropy_backend with OpenMP.")
    target_link_libraries(RenyiEntropy_backend PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(STATUS "Building RenyiEntropy_backend without OpenMP.")
endif ()

pybind11_add_module(fakeSampler_backend src/cpp/fake_sampler_binding.cpp)
if (ENABLE_OPENMP)
    message(STATUS "Linking fakeSampler_backend with OpenMP.")
    target_link_libraries(fakeSampler_backend PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(STATUS "Building fakeSampler_backend without OpenMP.")
endif ()

pybind11_add_module(ShadowState_backend src/cpp/ShadowState.cpp)
if (ENABLE_OPENMP)
    message(STATUS "Linking ShadowState_backend with OpenMP.")
    target_link_libraries(ShadowState_backend PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(STATUS "Building ShadowState_backend without OpenMP.")
endif ()

add_executable(TESTOPEMMP src/cpp/TESTOPENMP.cpp)
if (ENABLE_OPENMP)
    message(STATUS "Linking TESTOPEMMP with OpenMP.")
    target_link_libraries(TESTOPEMMP PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(STATUS "Building TESTOPEMMP without OpenMP.")
endif ()
