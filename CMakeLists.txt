cmake_minimum_required(VERSION 3.28)
project(PurityFromShadow)

# Set C++ standard and build options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Output separator for better visibility
function(print_separator message)
    message(STATUS "------------------------------------------")
    message(STATUS "${message}")
    message(STATUS "------------------------------------------")
endfunction()

# Find Python3 interpreter and development components
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python 3 not found.")
else()
    print_separator("Python 3 Configuration")
    message(STATUS "Python 3 Executable: ${Python3_EXECUTABLE}")
    message(STATUS "Python 3 Version: ${Python3_VERSION}")
endif()

# Set Python version and paths
set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})

# Detect OS and set OpenMP options
if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    # macOS: Do not enable OpenMP
    print_separator("macOS detected. OpenMP will not be enabled.")
    set(ENABLE_OPENMP FALSE)
else()
    # Linux and Windows: Try to find OpenMP
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        print_separator("OpenMP Configuration")
        message(STATUS "OpenMP found: Enabling OpenMP support.")
        set(ENABLE_OPENMP TRUE)
    else()
        print_separator("OpenMP Configuration")
        message(STATUS "OpenMP not found: Proceeding without OpenMP support.")
        set(ENABLE_OPENMP FALSE)
    endif()
endif()

# Get the number of logical CPU cores
include(ProcessorCount)
ProcessorCount(CPU_CORES)
print_separator("System Configuration")
message(STATUS "Number of CPU cores detected: ${CPU_CORES}")

# Add pybind11 and Eigen directories
add_subdirectory("./extern/pybind11")
add_subdirectory("./extern/eigen")
include_directories(${CMAKE_SOURCE_DIR}/extern/eigen)

# Add the pybind11 module and ensure ShadowState.cpp is compiled as part of the module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/dependency)

# pybind11 modules
pybind11_add_module(RenyiEntropy_backend src/cpp/RenyiEntropy.cpp src/cpp/ShadowState.cpp)
if(ENABLE_OPENMP)
    message(STATUS "Linking RenyiEntropy_backend with OpenMP.")
    target_link_libraries(RenyiEntropy_backend PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "Building RenyiEntropy_backend without OpenMP.")
endif()

pybind11_add_module(fakeSampler_backend src/cpp/fake_sampler_binding.cpp)
if(ENABLE_OPENMP)
    message(STATUS "Linking fakeSampler_backend with OpenMP.")
    target_link_libraries(fakeSampler_backend PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "Building fakeSampler_backend without OpenMP.")
endif()

pybind11_add_module(ShadowState_backend src/cpp/ShadowState.cpp)
if(ENABLE_OPENMP)
    message(STATUS "Linking ShadowState_backend with OpenMP.")
    target_link_libraries(ShadowState_backend PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "Building ShadowState_backend without OpenMP.")
endif()

# Standalone executable (if needed in the future)
# include_directories(/opt/homebrew/Cellar/quantum++/5.1/include/qpp)
# add_executable(standalone src/cpp/main.cpp)
