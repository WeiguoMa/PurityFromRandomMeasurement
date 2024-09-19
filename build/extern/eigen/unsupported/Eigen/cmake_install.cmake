# Install script for directory: /Users/weiguo_ma/CppProjects/PurityFromRandomMeasurement/extern/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/homebrew/Cellar/llvm/18.1.8/bin/llvm-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/AdolcForward"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/AlignedVector3"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/ArpackSupport"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/AutoDiff"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/BVH"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/EulerAngles"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/FFT"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/IterativeSolvers"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/KroneckerProduct"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/MatrixFunctions"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/MPRealSupport"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/NNLS"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/NonLinearOptimization"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/NumericalDiff"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/OpenGLSupport"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/Polynomials"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/SparseExtra"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/SpecialFunctions"
    "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/Users/weiguo_ma/CppProjects/PurityFromShadow/extern/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/weiguo_ma/CppProjects/PurityFromShadow/build/extern/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

