# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/weiguo_ma/CppProjects/PurityFromShadow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/weiguo_ma/CppProjects/PurityFromShadow/build

# Include any dependencies generated for this target.
include CMakeFiles/RenyiEntropy.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/RenyiEntropy.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/RenyiEntropy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RenyiEntropy.dir/flags.make

CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o: CMakeFiles/RenyiEntropy.dir/flags.make
CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o: /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/RenyiEntropy.cpp
CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o: CMakeFiles/RenyiEntropy.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/weiguo_ma/CppProjects/PurityFromShadow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o"
	/opt/homebrew/Cellar/llvm/18.1.8/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o -MF CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o.d -o CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o -c /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/RenyiEntropy.cpp

CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.i"
	/opt/homebrew/Cellar/llvm/18.1.8/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/RenyiEntropy.cpp > CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.i

CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.s"
	/opt/homebrew/Cellar/llvm/18.1.8/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/RenyiEntropy.cpp -o CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.s

CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o: CMakeFiles/RenyiEntropy.dir/flags.make
CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o: /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/ShadowState.cpp
CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o: CMakeFiles/RenyiEntropy.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/weiguo_ma/CppProjects/PurityFromShadow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o"
	/opt/homebrew/Cellar/llvm/18.1.8/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o -MF CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o.d -o CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o -c /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/ShadowState.cpp

CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.i"
	/opt/homebrew/Cellar/llvm/18.1.8/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/ShadowState.cpp > CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.i

CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.s"
	/opt/homebrew/Cellar/llvm/18.1.8/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/weiguo_ma/CppProjects/PurityFromShadow/src/cpp/ShadowState.cpp -o CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.s

# Object files for target RenyiEntropy
RenyiEntropy_OBJECTS = \
"CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o" \
"CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o"

# External object files for target RenyiEntropy
RenyiEntropy_EXTERNAL_OBJECTS =

RenyiEntropy.cpython-312-darwin.so: CMakeFiles/RenyiEntropy.dir/src/cpp/RenyiEntropy.cpp.o
RenyiEntropy.cpython-312-darwin.so: CMakeFiles/RenyiEntropy.dir/src/cpp/ShadowState.cpp.o
RenyiEntropy.cpython-312-darwin.so: CMakeFiles/RenyiEntropy.dir/build.make
RenyiEntropy.cpython-312-darwin.so: CMakeFiles/RenyiEntropy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/weiguo_ma/CppProjects/PurityFromShadow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module RenyiEntropy.cpython-312-darwin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RenyiEntropy.dir/link.txt --verbose=$(VERBOSE)
	/opt/anaconda3/bin/strip -x /Users/weiguo_ma/CppProjects/PurityFromShadow/build/RenyiEntropy.cpython-312-darwin.so

# Rule to build all files generated by this target.
CMakeFiles/RenyiEntropy.dir/build: RenyiEntropy.cpython-312-darwin.so
.PHONY : CMakeFiles/RenyiEntropy.dir/build

CMakeFiles/RenyiEntropy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RenyiEntropy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RenyiEntropy.dir/clean

CMakeFiles/RenyiEntropy.dir/depend:
	cd /Users/weiguo_ma/CppProjects/PurityFromShadow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/weiguo_ma/CppProjects/PurityFromShadow /Users/weiguo_ma/CppProjects/PurityFromShadow /Users/weiguo_ma/CppProjects/PurityFromShadow/build /Users/weiguo_ma/CppProjects/PurityFromShadow/build /Users/weiguo_ma/CppProjects/PurityFromShadow/build/CMakeFiles/RenyiEntropy.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/RenyiEntropy.dir/depend

