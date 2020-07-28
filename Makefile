# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/114/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/114/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xetql/optimal_lb_nbody

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xetql/optimal_lb_nbody

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/snap/clion/114/bin/cmake/linux/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/snap/clion/114/bin/cmake/linux/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xetql/optimal_lb_nbody/CMakeFiles /home/xetql/optimal_lb_nbody/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xetql/optimal_lb_nbody/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/build
.PHONY : test/fast

#=============================================================================
# Target rules for targets named yalbb_lbopt

# Build rule for target.
yalbb_lbopt: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 yalbb_lbopt
.PHONY : yalbb_lbopt

# fast build rule for target.
yalbb_lbopt/fast:
	$(MAKE) -f CMakeFiles/yalbb_lbopt.dir/build.make CMakeFiles/yalbb_lbopt.dir/build
.PHONY : yalbb_lbopt/fast

src/main.o: src/main.cpp.o

.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/yalbb_lbopt.dir/build.make CMakeFiles/yalbb_lbopt.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i

.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/yalbb_lbopt.dir/build.make CMakeFiles/yalbb_lbopt.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s

.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/yalbb_lbopt.dir/build.make CMakeFiles/yalbb_lbopt.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/main_test.o: src/main_test.cpp.o

.PHONY : src/main_test.o

# target to build an object file
src/main_test.cpp.o:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/main_test.cpp.o
.PHONY : src/main_test.cpp.o

src/main_test.i: src/main_test.cpp.i

.PHONY : src/main_test.i

# target to preprocess a source file
src/main_test.cpp.i:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/main_test.cpp.i
.PHONY : src/main_test.cpp.i

src/main_test.s: src/main_test.cpp.s

.PHONY : src/main_test.s

# target to generate assembly for a file
src/main_test.cpp.s:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/main_test.cpp.s
.PHONY : src/main_test.cpp.s

src/zoltan_fn.o: src/zoltan_fn.cpp.o

.PHONY : src/zoltan_fn.o

# target to build an object file
src/zoltan_fn.cpp.o:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/zoltan_fn.cpp.o
	$(MAKE) -f CMakeFiles/yalbb_lbopt.dir/build.make CMakeFiles/yalbb_lbopt.dir/src/zoltan_fn.cpp.o
.PHONY : src/zoltan_fn.cpp.o

src/zoltan_fn.i: src/zoltan_fn.cpp.i

.PHONY : src/zoltan_fn.i

# target to preprocess a source file
src/zoltan_fn.cpp.i:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/zoltan_fn.cpp.i
	$(MAKE) -f CMakeFiles/yalbb_lbopt.dir/build.make CMakeFiles/yalbb_lbopt.dir/src/zoltan_fn.cpp.i
.PHONY : src/zoltan_fn.cpp.i

src/zoltan_fn.s: src/zoltan_fn.cpp.s

.PHONY : src/zoltan_fn.s

# target to generate assembly for a file
src/zoltan_fn.cpp.s:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/src/zoltan_fn.cpp.s
	$(MAKE) -f CMakeFiles/yalbb_lbopt.dir/build.make CMakeFiles/yalbb_lbopt.dir/src/zoltan_fn.cpp.s
.PHONY : src/zoltan_fn.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... test"
	@echo "... yalbb_lbopt"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/main_test.o"
	@echo "... src/main_test.i"
	@echo "... src/main_test.s"
	@echo "... src/zoltan_fn.o"
	@echo "... src/zoltan_fn.i"
	@echo "... src/zoltan_fn.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
