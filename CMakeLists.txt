cmake_minimum_required(VERSION 3.15)
project(yalbb_example)

# Set compiler params
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_BUILD_TYPE Release)

# find sources
file(GLOB SRC src/*.cpp)
# find headers
file(GLOB HEADERS src/*.hpp)

# Generate YALBB library
execute_process(COMMAND mkdir -p build WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} . WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)
execute_process(COMMAND make -j 4 yalbb WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/yalbb)

# find Threads
find_package(Threads)
# find MPI
find_package(MPI REQUIRED)
# find YALBB
find_library(YALBB_LIBRARY_PATH NAMES yalbb HINTS ${CMAKE_CURRENT_LIST_DIR}/yalbb/lib)
find_path(YALBB_INCLUDE_DIRECTORY NAMES yalbb HINTS  ${CMAKE_CURRENT_LIST_DIR}/yalbb/includes)

# find my LB library (zoltan as an example)
set(ZOLTAN_INSTALL_FOLDER /usr/local CACHE PATH "zoltan install folder")
find_library(ZOLTAN_LIBRARY_PATH NAMES zoltan HINTS ${ZOLTAN_INSTALL_FOLDER}/lib)
find_path(ZOLTAN_INCLUDE_DIRECTORY NAMES zoltan.h HINTS ${ZOLTAN_INSTALL_FOLDER}/include)

# Build
set(BINARY_NAME "yalbb_lbopt")

add_executable(${BINARY_NAME} src/main.cpp ${SRC} ${HEADERS} src/loadbalancing.hpp)
target_include_directories(${BINARY_NAME} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY} useful-algorithms/include)
target_link_libraries(${BINARY_NAME} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

















