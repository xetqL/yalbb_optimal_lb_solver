cmake_minimum_required(VERSION 3.10)
project(yalbb_example)
# Set compiler params
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpic++)
# Look for my files
file(GLOB SRC src/*.cpp)
list(FILTER SRC EXCLUDE REGEX ".*main.*.cpp")
message(${SRC})
file(GLOB HEADERS src/*.hpp)
# Produce YALBB library
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
set(BIN_ASTAR   "yalbb_astar")
set(BIN_MENON   "yalbb_menon")
set(BIN_PROCA   "yalbb_proca")
set(BIN_MARQU   "yalbb_marqu")

set(BINARY_NAME_TEST "yalbb_otest")

add_executable(${BINARY_NAME} src/main.cpp ${SRC} ${HEADERS} src/StripeLB.hpp)
target_include_directories(${BINARY_NAME} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(${BINARY_NAME} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${BIN_ASTAR} src/main_astar.cpp ${SRC} ${HEADERS})
target_include_directories(${BIN_ASTAR} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(${BIN_ASTAR} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${BIN_MENON} src/main_menon.cpp ${SRC} ${HEADERS})
target_include_directories(${BIN_MENON} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(${BIN_MENON} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${BIN_PROCA} src/main_procassini.cpp ${SRC} ${HEADERS})
target_include_directories(${BIN_PROCA} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(${BIN_PROCA} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${BIN_MARQU} src/main_marquez.cpp ${SRC} ${HEADERS})
target_include_directories(${BIN_MARQU} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(${BIN_MARQU} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${BINARY_NAME_TEST} src/main_test.cpp ${SRC} ${HEADERS})
target_include_directories(${BINARY_NAME_TEST} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY})
target_link_libraries(${BINARY_NAME_TEST} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})
