cmake_minimum_required(VERSION 3.15)
project(yalbb)

# Set compiler params
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_BUILD_TYPE Release)

# Set build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

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
set(YALBB_INSTALL_FOLDER /home/xetql/projects/cpp)
find_library(YALBB_LIBRARY_PATH NAMES yalbb HINTS ${YALBB_INSTALL_FOLDER}/yalbb/lib)
find_path(YALBB_INCLUDE_DIRECTORY NAMES yalbb HINTS  ${YALBB_INSTALL_FOLDER}/yalbb/includes)
# find my LB library (zoltan as an example)
set(ZOLTAN_INSTALL_FOLDER /usr/local CACHE PATH "zoltan install folder")
find_library(ZOLTAN_LIBRARY_PATH NAMES zoltan HINTS ${ZOLTAN_INSTALL_FOLDER}/lib)
find_path(ZOLTAN_INCLUDE_DIRECTORY NAMES zoltan.h HINTS ${ZOLTAN_INSTALL_FOLDER}/include)
# additional lib
set(UF_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/useful-algorithms/include)

include_directories(src)

# define experiments entry file
set(EXPERIMENTS_FOLDER "src/experiments")

set(EXPA_COMP_BINARY_NAME "ExpansionCompression")
set(EXPA_COMP_ENTRY "${EXPERIMENTS_FOLDER}/ExpansionCompression.cpp")

set(COMP_BINARY_NAME "Compression")
set(COMP_ENTRY "${EXPERIMENTS_FOLDER}/Compression.cpp")

set(EXPA_BINARY_NAME "Expansion")
set(EXPA_ENTRY "${EXPERIMENTS_FOLDER}/Expansion.cpp")

set(GRAV_BINARY_NAME "Gravity")
set(GRAV_ENTRY "${EXPERIMENTS_FOLDER}/Gravity.cpp")

add_executable(${EXPA_COMP_BINARY_NAME} ${EXPA_COMP_ENTRY} ${SRC} ${HEADERS})
target_include_directories(${EXPA_COMP_BINARY_NAME} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY} ${UF_INCLUDE_DIR})
target_link_libraries(${EXPA_COMP_BINARY_NAME} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${COMP_BINARY_NAME} ${COMP_ENTRY} ${SRC} ${HEADERS})
target_include_directories(${COMP_BINARY_NAME} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY} ${UF_INCLUDE_DIR})
target_link_libraries(${COMP_BINARY_NAME} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${EXPA_BINARY_NAME} ${EXPA_ENTRY} ${SRC} ${HEADERS})
target_include_directories(${EXPA_BINARY_NAME} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY} ${UF_INCLUDE_DIR})
target_link_libraries(${EXPA_BINARY_NAME} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

add_executable(${GRAV_BINARY_NAME} ${GRAV_ENTRY} ${SRC} ${HEADERS})
target_include_directories(${GRAV_BINARY_NAME} PRIVATE ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY} ${UF_INCLUDE_DIR})
target_link_libraries(${GRAV_BINARY_NAME} PRIVATE ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

















