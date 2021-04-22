cmake_minimum_required(VERSION 3.15)
project(yalbb)

# find MPI
find_package(MPI REQUIRED)

# Set compiler params
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpic++)

if("${CMAKE_BUILD_TYPE}" STREQUAL Sanitize)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    message("Flags for build type \"${CMAKE_BUILD_TYPE}\" are: ${CMAKE_CXX_FLAGS}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL Profile)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p ${CMAKE_CXX_FLAGS_DEBUG} -frounding-math")
    message("Flags for build type \"${CMAKE_BUILD_TYPE}\" are: ${CMAKE_CXX_FLAGS}")
endif()

# Set build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

# find sources
file(GLOB SRC src/*.cpp)

# find headers
file(GLOB HEADERS src/*.hpp)

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

    message(STATUS "This project requires the CGAL library, and will not be compiled.")
    return()

endif()

# Boost and its components
find_package( Boost )

if ( NOT Boost_FOUND )
    message(STATUS "This project requires the Boost library, and will not be compiled.")
    return()
endif()

# find Threads
find_package(Threads)

# find YALBB
set(YALBB_INSTALL_FOLDER ${CMAKE_CURRENT_LIST_DIR})
find_library(YALBB_LIBRARY_PATH NAMES yalbb HINTS ${YALBB_INSTALL_FOLDER}/yalbb/lib)
find_path(YALBB_INCLUDE_DIRECTORY NAMES yalbb HINTS  ${YALBB_INSTALL_FOLDER}/yalbb/includes)

set(ZOLTAN_INSTALL_FOLDER /usr/local CACHE PATH "zoltan install folder")
find_library(ZOLTAN_LIBRARY_PATH NAMES zoltan HINTS ${ZOLTAN_INSTALL_FOLDER}/lib)
find_path(ZOLTAN_INCLUDE_DIRECTORY NAMES zoltan.h HINTS ${ZOLTAN_INSTALL_FOLDER}/include)

# additional lib
set(UF_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/NoRCB/useful-algorithms/include)
set(NORCB_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/NoRCB/include)
set(NORCB_LIB ${CMAKE_CURRENT_LIST_DIR}/NoRCB/libnorcb.a)

include_directories(src ${NORCB_INCLUDE_DIR} yalbb/zupply/src ${MPI_C_INCLUDE_PATH} ${YALBB_INCLUDE_DIRECTORY} ${ZOLTAN_INCLUDE_DIRECTORY} ${UF_INCLUDE_DIR})
link_libraries(${NORCB_LIB} CGAL::CGAL ${MPI_C_LIBRARY} ${ZOLTAN_LIBRARY_PATH}  ${YALBB_LIBRARY_PATH} stdc++fs ${ZOLTAN_LIBRARY_PATH} ${CMAKE_THREAD_LIBS_INIT})

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

add_executable(${COMP_BINARY_NAME} ${COMP_ENTRY} ${SRC} ${HEADERS})

add_executable(${EXPA_BINARY_NAME} ${EXPA_ENTRY} ${SRC} ${HEADERS})

add_executable(${GRAV_BINARY_NAME} ${GRAV_ENTRY} ${SRC} ${HEADERS})

add_executable(Collision src/experiments/Collision.cpp ${SRC} ${HEADERS})
