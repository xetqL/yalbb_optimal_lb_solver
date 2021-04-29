cmake_minimum_required(VERSION 3.15)
project(yalbb)

add_subdirectory(NoRCB)
add_subdirectory(yalbb)

set(YALBB_DIMENSION "2" CACHE STRING "number of dimensions")

add_compile_definitions(YALBB_DIMENSION=${YALBB_DIMENSION})

find_package( CGAL QUIET COMPONENTS  )
if ( NOT CGAL_FOUND )
    message(STATUS "This project requires the CGAL library, and will not be compiled.")
    return()
endif()

# Set compiler params
set(CMAKE_CXX_STANDARD 17)

if("${CMAKE_BUILD_TYPE}" STREQUAL Sanitize)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    message("Flags for build type \"${CMAKE_BUILD_TYPE}\" are: ${CMAKE_CXX_FLAGS}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL Profile)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p ${CMAKE_CXX_FLAGS_DEBUG} -frounding-math")
    message("Flags for build type \"${CMAKE_BUILD_TYPE}\" are: ${CMAKE_CXX_FLAGS}")
endif()

# Set build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# find sources
file(GLOB SRC src/*.cpp)

# find headers
file(GLOB HEADERS src/*.hpp)

# Boost and its components
find_package( Boost )
if ( NOT Boost_FOUND )
    message(STATUS "This project requires the Boost library, and will not be compiled.")
    return()
endif()

# MPI
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

# find Threads
find_package(Threads)

set(ZOLTAN_INSTALL_FOLDER /usr/local CACHE PATH "zoltan install folder")
find_library(ZOLTAN_LIBRARY_PATH NAMES zoltan HINTS ${ZOLTAN_INSTALL_FOLDER}/lib)
find_path(ZOLTAN_INCLUDE_DIRECTORY NAMES zoltan.h HINTS ${ZOLTAN_INSTALL_FOLDER}/include)

# additional lib
set(UF_INCLUDE_DIR useful-algorithms/include)
set(NORCB_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/NoRCB/include)

include_directories(src ${NORCB_INCLUDE_DIR} yalbb/zupply/src ${MPI_INCLUDE_PATH} yalbb/include ${ZOLTAN_INCLUDE_DIRECTORY} ${UF_INCLUDE_DIR})

add_subdirectory(experiments/Collision)
add_subdirectory(experiments/Compression)
add_subdirectory(experiments/Expansion)
add_subdirectory(experiments/ExpansionCompression)
add_subdirectory(experiments/Gravity)
