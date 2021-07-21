cmake_minimum_required(VERSION 3.15)
project(compression)

add_executable(compression Compression.cpp ${SRC} )
target_link_libraries(compression norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)