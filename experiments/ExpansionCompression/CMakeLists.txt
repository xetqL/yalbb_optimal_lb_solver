cmake_minimum_required(VERSION 3.15)
project(expansioncompression)

add_executable(expansioncompression ExpansionCompression.cpp ${SRC})

target_link_libraries(expansioncompression norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)