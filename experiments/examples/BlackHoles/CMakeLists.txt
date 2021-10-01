cmake_minimum_required(VERSION 3.15)
project(blackhole)

add_executable(blackhole Blackholes.cpp ${SRC}   )

target_link_libraries(blackhole norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)