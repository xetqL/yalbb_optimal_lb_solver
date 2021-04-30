cmake_minimum_required(VERSION 3.15)
project(gravity)

add_executable(gravity Blackholes.cpp ${SRC})

target_link_libraries(gravity norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)