cmake_minimum_required(VERSION 3.15)
project(collision)

add_executable(collision Collision.cpp ${SRC} )

target_link_libraries(collision norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)