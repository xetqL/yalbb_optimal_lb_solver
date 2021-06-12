cmake_minimum_required(VERSION 3.15)
project(gravitycircle)

add_executable(gravitycircle GravityCircle.cpp ${SRC})

target_link_libraries(gravitycircle norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)