cmake_minimum_required(VERSION 3.15)
project(expandingcircles)

add_executable(expandingcircles ExpandingCircles.cpp ${SRC})

target_link_libraries(expandingcircles norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)