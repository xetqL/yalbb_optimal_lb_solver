cmake_minimum_required(VERSION 3.15)
project(expansion)

add_executable(expansion Expansion.cpp ${SRC})

target_link_libraries(expansion norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)