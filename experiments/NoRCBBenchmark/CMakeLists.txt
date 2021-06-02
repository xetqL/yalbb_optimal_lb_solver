cmake_minimum_required(VERSION 3.15)

project(norcbbenchmark)

add_executable(norcbbenchmark NoRCBBenchmark.cpp ${SRC})

target_link_libraries(norcbbenchmark norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)