cmake_minimum_required(VERSION 3.15)
project(example)

add_executable(example PaperExampleExp.cpp ${SRC} )

target_link_libraries(example norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)