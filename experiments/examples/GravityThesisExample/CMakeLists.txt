cmake_minimum_required(VERSION 3.15)
project(thesisexample)

add_executable(thesisexample Gravity.cpp ${SRC} )

target_link_libraries(thesisexample norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)