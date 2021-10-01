cmake_minimum_required(VERSION 3.15)
project(contractingcircles)

add_executable(contractingcircles ContractingCircles.cpp ${SRC} )

target_link_libraries(contractingcircles norcb CGAL::CGAL yalbb ${MPI_LIBRARIES} ${ZOLTAN_LIBRARY_PATH} stdc++fs)