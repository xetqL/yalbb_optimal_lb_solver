
find_package( CGAL QUIET COMPONENTS  )
if ( NOT CGAL_FOUND )
    message(STATUS "This project requires the CGAL library, and will not be compiled.")
    return()
endif()

add_compile_definitions(YALBB_DIMENSION=2)

include_directories(${NORCB_INCLUDE_DIR})

add_subdirectory(BlackHoles)
add_subdirectory(Gravity)
add_subdirectory(ContractingCircles)
