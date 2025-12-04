# Main CMakeLists.txt for the Run3ModelGen project

# Mandatory setting for minimal CMake requirement
cmake_minimum_required( VERSION 3.20 )

# Create a project
project( Run3ModelGen )

include( ExternalProject )

# Add subdirectories for tools that need tweaking
add_subdirectory( micromegas_5.2.1 )

# Add ExternalProjects for tools that run out-of-the-box

# Add softsusy as ExternalProject
ExternalProject_Add( softsusy
    PREFIX softsusy-4.1.11
    URL http://softsusy.hepforge.org/downloads/softsusy-4.1.11.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND <SOURCE_DIR>/configure
    BUILD_COMMAND ${CMAKE_COMMAND} -E env "CMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_MAKE_PROGRAM}
    INSTALL_COMMAND ""
)

# Add superiso as ExternalProject
ExternalProject_Add( superiso
    PREFIX superiso_v4.0
    URL "${CMAKE_SOURCE_DIR}/superiso_v4.0.tgz"
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
    CMAKE_ARGS -j1
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} && ${CMAKE_MAKE_PROGRAM} slha
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
)

# # Add FeynHiggs as ExternalProject
# ExternalProject_Add( feyngiggs
#     PREFIX FeynHiggs-2.19.0
#     URL https://wwwth.mpp.mpg.de/members/heinemey/feynhiggs/newversion/FeynHiggs-2.19.0.tar.gz
#     DOWNLOAD_EXTRACT_TIMESTAMP NEW
#     UPDATE_COMMAND ""
#     CONFIGURE_COMMAND <SOURCE_DIR>/configure
#     BUILD_COMMAND ${CMAKE_COMMAND} -E env "CMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_MAKE_PROGRAM} CFLAGS="-z noexecstack"
#     INSTALL_COMMAND ${CMAKE_COMMAND} -E env "CMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_MAKE_PROGRAM} install
# )

# Add GM2Calc as ExternalProject
ExternalProject_Add( gm2calc
    PREFIX GM2Calc-2.0.0
    URL https://github.com/GM2Calc/GM2Calc/archive/v2.0.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
    UPDATE_COMMAND ""
    BUILD_COMMAND ${CMAKE_COMMAND} -E env "CMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_MAKE_PROGRAM}
    INSTALL_COMMAND ""
)

# # Add EVADE as ExternalProject
# ExternalProject_Add( evade
#     PREFIX EVADE-0.2.0
#     GIT_REPOSITORY https://gitlab.com/jonaswittbrodt/EVADE.git
#     DOWNLOAD_EXTRACT_TIMESTAMP NEW
#     CMAKE_ARGS -Wno-dev
#     UPDATE_COMMAND ""
#     BUILD_COMMAND ${CMAKE_COMMAND} -E env "CMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_MAKE_PROGRAM}
#     INSTALL_COMMAND ""
# )

# Add shell script to place all relevant files in build
add_custom_target( Run3ModelGen
    COMMENT "Building Run3ModelGen"
    COMMAND ${CMAKE_SOURCE_DIR}/cmake_build_env.sh -s ${CMAKE_SOURCE_DIR} -b ${CMAKE_BINARY_DIR}
    DEPENDS micromegas
)

# make Run3ModelGen the default target for "make"
add_custom_target(default_target DEPENDS Run3ModelGen)