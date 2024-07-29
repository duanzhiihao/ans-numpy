cmake_minimum_required(VERSION 3.5)
project(main_project)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})


# add_subdirectory(src)

find_package(pybind11 REQUIRED)

pybind11_add_module(ansnp ${PROJECT_SOURCE_DIR}/src/rans_interface.cpp)
