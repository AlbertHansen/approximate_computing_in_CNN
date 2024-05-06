cmake_minimum_required(VERSION 3.12)
project(test)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)
add_library(test MODULE test.cpp)

target_link_libraries(test PRIVATE pybind11::module)
