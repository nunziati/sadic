cmake_minimum_required(VERSION 3.1)
project(my_project)

cmake_policy(SET CMP0057 NEW)

# Find the DGtal package
find_package(DGtal REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

include(CTest)
enable_testing()
add_subdirectory(pybind11)
pybind11_add_module(my_module bindings.cpp)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
target_link_libraries(my_module PRIVATE pybind11::module DGtal)
include(CPack)