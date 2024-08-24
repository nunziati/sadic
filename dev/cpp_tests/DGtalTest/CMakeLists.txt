cmake_minimum_required(VERSION 3.1)
project(DGtalTest)

cmake_policy(SET CMP0057 NEW)

# Find the DGtal package
find_package(DGtal REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the executable
add_executable(MyProgram main.cpp)

# Link against the DGtal libraries
target_link_libraries(MyProgram DGtal)