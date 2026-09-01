cmake_minimum_required(VERSION 3.16)
project(sim2sim_logger CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(unitree_sdk2 REQUIRED)

add_executable(sim2sim_logger main.cpp)
target_link_libraries(sim2sim_logger PRIVATE unitree_sdk2)
