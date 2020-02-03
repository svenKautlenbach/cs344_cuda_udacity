cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(udacity
        LANGUAGES CXX CUDA
        DESCRIPTION "Learning parallel computing way of life"
        )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Debug)

include(${CMAKE_CURRENT_LIST_DIR}/squarin_numbers/CMakeLists.txt)

add_executable(udacity_test ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(udacity_test udacity-static)
