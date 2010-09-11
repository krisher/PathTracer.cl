cmake_minimum_required(VERSION 2.6)
project(raytracer.cl)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(GLUT)
find_package(OpenGL)
find_package(OpenCL)

set(CMAKE_CXX_FLAGS "-Wall")
add_subdirectory(src)
