cmake_minimum_required(VERSION 2.6)
project(raytracer.cl)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(GLUT)
find_package(OpenGL)
find_package(OpenCL)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g")

add_subdirectory(clrt)

ADD_CUSTOM_TARGET(debug
  ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=Debug ${PROJECT_SOURCE_DIR})

ADD_CUSTOM_TARGET(release
  ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=Release ${PROJECT_SOURCE_DIR})