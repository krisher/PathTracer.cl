

include_directories (${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/ocl) 

add_executable(clrt 
  main.cpp 
  GlutWindow.cpp
  RayTracer.cpp
  PlatformInfo.cpp
  GlutCLWindow.cpp
  RayTracerCL.cpp
)

add_custom_command(
  TARGET clrt
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ocl ${CMAKE_CURRENT_BINARY_DIR}/ocl
)


target_link_libraries(clrt ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${OPENCL_LIBRARIES} rt)


add_executable(plyclrt 
  plymain.cpp 
  GlutWindow.cpp
  RayTracer.cpp
  PlatformInfo.cpp
  GlutCLWindow.cpp
  RayTracerCL.cpp
  PLYLoader.cpp
  ../include/ply/ply.c
)

add_custom_command(
  TARGET plyclrt
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ocl ${CMAKE_CURRENT_BINARY_DIR}/ocl
)


target_link_libraries(plyclrt ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${OPENCL_LIBRARIES} rt)
