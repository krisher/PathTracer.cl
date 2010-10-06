

include_directories (${PROJECT_SOURCE_DIR}/include) 

add_executable(clrt 
  main.cpp 
  GlutWindow.cpp
  RayTracer.cpp
  PlatformInfo.cpp
  GlutCLWindow.cpp
  RayTracerCL.cpp
)
target_link_libraries(clrt ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${OPENCL_LIBRARIES})
