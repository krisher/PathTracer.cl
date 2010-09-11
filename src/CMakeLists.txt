

include_directories (${PROJECT_SOURCE_DIR}/include) 

add_executable(raytracer.cl 
  main.cpp 
  GlutWindow.cpp
  RayTracer.cpp
  PlatformInfo.cpp
  GlutCLWindow.cpp
  RayTracerCL.cpp
)
target_link_libraries(raytracer.cl ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${OPENCL_LIBRARIES})
