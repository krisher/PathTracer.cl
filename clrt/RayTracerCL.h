/*!
 * \file RayTracerCL.h
 * \brief Encapsulation of an OpenCL based ray-tracer.
 */
#ifndef _RAYTRACER_CL_H
#define _RAYTRACER_CL_H
#define __CL_ENABLE_EXCEPTIONS
/*
 * OpenCL C++ bindings...
 */
#include <CL/cl.hpp>

#include "geometry.h"
#include "RayTracer.h"


/*!
 * \brief RayTracer extension that does its work using OpenCL.
 */
class RayTracerCL : public RayTracer
{

 private:
  
  // OpenCL host API objects

  /*!
   * \brief OpenCL context
   */
  cl::Context context;
  /*!
   *\brief Command Queue bound to the first detected CL device.
   */ 
  cl::CommandQueue cmdQueue;
  /*!
   *\brief The compiled OpenCL raytracer entry function.
   */
  cl::Kernel raytracer_kernel;
  /*!
   * \brief Flag indicating whether OpenCL/OpenGL sharing is both supported and enabled for the command queue/device used for processing.
   */
  bool glSharing;
  /*!
   * \brief ND Range size, determined automatically from CL context.
   */
  size_t ndRangeSizes[2];

  
  // Structures used to transfer data to OpenCL.

  /*!
   * \brief A buffer that stores the camera parameters in a Camera struct.
   */
  cl::Buffer cameraBufferCL;
  /*!
   * \brief A buffer that stores the scene objects
   */
  cl::Buffer sceneBufferCL;
  /*!
   * \brief A buffer that stores seed values for the rng.
   */
  cl::Buffer seedBufferCL;


  /*!
   * \brief A flag indicating that the camera has been changed in host code, and must be updated on the OpenCL side.
   */
  bool camDirty;
  /*!
   * \brief The width of the image to render, in pixels
   */
  uint width;
  /*!
   * \brief The height of the image to render, in pixels
   */
  uint height;
  /*!
   * The size of the current geometry buffer 
   */
  uint geomBufferSize;

  void updateCLCamera();
  void init(cl::Platform const &platform);

 protected:
  virtual void cameraChanged();

 public:
  RayTracerCL();
  RayTracerCL(cl::Platform const &platform);
  ~RayTracerCL();

  /*!
   * \brief Accessor to determine whether GL/CL interoperability is both supported and enabled for this RayTracer.
   */
  bool supportsGLSharing() {return glSharing;};
  void disableGLSharing() {glSharing = false;};

  const cl::Context getCLContext();
  const cl::CommandQueue getCLCommandQueue() {return cmdQueue;};

  /*!
   * \brief Executes the OpenCL kernel to render the scene into the specified buffer.
   */
  void rayTrace(cl_mem *buff, uint const width, uint const height, uint const progression);


  /*!
   *\brief returns a new cl::Platform object representing the default OpenCL platform, or null if no default is defined.
   *
   */
  static cl::Platform * getDefaultPlatform();

  /*!
   * \brief Loads the contents of the specified file into memory as a std::string.
   */
  static std::string * loadSourceFromFile(const char *filename);
};

#endif /* _RAYTRACER_CL_H */
