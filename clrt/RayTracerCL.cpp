/*! 
 * \file RayTracer.cpp
 * \brief OpenCL based ray tracer main program
 *
 * http://developer.amd.com/GPU/ATISTREAMSDK/pages/TutorialOpenCL.aspx
 */
#include <utility>
/*
 * Enable C++ exceptions thrown from CL C++ bindings
 */


#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

// TODO: Linux only!
#include <GL/glx.h>

#include "PlatformInfo.h"
#include "RayTracerCL.h"

#define RAYTRACER_CL "raytracer.cl"
//Comment this out to disable debugging and timing info
//#define _DEBUG_RT
#include "Timing.h"
#ifdef _DEBUG_RT

#endif /* _DEBUG_RT */


// TODO:
// These definitions are missing from the NVidia GL/CL headers (195.53) Should be in cl_gl.h?
#ifndef CL_GL_CONTEXT_KHR
#define CL_GL_CONTEXT_KHR 0x2008
#define CL_GLX_DISPLAY_KHR 0x200A
#endif

#define D2R(x) (x * M_PI/180.0f)

#define OUTBUFF_CL_PARAM 0
#define CAMERA_CL_PARAM 1
#define GEOM_CL_PARAM 2
#define GEOMCOUNT_CL_PARAM 3
#define IMWIDTH_CL_PARAM 4
#define IMHEIGHT_CL_PARAM 5
#define SAMPLERATE_CL_PARAM 6
#define MAXDEPTH_CL_PARAM 7
#define PROGRESSION_CL_PARAM 8
#define SEEDS_CL_PARAM 9


/*!
 * \brief Initializes RayTracer using the default CL Platform.
 */
RayTracerCL::RayTracerCL() : RayTracer()
{
  init(*RayTracerCL::getDefaultPlatform());
}

RayTracerCL::RayTracerCL(cl::Platform const &platform) : RayTracer()
{
  init(platform);
}

RayTracerCL::~RayTracerCL() 
{
}

inline void RayTracerCL::init(cl::Platform const &platform) 
{
    cl_context_properties cps[7] = 
                                 {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),  
// TODO: Linux only!
				  CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
				  CL_GLX_DISPLAY_KHR,  (cl_context_properties) glXGetCurrentDisplay(), 
				  0 };
  /*
   * Create an OpenCL context.
   * TODO: Improved detection for GL sharing support.
   */
#ifndef RT_CL_DEVICE_TYPE
  try 
    {
      // TODO: NVidia does not define CL_GL_CONTEXT_KHR in their cl_gl headers.  Examples show this is not necessary,
      // but the spec states otherwise.
      context = cl::Context(CL_DEVICE_TYPE_GPU, cps, NULL, NULL, NULL);
    }
  catch (cl::Error err) 
    {
      std::cout << "Unable to create context for GPU device, falling back to CPU device." << std::endl;
      /*
       * CPU is rather unlikely to support GL sharing...
       */
      cps[2] = 0;
      context = cl::Context(CL_DEVICE_TYPE_CPU, cps, NULL, NULL, NULL);
      glSharing = false;
    }

#else
  context = cl::Context(RT_CL_DEVICE_TYPE, cps, NULL, NULL, NULL);
#endif /* RT_CL_DEVICE_TYPE */

  /*
   * Acquire a list of all of the devices available in the context.
   */
  std::vector<cl::Device> clDevices = context.getInfo<CL_CONTEXT_DEVICES>();
#ifdef _DEBUG_RT
  std::cout << std::endl << "OpenCL Devices: " << std::endl;
  std::vector<cl::Device>::iterator devItr;
  for (devItr = clDevices.begin(); devItr != clDevices.end(); ++devItr) 
    {
      ::printDeviceInfo(&(*devItr));
    }
#endif /* _DEBUG_RT */
  /*
   * Just pick the first device.
   *
   * TODO: allow user to specify which device to use...
   */
  if (clDevices.size() == 0) 
    {
      //TODO: this is a fatal error...

    }

  /*
   * Load CL files from disk, create a program from them, and build them for our selected device.
   */
  std::string raytracer_sources = *loadSourceFromFile(RAYTRACER_CL);
  cl::Program::Sources cl_source(1, std::make_pair(raytracer_sources.c_str(), raytracer_sources.length() + 1));
  cl::Program rtProgram(context, cl_source);


#ifdef _DEBUG_RT
  //  std::cout <<std::endl << "OpenCL Program Source (" << RAYTRACER_CL << "):" << std::endl;
  //  std::cout << "======================================" << std::endl;
  //  std::cout << raytracer_sources.c_str() << std::endl;
  //  std::cout << "======================================" << std::endl;
#endif /* _DEBUG_RT */


  /*
   * Compile the program for all devices in the list.
   * This will throw an exception if compilation fails, in which
   * case more specific information is recorded in the build log.
   *
   * Note: NVidia beta CUDA/OpenCL driver crashes during build sometimes...
   *
   * TODO: Only compile for our selected target device.
   */

  try 
    { 
      rtProgram.build(clDevices);
      std::cout << "Building Done!!!" << std::endl;
      std::string buildLog = rtProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevices[0]);
      std::cout << "Build Log: " << std::endl << buildLog << std::endl;
    }
  catch (cl::Error buildErr)
    {
      std::string buildLog = rtProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevices[0]);
      std::cerr << "Build Error: " << std::endl << buildLog << std::endl;
      throw (buildErr);
    }

  raytracer_kernel = cl::Kernel(rtProgram, "raytrace", NULL);
  
  /*
   * Create an OpenCL command queue
   */
  cmdQueue = cl::CommandQueue(context, clDevices[0], NULL /*properties...*/, NULL);

  glSharing = ::supportsGLSharing(clDevices[0]);

#ifdef _DEBUG_RT
  std::cout << "CL/GL Interoperability: " << (glSharing ? "Yes":"No") << std::endl;
#endif

  /*
   * Allocate a constant buffer to pass the camera parameters.
   *
   * Note that the camera data is transferred in the rayTrace method, so no need to populate the data now.
   */
  cameraBufferCL = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Camera));
  geomBufferSize = 0;
  sceneBufferCL = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Sphere));

}


void RayTracerCL::cameraChanged()
{
  RayTracer::cameraChanged();
  camDirty = true;
  //std::cout << "TODO: Update camera buffer in CL" << std::endl;
}

void RayTracerCL::updateCLCamera()
{
  gmtl::Vec3f view(0.0f,0.0f,-1.0f);
  gmtl::Vec3f up(0.0f, 1.0f, 0.0f);
  
  view = viewMatrix * view;
  up = viewMatrix * up;
  gmtl::Vec3f right;
  gmtl::cross(right, view, up);

  Camera cameraCL;

  cameraCL.position.x = viewMatrix(0,3);
  cameraCL.position.y = viewMatrix(1,3);
  cameraCL.position.z = viewMatrix(2,3);
  cameraCL.position.w = 0;
  
  cameraCL.up.x = up[0];
  cameraCL.up.y = up[1];
  cameraCL.up.z = up[2];
  cameraCL.up.w = 0.0f;

  cameraCL.right.x = right[0];
  cameraCL.right.y = right[1];
  cameraCL.right.z = right[2];
  cameraCL.right.w = 0.0f;

  view *= (float)((width/2.0)/tan(D2R(fovAngle)/2.0));
  cameraCL.view.x = view[0];
  cameraCL.view.y = view[1];
  cameraCL.view.z = view[2];
  cameraCL.view.w = 0;

  //Synchronous Write...
  cmdQueue.enqueueWriteBuffer(cameraBufferCL, CL_TRUE, 0, sizeof(Camera), (void *)&cameraCL);
  raytracer_kernel.setArg(CAMERA_CL_PARAM, sizeof(cl_mem), &(cameraBufferCL()));
  

  /*
   * Seed buffer
   */
  uint pixelCount = width * height * 2;
  uint seeds[pixelCount];
  for (uint i = 0; i < pixelCount ; ++i) 
      {
	seeds[i ] = rand();
	if (seeds[i] < 2)
	  seeds[i] = 2;
      }
  seedBufferCL = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(uint) * pixelCount, (void *)seeds);
  try {
  raytracer_kernel.setArg(SEEDS_CL_PARAM, sizeof(cl_mem), &(seedBufferCL()));
  } catch (cl::Error err) { std::cout <<"Err: " << err.err() << std::endl;}

#ifdef _DEBUG_RT
  //  std::cout << "Camera Changed: " << std::endl;
  //  std::cout << "\tposition: " << cameraCL.position[0] << ", " << cameraCL.position[1] << ", " << cameraCL.position[2] << std::endl;
  //  std::cout << "\tview: " << cameraCL.view[0] << ", " << cameraCL.view[1] << ", " << cameraCL.view[2] << std::endl;
  //  std::cout << "\tup: " << cameraCL.up[0] << ", " << cameraCL.up[1] << ", " << cameraCL.up[2] << std::endl;
#endif /* _DEBUG_RT */

}

void RayTracerCL::rayTrace(cl_mem *buff, uint const width, uint const height, uint const progression) 
{
  if (width == 0 || height == 0) return;
#ifdef _DEBUG_RT
  clock_t startTicks = clock();
#endif /* _DEBUG_RT   */
  /*
   * Update the camera
   */
  if (camDirty || width != this->width || height != this->height)
    {
      camDirty = false;
      this->width = width;
      this->height = height;
      updateCLCamera();
    }

  if (sceneObjects.size() > 0 && geomBufferSize != (sizeof(Sphere) * sceneObjects.size()))
    
    {
      geomBufferSize =(sizeof(Sphere) * sceneObjects.size());
      sceneBufferCL = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, geomBufferSize, (void *) &(sceneObjects.front()));
      raytracer_kernel.setArg(GEOM_CL_PARAM, sizeof(cl_mem), &(sceneBufferCL()));
      raytracer_kernel.setArg(GEOMCOUNT_CL_PARAM, (uint)sceneObjects.size());
    }
  else 
    {
      //TODO: check whether the scene content has changed but #elements did not
    }
  

  /*
   * Transfer the buffer and image size info.
   */
  raytracer_kernel.setArg(OUTBUFF_CL_PARAM, sizeof(cl_mem), buff);
  raytracer_kernel.setArg(IMWIDTH_CL_PARAM, width);
  raytracer_kernel.setArg(IMHEIGHT_CL_PARAM, height);
  raytracer_kernel.setArg(SAMPLERATE_CL_PARAM, sampleRate);
  raytracer_kernel.setArg(MAXDEPTH_CL_PARAM, maxPathDepth);
  raytracer_kernel.setArg(PROGRESSION_CL_PARAM, progression);
  //TODO: use KernelFunctor -- fixed NDRange...?
  try 
    {
      /*
       * Ensure width is a multiple of 32
       */
      int wgMultipleWidth = ((width & 0x1F)  == 0) ? width : ((width & 0xFFFFFFE0) + 0x20);
      /*
       * And height is a multiple of 6.
       *
       * TODO: NVidia QuadroFX 5800 supports 512 work-items per wg (32 x 16), however the ray tracer code uses too many registers
       * for 512 items (register file is shared among items in the same group).  6x32 is the minimum value to hide certain latencies
       * when accessing memory or read-after-write registers according to their OpenCL best-practices guide.
       */
      int wgMutipleHeight = (height / 6 + 1) * 6;
      cmdQueue.enqueueNDRangeKernel(raytracer_kernel, cl::NullRange, cl::NDRange(wgMultipleWidth, wgMutipleHeight), cl::NDRange(32,6));
      cmdQueue.finish();
#ifdef _DEBUG_RT
      std::cout << "CL Render Time: " << timeElapsed(startTicks) << "s" << std::endl;
#endif /* _DEBUG_RT */
    } 
  catch (cl::Error err) 
    {
      std::cerr << "Error submitting kernel for execution: " << err.err() << std::endl;
      std::cerr << "Global x: " << width << ", Global y: " << height << std::endl;
      throw(err);
    }

  camDirty = true;
}


const cl::Context RayTracerCL::getCLContext()  
{
  return this->context;
}

cl::Platform *RayTracerCL::getDefaultPlatform() 
{
  /*
   * Get a list of the available platforms.  This is required by the ATI Stream 2.0 final release, which
   * no longer allows selection of a default platform by passing null to functions that require it.
   */
  std::vector<cl::Platform> platforms;
  try 
    {
      cl::Platform::get(&platforms);
    }
  catch (cl::Error err)
    {
      std::cerr << "Error getting platform IDs: " << err.what() << std::endl;
      throw (err);
    }
  /*
   * Copied from AMD/ATI sample code, 
   * TODO: select an appropriate platform
   */
  std::vector<cl::Platform>::iterator i;
  // TODO: clean up un-used cl::Platform objects?
  if(platforms.size() > 0)
    {
#ifdef _DEBUG_RT
      std::cout << "Available Platforms: " << std::endl;
      for(i = platforms.begin(); i != platforms.end(); ++i)
        {
	      ::printPlatformInfo(&(*i));
        }
      std::cout << "Selected Platform: " << std::endl;
      ::printPlatformInfo(&platforms[0]);
#endif /* _DEBUG_RT */	      
      return new cl::Platform(platforms[0]);
    }
  // TODO: throw exception if out not initialized...
  return NULL;
}


/*!
 * Returns a cl::Program::Sources vector containing the contents of the specified file.
 */
std::string *RayTracerCL::loadSourceFromFile(const char *filename) 
{
  std::ifstream kernel_file(filename);
  if (!kernel_file.is_open()) 
    {
#ifdef _DEBUG_RT
      std::cerr << "Unable to open file: " << filename << std::endl;
#endif
      return NULL;
    }
  return new std::string(std::istreambuf_iterator<char>(kernel_file), 
			 (std::istreambuf_iterator<char>()));
}

