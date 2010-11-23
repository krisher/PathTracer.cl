
#ifndef _PLATFORM_INFO_HPP
#define _PLATFORM_INFO_HPP

/*
 * Enable C++ exceptions
 */
//#define __CL_ENABLE_EXCEPTIONS

namespace cl {

  class Device;
  class Platform;
  class Kernel;
  template <int N> struct size_t;
};


void printDeviceInfo(cl::Device * device);
void printPlatformInfo(cl::Platform * platform);
bool supportsGLSharing(cl::Device &device);
size_t getDeviceMaxWGSize(const cl::Device *device);
size_t getKernelMaxWGSize(const cl::Device *device, const cl::Kernel *kernel);


#endif /* _PLATFORM_INFO_HPP */
