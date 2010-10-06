
#ifndef _PLATFORM_INFO_HPP
#define _PLATFORM_INFO_HPP

/*
 * Enable C++ exceptions
 */
//#define __CL_ENABLE_EXCEPTIONS

namespace cl {
 
  class Device;
  class Platform;
};

void printDeviceInfo(cl::Device * device);
void printPlatformInfo(cl::Platform * platform);
bool supportsGLSharing(cl::Device &device);


#endif /* _PLATFORM_INFO_HPP */
