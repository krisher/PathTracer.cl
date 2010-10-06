/*! 
 * \file PlatformInfo.cpp
 * \brief Simple CL app to query platform info
 */
#include <utility>

#include <CL/cl.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

#include "PlatformInfo.h"

void printPlatformInfo(cl::Platform * platform)
{
  std::cout << "Platform: " << std::endl;
  std::string infoStr;
  /*
   * Print some information regarding OpenCL support by this
   * platform
   */
  platform->getInfo(CL_PLATFORM_VERSION, &infoStr);
  std::cout << "\tOpenCL version: " << infoStr << std::endl; 
  infoStr.erase();
  
  platform->getInfo(CL_PLATFORM_PROFILE, &infoStr);
  std::cout << "\tProfile: " << infoStr << std::endl; 
  infoStr.erase();      

  platform->getInfo(CL_PLATFORM_EXTENSIONS, &infoStr);
  std::cout << "\tExtensions: " << infoStr << std::endl; 
  infoStr.erase();      
}



void devTypeStr(cl_device_type type, std::string *out) 
{
  if (type & CL_DEVICE_TYPE_GPU) out->append( "GPU ");
  else if (type & CL_DEVICE_TYPE_CPU) out->append( "CPU ");
  else if (type & CL_DEVICE_TYPE_ACCELERATOR) out->append( "ACCELERATOR ");
  else out->append("Unknown Device Type ");

  if (type & CL_DEVICE_TYPE_DEFAULT) out->append( "*DEFAULT ");
}

void printDeviceInfo(cl::Device * device) 
{
  cl_device_type devType;
  std::string infoStr;
  cl_uint intData;
  size_t sizeData;

  device->getInfo(CL_DEVICE_NAME, &infoStr);
  std::cout << "Device: " << infoStr << std::endl;
  infoStr.clear();

  device->getInfo(CL_DEVICE_TYPE, &devType);
  devTypeStr(devType, &infoStr);
  std::cout << "\tType: " << infoStr << "(" << devType << ")" << std::endl;


  device->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &intData);
  std::cout << "\tMax Compute Units: " << intData << std::endl;
  
  device->getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &intData);
  std::cout << "\tMax Work Item Dimensions: " << intData << std::endl;

  std::vector<size_t> maxWorkItemSizes;
  device->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSizes);
  std::cout << "\tMax Work Item Sizes (per dimension, per work group): ";
  for (uint i=0; i < maxWorkItemSizes.size(); i++) {
    std::cout << maxWorkItemSizes[i] << ", ";
  }
  std::cout << std::endl;

  device->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &sizeData);
  std::cout << "\tMax Work Group Size: " << sizeData << std::endl;

  device->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &intData);
  std::cout << "\tMax Clock Freq: " << intData << "MHz" << std::endl;

  device->getInfo(CL_DEVICE_EXTENSIONS, &infoStr);
  std::cout << "\tExtensions: " << infoStr << std::endl;

  bool supportsGL = supportsGLSharing(*device);
  std::cout << "\tGL Sharing supported: " << ((supportsGL) ? "yes" : "no") << std::endl;
}


bool supportsGLSharing(cl::Device &device)
{
  std::string extStr;
  device.getInfo(CL_DEVICE_EXTENSIONS, &extStr);
  return extStr.find("cl_khr_gl_sharing") != std::string::npos;
}
