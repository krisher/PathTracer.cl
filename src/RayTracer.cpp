/*!
 * \file RayTracer.cpp
 */

#include <iostream>

#include "RayTracer.h"
#include "gmtl/Xforms.h"
#include "gmtl/EulerAngle.h"
#include <math.h>

#include <iostream>

#define D2R(x) (x * M_PI/180.0f)

RayTracer::RayTracer() 
{

  setFoVAngle(53.0f);
  sampleRate = 8;
  maxPathDepth = 4;
}



void RayTracer::setCameraMatrix(gmtl::Matrix44f const &mat) 
{
  viewMatrix = mat;
  cameraChanged();
}


void RayTracer::setCameraSpherical(gmtl::Point3f const &target, float elevationDeg, float azimuthDeg, float distance)
{
  /*
   * 
   */
  gmtl::Quatf rotation;
  gmtl::setRot(rotation, gmtl::EulerAngle<float, gmtl::ZYX>(0.0f, -D2R(azimuthDeg) + M_PI, -D2R(elevationDeg)));
  gmtl::Vec3f position(0.0f, 0.0f, distance);
  position *= rotation;

  gmtl::setRot(viewMatrix, rotation);
  gmtl::setTrans(viewMatrix, position + target);

  cameraChanged();
}


void RayTracer::addSphere(Sphere const &sphere)
{
  sceneObjects.push_back(sphere);
}

void RayTracer::removeSphere(Sphere const &sphere)
{
  //TODO: Implement remove sphere.
}

void RayTracer::clearSpheres()
{
  sceneObjects.clear();
}


void RayTracer::cameraChanged() 
{
}




