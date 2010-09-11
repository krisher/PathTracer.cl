/*!
 * \file RayTracer.h
 * \brief Encapsulation of an OpenCL based ray-tracer.
 */
#ifndef _RAYTRACER_H
#define _RAYTRACER_H


#include "geometry.h"

#include "gmtl/gmtl.h"
#include "gmtl/Matrix.h"
#include "gmtl/Point.h"


class RayTracer
{
 private:
  
 protected:
  //Camera Model
  /*!
   * \brief The field of view angle (horizontal) in degrees.
   */
  float fovAngle;
  /*!
   * \brief The viewing transform matrix.
   */
  gmtl::Matrix44f viewMatrix;

  //Ray Tracing settings
  /*!
   * \brief The number of samples per linear pixel (total number of samples per pixel is this squared)
   */
  uint sampleRate;
  /*!
   * \brief The maximum number of bounces before a ray path is terminated.
   */
  uint maxPathDepth;


  //TODO: use C++ abstractions, should not care about CL here...
  std::vector<Sphere> sceneObjects;

  /*!
   * \brief Callback to notify sub-classes that the camera has been updated.
   */
  virtual void cameraChanged();
  
 public:
  RayTracer();

  /*!
   * Specifies the camera transformation matrix directly.
   */
  void setCameraMatrix(gmtl::Matrix44f const &mat);
  void setCameraSpherical(gmtl::Point3f const &target, float elevationDeg, float azimuthDeg, float distance);

  void setFoVAngle(float fovDeg) {fovAngle = fovDeg; cameraChanged();};
  float getFoVAngle() {return fovAngle;};

  void setSampleRate(uint sampleRate) {this->sampleRate = sampleRate;};
  uint getSampleRate() { return sampleRate;};
  
  void setMaxPathDepth(uint depth) {this->maxPathDepth = depth;};
  uint getMaxPathDepth() { return maxPathDepth;};

  void addSphere(Sphere const &sphere);
  void removeSphere(Sphere const &sphere);
  void clearSpheres();
};

#endif /* _RAYTRACER_H */
