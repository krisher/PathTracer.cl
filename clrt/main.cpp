/*!
 *
 *\file main.cpp
 *
 */
#include <GL/glut.h>
#include "GlutCLWindow.h"
#include "gmtl/gmtl.h"
#include "gmtl/Point.h"
#include "geometry.h"

int main(void) 
{
  GlutCLWindow window(512,512);

  Sphere sphere;
  initSphere(sphere);
  sphere.diffuse.x = 0.0f;
  sphere.diffuse.y = 0.7f;
  sphere.diffuse.z = 0.7f;
  sphere.diffuse.w = 1.0f;
  sphere.center.y = -4.0f;
  sphere.center.x = -2.0f;
  sphere.center.z = -2.0f;
  sphere.ks = 0.f;
  sphere.radius = 1.0f;
  window.rayTracer.addSphere(sphere);

//  sphere.diffuse.x = 0.7f;
//  sphere.diffuse.y = 0.7f;
//  sphere.diffuse.z = 0.0f;
//  sphere.diffuse.w = 1.0f;
//  sphere.center.y = -4.0f;
//  sphere.center.x = 2.0f;
//  sphere.center.z = 2.0f;
//  sphere.ks = 0.f;
//  sphere.radius = 1.0f;
//  window.rayTracer.addSphere(sphere);
//
  sphere.diffuse.x = 0.0f;
  sphere.diffuse.y = 0.0f;
  sphere.diffuse.z = 0.0f;
  sphere.diffuse.w = 0.0f;
  sphere.center.y = -4.0f;
  sphere.center.x = 2.0f;
  sphere.center.z = 2.0f;
  sphere.ks = 0.0f;
  sphere.ior = 1.4f;
  sphere.extinction.x = 0.95f;
  sphere.extinction.y = 0.65f;
  sphere.extinction.z = 0.65f;
  sphere.extinction.w = 1.0f;
  sphere.specExp = 100.0f;
  sphere.radius = 1.0f;
  window.rayTracer.addSphere(sphere);



  sphere.diffuse.x = 0.8f;
  sphere.diffuse.y = 0.8f;
  sphere.diffuse.z = 0.8f;
  sphere.diffuse.w = 0.0f;
  sphere.center.y = -4.0f;
  sphere.center.x = 0.0f;
  sphere.center.z = 0.0f;
  sphere.ks = 1.0f;
  sphere.specExp = 1000000.0f;
  sphere.radius = 1.0f;
  window.rayTracer.addSphere(sphere);

  sphere.diffuse.x = 0.8f;
  sphere.diffuse.y = 0.8f;
  sphere.diffuse.z = 0.9f;
  sphere.diffuse.w = 0.2f;
  sphere.center.y = -4.0f;
  sphere.center.x = 2.0f;
  sphere.center.z = -2.0f;
  sphere.ks = 0.8f;
  sphere.specExp = 100.0f;
  sphere.radius = 1.0f;
  window.rayTracer.addSphere(sphere);


  sphere.diffuse.x = 0.7f;
  sphere.diffuse.y = 0.0f;
  sphere.diffuse.z = 0.8f;
  sphere.diffuse.w = 0.6f;
  sphere.center.y =  -4.0f;
  sphere.center.x = -2.0f;
  sphere.center.z =  2.0f;
  sphere.ks = 0.4f;
  sphere.specExp = 1000.0f;
  sphere.radius = 1.0f;
  window.rayTracer.addSphere(sphere);



  sphere.center.x = 0.0f;
  sphere.center.y = 4.0f;
  sphere.center.z = 2.0f;
  sphere.diffuse.w = 0.0;
  sphere.emission.x = 1.0f;
  sphere.emission.y = 1.0f;
  sphere.emission.z = 1.0f;
  sphere.emission.w = 1.0;
  sphere.radius = 0.5f;
  sphere.ks = 0.0f;
  sphere.extinction.w = 0;
  window.rayTracer.addSphere(sphere);

  window.rayTracer.setSampleRate(2);
  window.rayTracer.setMaxPathDepth(6);
  window.rayTracer.setCameraSpherical(gmtl::Point3f(0,-4,-0), 40.0f, 105.0f, 6);
  window.setProgressive(10000);

  glutMainLoop();

}
