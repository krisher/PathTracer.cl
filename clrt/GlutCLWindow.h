/*!
 * \file GlutCLWindow.h
 * \brief Extension of GlutWindow to draw OpenCL buffer in GL
 *
 * 
 */

#include "GlutWindow.h"
#include "RayTracerCL.h"

class GlutCLWindow : public GlutWindow
{

 private:

  GLuint pbo;
  /*!
   *\brief CL Buffer handle used when GL Sharing is enabled.
   */
  cl_mem clPBO;
  /*!
   *\brief CL Buffer handle used when GL Sharing is NOT enabled.
   */
  cl::Buffer *clPBOBuff;
  /*!
   * \brief Flag indicating whether the PBO contains the rendered scene (i.e. do we need to re-trace with the next display call).
   */
  bool reallocPBO;
  /*!
   * \brief The number of times the current frame has been refined with progressive sampling.
   */
  uint progression;
  uint maxProgression;
  
  //TODO: These should be accessible from the camera.
  float azimuth;
  float elevation;
  float distance;

  void allocatePBO();
  void rayTrace();

 public:
  /*!
   * \brief Ray Tracer
   */
  RayTracerCL rayTracer;


  GlutCLWindow(int width, int height);
  ~GlutCLWindow();
  virtual void glutDisplayCallback();
  virtual void glutReshapeCallback(int w, int h);
  virtual void glutSpecialKeypressCallback(int key, int x, int y);

  void setProgressive(uint const maxPasses) {maxProgression = maxPasses;};
  uint const isProgressive() {return maxProgression;};
};
