/*!
 * \file GlutCLWindow.h
 * \brief Extension of GlutWindow to draw OpenCL buffer in GL
 *
 * 
 */

#include "GlutWindow.h"
#include "RayTracerCL.h"
#include <time.h>

class GlutCLWindow: public GlutWindow {

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
    /*!
     * \brief the progression at which to stop rendering.
     */
    uint maxProgression;

    /*!
     * \brief The x and y coordinates of the last button press to trigger motion.
     */
    unsigned int drag_start[2];

    //TODO: These should be accessible from the camera.
    float azimuth;
    float elevation;
    float distance;

    timespec last_render_end;
    unsigned int frame_counter;
    double fps;


    static const bool reportFPS = true;
    static const unsigned int fpsAvgFrames = 20;

    void allocatePBO();
    void rayTrace();
    void restart();

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
    virtual void glutKeypressCallback(unsigned char key, int x, int y);
    virtual void glutMotionCallback(int x, int y);
    virtual void glutMouseCallback(int button, int state, int x, int y);

    void setProgressive(uint const maxPasses) {
        maxProgression = maxPasses;
    }

    uint const isProgressive() {
        return maxProgression;
    }
};
