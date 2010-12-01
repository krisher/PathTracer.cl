/*!
 * \file GlutCLWindow.cpp
 * \brief Glut window rendering of OpenCL output
 *
 */
#define GL_GLEXT_PROTOTYPES

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include "GlutCLWindow.h"

#include <iostream>
#include <time.h>
#include <string>
#include <sstream>

GlutCLWindow::GlutCLWindow(int width, int height) :
    GlutWindow(width, height, "OpenCL Ray Tracer") {
    /*
     * GL Context must be current for GL sharing to work...
     */
    //  rayTracer = RayTracerCL();
    azimuth = 105.0f;
    elevation = 40.0f;
    distance = 5.0f;
    rayTracer.setCameraSpherical(gmtl::Point3f(0, -4, -1), elevation, azimuth,
            5);
    maxProgression = 10000;
    pbo = NULL;
    clPBOBuff = NULL;
    frame_counter = 0;
    fps = 0;
    allocatePBO();
}

GlutCLWindow::~GlutCLWindow() {
    // TODO: delete pbo, cl pbo.
    // Need GL Context current to delete, however since context is not shared, pbo should be destroyed when glut window is closed.
}

void drawString(float x, float y, const std::string &str) {
    glRasterPos2f(x, y);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    for (std::string::const_iterator itr = str.begin(); itr != str.end(); itr++) {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *itr); //Automatically shifts matrix position for next char.
    }
}

void GlutCLWindow::allocatePBO() {
    // TODO: use FBO/RenderBuffer instead of PBO...
    if (pbo) {
        /*
         * Free existing PBO, we probably resized...
         *
         * Also need to remove reference from OpenCL to this buffer...
         */
        if (rayTracer.supportsGLSharing())
            clReleaseMemObject(clPBO);
        else if (clPBOBuff)
            delete clPBOBuff;
        glDeleteBuffers(1, &pbo);
    }

    /*
     * Allocate a new OpenGL PBO with our required dimension,
     * and no data.  Use stream draw since we will need to update
     * the image every time the view or scene changes.
     */
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    std::cout << "Allocating PBO with " << width * height * sizeof(GLfloat) * 4
            << " bytes" << std::endl;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLfloat) * 4,
            0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    /*
     * Bind the new PBO to the OpenCL output buffer for the ray tracer...
     *
     * It does not look like the C++ bindings for OpenCL support this call,
     * so we can manage this one buffer explicitly...
     */
    if (rayTracer.supportsGLSharing()) {
        int err = 0;
        clPBO = clCreateFromGLBuffer((rayTracer.getCLContext())(),
                CL_MEM_READ_WRITE, pbo, &err);
        //std::cout << "Created shared CL/GL buffer." << std::endl;
        if (err) {
            std::cerr << "Error binding OpenGL PBO to OpenCL Buffer (Error "
                    << err << "), falling back to failsafe method."
                    << std::endl;
            std::cerr << "PBO ID: " << pbo << std::endl;
            std::cerr << "CL Ctx: " << (rayTracer.getCLContext())()
                    << std::endl;
        }
    } else {
        try {
            clPBOBuff = new cl::Buffer((rayTracer.getCLContext()),
                    CL_MEM_WRITE_ONLY, width * height * sizeof(GLfloat) * 4);
            clPBO = (*clPBOBuff)();
        } catch (cl::Error err) {
            std::cerr << "Error allocating CL PBO: " << err.what() << " ("
                    << err.err() << ")" << std::endl;
            throw(err);
        }
    }
    /*
     * TODO: update kernel binding for output buffer
     * TODO: Check err code.
     */
    //std::cout << "Allocated output buffer." << std::endl;
}

void GlutCLWindow::glutReshapeCallback(int w, int h) {
    /*
     * TODO: w and h must be divisible by wg size, ideally these will always be multiples of 32.
     */
    GlutWindow::glutReshapeCallback(w, h);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    /*
     * Set up an ortho-projection such that the bottom/left corner
     * of the image plane is 0,0.
     */
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    reallocPBO = true;
    last_render_end = clock();
}

void GlutCLWindow::glutDisplayCallback() {
    /*
     * Only render the scene to the PBO if the PBO has not been populated yet.
     *
     * TODO: also need to check for changes in the scene content and the camera position, however
     * with animated or interactive scenes we can just always render.  This should be fast enough
     * at some point so that won't be a problem.
     */
    if (reallocPBO) {
        allocatePBO();
        progression = 0;
        rayTrace();
        reallocPBO = false;
        if (maxProgression > 0)
            glutPostRedisplay();
    } else if (progression < maxProgression) {
        /*
         * Progressive refinement, take more samples and accumulate the values into the PBO.
         */
        progression++;
        rayTrace();
        glutPostRedisplay();
    }
    /*
     * Draw PBO to screen as a full-window image.
     */
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (reportFPS) {
        if (++frame_counter == fpsAvgFrames) {
            clock_t time = clock();
            fps = CLOCKS_PER_SEC / ((time - last_render_end)
                    / (double) fpsAvgFrames);
            frame_counter = 0;
            last_render_end = time;
        }
        std::ostringstream fpsStr;
        fpsStr << "FPS: ";
        fpsStr << fps;
        drawString(0.0f + 20.0f / width, 1.0f - 20.0f / height, fpsStr.str());
    }
    /*
     * Call glFinish() and
     * Flip the back/front buffer
     */
    glutSwapBuffers();

}

void GlutCLWindow::rayTrace() {
    // TODO: Move rayTrace into RayTracerCL, support pluggable output buffer...

    /*
     * Take control of the PBO from OpenGL
     */
    if (rayTracer.supportsGLSharing()) {
        // TODO: C++ bindings do not have good support for this...
        clEnqueueAcquireGLObjects(rayTracer.getCLCommandQueue()(), 1, &clPBO,
                0, 0, 0);
    }

    /*
     * TODO: w and h must be divisible by wg size, ideally these will always be multiples of 32.
     */
    rayTracer.rayTrace(&clPBO, width, height, progression);

    if (rayTracer.supportsGLSharing()) {
        /*
         * Rendered image is already in pbo, just return control
         * to OpenGL for display.
         */
        clEnqueueReleaseGLObjects(rayTracer.getCLCommandQueue()(), 1, &clPBO,
                0, 0, 0);
    } else {
        /*
         * Map the GL PBO into client memory and copy the
         * CL results into it.
         */
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        GLfloat* ptr = (GLfloat*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER,
                GL_WRITE_ONLY);
        // Blocking read...
        rayTracer.getCLCommandQueue().enqueueReadBuffer(*clPBOBuff, CL_TRUE, 0,
                width * height * sizeof(GLfloat) * 4, ptr);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }
}

void GlutCLWindow::glutSpecialKeypressCallback(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_LEFT: {
        azimuth = fmod((azimuth + 3.0f), 360.0f);
        rayTracer.setCameraSpherical(gmtl::Point3f(0, -4, 0), elevation,
                azimuth, distance);
        progression = 0;
        break;
    }
    case GLUT_KEY_RIGHT: {
        azimuth = fmod((azimuth - 3.0f), 360.0f);
        rayTracer.setCameraSpherical(gmtl::Point3f(0, -4, 0), elevation,
                azimuth, distance);
        progression = 0;
        break;
    }
    case GLUT_KEY_UP: {
        elevation = fmin(elevation + 3.0f, 90.0f);
        rayTracer.setCameraSpherical(gmtl::Point3f(0, -4, 0), elevation,
                azimuth, distance);
        progression = 0;
        break;
    }
    case GLUT_KEY_DOWN: {
        elevation = fmax(elevation - 3.0f, 10.0f);
        rayTracer.setCameraSpherical(gmtl::Point3f(0, -4, 0), elevation,
                azimuth, distance);
        progression = 0;
        break;
    }
    }
}
