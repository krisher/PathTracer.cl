/*!
 * \file GlutWindow.cpp
 * \brief Basic GLUT window wrapper... 
 *
 */

#include <GL/glut.h>
#include "GlutWindow.h"

#include <iostream>

bool GlutWindow::glutInitialized = false;

GlutWindow * GlutWindow::glutWindows[MAX_GLUT_WINDOWS];

inline GlutWindow *GlutWindow::getGlutWindow() {
    int gWindowID = glutGetWindow();
    if (gWindowID < MAX_GLUT_WINDOWS && gWindowID >= 0
            && glutWindows[gWindowID] != NULL) {
        return glutWindows[gWindowID];
    } else {
        std::cerr << "Invalid GLUT Window ID: " << gWindowID << std::endl;
        return NULL;
    }
}

void GlutWindow::initGlut() {
    if (!glutInitialized) {
        char* argv[1] = { "" };
        int argc = 1;
        glutInit(&argc, argv);
        GlutWindow::glutInitialized = true;
    }
}

void GlutWindow::glutWindowDisplay() {
    getGlutWindow()->glutDisplayCallback();
}

void GlutWindow::glutWindowReshape(int w, int h) {
    getGlutWindow()->glutReshapeCallback(w, h);
}

void GlutWindow::glutKeypress(unsigned char key, int x, int y) {
    getGlutWindow()->glutKeypressCallback(key, x, y);
}
void GlutWindow::glutSpecialKeypress(int key, int x, int y) {
    getGlutWindow()->glutSpecialKeypressCallback(key, x, y);
}

void GlutWindow::glutMotion(int x, int y) {
    getGlutWindow()->glutMotionCallback(x, y);
}
void GlutWindow::glutMouse(int button, int state, int x, int y) {
    getGlutWindow()->glutMouseCallback(button, state, x, y);
}

GlutWindow::GlutWindow(unsigned int const width, unsigned int const height,
        char const * title) {
    this->width = width;
    this->height = height;

    initGlut();
    /*
     * Set up a basic window.  This is not very generic, but
     * exactly what is needed to render a raster image transferred via
     * OpenCL.
     */
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(glutGet(GLUT_SCREEN_WIDTH) / 2 - width / 2, glutGet(
            GLUT_SCREEN_HEIGHT) / 2 - height / 2);
    glutWindowID = glutCreateWindow(title);
    glutWindows[glutWindowID] = this;

    glutDisplayFunc(glutWindowDisplay);
    glutReshapeFunc(glutWindowReshape);
    glutKeyboardFunc(glutKeypress);
    glutSpecialFunc(glutSpecialKeypress);
    glutMouseFunc(glutMouse);
}

GlutWindow::~GlutWindow() {
    glutDestroyWindow(glutWindowID);
    GlutWindow::glutWindows[glutWindowID] = NULL;
}

void GlutWindow::glutReshapeCallback(int w, int h) {
    width = w;
    height = h;
}

void GlutWindow::enableMouseMotion() {
    glutMotionFunc(glutMotion);
}

void GlutWindow::disableMouseMotion() {
    glutMotionFunc(NULL);
}

void GlutWindow::glutKeypressCallback(unsigned char key, int x, int y) {}
void GlutWindow::glutSpecialKeypressCallback(int key, int x, int y) {}
void GlutWindow::glutMouseCallback(int button, int type, int x, int y) {}
void GlutWindow::glutMotionCallback(int x, int y) {}

