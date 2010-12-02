/*!
 * \file GlutWindow.h
 * \brief Basic GLUT window wrapper... 
 *
 * This provides a very simple Object-Oriented wrapper for a GLUT window,
 * including callbacks for the various glut functionality.
 *
 * This is very loosely based on GlutMaster (http://www.stetten.com/george/glutmaster/glutmaster.html)
 */

#ifndef _GLUT_WINDOW_HPP
#define _GLUT_WINDOW_HPP

#define MAX_GLUT_WINDOWS 128

class GlutWindow {

private:
    static bool glutInitialized;
    /*!
     *\brief Static initializer for GLUT
     */
    static void initGlut();

    static GlutWindow *glutWindows[];
    static inline GlutWindow *getGlutWindow();

    /*!
     *\brief GLUT display callback funciton dispatcher.
     */
    static void glutWindowDisplay();
    /*!
     *\brief GLUT reshape callback funciton dispatcher.
     */
    static void glutWindowReshape(int w, int h);
    /*!
     * \brief GLUT keyboard callback function dispatcher.
     */
    static void glutKeypress(unsigned char key, int x, int y);
    /*!
     * \brief GLUT special keyboard callback function dispatcher.
     */
    static void glutSpecialKeypress(int key, int x, int y);

    static void glutMotion(int x, int y);
    static void glutMouse(int button, int state, int x, int y);

    bool mouse_motion_registered;

protected:
    /*!
     *\brief GLUT Window ID for this window instance.
     */
    int glutWindowID;
    int width;
    int height;

    void enableMouseMotion();
    void disableMouseMotion();

public:
    GlutWindow(unsigned int const width, unsigned int const height, char const * title);
    ~GlutWindow();

    virtual void glutDisplayCallback() = 0;
    virtual void glutMotionCallback(int x, int y);
    virtual void glutMouseCallback(int button, int state, int x, int y);
    virtual void glutReshapeCallback(int w, int h);
    virtual void glutKeypressCallback(unsigned char key, int x, int y);
    virtual void glutSpecialKeypressCallback(int key, int x, int y);

};

#endif /* _GLUT_WINDOW_HPP */
