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


void init_material(material_t *mat) {
    mat->diffuse.x = 0.0f;
    mat->diffuse.y = 0.0f;
    mat->diffuse.z = 0.0f;
    mat->kd = 0.0f;

    mat->extinction.x = 0.0f;
    mat->extinction.y = 0.0f;
    mat->extinction.z = 0.0f;
    mat->kt = 0.0f;

    mat->emission.x = 0.0f;
    mat->emission.y = 0.0f;
    mat->emission.z = 0.0f;
    mat->emission_power = 0.0f;

    mat->ks = 0.0f;
    mat->specExp = 1000000.0f;
    mat->ior = 1.0f;
    mat->refExp = 1000000.0f;
}

void init_sphere(Sphere *sphere) {
  sphere->center.x = 0.0f;
  sphere->center.y = 0.0f;
  sphere->center.z = 0.0f;
  sphere->radius = 1.0f;

  init_material(&(sphere->mat));
}

int main(void) {
    GlutCLWindow window(512, 512);

    Sphere sphere;
    init_sphere(&sphere);
    sphere.center.y = -4.0f;
    sphere.center.x = -2.0f;
    sphere.center.z = -2.0f;
    sphere.mat.kd = 1.0f;
    sphere.mat.diffuse.x = 0.0f;
    sphere.mat.diffuse.y = 0.7f;
    sphere.mat.diffuse.z = 0.7f;
    window.rayTracer.addSphere(sphere);

    init_sphere(&sphere);
    sphere.center.y = -3.0f;
    sphere.center.x = 2.0f;
    sphere.center.z = 2.0f;
    sphere.mat.ks = 0.2f;
    sphere.mat.kt = 0.8f;
    sphere.mat.extinction.x = 0.99f;
    sphere.mat.extinction.y = 0.95f;
    sphere.mat.extinction.z = 0.95f;
    sphere.mat.ior = 1.1f;
    window.rayTracer.addSphere(sphere);

    init_sphere(&sphere);
    sphere.center.y = -4.0f;
    sphere.center.x = 0.0f;
    sphere.center.z = 0.0f;
    sphere.mat.ks = 1.0f;
    window.rayTracer.addSphere(sphere);

    init_sphere(&sphere);
    sphere.center.y = -4.0f;
    sphere.center.x = 2.0f;
    sphere.center.z = -2.0f;
    sphere.mat.kd = 0.2f;
    sphere.mat.ks = 0.8f;
    sphere.mat.diffuse.x = 0.7f;
    sphere.mat.diffuse.y = 0.7f;
    sphere.mat.diffuse.z = 0.0f;
    sphere.mat.specExp = 100.0f;
    window.rayTracer.addSphere(sphere);

    init_sphere(&sphere);
    sphere.center.y = -4.0f;
    sphere.center.x = -2.0f;
    sphere.center.z = 2.0f;
    sphere.mat.kd = 0.6f;
    sphere.mat.ks = 0.4f;
    sphere.mat.diffuse.x = 0.7f;
    sphere.mat.diffuse.y = 0.0f;
    sphere.mat.diffuse.z = 0.8f;
    sphere.mat.specExp = 1000.0f;
    window.rayTracer.addSphere(sphere);

    init_sphere(&sphere);
    sphere.center.x = 2.2f;
    sphere.center.y = 1.0f;
    sphere.center.z = 2.0f;
    sphere.radius = 0.5f;
    sphere.mat.emission_power = 1.0;
    sphere.mat.emission.x = 1.8f;
    sphere.mat.emission.y = 1.8f;
    sphere.mat.emission.z = 1.8f;
    window.rayTracer.addSphere(sphere);

//    init_sphere(&sphere);
//    sphere.center.x = 1.0f;
//    sphere.center.y = 3.0f;
//    sphere.center.z = -2.0f;
//    sphere.radius = 0.5f;
//    sphere.mat.emission_power = 1.0;
//    sphere.mat.emission.x = 0.4f;
//    sphere.mat.emission.y = 0.1f;
//    sphere.mat.emission.z = 0.6f;
//    window.rayTracer.addSphere(sphere);

    window.rayTracer.setSampleRate(1);
    window.rayTracer.setMaxPathDepth(6);
    window.rayTracer.setCameraSpherical(gmtl::Point3f(0, -4, -0), 14.0f,
            118.0f, 5);
    window.setProgressive(20000);

    glutMainLoop();

}
