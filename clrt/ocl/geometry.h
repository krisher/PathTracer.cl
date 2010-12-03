/*!
 * \file geometry.h
 * \brief Geometry structs and functions used by both OpenCL kernels and host code.
 */
#ifndef _GEOMETRY_H
#define _GEOMETRY_H

#ifdef __cplusplus
#include <CL/cl.h>

/*
 * Typedefs for float4 taken from ATI cl_platform.h v2.01.
 *
 * These definitions differ from the standard khronos definitions
 */
#if defined( __SSE__ )
#include <xmmintrin.h>
#if defined( __GNUC__ )
typedef float __cl_float4 __attribute__((vector_size(16)));
#else
typedef __m128 __cl_float4;
#endif
#define __CL_FLOAT4__   1
#endif
#if defined( __GNUC__ )
#define CL_ALIGNED(_x)      __attribute__ ((aligned(_x)))
#elif defined( _WIN32) && (_MSC_VER)
#include <crtdefs.h>
#define CL_ALIGNED(_x)         _CRT_ALIGN(_x)
#else
#warning  Need to implement some method to align data here
#define  CL_ALIGNED(_x)
#endif

typedef union {
    cl_float CL_ALIGNED(16) s[4];
#if defined( __GNUC__) && ! defined( __STRICT_ANSI__ )
    __extension__
    struct {
        cl_float x, y, z, w;
    };__extension__
    struct {
        cl_float s0, s1, s2, s3;
    };__extension__
    struct {
        cl_float2 lo, hi;
    };
#endif
#if defined( __CL_FLOAT2__)
    __cl_float2 v2[2];
#endif
#if defined( __CL_FLOAT4__)
    __cl_float4 v4;
#endif
} float4;
#endif /* __cplusplus */

/* Small floating point number used to offset ray origins to avoid roundoff error issues. */
#define SMALL_F 1e-4f
/*!
 * \brief 3D vector storage
 */
typedef struct {
    float x;
    float y;
    float z;
} vec3;

/*!
 * \brief Ray structure used in OpenCL.
 */
typedef struct {
    vec3 o;
    vec3 d;
    float tmin;
    float tmax;

    vec3 propagation;
    vec3 extinction;

    uint diffuse_bounce;

} ray_t;

/*!
 * \brief triangle geometry, a vertex and two edges.
 */
typedef struct {
    /*! The base vertex of the triangle. */
    vec3 v0;
    /*! Vector from the base vertex to the next triangle vertex. */
    vec3 e1;
    /*! Vector from the base vertex to the last triangle vertex. */
    vec3 e2;
} triangle_t;

/*!
 * \brief Struct to store ray/surface intersection information.
 */
typedef struct {
    /*!
     * \brief The location where an intersection occurred.
     */
    vec3 hit_pt;
    /*!
     * \brief The surface normal at the intersection point.
     */
    vec3 surface_normal;
} hit_info_t;

/*!
 * \brief camera model.
 *
 * up, right are normalized, view has the magnitude necessary so that a ray 
 * passing from position through ( up * (pixelY - height/2.0) + right * (pixelX - width/2.0) + view
 * produces the desired field of view.
 */
typedef struct {
    float4 view;
    float4 up;
    float4 right;
    float4 position;
} Camera;

/*!
 * Material parameters.
 *
 * TODO: precision can be reduced below the current 32bit per component values for all properties here if bandwidth becomes an issue.
 */
typedef struct { //64B
    /*! Diffuse reflectance */
    vec3 diffuse;
    /*! probability of diffuse reflectance */
    float kd;
    /*! extinction color (amount of each color component transmitted per unit distance) */
    vec3 extinction;
    /*! probability of transmission */
    float kt;
    /*! emission color (light source) */
    vec3 emission;
    /*! emission power, must be non-zero if this is an emissive material. */
    float emission_power;
    /*! probability of specular reflection */
    float ks;
    /*! specular exponent */
    float specExp;
    /*! index of refraction */
    float ior;
    /*! refractive exponent. */
    float refExp;
} material_t;

/*!
 * Sphere geometry, with material properties for diffuse, specular, refractive, and emissive components.
 */
typedef struct {
    /*! \brief sphere material */
    material_t mat;
    /*! \brief Center point of the sphere.  */
    vec3 center;
    /*! \brief Sphere radius.  */
    float radius;
} Sphere;


#ifdef __cplusplus
#undef float4
#endif /* __cplusplus */
#endif /* _GEOMETRY_H */
