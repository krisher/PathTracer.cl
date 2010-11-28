/*!
 * \file materials.h
 * \brief OpenCL material sampling functions
 */
#ifndef MATERIAL_H
#define MATERIAL_H

#include "geometryFuncs.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#define M_1_PI_F 0.318309886f
#endif /* M_PI_F */

#define M_2PI_F 6.283185307f
/*!
 * \brief Generates a vector sampled from the hemisphere surrounding the positive z axis, with a cosine distribution given two uniform random numbers.
 *
 * \return 4 floats, the vector (x,y,z), and the probability of the sample in the w component.
 */
float4 cosSampleHemisphere(const float r1, const float r2) {
    /*
     * Probability of direction Ko = 1/pi * cos(theta) where theta is the
     * angle between the surface normal and Ko.
     *
     * The polar angle about the normal is chosen from a uniform distribution
     * 0..2pi
     */
    const float cos_theta = sqrt(1.0f - r1);
    const float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    const float phi = M_2PI_F * r2;

    return (float4) (sin_theta * cos(phi), sin_theta * sin(phi), cos_theta, M_1_PI_F
            * cos_theta);
}

/*!
 * \brief Change of basis function to convert from shading coordinates (z => shading normal, x,y in tangent plane) to world coordinates, given a shading normal vector.
 *
 * \param result The vector to store the result in.
 * \param shading_normal The vector representing the world-space shading normal (will be mapped to s_z)
 *
 */
vec3 shading_to_world(const vec3 shading_normal, float s_x, float s_y,
        float s_z) {
    const vec3 tangentX = perpendicular_vector(shading_normal);
    const vec3 tangentY = cross(shading_normal, tangentX);
    return (vec3) {tangentX.x * s_x + tangentY.x * s_y + shading_normal.x * s_z,
        tangentX.y * s_x + tangentY.y * s_y + shading_normal.y * s_z,
        tangentX.z * s_x + tangentY.z * s_y + shading_normal.z * s_z};
}

/*!
 * \brief Samples irradiance direction for modified Phong specular reflectance model.
 *
 *
 * \param ray ray.d defines the vector wo, and is updated upon output to store the resulting sample direction, wi.
 * \param hit Contains the surface normal at the hit point.
 * \param specExp The Phong specular exponent term.
 * \param r1 A uniform random number
 * \param r2 A uniform random number
 */
float samplePhong(ray_t *ray, const hit_info_t *hit, float const specExp,
        float const r1, float const r2) {
    /*
     *
     */
    float4 surfaceNormal = as_float4(hit->surface_normal);
    float4 sampleDirection = as_float4(ray->d);
    if (dot(sampleDirection, surfaceNormal) > 0.0f) {
        surfaceNormal *= -1.0f;
    }
    if (specExp < 100000.0f) {

        /*
         * Compute the mirror reflection vector...
         */
        sampleDirection = normalize(sampleDirection + (surfaceNormal) * (-2.0f
                * dot(sampleDirection, surfaceNormal)));

        /*
         * Exponential cosine weighted sampling about the mirror reflection direction.
         *
         * PDF = Modified Phong PDF = ( (n + 1) / 2pi ) * cos(a) ^ n
         *
         * Where a is the angle between the output direction and the mirror reflection vector.
         */
        float const cosA = pow(r1, 1.0f / (specExp + 1.0f));

        /*
         * Generate another random value, uniform between 0 and 2pi, which is the angle around the mirror reflection
         * vector
         */
        float const phi = 2.0f * M_PI_F * r2;
        float const sinTheta = sqrt(1.0f - cosA * cosA);
        float const xb = cos(phi) * sinTheta;
        float const yb = sin(phi) * sinTheta;
        /*
         * Construct an ortho-normal basis using the reflection vector as one axis, and arbitrary (perpendicular)
         * vectors for the other two axes. The orientation of the coordinate system about the reflection vector is
         * irrelevant since xb and yb are generated from a uniform random variable.
         */
        float4 u = (float4) (0.0f, 1.0f, 0.0f, 0.0f);
        float const cosAng = dot(sampleDirection, u);
        if (cosAng > 0.9f || cosAng < -0.9f) {
            // Small angle, pick a better vector...
            u.x = -1.0f;
            u.y = 0.0f;
        }
        u = normalize(cross(u, sampleDirection));
        float4 const v = cross(u, sampleDirection);

        sampleDirection *= cosA;
        sampleDirection += u * xb + v * yb;

        if (dot(sampleDirection, surfaceNormal) < 0.0f)
            sampleDirection += -2.0f * xb * u + -2.0f * yb * v;
    } else {
        sampleDirection += (surfaceNormal) * (-2.0f * dot(sampleDirection,
                surfaceNormal));
    }
    ray->d.x = sampleDirection.x;
    ray->d.y = sampleDirection.y;
    ray->d.z = sampleDirection.z;

    ray->o.x = hit->hit_pt.x;
    ray->o.y = hit->hit_pt.y;
    ray->o.z = hit->hit_pt.z;

    ray->tmin = SMALL_F;
    ray->tmax = INFINITY;

    return 1.0f;
}

/*!
 * \brief Lambert diffuse reflectance model sampling function.
 *
 * Given a surface normal and two uniform random variables, samples the irradiance direction defined by the Lambert reflectance model.
 *
 * \param ray ray.d defines the vector wo, and is updated upon output to store the resulting sample direction, wi.
 * \param hit Contains the surface normal at the hit point.
 * \param r1 a uniform random variable.
 * \param r2 a uniform random variable.
 */
float sampleLambert(ray_t *ray, const hit_info_t *hit, float const r1,
        float const r2) {
    /*
     * Cosine-weighted sampling about the z axis.
     */
    const float4 hemisphereSample = cosSampleHemisphere(r1, r2);

    ray->d = shading_to_world(hit->surface_normal, hemisphereSample.x, hemisphereSample.y, hemisphereSample.z);

    ray->o.x = hit->hit_pt.x;
    ray->o.y = hit->hit_pt.y;
    ray->o.z = hit->hit_pt.z;

    ray->tmin = SMALL_F;
    ray->tmax = INFINITY;

    return hemisphereSample.w;
}

float evaluateLambert() {
    return M_1_PI_F;
}

/*!
 * \brief Samples irradiance direction for a refractive BRDF model.
 *
 * \param sampleDirection Contains the normalized direction vector of the sample on input, and the normalized irradiance sample direction on output.
 * \param surfaceNormal Contains the surface normal at the sample point.
 * \param ior The index of refraction, assumed to be embedded in air with IOR approximately == 1.0.
 * \param blurExp An exponent term similar to the specular exponent in the Phong model, here it controls the refraction direction.
 * \param r1 A uniform random number
 * \param r2 A uniform random number
 *
 * \return false if the generated sample direction crosses out of the refractive volume, true if it is inside.
 */
//bool sampleRefraction(ray_t *sampleDirection, const hit_info_t *hit,
//        float const ior, float const blurExp, float const r1, float const r2) {
//    float4 sNormal = *surfaceNormal;
//    float cosSampleAndNormal = dot(*sampleDirection, sNormal);
//    float rIdxRatio;
//    bool exiting;
//    if (cosSampleAndNormal <= 0.0f) {
//        rIdxRatio = 1.0f / ior;
//        cosSampleAndNormal = -cosSampleAndNormal;
//        exiting = false;
//    } else {
//        rIdxRatio = ior;
//        sNormal *= -1.0f;
//        exiting = true;
//    }
//
//    float snellRoot = 1.0f - (rIdxRatio * rIdxRatio * (1.0f
//            - cosSampleAndNormal * cosSampleAndNormal));
//    if (snellRoot < 0.0f) {
//        /*
//         * Total internal reflection
//         */
//        *sampleDirection = normalize(*sampleDirection + (sNormal) * (-2.0f
//                * dot(*sampleDirection, sNormal)));
//        return exiting;
//    } else {
//        /*
//         * Refraction
//         */
//        *sampleDirection *= rIdxRatio;
//        *sampleDirection += sNormal * (rIdxRatio * cosSampleAndNormal - sqrt(
//                snellRoot));
//        if (blurExp < 100000.0f) {
//            /*
//             * Idential to phong, except we substitude the refraction direction
//             * for the mirror reflection vector.
//             */
//            float cosA = pow(r1, 1.0f / (blurExp + 1.0f));
//
//            /*
//             * Generate another random value, uniform between 0 and 2pi, which
//             * is the angle around the mirror reflection vector
//             */
//            float phi = 2.0f * M_PI_F * r2;
//            float sinTheta = sqrt(1.0f - cosA * cosA);
//            float xb = cos(phi) * sinTheta;
//            float yb = sin(phi) * sinTheta;
//
//            /*
//             * Construct an ortho-normal basis using the reflection vector as
//             * one axis, and arbitrary (perpendicular) vectors for the other two
//             * axes. The orientation of the coordinate system about the
//             * reflection vector is irrelevant since xb and yb are generated
//             * from a uniform random variable.
//             */
//            float4 u = (float4) (0.0f, 1.0f, 0.0f, 0.0f);
//            float const cosAng = dot(*sampleDirection, u);
//            if (cosAng > 0.9f || cosAng < -0.9f) {
//                // Small angle, pick a better vector...
//                u.x = -1.0f;
//                u.y = 0.0f;
//            }
//            u = normalize(cross(u, *sampleDirection));
//            float4 const v = cross(u, *sampleDirection);
//            ;
//
//            *sampleDirection *= cosA;
//            *sampleDirection += u * xb + v * yb;
//            if (dot(*sampleDirection, *surfaceNormal) < 0.0f)
//                *sampleDirection -= 2.0f * (u * xb + v * yb);
//        }
//    }
//    return !exiting;
//}

/*!
 * \brief Given a spherical emissive light source, a sample point (to sample illumination for), computes a ray direction from 
 *        the sample point to a point on the light, the distance to that point, and the radiance that is emitted from the light
 *        toward the sample point.
 *
 * \param ray On input, ray.o indicates the point to sample illumination at, on output, ray.[d,tmin,tmax] are initialized.
 * \param sphereCenter the center of the emissive sphere.
 * \param radius The radius of the emissive sphere.
 * \param r1 A uniform random variable.
 * \param r2 A uniform random variable.
 * \return The inverse probability of the computed sample direction over all possible directions.
 */
float sphereEmissiveRadiance(ray_t *ray, float4 const sphere_center,
        float const radius, float const r1, float const r2) {

    vec3 direction = (vec3) {sphere_center.x - ray->o.x, sphere_center.y
        - ray->o.y, sphere_center.z - ray->o.z};

    const float light_dist_inv = rsqrt(direction.x * direction.x + direction.y
            * direction.y + direction.z * direction.z);
    direction.x *= light_dist_inv;
    direction.y *= light_dist_inv;
    direction.z *= light_dist_inv;

    /*
     * The maximum angle from originToCenter for a ray eminating from origin
     * that will hit the sphere.
     */
    float const sin_max_angle = radius * light_dist_inv;
    float const cos_max_angle = sqrt(1.0f - sin_max_angle * sin_max_angle);

    /*
     * Uniform sample density over the solid angle subtended by the sphere
     * wrt. the origin point. Taken from Shirley and Morely book.
     */
    float const cos_theta = 1.0f + r1 * (cos_max_angle - 1.0f);
    float const sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    float const phi = M_2PI_F * r2;

    ray->d = shading_to_world(direction, cos(phi) * sin_theta, sin(phi)
            * sin_theta, cos_theta);
    ray->tmin = SMALL_F;
    ray->tmax = intersectSphere(ray, sphere_center, radius) - SMALL_F;

    /*
     * Multiply by 1/distribution of light samples over the hemisphere around the hit point.
     *
     * Since this is only called for direct interaction with diffuse surfaces, it has been optimized based on the
     * assumption that the directIllumination function will call this
     */
    return (M_2PI_F * (1.0f - cos_max_angle));
}

#endif /* MATERIAL_H */
