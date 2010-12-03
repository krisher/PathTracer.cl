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
float4 cos_sample_hemisphere(const float r1, const float r2) {
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
vec3 shading_to_world(const vec3 to_convert, const vec3 shading_normal) {
    const vec3 tangentX = perpendicular_vector(shading_normal);
    const vec3 tangentY = cross_vec(shading_normal, tangentX);
    return (vec3) {tangentX.x * to_convert.x + tangentY.x * to_convert.y + shading_normal.x * to_convert.z,
        tangentX.y * to_convert.x + tangentY.y * to_convert.y + shading_normal.y * to_convert.z,
        tangentX.z * to_convert.x + tangentY.z * to_convert.y + shading_normal.z * to_convert.z};
}

/*!
 * \brief Change of basis function to convert from world coordinates to shading coordinates (z => shading normal, x,y in tangent plane), given a shading normal vector.
 *
 * \param result The vector to store the result in.
 * \param shading_normal The vector representing the world-space shading normal.
 *
 */
vec3 world_to_shading(const vec3 world_vec, const vec3 shading_normal) {
    const vec3 tangentX = perpendicular_vector(shading_normal);
    const vec3 tangentY = cross_vec(shading_normal, tangentX);
    return (vec3) {DOT(world_vec, tangentX),
        DOT(world_vec, tangentY),
        DOT(world_vec, shading_normal)};
}

/*!
 * \brief Samples irradiance direction for modified Phong specular reflectance model.
 *
 *
 * \param ray ray.d defines the vector wo, and is updated upon output to store the resulting sample direction, wi.  These are all defined in shading coordiantes (z == surface normal)
 * \param specExp The Phong specular exponent term.
 * \param r1 A uniform random number
 * \param r2 A uniform random number
 */
float samplePhong(vec3 *w, float const specExp, float const r1, float const r2) {
    /* Compute the mirror reflection vector, trivial in shading coordinates... */
    vec3 wi_shading = (vec3) {-w->x, -w->y, w->z};
    float pdf;
    if (specExp < 100000.0f) {

        /*
         * Exponential cosine weighted sampling about the mirror reflection direction.
         *
         * PDF = Modified Phong PDF = ( (n + 1) / 2pi ) * cos(a) ^ n
         *
         * Where a is the angle between the output direction and the mirror reflection vector.
         */
        float const cos_a = pow(r1, 1.0f / (specExp + 1.0f));
        float const sinTheta = sqrt(1.0f - cos_a * cos_a);
        float const phi = M_2PI_F * r2;

        wi_shading.x = cos(phi) * sinTheta;
        wi_shading.y = sin(phi) * sinTheta;
        wi_shading.z = cos_a;

        float wo_dot_wh = DOT((*w), wi_shading);
        wi_shading.x = -w->x + 2.0f * wo_dot_wh * wi_shading.x;
        wi_shading.y = -w->y + 2.0f * wo_dot_wh * wi_shading.y;
        wi_shading.z = -w->z + 2.0f * wo_dot_wh * wi_shading.z;

        pdf = 1.0f;
    } else {
        pdf = 1.0f;
    }
    *w = wi_shading;
    return pdf;
}

/*!
 * \brief Lambert diffuse reflectance model sampling function.
 *
 * Given a surface normal and two uniform random variables, samples the irradiance direction defined by the Lambert reflectance model.
 *
 * \param ray ray.d defines the vector wo, and is updated upon output to store the resulting sample direction, wi.  In shading coordinates (z == normal)
 * \param r1 a uniform random variable.
 * \param r2 a uniform random variable.
 */
float sampleLambert(vec3 *w, float const r1, float const r2) {
    /*
     * Cosine-weighted sampling about the z axis.
     */
    const float4 hemisphereSample = cos_sample_hemisphere(r1, r2);
    w->x = hemisphereSample.x;
    w->y = hemisphereSample.y;
    w->z = hemisphereSample.z;
    return hemisphereSample.w;
}

float evaluateLambert() {
    return M_1_PI_F;
}

/*!
 * \brief Samples irradiance direction for a refractive BRDF model.
 *
 * \param ray contains the incident ray on input, and the exitant ray on output.
 * \param hit The hit information at the incident ray's intersection point.
 * \param ior The index of refraction, assumed to be embedded in air with IOR approximately == 1.0.
 * \param blurExp An exponent term similar to the specular exponent in the Phong model, here it controls the refraction direction.
 * \param r1 A uniform random number
 * \param r2 A uniform random number
 *
 * \return false if the generated sample direction crosses out of the refractive volume, true if it is inside.
 */
bool sampleRefraction(vec3 *transmission, ray_t *ray, const hit_info_t *hit,
        float const ior, float const blurExp, float const r1, float const r2) {

    float cos_wo = fabs(ray->d.z);

    bool entering = ray->d.z > 0; /* wo in same hemisphere as surface normal? */
    float ei;
    float eo;
    if (entering) {
        ei = 1.0f;
        eo = ior;
    } else {
        ei = ior;
        eo = 1.0f;
    }
    float ref_idx_ratio = ei / eo;
    float cos_theta_sq = 1.0f - (ref_idx_ratio * ref_idx_ratio * (1.0f
            - ray->d.z * ray->d.z));

    if (cos_theta_sq < 0.0f) {
        /*
         * Internal reflection
         */
        ray->d.x *= -1;
        ray->d.y *= -1;
        entering = !entering;
    } else {
        /*
         * Refraction
         */
        float cos_theta = sqrt(cos_theta_sq);
        if (entering)
            cos_theta = -cos_theta; /* exitant ray should point away from surface normal */

        vec3
                wi_shading =
                        (vec3) {-ray->d.x * ref_idx_ratio, -ray->d.y * ref_idx_ratio, cos_theta};
        if (blurExp < 100000.0f) {
            /*
             * Identical to Phong, except we substitute the refraction direction
             * for the mirror reflection vector.
             *
             * TODO: This is broken...
             */
            float cos_a = pow(r1, 1.0f / (blurExp + 1.0f));
            float const sinTheta = sqrt(1.0f - cos_a * cos_a);
            float const phi = M_2PI_F * r2;

            wi_shading.x = cos(phi) * sinTheta;
            wi_shading.y = sin(phi) * sinTheta;
            wi_shading.z = cos_a;

            float wo_dot_wh = DOT(ray->d, wi_shading);
            wi_shading.x = -ray->d.x + 2.0f * wo_dot_wh * wi_shading.x;
            wi_shading.y = -ray->d.y + 2.0f * wo_dot_wh * wi_shading.y;
            wi_shading.z = -ray->d.z + 2.0f * wo_dot_wh * wi_shading.z;
        }
        ray->d = wi_shading;

        /* Fresnel Dielectric transmission based on PBRT */
        cos_theta = fabs(cos_theta);
        float parl = (eo * cos_wo - ei * cos_theta) / (eo * cos_wo + ei
                * cos_theta);
        float perp = (ei * cos_wo - eo * cos_theta) / (ei * cos_wo + eo
                * cos_theta);
        float fresnel = (parl * parl + perp * perp) * 0.5f;
        fresnel = (1.0f - fresnel) / cos_theta;
        transmission->x *= fresnel;
        transmission->y *= fresnel;
        transmission->z *= fresnel;
    }
    return entering;
}

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
float sphereEmissiveRadiance(ray_t *ray, vec3 const sphere_center,
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

    ray->d = shading_to_world((vec3) {cos(phi) * sin_theta, sin(phi)
        * sin_theta, cos_theta}, direction);
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
