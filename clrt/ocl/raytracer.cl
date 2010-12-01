#pragma OPENCL EXTENSION cl_amd_printf : enable
/*!
 * \file raytracer.cl
 * \brief OpenCL based Path-Tracer kernel.
 *
 */
#include "geometry.h"
#include "rng.h"
#include "materials.h"
#include "geometryFuncs.h"

/*
 * Half-width of the cube that encloses the scene.
 */
#define BOX_WIDTH 6.0f
#define BOX_HEIGHT 5.0f



int scene_intersection( ray_t *ray, __constant Sphere *geometry, const uint n_geometry)
{
    int hitObject = -1;
    for (int sphereNum = 0; sphereNum < n_geometry; ++sphereNum)
    {
        __constant Sphere *sphere = &(geometry[sphereNum]);
        float d = intersectSphere(ray, sphere->center, sphere->radius);
        if (d > ray->tmin && d < ray->tmax)
        {
            hitObject = sphereNum;
            ray->tmax = d;
        }
    }
    return hitObject;
}

/*!
 * \brief Test to determine whether the specified ray intersects anything between it's tmin/tmax range values.
 *
 * \return true if the ray does not intersect anything in the range tmin -> tmax, false if it does intersect something.
 */
bool visibility_test(const ray_t *ray, __constant Sphere * geometry, const uint n_geometry)
{
    for (int sphereNum = 0; sphereNum < n_geometry; ++sphereNum) {
        __constant Sphere *sphere = &(geometry[sphereNum]);
        float d = intersectSphere(ray, sphere->center, sphere->radius);
        if (d > ray->tmin && d < ray->tmax) {
            return false;
        }
    }
    return true;
}


/*!
 * Computes the direct illumination incident on a specified point.
 * \param hit The location and surface normal of the point to sample direct illumination at.
 * \param geometry The scene geometry.
 * \param n_geometry The number of objects in the scene.
 * \param seed Random seed value, updated on output if used.
 * \return The irradiance incident on the specified hit point.
 */
float4 sample_direct_illumination(const hit_info_t *hit, __constant Sphere *geometry, const uint n_geometry, seed_value_t *seed) {
    ray_t ray;
    ray.o.x = hit->hit_pt.x + hit->surface_normal.x * SMALL_F;
    ray.o.y = hit->hit_pt.y + hit->surface_normal.y * SMALL_F;
    ray.o.z = hit->hit_pt.z + hit->surface_normal.z * SMALL_F;
    float4 irradiance = (float4)0.0f;
    for (uint sphereNum = 0; sphereNum < n_geometry; ++sphereNum)
    {
        constant Sphere *light = &(geometry[sphereNum]);
        if (light->emission.w != 0)
        {
            float pdf_inv = sphereEmissiveRadiance(&ray, light->center, light->radius, frand(seed), frand(seed));
            irradiance += light->emission * pdf_inv;
            if (visibility_test(&ray, geometry, n_geometry)) {
                const float cosWi = ray.d.x * hit->surface_normal.x + ray.d.y * hit->surface_normal.y + ray.d.z * hit->surface_normal.z;
                if (cosWi > 0)
                {
                    irradiance += light->emission * cosWi;
                }
            }
        }
    }
    return irradiance;
}

/*!
 * \brief Given an incident ray, intersection geometry, and a material, this computes a new ray by sampling the BRDF for the material.
 * \param ray On input, ray->d indicates the incident direction, on output, the ray is fully initialized to a new ray pointing away from the intersection point.
 * \param hit Description of the local geometry at the ray intersection point.
 * \param hitSphere TODO: this should be a material.
 * \param seed Seed data for uniform random number generation.
 * \return true if the new sample ray has been initialized, false indicates that the ray should be terminated, and no more material sampling should occur (the ray is absorbed).
 */
bool sample_material(ray_t *ray, const hit_info_t *hit,
        constant Sphere *hitSphere, seed_value_t *seed) {

    ray->o.x = hit->hit_pt.x;
    ray->o.y = hit->hit_pt.y;
    ray->o.z = hit->hit_pt.z;
    /* convert wi to a vector pointing away from the hit point */
    ray->d.x *= -1.0f;
    ray->d.y *= -1.0f;
    ray->d.z *= -1.0f;
    /* And into shading coords */
    ray->d = world_to_shading(ray->d, hit->surface_normal);

    ray->tmin = SMALL_F;
    ray->tmax = INFINITY;

    /*
     * Generate random number to determine whether to sample specular, diffuse, or transmissive directions,
     * and two more to pass to the sampling functions.  This is done outside of the conditional to avoid
     * excessive serialization of work-item threads.
     */
    float p = frand(seed);
    float r1 = frand(seed);
    float r2 = frand(seed);
    if (p < hitSphere->ks) {
        const float inv_pdf = 1.0 / samplePhong(&(ray->d), hitSphere->specExp,
                r1, r2);
        ray->diffuse_bounce = false;
        /* cos (Wi) term */
        ray->propagation.x *= ray->d.z * inv_pdf;
        ray->propagation.y *= ray->d.z * inv_pdf;
        ray->propagation.z *= ray->d.z * inv_pdf;
    } else if (p < (hitSphere->ks + hitSphere->diffuse.w)) {
        //const float pdf =
        sampleLambert(&(ray->d), r1, r2);
        ray->diffuse_bounce = true;
        /*
         * cos (Wi) * diffuse / PDF
         *
         * Assumes that sampleLambert has a cosine distribution, so that everything cancels except the diffuse color.
         */
        ray->propagation.x *= hitSphere->diffuse.x; //* evaluateLambert() * DOT(ray.d, hit.surface_normal)/ pdf;
        ray->propagation.y *= hitSphere->diffuse.y;
        ray->propagation.z *= hitSphere->diffuse.z;
    } else if (p < (hitSphere->ks + hitSphere->diffuse.w
                    + hitSphere->extinction.w)) {
        if (sampleRefraction(&(ray->propagation), ray, hit, hitSphere->ior,
                        1000000.0f, r1, r2)) {
            ray->extinction.x = hitSphere->extinction.x;
            ray->extinction.y = hitSphere->extinction.y;
            ray->extinction.z = hitSphere->extinction.z;
        } else {
            ray->extinction.x = 0;
            ray->extinction.y = 0;
            ray->extinction.z = 0;
        }
        /* multiply by cos(theta_wi) */
        const float cos_theta_wi = fabs(ray->d.z);
        ray->propagation.x *= cos_theta_wi;
        ray->propagation.y *= cos_theta_wi;
        ray->propagation.z *= cos_theta_wi;
    } else {
        return false;
    }
    ray->d = shading_to_world(ray->d, hit->surface_normal);
    return true;
}

/*!
 * \brief Ray Tracer entry point
 *
 * \param out An output buffer to store rgba color values (as one float4 per rgba value), with length imWidth * imHeight * 4.  32bits/component required to support progressive rendering.
 * \param camera The camera data used to generate eye rays.
 * \param spheres The scene content, including light sources.
 * \param sphereCount The number of objects in the scene.
 * \param imWidth The output image width, in pixels.
 * \param imHeight The output image height, in pixels.
 * \param sampleRate The rate at which to sample pixels, the actual paths traced per pixel is this value squared.
 * \param maxDepth The maximum path length.
 * \param progressive When 0, data in the out-buffer is overwritten, when non-zero data is accumulated for progressive rendering.  In this case,
 *        the value determines the mixing weight.
 * \param seeds Seed data for the random number generator.  There should be one seed per pixel.  These values are used to generate a uniform random number sequence for each pixel, however the seed data is not updated in global memory.
 */
__kernel void raytrace(__global float *out, __constant Camera *camera,
        __constant Sphere *spheres, uint const sphereCount,
        uint const imWidth, uint const imHeight, uint const sampleRate,
        uint const maxDepth, uint const progressive, __global uint *seeds)
{
    /*
     * The pixel x,y is identified by the global ids for the first two dimensions.
     */
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    /*
     * The rendering is overdrawn to ensure the work-group size is a factor of the global NDRange size.
     * Don't do anything with the overdraw region, there is no space in the buffer to store the result.
     */
    if (x >= imWidth || y >= imHeight) return;

    /*
     * Copy RNG seed data into private memory for faster access.
     */
    seed_value_t seed;

    const uint seed_offs = y * get_global_size(0) + x;
    seed.x = seeds[seed_offs];
    seed.y = seeds[(get_global_size(0) * get_global_size(1)) + seed_offs];

    /*
     * Color variable used to accumulate samples
     */
    float4 pixelColor = (float4)0.0f;

    for (uint sx = 0; sx < sampleRate; ++sx)
    {
        for (uint sy = 0; sy < sampleRate; ++sy)
        {
            //TODO: sample camera aperture for DoF
            float4 rdirection = normalize(camera->view +
                    camera->right * ((float)(x + strat_rand(&seed, sx, sampleRate)) - ((float)imWidth)/2.0f) +
                    camera->up * ((float)(y + strat_rand(&seed, sy, sampleRate)) - ((float)imHeight)/2.0f));
            ray_t ray = {camera->position.x, camera->position.y, camera->position.z, rdirection.x, rdirection.y, rdirection.z, SMALL_F, INFINITY, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, false};
            for (int rayDepth = 0; rayDepth <= maxDepth; ++rayDepth)
            {
                int hitIdx = scene_intersection( &ray, spheres, sphereCount);
                if (hitIdx >= 0)
                {

                    constant Sphere *hitSphere = &spheres[hitIdx];
                    /*
                     * Update origin to new intersect point
                     */
                    hit_info_t hit;
                    hit.hit_pt = (vec3) {ray.o.x + ray.d.x * ray.tmax, ray.o.y + ray.d.y * ray.tmax, ray.o.z + ray.d.z * ray.tmax};
                    sphereNormal(&hit, hitSphere->center, hitSphere->radius);

                    /* Apply extinction due to simple participating media */
                    if (ray.extinction.x > 0.0f) ray.propagation.x *= exp(log(ray.extinction.x) * ray.tmax);
                    if (ray.extinction.y > 0.0f) ray.propagation.y *= exp(log(ray.extinction.y) * ray.tmax);
                    if (ray.extinction.z > 0.0f) ray.propagation.z *= exp(log(ray.extinction.z) * ray.tmax);

                    /*
                     * Emissive contribution on surfaces that are not sampled for direct illumination.
                     */
                    if (!ray.diffuse_bounce)
                    {
                        pixelColor.x += ray.propagation.x * hitSphere->emission.x;
                        pixelColor.y += ray.propagation.y * hitSphere->emission.y;
                        pixelColor.z += ray.propagation.z * hitSphere->emission.z;
                    }

                    /*
                     * Sample direct illumination
                     */
                    if (hitSphere->diffuse.w > 0.0f)
                    {
                        const float4 direct = sample_direct_illumination(&hit, spheres, sphereCount, &seed);
                        const float scale = hitSphere->diffuse.w * evaluateLambert();
                        pixelColor.x += ray.propagation.x * direct.x * hitSphere->diffuse.x * scale;
                        pixelColor.y += ray.propagation.y * direct.y * hitSphere->diffuse.y * scale;
                        pixelColor.z += ray.propagation.z * direct.z * hitSphere->diffuse.z * scale;
                    }

                    if (rayDepth == maxDepth) break;

                    if (!sample_material(&ray, &hit, hitSphere, &seed)) {
                        break;
                    }

                } // if hit (sphere) object
                else // No object hit (process hit for box).
                {
                    float hitDistance = intersectsBox(&ray, (float4)0.0f, BOX_WIDTH, BOX_HEIGHT, BOX_WIDTH);
                    if (hitDistance > ray.tmin && hitDistance < ray.tmax)
                    {
                        ray.tmax = hitDistance;
                        hit_info_t hit;
                        hit.hit_pt = (vec3) {ray.o.x + ray.d.x * ray.tmax, ray.o.y + ray.d.y * ray.tmax, ray.o.z + ray.d.z * ray.tmax};
                        boxNormal(&ray, &hit, BOX_WIDTH, BOX_HEIGHT, BOX_WIDTH); /* populate surface_normal */

                        const float4 direct = sample_direct_illumination(&hit, spheres, sphereCount, &seed);
                        const float scale = evaluateLambert();
                        ray.propagation.x *= 0.7f;
                        ray.propagation.y *= 0.7f;
                        ray.propagation.z *= 0.7f;
                        pixelColor.x += ray.propagation.x * direct.x * scale;
                        pixelColor.y += ray.propagation.y * direct.y * scale;
                        pixelColor.z += ray.propagation.z * direct.z * scale;

                        ray.o.x = hit.hit_pt.x;
                        ray.o.y = hit.hit_pt.y;
                        ray.o.z = hit.hit_pt.z;
                        //ray.d is not used by sampleLambert
                        //                        /* convert wi to a vector pointing away from the hit point */
                        //                        ray.d.x *= -1.0f;
                        //                        ray.d.y *= -1.0f;
                        //                        ray.d.z *= -1.0f;
                        //                        /* And into shading coords */
                        //                        ray.d = world_to_shading(ray.d, hit.surface_normal);

                        ray.tmin = SMALL_F;
                        ray.tmax = INFINITY;

                        //const float pdf =
                        sampleLambert(&(ray.d), frand(&seed), frand(&seed));
                        ray.d = shading_to_world(ray.d, hit.surface_normal);
                        ray.diffuse_bounce=true;
                    }
                    else
                    {
                        break; // Causes path to terminate.
                    }
                }
            } // foreach ray in path
        } // foreach sy
    } // foreach sx

    pixelColor /= (float)(sampleRate * sampleRate);
    if (progressive > 0)
    {
        float4 color = vload4((y * imWidth + x), out);
        pixelColor = mix(color , pixelColor, 1.0f / (float)(progressive));
    }
    vstore4(pixelColor, (y * imWidth + x), out);
    seeds[seed_offs] = seed.x;
    seeds[get_global_size(0) * get_global_size(1) + seed_offs] = seed.y;
    //    seeds[y * imWidth + x] = seed;
}

