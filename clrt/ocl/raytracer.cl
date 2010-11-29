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
#define BOX_SIZE 6.0f

/*!
 * Computes the direct illumination incident on a specified point.
 * \param hit The location and surface normal of the point to sample direct illumination at.
 * \param geometry The scene geometry.
 * \param n_geometry The number of objects in the scene.
 * \param seed Random seed value, updated on output if used.
 * \return The irradiance incident on the specified hit point.
 */
float4 directIllumination(const hit_info_t *hit, __constant Sphere *geometry, const uint n_geometry, seed_value_t *seed) {
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
            if (visibilityTest(&ray, geometry, n_geometry)) {
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
        uint const maxDepth, uint const progressive, __global seed_value_t *seeds)
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
    seed_value_t seed = seeds[y * imWidth + x];

    /*
     * Color variable used to accumulate samples
     */
    float4 pixelColor = (float4)0.0f;

    for (uint sx = 0; sx < sampleRate; ++sx)
    {
        for (uint sy = 0; sy < sampleRate; ++sy)
        {
            //TODO: sample camera aperture for DoF
            //TODO: There appears to be a bug with the initial hit on some surfaces, may be due to bad eye-rays.
            float4 rdirection = normalize(camera->view +
                    camera->right * ((float)(x + strat_rand(&seed, sx, sampleRate)) - ((float)imWidth)/2.0f) +
                    camera->up * ((float)(y + strat_rand(&seed, sy, sampleRate)) - ((float)imHeight)/2.0f));
            ray_t ray = {camera->position.x, camera->position.y, camera->position.z, rdirection.x, rdirection.y, rdirection.z, SMALL_F, INFINITY};

            float4 transmissionColor = {1.0f, 1.0f, 1.0f, 0.0f}; //All light is initially transmitted.
            float4 extinction = {0.0f, 0.0f, 0.0f, 0.0f};
            bool emissiveContributes = true;

            for (int rayDepth = 0; rayDepth <= maxDepth; ++rayDepth)
            {
                int hitIdx = sceneIntersection( &ray, spheres, sphereCount);
                if (hitIdx >= 0)
                {
                    if (extinction.x > 0.0f) transmissionColor.x *= exp(log(extinction.x) * ray.tmax);
                    if (extinction.y > 0.0f) transmissionColor.y *= exp(log(extinction.y) * ray.tmax);
                    if (extinction.z > 0.0f) transmissionColor.z *= exp(log(extinction.z) * ray.tmax);

                    constant Sphere *hitSphere = &spheres[hitIdx];
                    /*
                     * Emissive contribution
                     */
                    if (emissiveContributes)
                    {
                        pixelColor += transmissionColor * hitSphere->emission;
                    }
                    /*
                     * Update origin to new intersect point
                     */
                    hit_info_t hit;
                    hit.hit_pt = (vec3) {ray.o.x + ray.d.x * ray.tmax, ray.o.y + ray.d.y * ray.tmax, ray.o.z + ray.d.z * ray.tmax};
                    sphereNormal(&hit, hitSphere->center, hitSphere->radius);

                    /*
                     * Sample direct illumination
                     */
                    if (hitSphere->diffuse.w > 0.0f)
                    {
                        float4 direct = directIllumination(&hit, spheres, sphereCount, &seed);
                        pixelColor += transmissionColor * direct * hitSphere->diffuse * hitSphere->diffuse.w * evaluateLambert(); /* Diffuse BRDF */
                    }

                    /*
                     * Generate random number to determine whether to sample specular, diffuse, or transmissive directions,
                     * and two more to pass to the sampling functions.  This is done outside of the conditional to avoid
                     * excessive serialization of work-item threads.
                     */
                    float p = frand(&seed);
                    float r1 = frand(&seed);
                    float r2 = frand(&seed);
                    if (p < hitSphere->ks)
                    {
                        const float pdf = samplePhong(&ray, &hit, hitSphere->specExp,r1, r2);
                        emissiveContributes = true;
                        /* cos (Wi) term */
                        transmissionColor *= DOT(ray.d, hit.surface_normal) / pdf;
                    }
                    else if (p < (hitSphere->ks + hitSphere->diffuse.w))
                    {
                        //const float pdf =
                        sampleLambert(&ray, &hit, r1, r2);
                        emissiveContributes = false;
                        /*
                         * cos (Wi) * diffuse / PDF
                         *
                         * Assumes that sampleLambert has a cosine distribution, so that everything cancels except the diffuse color.
                         */
                        transmissionColor *= hitSphere->diffuse; //* evaluateLambert() * DOT(ray.d, hit.surface_normal)/ pdf;
                    }
                    else if (p < (hitSphere->ks + hitSphere->diffuse.w + hitSphere->extinction.w))
                    {
                        if (sampleRefraction(&transmissionColor, &ray, &hit, hitSphere->ior, 1000000.0f, r1, r2)) {
//                            extinction = hitSphere->extinction;
                        } else {
                            emissiveContributes = true;
                        }
                        transmissionColor *= fabs(DOT(ray.d, hit.surface_normal));
                    }
                    else
                    {
                        rayDepth = maxDepth + 1; /* Terminate ray. */
                    }
                } // if hit (sphere) object
                else // No object hit (process hit for box).
                {
                    float hitDistance = intersectsBox(&ray, (float4)0.0f, BOX_SIZE, BOX_SIZE, BOX_SIZE);
                    if (hitDistance > ray.tmin && hitDistance < ray.tmax)
                    {
                        ray.tmax = hitDistance;
                        hit_info_t hit;
                        hit.hit_pt = (vec3) {ray.o.x + ray.d.x * ray.tmax, ray.o.y + ray.d.y * ray.tmax, ray.o.z + ray.d.z * ray.tmax};
                        boxNormal(&ray, &hit, BOX_SIZE, BOX_SIZE, BOX_SIZE); /* populate surface_normal */

                        float4 direct = directIllumination(&hit, spheres, sphereCount, &seed);
                        pixelColor += 0.7f * direct * transmissionColor * evaluateLambert(); /* Diffuse BRDF */
                        //const float pdf =
                        sampleLambert(&ray, &hit, frand(&seed), frand(&seed));
                        transmissionColor *= 0.7f;// * fabs(ray.d.x * hit.surface_normal.x + ray.d.y * hit.surface_normal.y + ray.d.z * hit.surface_normal.z) * evaluateLambert() / pdf;
                        emissiveContributes = false;
                    }
                    else
                    {
                        rayDepth = maxDepth + 1; // Causes path to terminate.
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
    seeds[y * imWidth + x] = seed;
}

