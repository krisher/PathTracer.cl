/*!
 * \file raytracer.cl
 * \brief OpenCL based Path-Tracer kernel.
 *
 */
#include "geometry.h"
#include "rng.h"
#include "materials.h"
#include "geometryFuncs.h"

/* Small floating point number used to offset ray origins to avoid roundoff error issues. */
#define SMALL_F 1e-5f
/* Stratified pixel sampling in a jittered regular grid. */
#define JITTER(pixel, sample, sampleRate, rand) (((((float)sample) + rand) / (float)sampleRate) + (float)pixel)
/*
 * Half-width of the cube that encloses the scene.
 */
#define BOX_SIZE 5.0f

/*!
 * \brief Computes diffuse response to direct illumination and adds the result to pixel color.
 *
 * \param pixelColor Diffuse response to each light will be accumulated into this pixel color.
 * \param spheres The scene including the lights to sample, and the objects that may occlude the light.
 * \param sphereCount The number of objects in the scene
 * \param origin The point that we are sampling direct illumination for.
 * \param hitNormal the surface normal for the sample point.
 * \param transmissionColor The color that is actually transmitted back to the pixel from the sample point.  Radiance from direct illumination is modulated by this color.
 * \param diffuse The diffuse response of the sample point.
 * \param seed A random seed used with frand() to generate random numbers as needed.
 */
void directIllumination(float4 *pixelColor, __constant Sphere const *spheres, uint sphereCount, float4 const origin, float4 const hitNormal, float4 const transmissionColor, float4 const diffuse, seed_value_t *seed);

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
__kernel void raytrace(__global float *out, __constant Camera const *camera,
        __constant Sphere const *spheres, uint const sphereCount,
        uint const imWidth, uint const imHeight, uint const sampleRate,
        uint const maxDepth, uint const progressive, __global seed_value_t *seeds)
{
    /*
     * The pixel we are rendering is identified to the global ids for the first two dimensions.
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
            //TODO: sample camera aperture
            //TODO: There appears to be a bug with the initial hit on some surfaces, may be due to bad eye-rays.
            float4 origin = camera->position;
            float4 direction = normalize(camera->view +
                    camera->right * (float)(JITTER(x, sx, sampleRate, frand(&seed)) - ((float)imWidth)/2.0f) +
                    camera->up * (float)(JITTER(y, sy, sampleRate, frand(&seed)) - ((float)imHeight)/2.0f));
            /*
             * A color filter for the reflected light that will actually reach the image plane.
             */
            float4 transmissionColor = {1.0f, 1.0f, 1.0f, 0.0f};
            float4 extinction = {0.0f, 0.0f, 0.0f, 0.0f};
            bool emissiveContributes = true;
            for (int rayDepth = 0; rayDepth <= 6; ++rayDepth)
            {
                float hitDistance = 0.0f;
                int hitIdx = sceneIntersection( &hitDistance, spheres, sphereCount, origin + direction * SMALL_F, direction);
                if (hitDistance > 0.0f)
                {
                    if (extinction.x > 0.0f) transmissionColor.x *= exp(log(extinction.x) * hitDistance);
                    if (extinction.y > 0.0f) transmissionColor.y *= exp(log(extinction.y) * hitDistance);
                    if (extinction.z > 0.0f) transmissionColor.z *= exp(log(extinction.z) * hitDistance);

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
                    origin += (direction * hitDistance);
                    float4 hitNormal = (origin - hitSphere->center) / hitSphere->radius;

                    /*
                     * Sample direct illumination
                     */
                    if (hitSphere->diffuse.w > 0.0f)
                    {
                        directIllumination(&pixelColor, spheres, sphereCount, origin + SMALL_F * hitNormal, hitNormal, transmissionColor, hitSphere->diffuse, &seed);
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
                        samplePhong(&direction, &hitNormal, hitSphere->specExp,r1, r2);
                        emissiveContributes = true;
                        //all light reflected, don't need to adjust transmission color.
                        transmissionColor *= fabs(dot(direction, hitNormal));
                    }
                    else if (p < (hitSphere->ks + hitSphere->diffuse.w))
                    {
                        sampleLambert(&direction, &hitNormal, r1, r2);
                        emissiveContributes = false;
                        transmissionColor *= hitSphere->diffuse;// * fabs(dot(direction, hitNormal));
                    }
                    else if (p < (hitSphere->ks + hitSphere->diffuse.w + hitSphere->extinction.w))
                    {
                        if (sampleRefraction(&direction, &hitNormal, hitSphere->ior, 1000000.0f, r1, r2)) {
                            extinction = hitSphere->extinction;
                        } else {
                            emissiveContributes = true;
                        }
                    }
                    else
                    {
                        rayDepth = maxDepth + 1;
                    }
                } // if hit object
                else // No object hit.
                {
                    hitDistance = intersectsBox(origin + direction * SMALL_F, direction, (float4)0.0f, BOX_SIZE, BOX_SIZE, BOX_SIZE);
                    if (hitDistance > 0)
                    {
                        origin += direction * hitDistance;

                        float4 boxNorm = boxNormal( (float4)0.0f, BOX_SIZE, BOX_SIZE, BOX_SIZE, origin, direction);

                        directIllumination(&pixelColor, spheres, sphereCount, origin + SMALL_F * boxNorm, boxNorm, transmissionColor, (float4)(1.0f, 1.0f, 1.0f, 1.0f), &seed);

                        sampleLambert(&direction, &boxNorm, frand(&seed), frand(&seed));
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

/*!
 *
 * \param pixelColor the irradiance of the sample point from the light source is added to this parameter.
 * \param spheres The scene geometry
 * \param sphereCount The number of spheres in 'spheres'
 * \param samplePoint The location to find irradiance at.
 * \param hitNormal The surface normal at the sample point.
 * \param transmissionColor
 * \param diffuse
 * \param seed Random seed value, updated on output if used.
 */
void directIllumination(float4 *pixelColor, __constant Sphere const *spheres, uint sphereCount, float4 const samplePoint, float4 const hitNormal, float4 const transmissionColor, float4 const diffuse, seed_value_t *seed)
{
    /*
     * Sample direct illumination
     */
    for (uint sphereNum = 0; sphereNum < sphereCount; ++sphereNum)
    {
        constant Sphere *light = &(spheres[sphereNum]);
        if (dot(light->emission, light->emission) != 0)
        {
            float4 lightDirection;
            float lightDist;
            float radiance = sphereEmissiveRadiance(&lightDirection, &lightDist, light->center, light->radius, samplePoint, frand(seed), frand(seed));
            float4 lightHitPt = samplePoint + lightDirection * lightDist;
            float cosWincident = fabs(dot(lightDirection, hitNormal));
            if (cosWincident > 0 && lightDist > 0)
            {
                float isectDist;
                uint isectObj = sceneIntersection(&isectDist, spheres, sphereCount, samplePoint, lightDirection);
                if ( isectObj == sphereNum )
                {

                    *pixelColor += transmissionColor *
                    light->emission * radiance * //Radiance emitted toward samplePoint
                    ((diffuse.w / PI) * diffuse) * //Material
                    ((cosWincident));
                }
            }
        }
    }

}

typedef struct {
    float4 hit_pt;
    float4 surface_normal;
} hit_info_t;

void directIllumination(float4 *irradiance, const hit_info_t *hit, __constant Sphere *geometry, const uint n_geometry) {
   for (uint sphereNum = 0; sphereNum < n_geometry; ++sphereNum)
    {
        constant Sphere *light = &(geometry[sphereNum]);
        if (light->emission.w != 0)
        {
            float4 lightDirection;
            float lightDist;
            float radiance = sphereEmissiveRadiance(&lightDirection, &lightDist, light->center, light->radius, hit->hit_pt, frand(seed), frand(seed));
            float4 lightHitPt = hit->hit_pt + lightDirection * lightDist;
            float cosWincident = fabs(dot(lightDirection, hit->surface_normal));
            if (cosWincident > 0 && lightDist > 0)
            {
                float isectDist;
                uint isectObj = sceneIntersection(&isectDist, geometry, n_geometry, hit->hit_pt, lightDirection);
                if ( isectObj == sphereNum )
                {
                    *irradiance += light->emission * radiance * (cosWincident);
                }
            }
        }
    }

}
