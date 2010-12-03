#pragma OPENCL EXTENSION cl_amd_printf : enable
/*!
 * \file raytracer.cl
 * \brief OpenCL based Path-Tracer kernel.
 *
 */
#include "rtcommon.h"
#include "geometry.h"
#include "rng.h"
#include "materials.h"
#include "geometryFuncs.h"

/*
 * Half-width of the cube that encloses the scene.
 */
#define BOX_WIDTH 6.0f
#define BOX_HEIGHT 5.0f


void get_seed(seed_value_t *seed, __global const uint *seeds, const uint row_shift) {
    const uint seed_offs = ((get_global_id(1) + row_shift) % get_global_size(1)) * get_global_size(0) + get_global_id(0);
    seed->x = seeds[seed_offs];
    seed->y = seeds[(get_global_size(0) * get_global_size(1)) + seed_offs];
}

void put_seed(const seed_value_t seed, __global uint *seeds, const uint row_shift) {
    const uint seed_offs = ((get_global_id(1) + row_shift) % get_global_size(1)) * get_global_size(0) + get_global_id(0);
    seeds[seed_offs] = seed.x;
    seeds[(get_global_size(0) * get_global_size(1)) + seed_offs] = seed.y;
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
    get_seed(&seed, seeds, progressive);
//    const uint seed_offs = y * get_global_size(0) + x;
//    seed.x = seeds[seed_offs];
//    seed.y = seeds[(get_global_size(0) * get_global_size(1)) + seed_offs];

    /*
     * Color variable used to accumulate samples
     */
    float4 pixelColor = (float4)0.0f;
    for (uint sx = 0; sx < sampleRate; ++sx)
    {
        for (uint sy = 0; sy < sampleRate; ++sy)
        {
            //    TODO: sample camera aperture for DoF
            float4 rdirection = normalize(camera->view +
                    camera->right * ((float)(x + strat_rand(&seed, sx, sampleRate)) - ((float)imWidth)/2.0f) +
                    camera->up * ((float)(y + strat_rand(&seed, sy, sampleRate)) - ((float)imHeight)/2.0f));

            ray_t ray = {camera->position.x, camera->position.y, camera->position.z, rdirection.x, rdirection.y, rdirection.z, SMALL_F, INFINITY, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, false};
            vec3 result = trace_path(ray, spheres, sphereCount, maxDepth, BOX_WIDTH, BOX_HEIGHT, &seed);
            pixelColor.x += result.x;
            pixelColor.y += result.y;
            pixelColor.z += result.z;
        } // foreach sy
    } // foreach sx

    if (progressive > 0)
    {
        float4 color = vload4((y * imWidth + x), out);
        pixelColor = mix(color , pixelColor, 1.0f / (float)(progressive));
    }

    vstore4(pixelColor, (y * imWidth + x), out);
    put_seed(seed, seeds, progressive);
//    seeds[seed_offs] = seed.x;
//    seeds[get_global_size(0) * get_global_size(1) + seed_offs] = seed.y;
}

/*!
 * \brief Ray Tracer entry point (single sample)
 *
 * \param out An output buffer to store rgba color values (as one float4 per rgba value), with length imWidth * imHeight * 4.  32bits/component required to support progressive rendering.
 * \param camera The camera data used to generate eye rays.
 * \param spheres The scene content, including light sources.
 * \param sphereCount The number of objects in the scene.
 * \param imWidth The output image width, in pixels.
 * \param imHeight The output image height, in pixels.
 * \param maxDepth The maximum path length.
 * \param progressive When 0, data in the out-buffer is overwritten, when non-zero data is accumulated for progressive rendering.  In this case,
 *        the value determines the mixing weight.
 * \param seeds Seed data for the random number generator.  There should be one seed per pixel.  These values are used to generate a uniform random number sequence for each pixel, however the seed data is not updated in global memory.
 */
__kernel void raytrace_ss(__global float *out, __constant Camera *camera,
        __constant Sphere *spheres, uint const sphereCount,
        uint const imWidth, uint const imHeight,
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
     * Initial ray direction based on pixel location
     */
    float4 rdirection = normalize(camera->view +
            camera->right * ((float)(x + frand(&seed)) - ((float)imWidth)/2.0f) +
            camera->up * ((float)(y + frand(&seed)) - ((float)imHeight)/2.0f));

    ray_t ray = {camera->position.x, camera->position.y, camera->position.z, rdirection.x, rdirection.y, rdirection.z, SMALL_F, INFINITY, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, false};

    vec3 color = trace_path(ray, spheres, sphereCount, maxDepth, BOX_WIDTH, BOX_HEIGHT, &seed);

    float4 pixelColor = (float4)(color.x, color.y, color.z, 0.0f);
    if (progressive > 0)
    {
        float4 color = vload4((y * imWidth + x), out);
        pixelColor = mix(color , pixelColor, 1.0f / (float)(progressive));
    }
    vstore4(pixelColor, (y * imWidth + x), out);
    seeds[seed_offs] = seed.x;
    seeds[get_global_size(0) * get_global_size(1) + seed_offs] = seed.y;
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
__kernel void raytrace_tris(__global float *out, __constant Camera *camera,
		__constant Sphere *spheres, uint const sphereCount,
		uint const imWidth, uint const imHeight, uint const sampleRate,
		uint const maxDepth, uint const progressive, __global uint *seeds,
		__global const vec3 *tri_verts, __global const int *tri_vert_idx, uint const n_tris)
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


			vec3 result = trace_path_tri(ray, tri_verts, tri_vert_idx, n_tris, spheres, sphereCount, maxDepth, BOX_WIDTH, BOX_HEIGHT, &seed);
            pixelColor.x += result.x;
            pixelColor.y += result.y;
            pixelColor.z += result.z;
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
}
