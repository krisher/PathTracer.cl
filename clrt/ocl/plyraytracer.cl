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
					if (!ray.diffuse_bounce && hitSphere->mat.emission_power != 0)
					{
						pixelColor.x += ray.propagation.x * hitSphere->mat.emission.x;
						pixelColor.y += ray.propagation.y * hitSphere->mat.emission.y;
						pixelColor.z += ray.propagation.z * hitSphere->mat.emission.z;
					}

					/*
					 * Sample direct illumination
					 */
					if (hitSphere->mat.kd > 0.0f)
					{
						const float4 direct = sample_direct_illumination(&hit, spheres, sphereCount, &seed);
						const float scale = hitSphere->mat.kd * evaluateLambert();
						pixelColor.x += ray.propagation.x * direct.x * hitSphere->mat.diffuse.x * scale;
						pixelColor.y += ray.propagation.y * direct.y * hitSphere->mat.diffuse.y * scale;
						pixelColor.z += ray.propagation.z * direct.z * hitSphere->mat.diffuse.z * scale;
					}

					if (rayDepth == maxDepth) break;

					if (!sample_material(&ray, &hit, &(hitSphere->mat), &seed)) {
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

