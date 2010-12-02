#ifndef _RT_COMMON_H
#define _RT_COMMON_H

#include "geometry.h"
#include "rng.h"
#include "materials.h"
#include "geometryFuncs.h"

/*!
 * \brief Gets a triangle by triangle index from global arrays of vertex index and vertex data.
 * \param triangle struct to store the fetched triangle in.
 * \param triangle_idx The index of the triangle to get (triples in tri_vert_indicies).
 * \param tri_verts The vertex data for triangles (3 floats per vertex).
 * \param tri_vert_indicies The triangle data, represented as offsets into tri_verts for each vertex.
 *
 * TODO: This is a very inefficient memory access pattern for the triangle data...
 */
void get_triangle(triangle_t *triangle, const uint triangle_idx, const __global vec3 *tri_verts, const __global uint *tri_vert_indicies) {
	const uint triangleOffs = triangle_idx * 3;
	/* Load triangle verts from global memory */
	const __global vec3 *vert0 = &tri_verts[tri_vert_indicies[triangleOffs]];
	const __global vec3 *vert1 = &tri_verts[tri_vert_indicies[triangleOffs + 1]];
	const __global vec3 *vert2 = &tri_verts[tri_vert_indicies[triangleOffs + 2]];

	/* Compute edges */
	triangle->v0 = *vert0;
	triangle->e1 = *vert1;
	triangle->e1.x -= triangle->v0.x;
	triangle->e1.y -= triangle->v0.y;
	triangle->e1.z -= triangle->v0.z;
	triangle->e2 = *vert2;
	triangle->e2.x -= triangle->v0.x;
	triangle->e2.y -= triangle->v0.y;
	triangle->e2.z -= triangle->v0.z;
}

int scene_intersection_tri( ray_t *ray, __global const vec3 *verts, __global const int *tri_indices, const uint n_geometry)
{
	float u, v;
	triangle_t tri;
	int hitObject = -1;
	for (int geom_idx = 0; geom_idx < n_geometry; ++geom_idx)
	{
		getTriangle(&tri, geom_idx, verts, tri_indices);
		if (intersects_triangle(ray, &u, &v, &tri)) {
			hitObject = geom_idx;
		}
	}
	return hitObject;
}

/*!
 * \brief Test to determine whether the specified ray intersects anything between it's tmin/tmax range values.
 *
 * \return true if the ray does not intersect anything in the range tmin -> tmax, false if it does intersect something.
 */
bool visibility_test_tri(const ray_t *ray, __global const vec3 *verts, __global const int *tri_indices, const uint n_geometry)
{
	float u, v;
	triangle_t tri;
	for (int geom_idx = 0; geom_idx < n_geometry; ++geom_idx) {
		getTriangle(&tri, geom_idx, verts, tri_indices);
		if (intersects_triangle_p(ray, &tri)) return false;
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
float4 sample_direct_illumination_tri(const hit_info_t *hit, __global const vec3 *verts, __global const int *tri_indices, const uint n_geometry, __constant Sphere *lights, const uint n_lights, seed_value_t *seed) {
	ray_t ray;
	ray.o.x = hit->hit_pt.x + hit->surface_normal.x * SMALL_F;
	ray.o.y = hit->hit_pt.y + hit->surface_normal.y * SMALL_F;
	ray.o.z = hit->hit_pt.z + hit->surface_normal.z * SMALL_F;
	float4 irradiance = (float4)0.0f;
	float u, v;
	triangle_t tri;

	for (uint light_idx = 0; light_idx < n_lights; ++light_idx)
	{
		constant Sphere *light = &(lights[light_idx]);
		if (light->mat.emission_power != 0)
		{
			sphereEmissiveRadiance(&ray, light->center, light->radius, frand(seed), frand(seed));
			if (visibility_test_tri(&ray, verts, tri_indices, n_geometry)) {
				const float cosWi = ray.d.x * hit->surface_normal.x + ray.d.y * hit->surface_normal.y + ray.d.z * hit->surface_normal.z;
				if (cosWi > 0)
				{
					irradiance.x += light->mat.emission.x * cosWi;
					irradiance.y += light->mat.emission.y * cosWi;
					irradiance.z += light->mat.emission.z * cosWi;
				}
			}
		}
	}
	return irradiance;
}

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
		if (light->mat.emission_power != 0)
		{
			sphereEmissiveRadiance(&ray, light->center, light->radius, frand(seed), frand(seed));
			if (visibility_test(&ray, geometry, n_geometry)) {
				const float cosWi = ray.d.x * hit->surface_normal.x + ray.d.y * hit->surface_normal.y + ray.d.z * hit->surface_normal.z;
				if (cosWi > 0)
				{
					irradiance.x += light->mat.emission.x * cosWi;
					irradiance.y += light->mat.emission.y * cosWi;
					irradiance.z += light->mat.emission.z * cosWi;
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
 * \param mat The sphere material.
 * \param seed Seed data for uniform random number generation.
 * \return true if the new sample ray has been initialized, false indicates that the ray should be terminated, and no more material sampling should occur (the ray is absorbed).
 */
bool sample_material(ray_t *ray, const hit_info_t *hit,
		constant material_t *mat, seed_value_t *seed) {

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
	if (p < mat->ks) {
		const float inv_pdf = 1.0 / samplePhong(&(ray->d), mat->specExp,
				r1, r2);
		ray->diffuse_bounce = false;
		/* cos (Wi) term */
		ray->propagation.x *= ray->d.z * inv_pdf;
		ray->propagation.y *= ray->d.z * inv_pdf;
		ray->propagation.z *= ray->d.z * inv_pdf;
	} else if (p < (mat->ks + mat->kd)) {
		//const float pdf =
		sampleLambert(&(ray->d), r1, r2);
		ray->diffuse_bounce = true;
		/*
		 * cos (Wi) * diffuse / PDF
		 *
		 * Assumes that sampleLambert has a cosine distribution, so that everything cancels except the diffuse color.
		 */
		ray->propagation.x *= mat->diffuse.x; //* evaluateLambert() * DOT(ray.d, hit.surface_normal)/ pdf;
		ray->propagation.y *= mat->diffuse.y;
		ray->propagation.z *= mat->diffuse.z;
	} else if (p < (mat->ks + mat->kd
					+ mat->kt)) {
		if (sampleRefraction(&(ray->propagation), ray, hit, mat->ior,
						mat->refExp, r1, r2)) {
			ray->extinction.x = mat->extinction.x;
			ray->extinction.y = mat->extinction.y;
			ray->extinction.z = mat->extinction.z;
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
		ray->diffuse_bounce = false;
	} else {
		return false;
	}
	ray->d = shading_to_world(ray->d, hit->surface_normal);
	return true;
}

#endif /* _RT_COMMON_H */
