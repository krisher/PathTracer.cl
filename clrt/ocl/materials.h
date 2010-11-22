/*!
 * \file materials.h
 * \brief OpenCL material sampling functions
 */
#ifndef MATERIAL_H
#define MATERIAL_H

#include "geometryFuncs.h"

#define M_PI_F 3.14159265358979323846f
#define M_1_PI_F 0.318309886f


/*!
 * \brief Samples irradiance direction for modified Phong specular reflectance model.
 *
 *
 * \param ray ray.[dx,dy,dz] define the vector wo, and are updated upon output to store the resulting sample direction, wi.
 * \param hit Contains the surface normal at the hit point.
 * \param specExp The Phong specular exponent term.
 * \param r1 A uniform random number
 * \param r2 A uniform random number
 */
float samplePhong(ray_t *ray, const hit_info_t *hit,
		float const specExp, float const r1, float const r2) {
	/*
	 *
	 */
	float4 surfaceNormal = hit->surface_normal;
	float4 sampleDirection = (float4)(ray->dx, ray->dy, ray->dz, 0.0f);
	if (dot(sampleDirection, surfaceNormal) > 0.0f) {
		surfaceNormal *= -1.0f;
	}
	if (specExp < 100000.0f) {

		/*
		 * Compute the mirror reflection vector...
		 */
		sampleDirection = normalize(sampleDirection + (surfaceNormal)
				* (-2.0f * dot(sampleDirection, surfaceNormal)));

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
	ray->dx = sampleDirection.x;
	ray->dy = sampleDirection.y;
	ray->dz = sampleDirection.z;

	ray->ox = hit->hit_pt.x;
	ray->oy = hit->hit_pt.y;
	ray->oz = hit->hit_pt.z;

	ray->tmin = 1e-5f;
	ray->tmax = INFINITY;

	return 1.0f;
}

/*!
 * \brief Lambert diffuse reflectance model sampling function.
 *
 * Given a surface normal and two uniform random variables, samples the irradiance direction defined by the Lambert reflectance model.
 *
 * \param ray ray.[dx,dy,dz] define the vector wo, and are updated upon output to store the resulting sample direction, wi.
 * \param hit Contains the surface normal at the hit point.
 * \param r1 a uniform random variable.
 * \param r2 a uniform random variable.
 */
float sampleLambert(ray_t *ray, const hit_info_t *hit,
		float const r1, float const r2) {
	/*
	 * Cosine-weighted sampling about the surface normal:
	 *
	 * Probability of direction Ko = 1/pi * cos(theta) where theta is the
	 * angle between the surface normal and Ko.
	 *
	 * The polar angle about the normal is chosen from a uniform distribution
	 * 0..2pi
	 */
	const float cosTheta = sqrt(1.0f - r1);
	const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
	const float phi = 2.0f * M_PI_F * r2;
	const float xb = sinTheta * cos(phi);
	const float yb = sinTheta * sin(phi);

	/*
	 * Construct orthonormal basis with vectors:
	 *
	 * surfaceNormal, directionOut, nv
	 */
	float4 sampleDirection = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
	if (fabs(dot(sampleDirection, hit->surface_normal)) > 0.9f) {
		// Small angle, pick a better vector...
		sampleDirection.x = -1.0f;
		sampleDirection.y = 0.0f;
	}
	sampleDirection = normalize(cross(sampleDirection, hit->surface_normal));
	const float4 nv = cross(hit->surface_normal, sampleDirection);
	/*
	 * Use the x,y,z values calculated above as coordinates in the ONB...
	 */
	sampleDirection *= xb;
	sampleDirection += hit->surface_normal * cosTheta + nv * yb;

	ray->dx = sampleDirection.x;
	ray->dy = sampleDirection.y;
	ray->dz = sampleDirection.z;

	ray->ox = hit->hit_pt.x;
	ray->oy = hit->hit_pt.y;
	ray->oz = hit->hit_pt.z;

	ray->tmin = 1e-5f;
	ray->tmax = INFINITY;

	return cosTheta * M_1_PI_F;
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
bool sampleRefraction(float4 *sampleDirection, float4 const *surfaceNormal,
		float const ior, float const blurExp, float const r1, float const r2) {
	float4 sNormal = *surfaceNormal;
	float cosSampleAndNormal = dot(*sampleDirection, sNormal);
	float rIdxRatio;
	bool exiting;
	if (cosSampleAndNormal <= 0.0f) {
		rIdxRatio = 1.0f / ior;
		cosSampleAndNormal = -cosSampleAndNormal;
		exiting = false;
	} else {
		rIdxRatio = ior;
		sNormal *= -1.0f;
		exiting = true;
	}

	float snellRoot = 1.0f - (rIdxRatio * rIdxRatio * (1.0f
			- cosSampleAndNormal * cosSampleAndNormal));
	if (snellRoot < 0.0f) {
		/*
		 * Total internal reflection
		 */
		*sampleDirection = normalize(*sampleDirection + (sNormal) * (-2.0f
				* dot(*sampleDirection, sNormal)));
		return exiting;
	} else {
		/*
		 * Refraction
		 */
		*sampleDirection *= rIdxRatio;
		*sampleDirection += sNormal * (rIdxRatio * cosSampleAndNormal - sqrt(
				snellRoot));
		if (blurExp < 100000.0f) {
			/*
			 * Idential to phong, except we substitude the refraction direction
			 * for the mirror reflection vector.
			 */
			float cosA = pow(r1, 1.0f / (blurExp + 1.0f));

			/*
			 * Generate another random value, uniform between 0 and 2pi, which
			 * is the angle around the mirror reflection vector
			 */
			float phi = 2.0f * M_PI_F * r2;
			float sinTheta = sqrt(1.0f - cosA * cosA);
			float xb = cos(phi) * sinTheta;
			float yb = sin(phi) * sinTheta;

			/*
			 * Construct an ortho-normal basis using the reflection vector as
			 * one axis, and arbitrary (perpendicular) vectors for the other two
			 * axes. The orientation of the coordinate system about the
			 * reflection vector is irrelevant since xb and yb are generated
			 * from a uniform random variable.
			 */
			float4 u = (float4) (0.0f, 1.0f, 0.0f, 0.0f);
			float const cosAng = dot(*sampleDirection, u);
			if (cosAng > 0.9f || cosAng < -0.9f) {
				// Small angle, pick a better vector...
				u.x = -1.0f;
				u.y = 0.0f;
			}
			u = normalize(cross(u, *sampleDirection));
			float4 const v = cross(u, *sampleDirection);
			;

			*sampleDirection *= cosA;
			*sampleDirection += u * xb + v * yb;
			if (dot(*sampleDirection, *surfaceNormal) < 0.0f)
				*sampleDirection -= 2.0f * (u * xb + v * yb);
		}
	}
	return !exiting;
}

/*!
 * \brief Given a spherical emissive light source, a sample point (to sample illumination for), computes a ray direction from 
 *        the sample point to a point on the light, the distance to that point, and the radiance that is emitted from the light
 *        toward the sample point.
 *
 * \param directionOut On input, ray.[ox,oy,oz] indicate the point to sample illumination at, on output, ray.[dx,dy,dz,tmin,tmax] are initialized.
 * \param sphereCenter the center of the emissive sphere.
 * \param radius The radius of the emissive sphere.
 * \param r1 A uniform random variable.
 * \param r2 A uniform random variable.
 */
float sphereEmissiveRadiance(ray_t *ray, float4 const sphereCenter,
		float const radius, float const r1, float const r2) {
	float4 origin = (float4) (ray->ox, ray->oy, ray->oz, 0);
	float4 direction = sphereCenter - origin;
	ray->tmax = length(direction);
	/*
	 * The maximum angle from originToCenter for a ray eminating from origin
	 * that will hit the sphere.
	 */
	float const sinMaxAngle = radius / ray->tmax;
	float const cosMaxAngle = sqrt(1.0f - sinMaxAngle * sinMaxAngle);

	/*
	 * Uniform sample density over the solid angle subtended by the sphere
	 * wrt. the origin point. Taken from Shirley and Morely book.
	 */
	float const cosRandomAzimuth = 1.0f + r1 * (cosMaxAngle - 1.0f);
	float const sinRandomAzimuth = sqrt(1.0f - cosRandomAzimuth
			* cosRandomAzimuth);
	float const randomPolar = 2.0f * M_PI_F * r2;

	/*
	 * Construct an orthonormal basis around the direction vector
	 */
	direction /= ray->tmax;

	float4 nu = (float4) (0.0f, 1.0f, 0.0f, 0.0f);
	float cosAng = dot(nu, direction);
	if (cosAng < -0.9f || cosAng > 0.9f) {
		nu.x = 1.0f;
		nu.y = 0.0f;
	}
	nu = normalize(cross(nu, direction));
	float4 nv = cross(direction, nu);

	direction *= cosRandomAzimuth;
	direction += nu * cos(randomPolar) * sinRandomAzimuth + nv * sin(
			randomPolar) * sinRandomAzimuth;

	ray->dx = direction.x;
	ray->dy = direction.y;
	ray->dz = direction.z;
	ray->tmin = 1e-5f;
	ray->tmax = intersectSphere(ray, sphereCenter, radius)
			- 1e-5f;

	/*
	 * Multiply by 1/distribution of light samples over the hemisphere around the hit point.
	 *
	 * Since this is only called for direct interaction with diffuse surfaces, it has been optimized based on the
	 * assumption that the directIllumination function will call this
	 */
	return (2.0f * M_PI_F * (1.0f - cosMaxAngle));
}

#endif /* MATERIAL_H */
