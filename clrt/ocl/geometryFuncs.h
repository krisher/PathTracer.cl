
#ifndef GEOMETRY_FUNCS_H
#define GEOMETRY_FUNCS_H

/*!
 * \brief Ray structure used in OpenCL.
 */
typedef struct {
	float ox;
	float oy;
	float oz;
	float dx;
	float dy;
	float dz;
	float tmin;
	float tmax;
} ray_t;

typedef struct {
	float4 hit_pt;
	float4 surface_normal;
} hit_info_t;

/*!
 * \brief Ray-Sphere intersection test
 * \param ray The ray to test intersection with.
 * \param center The sphere center.
 * \param radius The sphere radius.
 * \return The distance to intersection, or 0 if there was no intersection.
 */
float intersectSphere(const ray_t *ray, float4 const center, float const radius) {
	float tOx = ray->ox - center.x;
	float tOy = ray->oy - center.y;
	float tOz = ray->oz - center.z;
	float originFromCenterDistSq = tOx * tOx + tOy * tOy + tOz * tOz;
	float B = tOx * ray->dx + tOy * ray->dy + tOz * ray->dz;
	float D = B * B - (originFromCenterDistSq - radius * radius);
	if (D > 0) {
		float sqrtD = sqrt(D);
		return (sqrtD < -B) ? -B - sqrtD : -B + sqrtD;
	}
	return 0.0f;
}

void boxNormal(const ray_t *ray, hit_info_t *hit, float4 const center, float const xSize, float const ySize,
		float const zSize) {
	float4 hitPt = hit->hit_pt;
	// Figure out which face the intersection occurred on
	float xDist = fabs(fabs(hitPt.x) - xSize);
	float yDist = fabs(fabs(hitPt.y) - ySize);
	float zDist = fabs(fabs(hitPt.z) - zSize);
	if (xDist < yDist) {
		if (xDist < zDist)
			hit->surface_normal = (float4) (1.0f, 0.0f, 0.0f, 0.0f);
		else
			hit->surface_normal = (float4) (0.0f, 0.0f, 1.0f, 0.0f);
	} else if (yDist < zDist) {
		hit->surface_normal = (float4) (0.0f, 1.0f, 0.0f, 0.0f);
	} else {
		hit->surface_normal = (float4) (0.0f, 0.0f, 1.0f, 0.0f);
	}
	if (hit->surface_normal.x * ray->dx + hit->surface_normal.y * ray->dy + hit->surface_normal.z * ray->dz > 0.0f)
		hit->surface_normal *= -1.0f;
}

float intersectsBox(const ray_t *ray, float4 center,
		float xSize, float ySize, float zSize) {
	float nearIsect = 0;
	float farIsect = 0;

	float t1, t2;
	if (ray->dx != 0) {
		t1 = (-xSize - ray->ox + center.x) / ray->dx;
		t2 = (xSize - ray->ox + center.x) / ray->dx;
		nearIsect = min(t1, t2);
		farIsect = max(t1, t2);
	} else {
		/*
		 * Ray runs parallel to x, can only intersect if origin x is between +/- xSize
		 */
		if (ray->ox + center.x > xSize || ray->ox + center.x < -xSize) {
			return 0;
		}
		nearIsect = 1e10f;
		farIsect = 0;
	}

	if (ray->dy != 0) {
		t1 = (-ySize - ray->oy + center.y) / ray->dy;
		t2 = (ySize - ray->oy + center.y) / ray->dy;
		if (t1 > t2) {
			nearIsect = max(t2, nearIsect);
			farIsect = min(t1, farIsect);
		} else {
			nearIsect = max(t1, nearIsect);
			farIsect = min(t2, farIsect);
		}
	} else {
		/*
		 * Ray runs parallel to y, can only intersect if origin y is between +/- ySize
		 */
		if (ray->oy + center.y > ySize || ray->oy + center.y < -ySize) {
			return 0;
		}
	}

	if (ray->dz != 0) {
		t1 = (-zSize - ray->oz + center.z) / ray->dz;
		t2 = (zSize - ray->oz + center.z) / ray->dz;
		if (t1 > t2) {
			nearIsect = max(t2, nearIsect);
			farIsect = min(t1, farIsect);
		} else {
			nearIsect = max(t1, nearIsect);
			farIsect = min(t2, farIsect);
		}
	} else {
		/*
		 * Ray runs parallel to z, can only intersect if origin z is between +/- zSize
		 */
		if (ray->oz + center.z > zSize || ray->oz + center.z < -zSize) {
			return 0;
		}
	}
	if (nearIsect > farIsect || farIsect < 0) {
		return 0;
	}
	if (nearIsect < 0) {
		return farIsect;
	}
	return nearIsect;
}

int sceneIntersection( ray_t *ray, __constant const Sphere *geometry, const uint n_geometry)
{
	int hitObject = -1;
	for (int sphereNum = 0; sphereNum < n_geometry; ++sphereNum)
	{
		__constant const Sphere *sphere = &(geometry[sphereNum]);
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
bool visibilityTest(const ray_t *ray, __constant Sphere * geometry, const uint n_geometry)
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

#endif /* GEOMETRY_FUNCS_H */
