#ifndef GEOMETRY_FUNCS_H
#define GEOMETRY_FUNCS_H

#include "geometry.h"

#define DOT(a,b) (a.x * b.x + a.y * b.y + a.z * b.z)

/*!
 * \brief Ray-Sphere intersection test
 * \param ray The ray to test intersection with.
 * \param center The sphere center.
 * \param radius The sphere radius.
 * \return The distance to intersection, or 0 if there was no intersection.
 */
float intersectSphere(const ray_t *ray, float4 const center, float const radius) {
    float tOx = ray->o.x - center.x;
    float tOy = ray->o.y - center.y;
    float tOz = ray->o.z - center.z;
    float originFromCenterDistSq = tOx * tOx + tOy * tOy + tOz * tOz;
    float B = tOx * ray->d.x + tOy * ray->d.y + tOz * ray->d.z;
    float D = B * B - (originFromCenterDistSq - radius * radius);
    if (D > 0) {
        float sqrtD = sqrt(D);
        return (sqrtD < -B) ? -B - sqrtD : -B + sqrtD;
    }
    return 0.0f;
}

void boxNormal(const ray_t *ray, hit_info_t *hit, float4 const center,
        float const xSize, float const ySize, float const zSize) {
    const float4 hitPt = hit->hit_pt;
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
    /*
     * Normal should point toward ray origin.
     */
    if (hit->surface_normal.x * ray->d.x + hit->surface_normal.y * ray->d.y
            + hit->surface_normal.z * ray->d.z > 0.0f)
        hit->surface_normal *= -1.0f;
}

float intersectsBox(const ray_t *ray, const float4 center, const float xSize,
        const float ySize, const float zSize) {
    float nearIsect = 0.0f;
    float farIsect = INFINITY;

    float t1, t2;
    if (ray->d.x != 0) {
        t1 = (center.x - xSize - ray->o.x) / ray->d.x;
        t2 = (center.x + xSize - ray->o.x) / ray->d.x;
        nearIsect = min(t1, t2);
        farIsect = max(t1, t2);
    } else {
        /*
         * Ray runs parallel to x, can only intersect if origin x is between +/- xSize
         */
        if (ray->o.x > (center.x + xSize) || ray->o.x < (center.x - xSize)) {
            return 0;
        }
    }

    if (ray->d.y != 0) {
        t1 = (center.y - ySize - ray->o.y) / ray->d.y;
        t2 = (center.y + ySize - ray->o.y) / ray->d.y;
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
        if (ray->o.y > (center.y + ySize) || ray->o.y < (center.y - ySize)) {
            return 0;
        }
    }

    if (ray->d.z != 0) {
        t1 = (center.z-zSize - ray->o.z) / ray->d.z;
        t2 = (center.z+zSize - ray->o.z) / ray->d.z;
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
        if (ray->o.z + center.z > zSize || ray->o.z + center.z < -zSize) {
            return 0;
        }
    }
    if (nearIsect > farIsect || farIsect < 0) {
        return INFINITY;
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
