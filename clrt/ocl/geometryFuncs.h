#ifndef GEOMETRY_FUNCS_H
#define GEOMETRY_FUNCS_H

#include "geometry.h"

#define DOT(a,b) (a.x * b.x + a.y * b.y + a.z * b.z)

/*!
 * \brief calculates a vector that is perpendicular to the specified vector.
 * \param vec A unit length vector.
 * \return A unit length vector that is perpendicular to vec.
 */
vec3 perpendicular_vector(const vec3 vec) {
    vec3 tangentX;
    if (fabs(vec.y) > 0.9f) { //Maximize the probability of taking one branch over the other for better branch coherency.
        const float inv_len = rsqrt(vec.z * vec.z + vec.y * vec.y);
        tangentX.x = 0.0f;
        tangentX.y = -vec.z * inv_len;
        tangentX.z = vec.y * inv_len;
    } else {
        const float inv_len = rsqrt(vec.z * vec.z + vec.x * vec.x);
        tangentX.x = vec.z * inv_len;
        tangentX.y = 0.0f;
        tangentX.z = -vec.x * inv_len;
    }
    return tangentX;
}

vec3 cross_vec(const vec3 v1, const vec3 v2) {
    return (vec3) {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x
        * v2.y - v1.y * v2.x};
}

/*!
 * \brief Ray-Sphere intersection test
 * \param ray The ray to test intersection with.
 * \param center The sphere center.
 * \param radius The sphere radius.
 * \return The distance to intersection, or 0 if there was no intersection.
 */
float intersectSphere(const ray_t *ray, vec3 const center, float const radius) {
    float tOx = ray->o.x - center.x;
    float tOy = ray->o.y - center.y;
    float tOz = ray->o.z - center.z;
    float originFromCenterDistSq = tOx * tOx + tOy * tOy + tOz * tOz;
    float B = tOx * ray->d.x + tOy * ray->d.y + tOz * ray->d.z;
    float D = B * B - (originFromCenterDistSq - radius * radius);
    if (D > 0) {
        float sqrtD = sqrt(D);
        /* return the smallest positive value */
        if (-B - sqrtD > ray->tmin)
            return -B - sqrtD;
        return -B + sqrtD;
    }
    return 0.0f;
}

void sphereNormal(hit_info_t *hit, vec3 const center, float const radius) {
    const float inv_radius = 1.0f / radius;
    hit->surface_normal.x = (hit->hit_pt.x - center.x) * inv_radius;
    hit->surface_normal.y = (hit->hit_pt.y - center.y) * inv_radius;
    hit->surface_normal.z = (hit->hit_pt.z - center.z) * inv_radius;
}

void boxNormal(const ray_t *ray, hit_info_t *hit, float const xSize,
        float const ySize, float const zSize) {
    const vec3 hitPt = hit->hit_pt;
    // Figure out which face the intersection occurred on
    const float x_side_dist = fabs(fabs(hitPt.x) - xSize);
    const float y_side_dist = fabs(fabs(hitPt.y) - ySize);
    const float z_side_dist = fabs(fabs(hitPt.z) - zSize);
    if (x_side_dist < y_side_dist && x_side_dist < z_side_dist)
        hit->surface_normal = (vec3) {-hitPt.x / fabs(hitPt.x), 0.0f, 0.0f};
    else if (y_side_dist < z_side_dist)
        hit->surface_normal = (vec3) {0.0f, -hitPt.y / fabs(hitPt.y), 0.0f};
    else
        hit->surface_normal = (vec3) {0.0f, 0.0f, -hitPt.z / fabs(hitPt.z)};
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
        if (fabs(ray->o.x - center.x) > xSize) {
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
        if (fabs(ray->o.y - center.y) > ySize) {
            return 0;
        }
    }

    if (ray->d.z != 0) {
        t1 = (center.z - zSize - ray->o.z) / ray->d.z;
        t2 = (center.z + zSize - ray->o.z) / ray->d.z;
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
        if (fabs(ray->o.z - center.z) > zSize) {
            return 0;
        }
    }
    if (nearIsect > farIsect || farIsect < ray->tmin) {
        return INFINITY;
    }
    if (nearIsect < ray->tmin) {
        return farIsect;
    }
    return nearIsect;
}

/*!
 * \brief Moller-Trumbore ray/triangle intersection (based on Java implementation).
 * \param ray The ray to test intersection with.  ray->tmax is updated with the intersection distance if one occurred.
 * \param u Output for the barycentric u coordinate (edge between v0 -> v1)
 * \param v Output for the barycentric v coordinate (edge between v0 -> v2)
 * \param triangle The triangle to test intersection with.
 * \return true if an intersection was found (and ray->tmax, u, and v were all assigned), false if no intersection was found (and none of the parameters were modified).
 */
bool intersects_triangle(ray_t *ray, float *u, float *v,
        const triangle_t *triangle) {
    const vec3 p = cross_vec(ray->d, triangle->e2);
    float divisor = DOT(p, triangle->e1);
    /*
     * Ray nearly parallel to triangle plane, or degenerate triangle...
     */
    if (fabs(divisor) < SMALL_F) {
        return false;
    }
    divisor = 1.0f / divisor;
    vec3 translated_origin = ray->o;
    translated_origin.x -= triangle->v0.x;
    translated_origin.y -= triangle->v0.y;
    translated_origin.z -= triangle->v0.z;

    const vec3 q = cross_vec(translated_origin, triangle->e1);
    /*
     * Barycentric coords also result from this formulation, which could be useful for interpolating attributes
     * defined at the vertex locations:
     */
    const float e0dist = DOT(p, translated_origin) * divisor;
    if (e0dist < 0 || e0dist > 1) {
        return false;
    }

    const float e1dist = DOT(q, ray->d) * divisor;
    if (e1dist < 0 || e1dist + e0dist > 1) {
        return false;
    }

    const float isectDist = DOT(q, triangle->e2) * divisor;

    if (isectDist > ray->tmax || isectDist < ray->tmin) {
        return false;
    }

    /* Found intersection, store tri index, isect dist, and barycentric coords. */
    ray->tmax = isectDist;
    *u = e0dist;
    *v = e1dist;
    return true;
}

#endif /* GEOMETRY_FUNCS_H */
