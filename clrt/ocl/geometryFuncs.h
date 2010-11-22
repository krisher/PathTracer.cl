

#ifndef GEOMETRY_FUNCS_H
#define GEOMETRY_FUNCS_H

float intersectSphere(float4 const center, float const radius, float4 const origin, float4 const direction)
{
  float4 tO = origin - center;
  float originFromCenterDistSq = dot(tO, tO);
  float B = dot(tO, direction);
  float D = B * B - (originFromCenterDistSq - radius * radius);
  if (D > 0) 
    {
      float sqrtD = sqrt(D);
      return (sqrtD < -B) ? -B - sqrtD : -B + sqrtD;
    }
  return 0.0f;
}

float4 boxNormal(float4 const center, float const xSize, float const ySize, float const zSize, float4 const hitPt, float4 const direction) 
{
      // Figure out which face the intersection occurred on
  float4 isectNormal;
  float xDist = fabs(fabs(hitPt.x) - xSize);
  float yDist = fabs(fabs(hitPt.y) - ySize);
  float zDist = fabs(fabs(hitPt.z) - zSize);
  if (xDist < yDist) 
    {
      if (xDist < zDist) 
	  isectNormal = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
      else 
	isectNormal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
    } 
  else if (yDist < zDist) 
    {
      isectNormal = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
    } 
  else 
    {
      isectNormal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
    }
  if (dot(isectNormal, direction) > 0.0f)
    isectNormal *= -1.0f;
  return isectNormal;
}



float intersectsBox(float4 const origin, float4 const direction, float4 center, float xSize, float ySize, float zSize) {
  float nearIsect = 0; 
  float farIsect = 0;

  float t1, t2;
      if (direction.x != 0) {
         t1 = (-xSize - origin.x + center.x) / direction.x;
         t2 = (xSize - origin.x + center.x) / direction.x;
	 nearIsect = min(t1, t2);
	 farIsect = max(t1, t2);
      } else {
         /*
          * Ray runs parallel to x, can only intersect if origin x is between +/- xSize
          */
         if (origin.x + center.x > xSize || origin.x + center.x < -xSize) {
            return 0;
         }
	 nearIsect = 1e10f;
	 farIsect = 0;
      }

      if (direction.y != 0) {
         t1 = (-ySize - origin.y + center.y) / direction.y;
         t2 = (ySize - origin.y + center.y) / direction.y;
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
         if (origin.y + center.y > ySize || origin.y + center.y < -ySize) {
            return 0;
         }
      }

      if (direction.z != 0) {
         t1 = (-zSize - origin.z + center.z) / direction.z;
         t2 = (zSize - origin.z + center.z) / direction.z;
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
         if (origin.z  + center.z> zSize || origin.z  + center.z< -zSize) {
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

int sceneIntersection( float *hitDistance, __constant const Sphere * spheres, uint sphereCount, float4 const origin, float4 const direction)
{
  *hitDistance = 0.0f;
  int hitObject = -1;
  for (int sphereNum = 0; sphereNum < sphereCount; ++sphereNum) 
    {
      __constant const Sphere *sphere = &(spheres[sphereNum]);
      float d = intersectSphere(sphere->center, sphere->radius, origin, direction);
      if (d > 1e-5f && (( d < *hitDistance) || *hitDistance == 0.0f))
	{
	  hitObject = sphereNum;
	  *hitDistance = d;
	}
    }
  return hitObject;
} 


#endif /* GEOMETRY_FUNCS_H */
