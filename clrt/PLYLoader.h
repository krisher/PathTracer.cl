#ifndef _PLY_LOADER_H
#define _PLY_LOADER_H

#include <stdio.h>
#include "ocl/geometry.h"
#include "ply/ply.h"

class PLYLoader {

private:
	static PlyProperty vert_props[];
	static PlyProperty face_props[];

	vec3 *vertexData;
	int *triangles;
	unsigned int vertCount;
	unsigned int triCount;

public:
	PLYLoader(): vertexData(NULL) {};
	PLYLoader(FILE *fp);
};


#endif /* _PLY_LOADER_H */
