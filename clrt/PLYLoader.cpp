#include "PLYLoader.h"
#include <iostream>

PlyProperty PLYLoader::vert_props[3] = { { "x", Float32, Float32,
		offsetof(vec3,x), 0, 0, 0, 0 }, { "y", Float32, Float32,
		offsetof(vec3,y), 0, 0, 0, 0 }, { "z", Float32, Float32,
		offsetof(vec3,z), 0, 0, 0, 0 } };

typedef struct {
	unsigned char vertCount;
	int* vertIdx;
} face_t;

PlyProperty PLYLoader::face_props[1] = { /* list of property information for a face */
{ "vertex_indices", Int32, Int32, offsetof(face_t,vertIdx), 1, Uint8, Uint8,
		offsetof(face_t,vertCount) }, };

PLYLoader::PLYLoader(FILE *fp) {
	std::cout << "Loading PLY from disk..." << std::endl;
	int i, j;
	int elem_count;
	char *elem_name;

	/*** Read in the original PLY object ***/

	PlyFile *in_ply = ::read_ply(fp);
	if (vertexData) {
		delete vertexData;
		vertexData = NULL;
	}

	for (i = 0; i < in_ply->num_elem_types; i++) {

		/* prepare to read the i'th list of elements */
		elem_name = setup_element_read_ply(in_ply, i, &elem_count);

		if (equal_strings("vertex", elem_name)) {

			/* create a vertex list to hold all the vertices */
			vertexData = new vec3[elem_count];//(vec3 *) malloc (sizeof (vec3 *) * elem_count);
			vertCount = elem_count;

			/* set up for getting vertex elements */

			setup_property_ply(in_ply, &vert_props[0]);
			setup_property_ply(in_ply, &vert_props[1]);
			setup_property_ply(in_ply, &vert_props[2]);
			//      vert_other = get_other_properties_ply (in_ply,
			//					     offsetof(Vertex,other_props));

			/* grab all the vertex elements */
			for (j = 0; j < elem_count; j++) {
				//vlist[j] = (Vertex *) malloc (sizeof (Vertex));
				get_element_ply(in_ply, (void *) &(vertexData[j]));
			}
		} else if (equal_strings("face", elem_name)) {
			/* create a list to hold all the face elements */
			face_t *flist = new face_t[elem_count];
			int *triangles = new int[elem_count * 3];
			triCount = elem_count;

			/* set up for getting face elements */

			setup_property_ply(in_ply, &face_props[0]);
//			face_other = get_other_properties_ply(in_ply,
//					offsetof(Face,other_props));

			/* grab all the face elements */
			for (j = 0; j < elem_count; j++) {
//				flist[j] = (Face *) malloc(sizeof(Face));
				get_element_ply(in_ply, (void *) &(flist[j]));
				if (flist[j].vertCount == 2) {
					triangles[j * 3] = flist[j].vertIdx[0];
					triangles[j * 3 + 1] = flist[j].vertIdx[1];
					triangles[j * 3 + 2] = flist[j].vertIdx[2];
				}
			}
			delete flist;
			this->triangles = triangles;
		} else
			get_other_element_ply(in_ply);
	}

	close_ply(in_ply);

	std::cout << "Loaded: " << vertCount << " vertices, " << triCount << " faces." << std::endl;
}
