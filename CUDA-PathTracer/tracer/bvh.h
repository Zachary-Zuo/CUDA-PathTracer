#ifndef H_BVH_H
#define H_BVH_H

#include "cutil_math.h"
#include "primitive.h"

struct BVHNode{
	BVHNode* left;
	BVHNode* right;  
	BBox bbox;
	bool is_leaf;
	std::vector<Primitive> primitives;

	BVHNode(){
		left = right = NULL;
	}
};

struct LinearBVHNode{
	BBox bbox;
	int second_child_offset;
	bool is_leaf;
	int start;
	int end;

	__host__ __device__ LinearBVHNode(){
		start = end = -1;
	}
};

class BVH{
public:
	LinearBVHNode* linear_root;
	int total_nodes;
	std::vector<Primitive> prims;
	BBox root_box;

public:
	BVH();

	void LoadOrBuildBVH(std::vector<Primitive>& primitives, std::string file);

private:
	void build(std::vector<Primitive>& primitives);
	BVHNode* split(std::vector<Primitive>& primitives, BBox& bbox);
	void flatten(BVHNode* node, int cur, int& next);
	void clearBVHNode(BVHNode* node);
};

#endif