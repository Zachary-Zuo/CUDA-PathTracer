#pragma once
#include "common.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "bbox.h"
#include "intersection.h"
#include "Sampling.h"
#include "Triangle.h"


class Scene;
class Mesh{
public:
	std::vector<Vertex> vertices;
	std::vector<Triangle> triangles;
	int matIdx;
	int bssrdfIdx;

public:
	void LoadObjFromFile(std::string filename, unsigned int flags, glm::mat4& trs);

private:
	void processNode(aiNode* node, const aiScene* scene, glm::mat4& trs);
	void processMesh(aiMesh* aimesh, const aiScene* scene, glm::mat4& trs);
	float3 genTangent(int idx1, int idx2, int idx3);
};