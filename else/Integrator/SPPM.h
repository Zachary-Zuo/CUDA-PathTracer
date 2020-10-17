#pragma once
#include "device_launch_parameters.h"
#include "../Sampling.h"
#include <thrust/random.h>
#include <cuda_runtime.h>
#include "../Resource.h"

struct VisiblePoint {
	float3 ld; //direct light
	float3 ind; //indirect light
	float3 beta; //throughput
	float3 dir;
	Intersection isect;

	float3 tau;
	float radius;
	float n;
	bool valid = false;
};

struct CPUGridNode {
	std::vector<int> vpIdx;
};

extern VisiblePoint* device_vps;
extern int* device_vpIdx;
extern int* device_vpOffset;


__global__ void StochasticProgressivePhotonmapperFP(int iter, int maxDepth, float initRadius = 0.5f);
void StochasticProgressivePhotonmapperBuildHashTable(int width, int height);
__global__ void StochasticProgressivePhotonmapperSP(int iter, int maxDepth);
__global__ void StochasticProgressivePhotonmapperTP(int iter, int photonsPerIteration);
__global__ void StochasticProgressivePhotonmapperInit(VisiblePoint* v, int* offset);