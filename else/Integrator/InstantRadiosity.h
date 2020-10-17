#pragma once
#include "device_launch_parameters.h"
#include "../Sampling.h"
#include <thrust/random.h>
#include <cuda_runtime.h>
#include "../Resource.h"

#define IR_MAX_VPLS 32

__global__ void GenerateVpl(int iter, int maxDepth);
__global__ void InstantRadiosity(int iter, int vplIter, int maxDepth, float bias);