#pragma once
#include "device_launch_parameters.h"
#include "../Sampling.h"
#include <thrust/random.h>
#include <cuda_runtime.h>
#include "../Resource.h"

__global__ void LightTracingInit();
__global__ void LightTracing(int iter, int maxDepth);