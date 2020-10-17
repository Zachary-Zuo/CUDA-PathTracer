#pragma once
#include "device_launch_parameters.h"
#include "../Sampling.h"
#include <thrust/random.h>
#include <cuda_runtime.h>
#include "../Resource.h"

__global__ void Mlt(int iter, int maxDepth);