#pragma once
#include "device_launch_parameters.h"
#include "../Sampling.h"
#include <thrust/random.h>
#include <cuda_runtime.h>
#include "../Resource.h"

__global__ void Ao(int iter, float maxDist);