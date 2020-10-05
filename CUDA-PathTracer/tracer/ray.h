#pragma once
#include "common.h"
#include "cutil_math.h"

class Medium;
class Ray
{
public:
	float3 origin;
	float3 destination;
	float tmin, tmax;
	Medium *medium;

public:
	__host__ __device__ Ray()
		: origin(make_float3(0, 0, 0)),
		  destination(make_float3(0, 0, 0)),
		  medium(nullptr),
		  tmin(0.001),
		  tmax(INFINITY) {}

	__host__ __device__ Ray(float3 &orig, float3 &dir, Medium *medium = nullptr, float min = 0.001, float max = INFINITY)
		: origin(orig),
		  destination(dir),
		  medium(medium),
		  tmin(min),
		  tmax(max) {}

	__host__ __device__ float3 operator()(float t)
	{
		return origin + t * destination;
	}
};