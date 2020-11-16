#pragma once
#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cutil_math.h"
#include <thrust\device_vector.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

#define PI                  3.14159265358f
#define TWOPI               6.28318530716f
#define FOURPI              12.56637061432f 
#define ONE_OVER_PI         0.3183098861847f
#define ONE_OVER_TWO_PI     0.1591549430923f
#define ONE_OVER_FOUR_PI    0.0795774715461f

#define MachineEpsilon (1.192092896e-07F * 0.5)

static void HandleError(cudaError_t err,
						const char* file,
						int line)
{
	if (err != cudaSuccess)
	{
		std::string s = cudaGetErrorString(err);
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);

		__debugbreak();
	}
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__host__ __device__ inline float RadiansToDegrees(float radians)
{
	float degrees = radians * 180.0 / PI;
	return degrees;
}

__host__ __device__ inline float DegreesToRadians(float degrees)
{
	float radians = degrees / 180.0 * PI;
	return radians;
}

__host__ __device__ inline void Swap(float& a, float& b)
{
	float temp;
	temp = a;
	a = b;
	b = temp;
}

__host__ __device__ inline float3 Clamp(float3 value, float minValue, float maxValue) {
	if (value.x < minValue) value.x = minValue;
	if (value.x > maxValue) value.x = maxValue;
	if (value.y < minValue) value.y = minValue;
	if (value.y > maxValue) value.y = maxValue;
	if (value.z < minValue) value.z = minValue;
	if (value.z > maxValue) value.z = maxValue;

	return value;
}

__host__ __device__ inline bool IsBlack(float3& c) {
	return c.x == 0 && c.y == 0 && c.z == 0;
}

__host__ __device__ inline bool IsNan(float3& c) {
	return isnan(c.x) || isnan(c.y) || isnan(c.z);
}

__host__ __device__ inline bool IsInf(float3& c) {
	return isinf(c.x) || isinf(c.y) || isinf(c.z);
}

__host__ __device__ inline float3 Exp(const float3& c) {
	float r = expf(c.x);
	float g = expf(c.y);
	float b = expf(c.z);
	return{ r, g, b };
}

__host__ __device__ inline float3 Sqrt(const float3& c) {
	float r = sqrt(c.x);
	float g = sqrt(c.y);
	float b = sqrt(c.z);
	return{ r, g, b };
}

template<typename T>
__host__ __device__ inline T Max(T c0, T c1) {
	return (c0 > c1 ? c0 : c1);
}

inline float3 VecToFloat3(glm::vec3& v) {
	return make_float3(v.x, v.y, v.z);
}

inline glm::vec3 Float3ToVec(float3& v) {
	return glm::vec3(v.x, v.y, v.z);
}

__host__ __device__ inline float gamma(int n)
{
	return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

__host__ __device__ inline void MakeCoordinate(float3& n, float3& u, float3& w)
{
	if (std::abs(n.x) > std::abs(n.y))
	{
		float invLen = 1.0f / std::sqrt(n.x * n.x + n.z * n.z);
		w = make_float3(n.z * invLen, 0.0f, -n.x * invLen);
	}
	else
	{
		float invLen = 1.0f / std::sqrt(n.y * n.y + n.z * n.z);
		w = make_float3(0.0f, n.z * invLen, -n.y * invLen);
	}
	u = cross(w, n);
}

__host__ __device__ inline float3 ToWorld(float3& dir, float3& u, float3& v, float3& w)
{
	return dir.x * u + dir.y * v + dir.z * w;
}

__host__ __device__ inline float3 ToLocal(float3& dir, float3& u, float3& v, float3& w)
{
	return make_float3(dot(dir, u), dot(dir, v), dot(dir, w));
}

__host__ __device__ inline float EquiAngular(float u, float D, float thetaA, float thetaB)
{
	return D * tanf(u * (thetaB - thetaA) + thetaA);
}

__host__ __device__ inline float EquiAngularPdf(float t, float D, float thetaA, float thetaB)
{
	return D / ((thetaB - thetaA) * (t * t + D * D));
}

__host__ __device__ inline float2 GaussianDiskInfinity(float u1, float u2, float falloff)
{
	float r = sqrtf(log(u1) / -falloff);
	float theta = TWOPI * u2;

	return { r * cos(theta), r * sin(theta) };
}

__host__ __device__ inline float GaussianDiskInfinityPdf(float x, float y, float falloff)
{
	return ONE_OVER_PI * falloff * exp(-falloff * (x * x + y * y));
}

__host__ __device__ inline float GaussianDiskInfinityPdf(float3& center, float3& sample, float3& n, float falloff)
{
	float3 d = sample - center;
	float3 projected = d - n * dot(d, n);
	return ONE_OVER_PI * falloff * exp(-falloff * dot(projected, projected));
}

__host__ __device__ inline float2 GaussianDisk(float u1, float u2, float falloff, float rmax)
{
	float r = sqrtf(log(1.0f - u1 * (1.0f - exp(-falloff * rmax * rmax))) /
		-falloff);
	float theta = TWOPI * u2;
	return { r * cos(theta), r * sin(theta) };
}

__host__ __device__ inline float GaussianDiskPdf(float x, float y, float falloff, float rmax)
{
	return GaussianDiskInfinityPdf(x, y, falloff) /
		(1.0f - exp(-falloff * rmax * rmax));
}

__host__ __device__ inline float GaussianDiskPdf(float3& center, float3& sample, float3& n, float falloff, float rmax)
{
	return GaussianDiskInfinityPdf(center, sample, n, falloff) / (1.f - exp(-falloff * rmax * rmax));
}

__host__ __device__ inline float Exponential(float u, float falloff)
{
	return -log(u) / falloff;
}

__host__ __device__ inline float ExponentialPdf(float x, float falloff)
{
	return falloff * exp(-falloff * x);
}

__device__ inline unsigned int WangHash(unsigned int seed)
{
	seed = (seed ^ 61) ^ (seed >> 16);
	seed = seed + (seed << 3);
	seed = seed ^ (seed >> 4);
	seed = seed * 0x27d4eb2d;
	seed = seed ^ (seed >> 15);

	return seed;
}