#pragma once
#include "common.h"

__host__ __device__ inline float3 UniformSampleSphere(float u1, float u2, float& pdf)
{
	float costheta = 1.f - 2.f * u1;
	float sintheta = sqrtf(1.f - costheta * costheta);
	float phi = TWOPI * u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = ONE_OVER_FOUR_PI;

	return make_float3(sintheta * cosphi, costheta, sintheta * sinphi);
}

__host__ __device__ inline float3 UniformSampleHemiSphere(float u1, float u2, float3& n, float& pdf)
{
	float costheta = u1;
	float sintheta = sqrtf(1.f - costheta * costheta);
	float phi = TWOPI * u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = ONE_OVER_TWO_PI;

	float3 dir = make_float3(sintheta * cosphi, costheta, sintheta * sinphi);
	return dir;
}

__host__ __device__ inline float3 CosineSampleHemiSphere(float u1, float u2, float3& n, float& pdf)
{
	float sintheta = sqrtf(u1);
	float costheta = sqrtf(1.f - u1);
	float phi = TWOPI * u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = costheta * ONE_OVER_PI;

	float3 dir = make_float3(sintheta * cosphi, costheta, sintheta * sinphi);
	return dir;
}

__host__ __device__ inline float3 UniformSampleCone(float u1, float u2, float costhetamax, float3& n, float& pdf)
{
	float costheta = 1.f - u1 * (1.f - costhetamax);
	float sintheta = sqrtf(1.f - costheta * costheta);
	float phi = TWOPI * u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = 1.f / (TWOPI * (1.f - costhetamax));

	float3 dir = make_float3(sintheta * cosphi, costheta, sintheta * sinphi);
	return dir;
}

__host__ __device__ inline float2 UniformSampleDisk(float u1, float u2, float& pdf)
{
	float r = sqrtf(u1);
	float phi = TWOPI * u2;

	pdf = ONE_OVER_PI;

	return make_float2(r * cosf(phi), r * sinf(phi));
}

__host__ __device__ inline float2 ConcentricSampleDisk(float u1, float u2, float& pdf)
{
	// Map uniform random numbers to $[-1,1]^2$
	float2 uOffset = 2.f * make_float2(u1, u2) - make_float2(1, 1);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0)
		return make_float2(0, 0);

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y))
	{
		r = uOffset.x;
		theta = PI * 0.25f * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = PI * 0.5f - PI * 0.25f * (uOffset.x / uOffset.y);
	}

	pdf = ONE_OVER_PI;

	return r * make_float2(std::cos(theta), std::sin(theta));
}

__host__ __device__ inline float2 UniformSampleTriangle(float u1, float u2)
{
	float su1 = sqrtf(u1);
	float u = 1.f - su1;
	float v = u2 * su1;
	return make_float2(u, v);
}
