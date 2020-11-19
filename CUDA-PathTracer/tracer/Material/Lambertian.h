#pragma once
#include "../common.h"
#include "../BxDFBase.h"
#include "../Intersection.h"
#include "../Spectrum.h"

class LambertianReflection {
public:
	// LambertianReflection Public Methods
	__host__ __device__ LambertianReflection(const Spectrum& R)
		: type(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {}
	__host__ __device__ Spectrum f(const float3& wo, const float3& wi) const;
	__host__ __device__ Spectrum rho(const float3&, int, const float2*) const { return R; }
	__host__ __device__ Spectrum rho(int, const float2*, const float2*) const { return R; }
private:
	// LambertianReflection Private Data
	const Spectrum R;
	BxDFType type;
};

Spectrum LambertianReflection::f(const float3& wo, const float3& wi) const {
	return R * ONE_OVER_PI;
}