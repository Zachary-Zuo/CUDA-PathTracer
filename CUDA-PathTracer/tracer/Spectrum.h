#pragma once
#include "common.h"

class RGBSpectrum {
public:
	// RGBSpectrum Public Methods
	__host__ __device__ RGBSpectrum(float3& v) {
		c[0] = v.x;
		c[1] = v.y;
		c[2] = v.z;
	}
	__host__ __device__ RGBSpectrum(float& r, float& g, float& b) {
		c[0] = r;
		c[1] = g;
		c[2] = b;
	}
	
	RGBSpectrum operator*(float a) const {
		RGBSpectrum ret = *this;
		ret.c[0] *= a;
		ret.c[1] *= a;
		ret.c[2] *= a;
		return ret;
	}

	float c[3];
};



typedef RGBSpectrum Spectrum;
