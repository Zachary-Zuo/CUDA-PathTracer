#pragma once
#include "common.h"
#include <assimp/Importer.hpp>
#include "bbox.h"
#include "intersection.h"
#include "Sampling.h"

struct Vertex{
	float3 v;
	float3 n;
	float2 uv;
	float3 t;
};

__host__ __device__ inline void CoordinateSystem(const float3& v1, float3& v2,
	float3& v3) {
	if (fabs(v1.x) > fabs(v1.y))
		v2 = make_float3(-v1.z, 0, v1.x) / sqrtf(v1.x * v1.x + v1.z * v1.z);
	else
		v2 = make_float3(0, v1.z, -v1.y) / sqrtf(v1.y * v1.y + v1.z * v1.z);
	v3 = cross(v1, v2);
}

__host__ __device__ inline float LengthSquared(const float3& v) {
	return v.x * v.x + v.y * v.y * v.z * v.z;
}

__host__ __device__ inline float MaxComponent(const float3& v) {
	return fmax(v.x, fmax(v.y, v.z));
}

__host__ __device__ inline int MaxDimension(const float3& v) {
	return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

__host__ __device__  inline float3 Permute(const float3& v, int x, int y, int z) {
	float fx, fy, fz;
	if (x == 0)
		fx = v.x;
	else if (x == 1)
		fx = v.y;
	else
		fx = v.z;

	if (y == 0)
		fy = v.x;
	else if (y == 1)
		fy = v.y;
	else
		fy = v.z;

	if (z == 0)
		fz = v.x;
	else if (z == 1)
		fz = v.y;
	else
		fz = v.z;

	return make_float3(fx, fy, fz);
}

class Triangle{
public:
	Vertex v1, v2, v3;
	int matIdx;
	int bssrdfIdx;
	int lightIdx;
	int mediumInside, mediumOutside;

public:
	__host__ __device__ BBox GetBBox() const{
		float3 f1 = v1.v, f2 = v2.v, f3 = v3.v;
		BBox bbox;
		bbox.Expand(f1);
		bbox.Expand(f2);
		bbox.Expand(f3);

		return bbox;
	}

	__host__ __device__ float GetSurfaceArea() const{
		float3 e1 = v2.v - v1.v;
		float3 e2 = v3.v - v1.v;
		return length(cross(e1, e2)) * 0.5f;
	}

	__host__ __device__ bool Intersect(Ray& ray, Intersection* isect) const{
		float3 e1 = v2.v - v1.v;
		float3 e2 = v3.v - v1.v;
		float3 s1 = cross(ray.destination, e2);
		float divisor = dot(s1, e1);
		if (fabs(divisor) < 1e-8f)
			return false;
		float invDivisor = 1.0 / divisor;
		float3 s = ray.origin - v1.v;
		float b1 = dot(s, s1)*invDivisor;
		if (b1 < 0.0 || b1 > 1.0)
			return false;

		float3 s2 = cross(s, e1);
		float b2 = dot(ray.destination, s2)*invDivisor;
		if (b2 < 0.0 || b1 + b2 > 1.0)
			return false;

		float maxD = fmaxf(fabs(ray.destination));
		float maxE = fmaxf(fmaxf(fabs(v1.v)), fmaxf(fabs(v2.v)), fmaxf(fabs(v3.v)));
		float deltaE = gamma(1) * maxE;
		float deltaS1 = 2 * (gamma(2) * maxD * maxE + deltaE * maxD);
		float deltaS = gamma(1) * fmaxf(fabs(v1.v));
		float maxS = fmaxf(fabs(s));
		float maxS1 = fmaxf(fabs(s1));
		float deltaS2 = 2 * (gamma(2) * maxS * maxS1 + deltaS * maxS1 + deltaS1 * maxS);
		float maxS2 = fmaxf(fabs(s2));
		float deltaT = 2 * (gamma(2) * maxE * maxS2 + deltaS2 * maxE + deltaE * maxS2) * abs(invDivisor);

		float tt = dot(e2, s2) * invDivisor;
		if (tt < ray.tmin+ deltaT || tt > ray.tmax)
			return false;

		ray.tmax = tt;
		if (isect){
            float3 ns = normalize(v1.n * (1.f - b1 - b2) + v2.n * b1 + v3.n * b2);
            float3 ss = normalize(v1.t * (1.f - b1 - b2) + v2.t * b1 + v3.t * b2);
            float3 ts = cross(ss, ns);
            
            if (length(ts) > 0.f)
            {
                ts = normalize(ts);
                ss = cross(ts, ns);
            }
			isect->pos = ray(tt);
			//不能默认文件里的法线已经归一化，这里需要手动归一化一下
            isect->shading.n = normalize(cross(ss, ts));
            isect->n = Faceforward(ns, isect->shading.n);
			isect->uv = v1.uv*(1.f - b1 - b2) + v2.uv*b1 + v3.uv*b2;
            isect->dpdu = ss;
			isect->matIdx = matIdx;
			isect->lightIdx = lightIdx;
			isect->bssrdf = bssrdfIdx;
			isect->mediumInside = mediumInside;
			isect->mediumOutside = mediumOutside;
		}
		return true;
	}

	__host__ __device__ void SampleShape(float3& pos, float2& u, float3& dir, float3& nor, float& pdf) const{
		float2 uv = UniformSampleTriangle(u.x, u.y);
		float3 p = uv.x * v1.v + uv.y * v2.v + (1 - uv.x - uv.y)*v3.v;
		float3 normal = normalize(uv.x*v1.n + uv.y*v2.n + (1 - uv.x - uv.y)*v3.n);
		dir = p - pos;
		nor = normal;
		pdf = 1.f / (GetSurfaceArea() * fabsf(dot(normal, normalize(dir)))) * dot(dir, dir);
		if (dot(normal, dir) >= 0.f)
			pdf = 0.f;
	}

	__host__ __device__ void SampleShape(float4& u, float3& pos, float3& dir, float3& nor, float& pdfA, float& pdfW){
		float2 uv = UniformSampleTriangle(u.x, u.y);
		pos = uv.x * v1.v + uv.y * v2.v + (1 - uv.x - uv.y)*v3.v;
	    nor = normalize(uv.x*v1.n + uv.y*v2.n + (1 - uv.x - uv.y)*v3.n);
		dir = CosineSampleHemiSphere(u.z, u.w, nor, pdfW);
		float3 uu, ww;
		MakeCoordinate(nor, uu, ww);
		dir = ToWorld(dir, uu, ww,nor);
		pdfA = 1.f / GetSurfaceArea();
	}
};