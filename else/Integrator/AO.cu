#include "AO.h"
#include "../CudaTools.h"

//**************************AO Integrator**************************
__global__ void Ao(int iter, float maxDist) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	//start
	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	float2 aperture = UniformSampleDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_Resource.kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, aperture);
	ray.tmin = kernel_Resource.kernel_epsilon;

	float3 L = { 0.f, 0.f, 0.f };
	Intersection isect;
	bool intersect = Intersect(ray, &isect);
	if (!intersect) {
		kernel_Resource.kernel_color[pixel] = { 0, 0, 0 };
		return;
	}

	float3 pos = isect.pos;
	float3 nor = isect.nor;
	float pdf = 0.f;
	if (dot(-ray.destination, nor) < 0.f)
		nor = -nor;
	float3 dir = CosineSampleHemiSphere(uniform(rng), uniform(rng), nor, pdf);
	float3 uu = isect.dpdu, ww;
	ww = cross(uu, nor);
	dir = ToWorld(dir, uu, nor, ww);
	float cosine = dot(dir, nor);
	Ray r(pos, dir, nullptr, kernel_Resource.kernel_epsilon, maxDist);
	intersect = IntersectP(r);
	if (!intersect) {
		float v = cosine * ONE_OVER_PI / pdf;
		L += make_float3(v, v, v);
	}

	if (!IsNan(L))
		kernel_Resource.kernel_color[pixel] = L;
}
//**************************AO End*********************************
