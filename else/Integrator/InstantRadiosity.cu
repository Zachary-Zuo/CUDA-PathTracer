#include "InstantRadiosity.h"
#include "../CudaTools.h"

//**************************Instant Radiosity Integrator************

struct Vpl {
	float3 beta;
	float3 dir;
	float3 pos;
	float3 nor;
	float2 uv;
	float3 dpdu;
	int matIdx;
};

__device__ Vpl vpls[IR_MAX_VPLS][IR_MAX_VPLS];
__device__ int numVpls[IR_MAX_VPLS];


__global__ void GenerateVpl(int iter, int maxDepth) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	numVpls[pixel] = 0;
	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int idx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_Resource.kernel_lights[idx];
	float3 radiance, lightNor;
	Ray ray;
	float4 lightUniform = make_float4(uniform(rng), uniform(rng), uniform(rng), uniform(rng));
	float pdfA, pdfW;
	light.SampleLight(lightUniform, ray, lightNor, radiance, pdfA, pdfW, kernel_Resource.kernel_epsilon);
	beta *= radiance * fabs(dot(lightNor, ray.destination)) / (pdfA * pdfW * choicePdf);
	{
		Vpl vpl;
		vpl.beta = radiance;
		vpl.dir.x = pdfA * choicePdf;
		vpl.pos = ray.origin;
		vpl.nor = lightNor;
		vpls[pixel][numVpls[pixel]++] = vpl;
	}

	Intersection isect;
	for (int bounces = 0; bounces < maxDepth; ++bounces) {
		if (!Intersect(ray, &isect)) {
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material mat = kernel_Resource.kernel_materials[isect.matIdx];

		{
			Vpl vpl;
			vpl.beta = beta;
			vpl.dir = -ray.destination;
			vpl.pos = isect.pos;
			vpl.nor = isect.nor;
			vpl.uv = isect.uv;
			vpl.dpdu = isect.dpdu;
			vpl.matIdx = isect.matIdx;
			vpls[pixel][numVpls[pixel]++] = vpl;
		}

		float3 fr, out;
		float3 bsdfUniform = make_float3(uniform(rng), uniform(rng), uniform(rng));
		float pdf;
		SampleBSDF(mat, -ray.destination, nor, uv, dpdu, bsdfUniform, out, fr, pdf, TransportMode::Importance);
		if (IsBlack(fr)) break;

		beta *= fr * fabs(dot(out, nor)) / pdf;

		ray = Ray(pos, out, nullptr, kernel_Resource.kernel_epsilon);

		if (bounces > 3) {
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}
}

__global__ void InstantRadiosity(int iter, int vplIter, int maxDepth, float bias) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	float2 aperture = UniformSampleDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_Resource.kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, aperture);
	ray.tmin = kernel_Resource.kernel_epsilon;
	float3 beta = { 1.f, 1.f, 1.f };
	float3 L = { 0.f, 0.f, 0.f };

	Intersection isect;
	for (int bounces = 0; bounces < maxDepth; ++bounces) {
		if (!Intersect(ray, &isect)) break;
		if (isect.lightIdx != -1) {
			L += kernel_Resource.kernel_lights[isect.lightIdx].Le(isect.nor, -ray.destination);
		}
		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material mat = kernel_Resource.kernel_materials[isect.matIdx];
		if (IsDelta(mat.type)) {
			float3 fr, out;
			float3 bsdfUniform = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float pdf;
			SampleBSDF(mat, -ray.destination, nor, uv, dpdu, bsdfUniform, out, fr, pdf);
			if (IsBlack(fr)) break;
			beta *= fr * fabs(dot(nor, out)) / pdf;

			ray = Ray(pos, out, nullptr, kernel_Resource.kernel_epsilon);
			continue;
		}

		for (int i = 0; i < numVpls[vplIter]; ++i) {
			Vpl vpl = vpls[vplIter][i];

			float3 dir = pos - vpl.pos;
			float3 out = normalize(dir);
			float squreDistance = dot(dir, dir);
			Ray shadowRay(pos, -out, nullptr, kernel_Resource.kernel_epsilon, sqrt(squreDistance) - kernel_Resource.kernel_epsilon);
			if (IntersectP(shadowRay)) continue;

			if (squreDistance < bias) squreDistance = bias;
			float c1 = fabs(dot(out, nor));
			float c2 = fabs(dot(out, vpl.nor));
			float G = c1 * c2 / squreDistance;
			float3 fr1, fr2;
			float pdf1, pdf2;
			Fr(mat, -ray.destination, -out, nor, uv, dpdu, fr1, pdf1);
			if (i == 0) {
				if (dot(dir, vpl.nor) > 0.f)
					L += beta * fr1 * G * vpl.beta / vpl.dir.x;
				continue;
			}
			Material m = kernel_Resource.kernel_materials[vpl.matIdx];
			if (IsDelta(m.type)) continue;
			Fr(m, vpl.dir, out, vpl.nor, vpl.uv, vpl.dpdu, fr2, pdf2);

			L += beta * fr1 * G * fr2 * vpl.beta;
		}

		break;
	}

	if (IsNan(L) || IsInf(L)) return;

	kernel_Resource.kernel_color[pixel] = L;
}
//**************************Instant Radiosity Integrator End********