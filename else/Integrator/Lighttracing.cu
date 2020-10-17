#include "Lighttracing.h"
#include "../CudaTools.h"

//**************************Lighttracing Integrator****************
__global__ void LightTracingInit() {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	kernel_Resource.kernel_color[pixel] = { 0.f, 0.f, 0.f };
}

__global__ void LightTracing(int iter, int maxDepth) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int lightIdx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_Resource.kernel_lights[lightIdx];
	float4 u = make_float4(uniform(rng), uniform(rng), uniform(rng), uniform(rng));
	Ray ray;
	float3 nor, radiance;
	float pdfA, pdfW;
	light.SampleLight(u, ray, nor, radiance, pdfA, pdfW, kernel_Resource.kernel_epsilon);
	ray.medium = light.medium == -1 ? nullptr : &kernel_Resource.kernel_mediums[light.medium];

	beta *= radiance * fabs(dot(ray.destination, nor)) / (pdfA * pdfW * choicePdf);

	Ray shadowRay;
	float we, cameraPdf;
	int raster;
	kernel_Resource.kernel_camera->SampleCamera(ray.origin, shadowRay, we, cameraPdf, raster, kernel_Resource.kernel_epsilon);
	shadowRay.medium = ray.medium;
	if (cameraPdf != 0.f) {
		float3 tr = Tr(shadowRay, uniform, rng);
		if (!IsBlack(tr))
			kernel_Resource.kernel_color[raster] += tr * radiance;
	}

	Intersection isect;
	for (int bounces = 0; bounces < maxDepth; ++bounces) {
		if (!Intersect(ray, &isect)) {
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float sampledDist;
		bool sampledMedium = false;
		if (ray.medium) {
			if (ray.medium->type == MT_HOMOGENEOUS)
				beta *= ray.medium->homogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
			else
				beta *= ray.medium->heterogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
		}
		if (IsBlack(beta)) break;
		if (sampledMedium) {
			float3 samplePos = ray(sampledDist);
			Ray shadowRay;
			float we, cameraPdf;
			int raster;
			kernel_Resource.kernel_camera->SampleCamera(samplePos, shadowRay, we, cameraPdf, raster, kernel_Resource.kernel_epsilon);
			shadowRay.medium = ray.medium;
			float3 tr = Tr(shadowRay, uniform, rng);
			float phase, unuse;
			ray.medium->Phase(-ray.destination, shadowRay.destination, phase, unuse);

			float3 L = beta * we * tr * phase / cameraPdf;
			if (!IsInf(L) && !IsNan(L)) {
				//kernel_Resource.kernel_color[raster] += L;
				atomicAdd(&kernel_Resource.kernel_color[raster].x, L.x);
				atomicAdd(&kernel_Resource.kernel_color[raster].y, L.y);
				atomicAdd(&kernel_Resource.kernel_color[raster].z, L.z);
			}

			float pdf;
			float2 phaseU = make_float2(uniform(rng), uniform(rng));
			float3 dir;
			ray.medium->SamplePhase(phaseU, dir, phase, pdf);
			ray = Ray(samplePos, dir, ray.medium, kernel_Resource.kernel_epsilon);
		}
		else {
			if (isect.matIdx == -1) {
				bounces--;
				Medium* m = dot(ray.destination, isect.nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumOutside])
					: (isect.mediumInside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumInside]);
				ray = Ray(pos, ray.destination, m, kernel_Resource.kernel_epsilon);

				continue;
			}

			Material mat = kernel_Resource.kernel_materials[isect.matIdx];

			//direct
			if (!IsDelta(mat.type)) {
				Ray shadowRay;
				float we, cameraPdf;
				int raster;
				kernel_Resource.kernel_camera->SampleCamera(pos, shadowRay, we, cameraPdf, raster, kernel_Resource.kernel_epsilon);
				shadowRay.medium = ray.medium;

				if (cameraPdf != 0.f) {
					float3 tr = Tr(shadowRay, uniform, rng);
					float3 fr;
					float unuse;
					Fr(mat, -ray.destination, shadowRay.destination, nor, uv, isect.dpdu, fr, unuse);

					float3 L = tr * beta * fr * we * fabs(dot(shadowRay.destination, nor)) / cameraPdf;
					if (!IsInf(L) && !IsNan(L)) {
						//kernel_Resource.kernel_color[raster] += L;
						atomicAdd(&kernel_Resource.kernel_color[raster].x, L.x);
						atomicAdd(&kernel_Resource.kernel_color[raster].y, L.y);
						atomicAdd(&kernel_Resource.kernel_color[raster].z, L.z);
					}
				}
			}

			float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.destination, nor, uv, isect.dpdu, u, out, fr, pdf, TransportMode::Importance);
			if (IsBlack(fr))
				break;
			beta *= fr * fabs(dot(out, nor)) / pdf;
			Medium* m = dot(out, nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumOutside])
				: (isect.mediumInside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumInside]);
			m = dot(-ray.destination, nor) * dot(out, nor) > 0 ? ray.medium : m;

			ray = Ray(pos, out, m, kernel_Resource.kernel_epsilon);
		}

		if (bounces > 3) {
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}
}
//**************************Lighttracing End***********************
