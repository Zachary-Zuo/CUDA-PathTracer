#include "Path.h"
#include "../CudaTools.h"

//**************************Path Integrator************************
__global__ void Path(int iter, int maxDepth) {
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

	float3 Li = make_float3(0.f, 0.f, 0.f);
	float3 beta = make_float3(1.f, 1.f, 1.f);
	Ray r = ray;
	Intersection isect;
	bool specular = false;
	for (int bounces = 0; bounces < maxDepth; ++bounces) {
		if (!Intersect(r, &isect)) {
			if ((bounces == 0 || specular) && kernel_Resource.kernel_infinite->isvalid)
				Li += beta * kernel_Resource.kernel_infinite->Le(r.destination);
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material material = kernel_Resource.kernel_materials[isect.matIdx];

		if (bounces == 0 || specular) {
			if (isect.lightIdx != -1) {
				Li += beta * kernel_Resource.kernel_lights[isect.lightIdx].Le(nor, -r.destination);
				break;
			}
		}

		//direct light with multiple importance sampling
		if (!IsDelta(material.type)) {
			float3 Ld = make_float3(0.f, 0.f, 0.f);
			bool inf = false;
			float u = uniform(rng);
			float choicePdf;
			int idx = LookUpLightDistribution(u, choicePdf);
			if (idx == kernel_Resource.kernel_light_size) inf = true;
			float2 u1 = make_float2(uniform(rng), uniform(rng));
			float3 radiance, lightNor;
			Ray shadowRay;
			float lightPdf;
			if (!inf)
				kernel_Resource.kernel_lights[idx].SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_Resource.kernel_epsilon);
			else
				kernel_Resource.kernel_infinite->SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_Resource.kernel_epsilon);
			shadowRay.medium = r.medium;

			if (!IsBlack(radiance) && !IntersectP(shadowRay)) {
				float3 fr;
				float samplePdf;

				//Fr(material, -r.destination, shadowRay.destination, nor, uv, dpdu, uniform(rng), fr, samplePdf);
				Fr(material, -r.destination, shadowRay.destination, nor, uv, dpdu, fr, samplePdf);

				float weight = PowerHeuristic(1, lightPdf * choicePdf, 1, samplePdf);
				Ld += weight * fr * radiance * fabs(dot(nor, shadowRay.destination)) / (lightPdf * choicePdf);
			}

			float3 us = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(material, -r.destination, nor, uv, dpdu, us, out, fr, pdf);
			if (!(IsBlack(fr) || pdf == 0)) {
				Intersection lightIsect;
				Ray lightRay(pos, out, r.medium, kernel_Resource.kernel_epsilon);
				if (Intersect(lightRay, &lightIsect)) {
					float3 p = lightIsect.pos;
					float3 n = lightIsect.nor;
					float3 radiance = { 0.f, 0.f, 0.f };
					if (lightIsect.lightIdx != -1)
						radiance = kernel_Resource.kernel_lights[lightIsect.lightIdx].Le(n, -lightRay.destination);
					if (!IsBlack(radiance)) {
						float pdfA, pdfW;
						kernel_Resource.kernel_lights[lightIsect.lightIdx].Pdf(Ray(p, -out, r.medium, kernel_Resource.kernel_epsilon), n, pdfA, pdfW);
						float choicePdf = PdfFromLightDistribution(lightIsect.lightIdx);
						float lenSquare = dot(p - pos, p - pos);
						float costheta = fabs(dot(n, lightRay.destination));
						float lPdf = pdfA * lenSquare / (costheta);
						float weight = PowerHeuristic(1, pdf, 1, lPdf * choicePdf);

						Ld += weight * fr * radiance * fabs(dot(out, nor)) / pdf;
					}
				}
				else {
					//infinite
					if (kernel_Resource.kernel_infinite->isvalid) {
						float3 radiance = { 0.f, 0.f, 0.f };
						radiance = kernel_Resource.kernel_infinite->Le(lightRay.destination);
						float choicePdf = PdfFromLightDistribution(kernel_Resource.kernel_light_size);
						float lightPdf, pdfA;
						float3 lightNor;
						kernel_Resource.kernel_infinite->Pdf(lightRay, lightNor, pdfA, lightPdf);
						float weight = PowerHeuristic(1, pdf, 1, lightPdf * choicePdf);

						Ld += weight * fr * radiance * fabs(dot(out, nor)) / pdf;
					}
				}
			}

			Li += beta * Ld;
		}

		float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
		float3 out, fr;
		float pdf;

		SampleBSDF(material, -r.destination, nor, uv, dpdu, u, out, fr, pdf);
		if (IsBlack(fr))
			break;

		beta *= fr * fabs(dot(nor, out)) / pdf;
		specular = IsDelta(material.type);

		r = Ray(pos, out, nullptr, kernel_Resource.kernel_epsilon);

		if (bounces > 3) {
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}

	if (!IsInf(Li) && !IsNan(Li))
		kernel_Resource.kernel_color[pixel] = Li;
}

