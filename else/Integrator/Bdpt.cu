#include "Bdpt.h"
#include "../CudaTools.h"

//**************************Bdpt Integrator************************
#define BDPT_MAX_DEPTH 65

struct BdptVertex {
	float3 beta;
	Intersection isect;
	Medium* medium = nullptr;
	bool delta;
	float fwd;
	float rev;
};

//convert pdf from area to omega
__device__ float ConvertPdf(float pdf, Intersection& prev, Intersection& cur) {
	float3 dir = prev.pos - cur.pos;
	float square = dot(dir, dir);
	dir = normalize(dir);
	float ret = pdf / square;
	if (!IsBlack(cur.nor))
		ret *= fabs(dot(dir, cur.nor));
	return ret;
}

__device__ int GenerateCameraPath(int x, int y, BdptVertex* path, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	//start
	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	//bdpt doesn't support dof now
	//float2 aperture = UniformSampleDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_Resource.kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, make_float2(0, 0));
	ray.tmin = kernel_Resource.kernel_epsilon;
	ray.medium = kernel_Resource.kernel_camera->medium == -1 ? nullptr : &kernel_Resource.kernel_mediums[kernel_Resource.kernel_camera->medium];
	float3 beta = { 1.f, 1.f, 1.f };

	int nVertex = 0;
	//set camera isect
	{
		Intersection cameraIsect;
		cameraIsect.pos = kernel_Resource.kernel_camera->position;
		cameraIsect.nor = -kernel_Resource.kernel_camera->w;
		BdptVertex vertex;
		vertex.beta = beta;
		vertex.isect = cameraIsect;
		vertex.delta = false;
		vertex.medium = ray.medium;
		vertex.fwd = 1.f;
		path[0] = vertex;
	}
	nVertex++;

	float forward = 0.f, rrPdf = 1.f;
	kernel_Resource.kernel_camera->PdfCamera(ray.destination, unuse, forward);
	Intersection isect;
	int bounces = 0;
	for (; bounces < BDPT_MAX_DEPTH; ++bounces) {
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

			float phase, pdf;
			float2 phaseU = make_float2(uniform(rng), uniform(rng));
			float3 dir;
			ray.medium->SamplePhase(phaseU, dir, phase, pdf);
			ray = Ray(samplePos, dir, ray.medium, kernel_Resource.kernel_epsilon);

			//set medium Intersection
			{
				BdptVertex vertex;
				Intersection mediumIsect;
				mediumIsect.pos = samplePos;
				mediumIsect.nor = { 0.f, 0.f, 0.f };
				mediumIsect.matIdx = -1;
				mediumIsect.lightIdx = -1;
				vertex.beta = beta;
				vertex.delta = false;
				vertex.isect = mediumIsect;
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
				forward = phase;
				path[bounces].rev = ConvertPdf(forward, path[bounces + 1].isect, path[bounces].isect);
			}
			nVertex++;
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

			{
				BdptVertex vertex;
				vertex.beta = beta;
				vertex.isect = isect;
				vertex.delta = IsDelta(mat.type);
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
			}
			nVertex++;

			float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.destination, nor, uv, isect.dpdu, u, out, fr, pdf);
			if (IsBlack(fr))
				break;
			beta *= fr * fabs(dot(out, nor)) / pdf;

			forward = pdf;
			if (IsDelta(mat.type)) forward = 0.f;
			//calc reverse pdf
			{
				float3 unuseFr;
				float pdf;
				Fr(mat, out, -ray.destination, nor, uv, isect.dpdu, unuseFr, pdf);
				path[bounces].rev = ConvertPdf(pdf, path[bounces + 1].isect, path[bounces].isect);
			}

			Medium* m = dot(out, nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumOutside])
				: (isect.mediumInside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumInside]);
			m = dot(-ray.destination, nor) * dot(out, nor) > 0 ? ray.medium : m;

			ray = Ray(pos, out, m, kernel_Resource.kernel_epsilon);
		}

		//russian roulette
		if (bounces > 3) {
			rrPdf = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < rrPdf)
				break;

			beta /= (1 - rrPdf);
		}
	}

	return nVertex;
}

__device__ int GenerateLightPath(BdptVertex* path, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int lightIdx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_Resource.kernel_lights[lightIdx];
	float4 u = make_float4(uniform(rng), uniform(rng), uniform(rng), uniform(rng));
	Ray ray;
	float3 lightNor, radiance;
	float pdfA, pdfW;
	light.SampleLight(u, ray, lightNor, radiance, pdfA, pdfW, kernel_Resource.kernel_epsilon);
	ray.medium = light.medium == -1 ? nullptr : &kernel_Resource.kernel_mediums[light.medium];

	int nVertex = 0;
	//set light isect
	{
		Intersection lightIsect;
		lightIsect.pos = ray.origin;
		lightIsect.nor = lightNor;
		lightIsect.lightIdx = lightIdx;
		BdptVertex vertex;
		vertex.beta = radiance;
		vertex.isect = lightIsect;
		vertex.delta = false;
		vertex.medium = ray.medium;
		vertex.fwd = pdfA * choicePdf;
		path[0] = vertex;
	}
	nVertex++;
	beta *= radiance * fabs(dot(ray.destination, lightNor)) / (pdfA * pdfW * choicePdf);

	Intersection isect;
	float forward = pdfW, rrPdf = 1.f;
	int bounces = 0;
	for (; bounces < BDPT_MAX_DEPTH; ++bounces) {
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

			float phase, pdf;
			float2 phaseU = make_float2(uniform(rng), uniform(rng));
			float3 dir;
			ray.medium->SamplePhase(phaseU, dir, phase, pdf);
			ray = Ray(samplePos, dir, ray.medium, kernel_Resource.kernel_epsilon);

			//set medium Intersection
			{
				BdptVertex vertex;
				Intersection mediumIsect;
				mediumIsect.pos = samplePos;
				mediumIsect.nor = { 0.f, 0.f, 0.f };
				mediumIsect.matIdx = -1;
				mediumIsect.lightIdx = -1;
				vertex.beta = beta;
				vertex.delta = false;
				vertex.isect = mediumIsect;
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
				forward = phase;
				path[bounces].rev = ConvertPdf(phase, path[bounces + 1].isect, path[bounces].isect);
			}
			nVertex++;
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

			{
				BdptVertex vertex;
				vertex.beta = beta;
				vertex.isect = isect;
				vertex.delta = IsDelta(mat.type);
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
			}
			nVertex++;

			float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.destination, nor, uv, isect.dpdu, u, out, fr, pdf, TransportMode::Importance);
			if (IsBlack(fr))
				break;
			beta *= fr * fabs(dot(out, nor)) / pdf;

			forward = pdf;
			if (IsDelta(mat.type)) forward = 0.f;
			//calc reverse pdf
			{
				float3 unuseFr;
				float pdf;
				Fr(mat, out, -ray.destination, nor, uv, isect.dpdu, unuseFr, pdf);
				path[bounces].rev = ConvertPdf(pdf, path[bounces + 1].isect, path[bounces].isect);
			}
			Medium* m = dot(out, nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumOutside])
				: (isect.mediumInside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumInside]);
			m = dot(-ray.destination, nor) * dot(out, nor) > 0 ? ray.medium : m;

			ray = Ray(pos, out, m, kernel_Resource.kernel_epsilon);
		}

		//russian roulette
		if (bounces > 3) {
			rrPdf = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < rrPdf)
				break;

			beta /= (1 - rrPdf);
		}
	}

	return nVertex;
}

__device__ float MisWeight(BdptVertex* cameraPath, BdptVertex* lightPath, int s, int t) {
	if (s + t == 2)//light source is directly visible
		return 1.f;

	//delta bsdf pdf is 0
	auto remap = [](float pdf)->float {
		return pdf == 0 ? 1.f : pdf;
	};

	float sumW = 0.f;
	float ri = 1.f;
	for (int i = s - 1; i > 0; --i) {
		ri *= remap(cameraPath[i].rev) / remap(cameraPath[i].fwd);

		if (!cameraPath[i].delta && !cameraPath[i - 1].delta)
			sumW += ri;
	}

	ri = 1.f;
	for (int i = t - 1; i >= 0; --i) {
		ri *= remap(lightPath[i].rev) / remap(lightPath[i].fwd);

		bool delta = lightPath[i == 0 ? 0 : i - 1].delta;
		if (!lightPath[i].delta && !delta)
			sumW += ri;
	}

	return 1.f / (sumW + 1.f);
}

__device__ float3 Connect(BdptVertex* cameraPath, BdptVertex* lightPath, int s, int t, int& raster,
	thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	float3 L = { 0.f, 0.f, 0.f };

	if (t == 0) {
		//naive path tracing
		BdptVertex& cur = cameraPath[s - 1];
		BdptVertex& prev = cameraPath[s - 2];
		if (cur.isect.lightIdx == -1) return{ 0.f, 0.f, 0.f };

		float3 dir = normalize(prev.isect.pos - cur.isect.pos);
		Area light = kernel_Resource.kernel_lights[cur.isect.lightIdx];
		L += cur.beta * light.Le(cur.isect.nor, dir);
		if (IsBlack(L)) return L;

		Ray ray(cur.isect.pos, dir);
		float choicePdf = PdfFromLightDistribution(cur.isect.lightIdx);
		float pdfA, pdfW;
		light.Pdf(ray, cur.isect.nor, pdfA, pdfW);
		float curRev = cur.rev;
		float prevRev = prev.rev;
		cur.rev = pdfA * choicePdf;
		prev.rev = ConvertPdf(pdfW, cur.isect, prev.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);
		//reset
		cur.rev = curRev;
		prev.rev = prevRev;

		return mis * L;
	}
	else if (t == 1) {
		//next event path tracing
		BdptVertex& prev = cameraPath[s - 2];
		BdptVertex& cur = cameraPath[s - 1];
		BdptVertex& next = lightPath[0];
		float3 in = normalize(prev.isect.pos - cur.isect.pos);
		bool isMedium = cur.isect.matIdx == -1;
		Material mat;
		if (!isMedium) mat = kernel_Resource.kernel_materials[cur.isect.matIdx];
		float choicePdf, lightPdf;
		int idx = LookUpLightDistribution(uniform(rng), choicePdf);
		Area light = kernel_Resource.kernel_lights[idx];
		float3 radiance, lightNor, lightPos;
		Ray shadowRay;
		float2 lightUniform = { uniform(rng), uniform(rng) };
		light.SampleLight(cur.isect.pos, lightUniform, radiance, shadowRay, lightNor, lightPdf, kernel_Resource.kernel_epsilon);
		lightPos = shadowRay(shadowRay.tmax + kernel_Resource.kernel_epsilon);
		shadowRay.medium = cur.medium;
		if (IsBlack(radiance)) return{ 0.f, 0.f, 0.f };
		if (!isMedium && IsDelta(mat.type)) return{ 0.f, 0.f, 0.f };
		float3 tr = Tr(shadowRay, uniform, rng);
		if (IsBlack(tr)) return{ 0.f, 0.f, 0.f };

		float3 fr;
		float nextPdf, G, phase;
		if (isMedium) {
			cur.medium->Phase(in, shadowRay.destination, phase, nextPdf);
			fr = make_float3(phase, phase, phase);
			G = 1.f;
		}
		else {
			Fr(mat, in, shadowRay.destination, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, nextPdf);
			G = fabs(dot(cur.isect.nor, shadowRay.destination));
		}
		L += cur.beta * tr * fr * radiance * G / (lightPdf * choicePdf);
		if (IsBlack(L)) return{ 0.f, 0.f, 0.f };

		BdptVertex tNext = next;
		float pdfA, pdfW;
		light.Pdf(shadowRay, lightNor, pdfA, pdfW);
		next.isect.pos = lightPos;
		next.isect.nor = lightNor;
		next.fwd = pdfA * choicePdf;
		next.rev = ConvertPdf(nextPdf, cur.isect, next.isect);
		float curRev = cur.rev;
		float prevRev = prev.rev;
		cur.rev = ConvertPdf(pdfW, next.isect, cur.isect);
		float pdf;
		if (isMedium) pdf = phase;
		else Fr(mat, shadowRay.destination, in, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, pdf);
		prev.rev = ConvertPdf(pdf, cur.isect, prev.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);

		cur.rev = curRev;
		prev.rev = prevRev;
		next = tNext;

		return mis * L;
	}
	else if (s == 1) {
		//light tracing
		BdptVertex& prev = lightPath[t - 2];
		BdptVertex& cur = lightPath[t - 1];
		BdptVertex& next = cameraPath[0];
		float3 in = normalize(prev.isect.pos - cur.isect.pos);
		bool isMedium = cur.isect.matIdx == -1;
		Material mat;
		if (!isMedium) mat = kernel_Resource.kernel_materials[cur.isect.matIdx];
		Ray shadowRay;
		float we, cameraPdf;
		kernel_Resource.kernel_camera->SampleCamera(cur.isect.pos, shadowRay, we, cameraPdf, raster, kernel_Resource.kernel_epsilon);
		shadowRay.medium = cur.medium;
		if (cameraPdf == 0) return{ 0.f, 0.f, 0.f };
		if (!isMedium && IsDelta(mat.type)) return{ 0.f, 0.f, 0.f };
		float3 tr = Tr(shadowRay, uniform, rng);
		if (IsBlack(tr)) return{ 0.f, 0.f, 0.f };

		float3 fr;
		float nextPdf, phase, costheta = fabs(dot(shadowRay.destination, cur.isect.nor));
		if (isMedium) {
			cur.medium->Phase(in, shadowRay.destination, phase, nextPdf);
			fr = make_float3(phase, phase, phase);
			costheta = 1.f;
		}
		else Fr(mat, in, shadowRay.destination, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, nextPdf);
		L += cur.beta * tr * fr * we * costheta / cameraPdf;
		if (IsBlack(L)) return{ 0.f, 0.f, 0.f };

		float nextRev = next.rev;
		float curRev = cur.rev;
		float prevRev = prev.rev;
		next.rev = ConvertPdf(nextPdf, cur.isect, next.isect);
		float pdfA, pdfW;
		kernel_Resource.kernel_camera->PdfCamera(-shadowRay.destination, pdfA, pdfW);
		cur.rev = ConvertPdf(pdfW, next.isect, cur.isect);
		float pdf;
		if (isMedium) pdf = phase;
		else Fr(mat, shadowRay.destination, in, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, pdf);
		prev.rev = ConvertPdf(pdf, cur.isect, prev.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);
		next.rev = nextRev;
		cur.rev = curRev;
		prev.rev = prevRev;

		return mis * L;
	}
	else {
		//other
		BdptVertex& c2 = cameraPath[s - 2];
		BdptVertex& c1 = cameraPath[s - 1];
		BdptVertex& l1 = lightPath[t - 1];
		BdptVertex& l2 = lightPath[t - 2];
		float3 l1Tol2 = normalize(l2.isect.pos - l1.isect.pos);
		float3 l1Toc1 = normalize(c1.isect.pos - l1.isect.pos);
		float3 c1Tol1 = -l1Toc1;
		float3 c1Toc2 = normalize(c2.isect.pos - c1.isect.pos);
		float3 dir = c1.isect.pos - l1.isect.pos;
		Material c1Mat, l1Mat;
		if (!c1.medium) c1Mat = kernel_Resource.kernel_materials[c1.isect.matIdx];
		if (!l1.medium) l1Mat = kernel_Resource.kernel_materials[l1.isect.matIdx];
		Ray shadowRay;
		shadowRay.origin = c1.isect.pos;
		shadowRay.destination = c1Tol1;
		shadowRay.medium = c1.medium;
		shadowRay.tmin = kernel_Resource.kernel_epsilon;
		shadowRay.tmax = length(dir) - kernel_Resource.kernel_epsilon;
		if (!c1.medium && IsDelta(c1Mat.type)) return{ 0.f, 0.f, 0.f };
		if (!l1.medium && IsDelta(l1Mat.type)) return{ 0.f, 0.f, 0.f };
		float3 tr = Tr(shadowRay, uniform, rng);
		if (IsBlack(tr)) return{ 0.f, 0.f, 0.f };
		float cos1 = l1.medium ? 1.f : fabs(dot(l1Toc1, l1.isect.nor));
		float cos2 = c1.medium ? 1.f : fabs(dot(c1Tol1, c1.isect.nor));

		float3 c1Fr, l1Fr;
		float l1Pdf, c1Pdf;
		float l1Phase, c1Phase;
		if (c1.medium) {
			c1.medium->Phase(c1Toc2, c1Tol1, c1Phase, l1Pdf);
			c1Fr = make_float3(c1Phase, c1Phase, c1Phase);
		}
		else Fr(c1Mat, c1Toc2, c1Tol1, c1.isect.nor, c1.isect.uv, c1.isect.dpdu, c1Fr, l1Pdf);
		if (l1.medium) {
			l1.medium->Phase(l1Tol2, l1Toc1, l1Phase, c1Pdf);
			l1Fr = make_float3(l1Phase, l1Phase, l1Phase);
		}
		else Fr(l1Mat, l1Tol2, l1Toc1, l1.isect.nor, l1.isect.uv, l1.isect.dpdu, l1Fr, c1Pdf);
		float3 G = tr * cos1 * cos2 / dot(dir, dir);
		L += c1.beta * c1Fr * G * l1Fr * l1.beta;
		if (IsBlack(L)) return{ 0.f, 0.f, 0.f };

		float c2Rev = c2.rev;
		float c1Rev = c1.rev;
		float l1Rev = l1.rev;
		float l2Rev = l2.rev;
		c1.rev = ConvertPdf(c1Pdf, l1.isect, c1.isect);
		l1.rev = ConvertPdf(l1Pdf, c1.isect, l1.isect);
		float l2Pdf, c2Pdf;
		if (l1.medium) l1.medium->Phase(l1Toc1, l1Tol2, l1Phase, l2Pdf);
		else Fr(l1Mat, l1Toc1, l1Tol2, l1.isect.nor, l1.isect.uv, l1.isect.dpdu, l1Fr, l2Pdf);
		if (c1.medium) c1.medium->Phase(c1Tol1, c1Toc2, c1Phase, c2Pdf);
		else Fr(c1Mat, c1Tol1, c1Toc2, c1.isect.nor, c1.isect.uv, c1.isect.dpdu, c1Fr, c2Pdf);
		l2.rev = ConvertPdf(l2Pdf, l1.isect, l2.isect);
		c2.rev = ConvertPdf(c2Pdf, c1.isect, c2.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);
		c2.rev = c2Rev;
		c1.rev = c1Rev;
		l1.rev = l1Rev;
		l2.rev = l2Rev;

		return mis * L;
	}

	return L;
}

__global__ void BdptInit() {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	kernel_Resource.kernel_color[pixel] = { 0.f, 0.f, 0.f };
}

__global__ void Bdpt(int iter, int maxDepth) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	//too slow to use dynamic allocate
	BdptVertex cameraPath[BDPT_MAX_DEPTH + 2];
	BdptVertex lightPath[BDPT_MAX_DEPTH + 2];
	int nCamera = GenerateCameraPath(x, y, cameraPath, uniform, rng);
	int nLight = GenerateLightPath(lightPath, uniform, rng);
	for (int s = 1; s <= nCamera; ++s) {
		for (int t = 0; t <= nLight; ++t) {
			if ((s == 1 && t == 0) || (s == 1 && t == 1))
				continue;

			int raster;
			float3 L = Connect(cameraPath, lightPath, s, t, raster, uniform, rng);
			if (IsInf(L) || IsNan(L) || IsBlack(L))
				continue;

			if (s == 1) {
				atomicAdd(&kernel_Resource.kernel_color[raster].x, L.x);
				atomicAdd(&kernel_Resource.kernel_color[raster].y, L.y);
				atomicAdd(&kernel_Resource.kernel_color[raster].z, L.z);
				continue;
			}

			atomicAdd(&kernel_Resource.kernel_color[pixel].x, L.x);
			atomicAdd(&kernel_Resource.kernel_color[pixel].y, L.y);
			atomicAdd(&kernel_Resource.kernel_color[pixel].z, L.z);
		}
	}
}
//**************************Bdpt End*******************************