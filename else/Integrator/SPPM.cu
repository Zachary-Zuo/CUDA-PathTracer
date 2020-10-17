#include "SPPM.h"
#include "../CudaTools.h"


//**************************PPM Integrator*************************
__device__ VisiblePoint* vps;
__device__ int* vpIdx, * vpOffset;//grid info
int totalNodes = 0;
VisiblePoint* device_vps;
int* device_vpIdx, * device_vpOffset;
__device__ float3 boundsMin, boundsMax;
__device__ int gridRes[3], hashSize;
__global__ void SPPMSetParam(int* idx, float3 fmin, float3 fmax, int x, int y, int z, int hsize) {
	vpIdx = idx;
	boundsMin = fmin;
	boundsMax = fmax;
	gridRes[0] = x;
	gridRes[1] = y;
	gridRes[2] = z;
	hashSize = hsize;
}

//from pbrt-v3
__host__ __device__ bool ToGrid(float3& p, BBox& bounds, int gridRes[3], float3& pi) {
	bool inBounds = true;
	float3 pg = bounds.Offset(p);
	for (int i = 0; i < 3; ++i) {
		(&pi.x)[i] = (int)(gridRes[i] * (&pg.x)[i]);
		inBounds &= ((&pi.x)[i] >= 0 && (&pi.x)[i] < gridRes[i]);
		(&pi.x)[i] = clamp((int)(&pi.x)[i], 0, gridRes[i] - 1);
	}

	return inBounds;
}

__host__ __device__ unsigned int Hash(int x, int y, int z, int hashSize) {
	//those magic number are some large primes
	return (unsigned int)((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) % hashSize;
}

//Still too slow, i will be very grateful if someone tells me how to optimize!!
void BuildHashTable(int width, int height) {
	VisiblePoint* host_vps = new VisiblePoint[width * height];
	HANDLE_ERROR(cudaMemcpy(host_vps, device_vps, width * height * sizeof(VisiblePoint), cudaMemcpyDeviceToHost));

	int hSize = width * height;
	CPUGridNode* grid = new CPUGridNode[hSize];

	BBox gridBounds;
	float initRadius = 0.f;
	for (int i = 0; i < width * height; ++i) {
		gridBounds.Expand(host_vps[i].isect.pos);
		if (host_vps[i].radius > initRadius) initRadius = host_vps[i].radius;
	}

	float3 radius3f = make_float3(initRadius, initRadius, initRadius);
	gridBounds.fmin -= radius3f;
	gridBounds.fmax += radius3f;
	float3 diag = gridBounds.Diagonal();
	float maxDiag = (&diag.x)[gridBounds.GetMaxExtent()];
	int baseGridRes = (int)(maxDiag / initRadius);
	int gRes[3];
	for (int i = 0; i < 3; ++i)
		gRes[i] = Max((int)(baseGridRes * (&diag.x)[i] / maxDiag), 1);

	int total = 0;
	for (int i = 0; i < width * height; ++i) {
		VisiblePoint vp = host_vps[i];
		float3 pMin, pMax;
		ToGrid(vp.isect.pos - radius3f, gridBounds, gRes, pMin);
		ToGrid(vp.isect.pos + radius3f, gridBounds, gRes, pMax);
		for (int z = pMin.z; z <= pMax.z; ++z) {
			for (int y = pMin.y; y <= pMax.y; ++y) {
				for (int x = pMin.x; x <= pMax.x; ++x) {
					int h = Hash(x, y, z, hSize);
					grid[h].vpIdx.push_back(i);
					total++;
				}
			}
		}
	}

	std::vector<int> temp(total), off(hSize + 1); off[0] = 0;
	int* start = &temp[0], offset = 0;
	for (int i = 0; i < hSize; ++i) {
		memcpy(start + offset, &grid[i].vpIdx[0], grid[i].vpIdx.size() * sizeof(int));
		offset += grid[i].vpIdx.size();
		off[i + 1] = offset;
	}

	if (total != totalNodes) {
		HANDLE_ERROR(cudaFree(device_vpIdx));
		HANDLE_ERROR(cudaMalloc(&device_vpIdx, total * sizeof(int)));
	}
	HANDLE_ERROR(cudaMemcpy(device_vpIdx, &temp[0], total * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_vpOffset, &off[0], (hSize + 1) * sizeof(int), cudaMemcpyHostToDevice));

	SPPMSetParam << <1, 1 >> > (device_vpIdx, gridBounds.fmin, gridBounds.fmax, gRes[0], gRes[1], gRes[2], hSize);
	delete[] host_vps;

	delete[] grid;
}

__device__ void TraceRay(int pixel, Ray r, int iter, int maxDepth, float initRadius, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	VisiblePoint& vp = vps[pixel];
	if (iter == 1) {
		vp.radius = initRadius;
		vp.n = 0.f;
		vp.ld = { 0.f, 0.f, 0.f };
		vp.tau = { 0.f, 0.f, 0.f };
		vp.valid = false;
	}

	float3 beta = { 1.f, 1.f, 1.f };
	Ray ray = r;
	bool specular = false;
	for (int bounces = 0; bounces < maxDepth; ++bounces) {
		Intersection isect;
		if (!Intersect(ray, &isect)) {
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material mat = kernel_Resource.kernel_materials[isect.matIdx];

		float3 Ld = { 0.f, 0.f, 0.f };
		if (!IsDelta(mat.type) && isect.lightIdx == -1) {
			float u = uniform(rng);
			float choicePdf;
			int idx = LookUpLightDistribution(u, choicePdf);
			float2 u1 = make_float2(uniform(rng), uniform(rng));
			float3 radiance, lightNor;
			Ray shadowRay;
			float lightPdf;
			kernel_Resource.kernel_lights[idx].SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_Resource.kernel_epsilon);

			if (!IsBlack(radiance) && !IntersectP(shadowRay)) {
				float3 fr;
				float samplePdf;

				Fr(mat, -ray.destination, shadowRay.destination, nor, uv, dpdu, fr, samplePdf);

				float weight = PowerHeuristic(1, lightPdf * choicePdf, 1, samplePdf);
				Ld += weight * fr * radiance * fabs(dot(nor, shadowRay.destination)) / (lightPdf * choicePdf);
			}

			float3 us = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.destination, nor, uv, dpdu, us, out, fr, pdf);
			if (!(IsBlack(fr) || pdf == 0)) {
				Intersection lightIsect;
				Ray lightRay(pos, out, nullptr, kernel_Resource.kernel_epsilon);
				if (Intersect(lightRay, &lightIsect) && lightIsect.lightIdx != -1) {
					float3 p = lightIsect.pos;
					float3 n = lightIsect.nor;
					float3 radiance = { 0.f, 0.f, 0.f };
					radiance = kernel_Resource.kernel_lights[lightIsect.lightIdx].Le(n, -lightRay.destination);
					if (!IsBlack(radiance)) {
						float pdfA, pdfW;
						kernel_Resource.kernel_lights[lightIsect.lightIdx].Pdf(Ray(p, -out, nullptr, kernel_Resource.kernel_epsilon), n, pdfA, pdfW);
						float choicePdf = PdfFromLightDistribution(lightIsect.lightIdx);
						float lenSquare = dot(p - pos, p - pos);
						float costheta = fabs(dot(n, lightRay.destination));
						float lPdf = pdfA * lenSquare / (costheta);
						float weight = PowerHeuristic(1, pdf, 1, lPdf * choicePdf);

						Ld += weight * fr * radiance * fabs(dot(out, nor)) / pdf;
					}
				}

			}
		}

		//light vp
		if (bounces == 0 || (specular && isect.lightIdx != -1)) {
			Ld += kernel_Resource.kernel_lights[isect.lightIdx].Le(nor, -ray.destination);
		}

		if (!IsNan(Ld)) vp.ld += beta * Ld;

		//delta material should be more careful
		if (IsDelta(mat.type) || (IsGlossy(mat.type) && mat.alphaU < 0.2f)) {
			float3 fr, out;
			float pdf;
			float3 uniformBsdf = make_float3(uniform(rng), uniform(rng), uniform(rng));
			SampleBSDF(mat, -ray.destination, nor, uv, dpdu, uniformBsdf, out, fr, pdf);
			if (IsBlack(fr)) return;

			beta *= fr * fabs(dot(out, nor)) / pdf;
			specular = IsDelta(mat.type);

			ray = Ray(pos, out, nullptr, kernel_Resource.kernel_epsilon);

			continue;
		}

		vp.beta = beta;
		vp.dir = -ray.destination;
		vp.isect = isect;
		vp.valid = true;

		break;
	};
}

__device__ void TracePhoton(int maxDepth, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int idx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_Resource.kernel_lights[idx];
	float3 radiance, lightNor;
	float4 lightUniform = { uniform(rng), uniform(rng), uniform(rng), uniform(rng) };
	Ray ray;
	float pdfA, pdfW;
	light.SampleLight(lightUniform, ray, lightNor, radiance, pdfA, pdfW, kernel_Resource.kernel_epsilon);
	beta *= radiance * fabs(dot(lightNor, ray.destination)) / (pdfA * pdfW * choicePdf);

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
		if (bounces > 0) {//bounces = 0 are already taken into account
			float3 gridCoord;
			BBox gridBounds(boundsMin, boundsMax);
			if (ToGrid(pos, gridBounds, gridRes, gridCoord)) {
				int h = Hash(gridCoord.x, gridCoord.y, gridCoord.z, hashSize);
				int start = vpOffset[h];
				int vpSize = vpOffset[h + 1] - start;
				for (int i = 0; i < vpSize; ++i) {
					int idx = vpIdx[start + i];
					VisiblePoint& vp = vps[idx];
					if (!vp.valid) continue;
					float3 out = pos - vp.isect.pos;
					float distanceSquare = dot(out, out);
					if (distanceSquare > vp.radius * vp.radius) continue;
					Material vpMat = kernel_Resource.kernel_materials[vp.isect.matIdx];
					float3 fr;
					float pdf;
					Fr(vpMat, vp.dir, -ray.destination, vp.isect.nor, vp.isect.uv, vp.isect.dpdu, fr, pdf);
					if (IsBlack(fr) || IsNan(fr)) continue;
					float3 b = fr * beta * vp.beta;
					b += vp.tau;

					//suppose just a photon hit the same visible point at the same time
					float alpha = 0.7f;
					float g = (vp.n + alpha) / (vp.n + 1.f);
					float rnew = vp.radius * sqrt(g);
					vp.tau = b * g;
					vp.n += alpha;
					vp.radius = rnew;
				}
			}
		}

		float3 fr, out;
		float3 bsdfUniform = make_float3(uniform(rng), uniform(rng), uniform(rng));
		float pdf;
		SampleBSDF(mat, -ray.destination, nor, uv, dpdu, bsdfUniform, out, fr, pdf, TransportMode::Importance);
		if (pdf == 0) break;

		beta *= fr * fabs(dot(nor, out)) / pdf;

		ray = Ray(pos, out, nullptr, kernel_Resource.kernel_epsilon);

		if (bounces > 3) {
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}
}

__global__ void StochasticProgressivePhotonmapperInit(VisiblePoint* v, int* offset) {
	vps = v;
	vpOffset = offset;
}

//first pass trace eye ray
__global__ void StochasticProgressivePhotonmapperFP(int iter, int maxDepth, float initRadius) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	//ppm doesn't support dof now
	//float2 aperture = UniformSampleDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_Resource.kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, make_float2(0, 0));
	ray.tmin = kernel_Resource.kernel_epsilon;

	TraceRay(pixel, ray, iter, maxDepth, initRadius, uniform, rng);
}

//build hash table for vp
void StochasticProgressivePhotonmapperBuildHashTable(int width, int height) {
	BuildHashTable(width, height);
}

//second pass trace photon
__global__ void StochasticProgressivePhotonmapperSP(int iter, int maxDepth) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter * iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	TracePhoton(maxDepth, uniform, rng);
}

//third pass density evaluate
__global__ void StochasticProgressivePhotonmapperTP(int iter, int photonsPerIteration) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned pixel = x + y * blockDim.x * gridDim.x;

	VisiblePoint& vp = vps[pixel];

	float3 L = { 0.f, 0.f, 0.f };
	if (vp.valid) {
		//as the number of iterations increases, the radius becomes
		//smaller and samller, eventually producing infinity indirect
		float3 indirect = vp.tau / (PI * vp.radius * vp.radius * photonsPerIteration * iter);
		//skip if color is not a number
		if (IsNan(indirect) || IsInf(indirect)) indirect = vp.ind;
		vp.ind = indirect;
		L = vp.ld / iter + indirect;
	}
	kernel_Resource.kernel_color[pixel] = L;
}
//**************************SPPM End********************************
