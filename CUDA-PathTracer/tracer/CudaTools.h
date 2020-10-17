#pragma once
#include "camera.h"
#include "scene.h"
#include "bvh.h"
#include "device_launch_parameters.h"
#include "Sampling.h"
#include <thrust/random.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include "Resource.h"
//#include "externParam.h"

__device__ inline unsigned int WangHash(unsigned int seed)
{
	seed = (seed ^ 61) ^ (seed >> 16);
	seed = seed + (seed << 3);
	seed = seed ^ (seed >> 4);
	seed = seed * 0x27d4eb2d;
	seed = seed ^ (seed >> 15);

	return seed;
}

__host__ __device__ inline float DielectricFresnel(float cosi, float cost, const float& etai, const float& etat) {
	float Rparl = (etat * cosi - etai * cost) / (etat * cosi + etai * cost);
	float Rperp = (etai * cosi - etat * cost) / (etai * cosi + etat * cost);

	return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

__device__ inline float3 ConductFresnel(float cosi, const float3& eta, const float3& k) {
	float3 tmp = (eta * eta + k * k) * cosi * cosi;
	float3 Rparl2 = (tmp - eta * cosi * 2.f + 1.f) /
		(tmp + eta * cosi * 2.f + 1.f);
	float3 tmp_f = (eta * eta + k * k);
	float3 Rperp2 = (tmp_f - eta * cosi * 2.f + cosi * cosi) /
		(tmp_f + eta * cosi * 2.f + cosi * cosi);
	return (Rparl2 + Rperp2) * 0.5f;
}

__device__ inline float GGX_D(float3& wh, float3& normal, float3 dpdu, float alphaU, float alphaV) {
	float costheta = dot(wh, normal);
	if (costheta <= 0.f) return 0.f;
	costheta = clamp(costheta, 0.f, 1.f);
	float costheta2 = costheta * costheta;
	float sintheta2 = 1.f - costheta2;
	float costheta4 = costheta2 * costheta2;
	float tantheta2 = sintheta2 / costheta2;

	float3 uu = dpdu;
	float3 dir = normalize(wh - costheta * normal);
	float cosphi = dot(dir, uu);
	float cosphi2 = cosphi * cosphi;
	float sinphi2 = 1.f - cosphi2;
	float sqrD = 1.f + tantheta2 * (cosphi2 / (alphaU * alphaU) + sinphi2 / (alphaV * alphaV));
	return 1.f / (PI * alphaU * alphaV * costheta4 * sqrD * sqrD);
}

__device__ inline float SmithG(float3& w, float3& normal, float3& wh, float3 dpdu, float alphaU, float alphaV) {
	float wdn = dot(w, normal);
	if (wdn * dot(w, wh) < 0.f)	return 0.f;
	float sintheta = sqrtf(clamp(1.f - wdn * wdn, 0.f, 1.f));
	float tantheta = sintheta / wdn;
	if (isinf(tantheta)) return 0.f;

	float3 uu = dpdu;
	float3 dir = normalize(w - wdn * normal);
	float cosphi = dot(dir, uu);
	float cosphi2 = cosphi * cosphi;
	float sinphi2 = 1.f - cosphi2;
	float alpha2 = cosphi2 * (alphaU * alphaU) + sinphi2 * (alphaV * alphaV);
	float sqrD = alpha2 * tantheta * tantheta;
	return 2.f / (1.f + sqrtf(1 + sqrD));
}

__device__ inline float GGX_G(float3& wo, float3& wi, float3& normal, float3& wh, float3 dpdu, float alphaU, float alphaV) {
	return SmithG(wo, normal, wh, dpdu, alphaU, alphaV) * SmithG(wi, normal, wh, dpdu, alphaU, alphaV);
}

__device__ inline float3 SampleGGX(float alphaU, float alphaV, float u1, float u2) {
	if (alphaU == alphaV) {
		float costheta = sqrtf((1.f - u1) / (u1 * (alphaU * alphaV - 1.f) + 1.f));
		float sintheta = sqrtf(1.f - costheta * costheta);
		float phi = 2 * PI * u2;
		float cosphi = cosf(phi);
		float sinphi = sinf(phi);

		return{
			sintheta * cosphi,
			costheta,
			sintheta * sinphi
		};
	}
	else {
		float phi;
		if (u2 <= 0.25) phi = atan(alphaV / alphaU * tan(TWOPI * u2));
		else if (u2 >= 0.75f) phi = atan(alphaV / alphaU * tan(TWOPI * u2)) + TWOPI;
		else phi = atan(alphaV / alphaU * tan(TWOPI * u2)) + PI;
		float sinphi = sin(phi), cosphi = cos(phi);
		float sinphi2 = sinphi * sinphi;
		float cosphi2 = 1.0f - sinphi2;
		float inverseA = 1.0f / (cosphi2 / (alphaU * alphaU) + sinphi2 / (alphaV * alphaV));
		float theta = atan(sqrt(inverseA * u1 / (1.0f - u1)));
		float sintheta = sin(theta), costheta = cos(theta);
		return{
			sintheta * cosphi,
			costheta,
			sintheta * sinphi
		};
	}
}

__host__ __device__ inline float3 Reflect(float3 in, float3 nor) {
	return 2.f * dot(in, nor) * nor - in;
}

__host__ __device__ inline float3 Refract(float3 in, float3 nor, float etai, float etat) {
	float cosi = dot(in, nor);
	bool enter = cosi > 0;
	if (!enter) {
		float t = etai;
		etai = etat;
		etat = t;
	}

	float eta = etai / etat;
	float sini2 = 1.f - cosi * cosi;
	float sint2 = sini2 * eta * eta;
	float cost = sqrtf(1.f - sint2);
	return normalize((nor * cosi - in) * eta + (enter ? -cost : cost) * nor);
}

__device__ inline float3 SchlickFresnel(float3 specular, float costheta) {
	float3 rs = specular;
	float c = 1.f - costheta;
	return rs + c * c * c * c * c * (make_float3(1.f, 1.f, 1.f) - rs);
}

__device__ inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
	float f = nf * fPdf, g = ng * gPdf;
	return (f * f) / (f * f + g * g);
}

//当光源多的时候可以使用二分法加速
__device__ inline int LookUpLightDistribution(float u, float& pdf) {
	for (int i = 0; i < kernel_Resource.kernel_light_distribution_size; ++i) {
		float s = kernel_Resource.kernel_light_distribution[i];
		float e = kernel_Resource.kernel_light_distribution[i + 1];
		if (u >= s && u <= e) {
			pdf = e - s;
			return i;
		}
	}
}

__device__ inline float PdfFromLightDistribution(int idx) {
	return kernel_Resource.kernel_light_distribution[idx + 1] - kernel_Resource.kernel_light_distribution[idx];
}

__device__ inline float Luminance(const float3& c) {
	return dot(c, { 0.212671f, 0.715160f, 0.072169f });
}

__device__ inline bool SameHemiSphere(float3& in, float3& out, float3& nor) {
	return dot(in, nor) * dot(out, nor) > 0 ? true : false;
}

__device__ inline bool Intersect(Ray& ray, Intersection* isect) {
	int stack[64];
	int* stack_top = stack;
	int* stack_bottom = stack;

	bool ret = false;
	int node_idx = 0;
	do {
		LinearBVHNode node = kernel_Resource.kernel_linear[node_idx];
		bool intersect = node.bbox.Intersect(ray);
		if (intersect) {
			if (!node.is_leaf) {
				*stack_top++ = node.second_child_offset;
				*stack_top++ = node_idx + 1;
			}
			else {
				for (int i = node.start; i <= node.end; ++i) {
					Primitive prim = kernel_Resource.kernel_primitives[i];

					if (prim.type == GT_TRIANGLE) {
						if (prim.triangle.Intersect(ray, isect))
							ret = true;
					}
					else if (prim.type == GT_LINES) {
						if (prim.line.Intersect(ray, isect))
							ret = true;
					}
					else if (prim.type == GT_SPHERE) {
						if (prim.sphere.Intersect(ray, isect))
							ret = true;
					}
				}
			}
		}

		if (stack_top == stack_bottom)
			break;
		node_idx = *--stack_top;
	} while (true);

	return ret;
}

__device__ inline bool IntersectP(Ray& ray) {
	int stack[64];
	int* stack_top = stack;
	int* stack_bottom = stack;

	int node_idx = 0;
	do {
		LinearBVHNode node = kernel_Resource.kernel_linear[node_idx];
		bool intersect = node.bbox.Intersect(ray);
		if (intersect) {
			if (!node.is_leaf) {
				*stack_top++ = node.second_child_offset;
				*stack_top++ = node_idx + 1;
			}
			else {
				for (int i = node.start; i <= node.end; ++i) {
					Primitive prim = kernel_Resource.kernel_primitives[i];
					if (prim.type == GT_TRIANGLE) {
						if (prim.triangle.Intersect(ray, nullptr))
							return true;
					}
					else if (prim.type == GT_LINES) {
						if (prim.line.Intersect(ray, nullptr))
							return true;
					}
					else if (prim.type == GT_SPHERE) {
						if (prim.sphere.Intersect(ray, nullptr))
							return true;
					}
				}
			}
		}

		if (stack_top == stack_bottom)
			break;
		node_idx = *--stack_top;
	} while (true);

	return false;
}

__device__ inline float3 Tr(Ray& ray, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	float3 tr = make_float3(1, 1, 1);
	float tmax = ray.tmax;
	while (true) {
		Intersection isect;
		bool invisible = Intersect(ray, &isect);
		if (invisible && isect.matIdx != -1)
			return{ 0, 0, 0 };

		if (ray.medium) {
			if (ray.medium->type == MT_HOMOGENEOUS)
				tr *= ray.medium->homogeneous.Tr(ray, uniform, rng);
			else
				tr *= ray.medium->heterogeneous.Tr(ray, uniform, rng);
		}

		if (!invisible) break;
		Medium* m = dot(ray.destination, isect.nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumOutside])
			: (isect.mediumInside == -1 ? nullptr : &kernel_Resource.kernel_mediums[isect.mediumInside]);
		tmax -= ray.tmax;
		ray = Ray(ray(ray.tmax), ray.destination, m, kernel_Resource.kernel_epsilon, tmax);
	}

	return tr;
}

__device__ inline float4 getTexel(Material material, int w, int h, int2 uv) {
	float inv = 1.f / 255.f;

	int x = uv.x, y = uv.y;
	float rx = x - (x / w) * w;
	float ry = y - (y / h) * h;
	x = (rx < 0) ? rx + w : rx;
	y = (ry < 0) ? ry + h : ry;
	if (x < 0) x = 0;
	if (x > w - 1) x = w - 1;
	if (y < 0) y = 0;
	if (y > h - 1) y = h - 1;

	uchar4 c = kernel_Resource.kernel_textures[material.textureIdx][y * w + x];
	return make_float4(c.x * inv, c.y * inv, c.z * inv, c.w * inv);
}

__device__ inline float4 GetTexel(Material material, float2 uv) {
	if (material.textureIdx == -1)
		return make_float4(material.diffuse, 1.f);

	int w = kernel_Resource.kernel_texture_size[material.textureIdx * 2];
	int h = kernel_Resource.kernel_texture_size[material.textureIdx * 2 + 1];
	float xx = w * uv.x;
	float yy = h * uv.y;
	int x = floor(xx);
	int y = floor(yy);
	float dx = fabs(xx - x);
	float dy = fabs(yy - y);
	float4 c00 = getTexel(material, w, h, make_int2(x, y));
	float4 c10 = getTexel(material, w, h, make_int2(x + 1, y));
	float4 c01 = getTexel(material, w, h, make_int2(x, y + 1));
	float4 c11 = getTexel(material, w, h, make_int2(x + 1, y + 1));
	return (1 - dy) * ((1 - dx) * c00 + dx * c10)
		+ dy * ((1 - dx) * c01 + dx * c11);
}

//**************************bssrdf*****************
__device__ inline float3 SingleScatter(Intersection* isect, float3 in, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	float3 pos = isect->pos;
	float3 nor = isect->nor;
	float coso = fabs(dot(in, nor));
	Bssrdf bssrdf = kernel_Resource.kernel_bssrdfs[isect->bssrdf];
	float eta = bssrdf.eta;
	float sino2 = 1.f - coso * coso;
	float cosi = sqrtf(1.f - sino2 / (eta * eta));
	float fresnel = 1.f - DielectricFresnel(coso, cosi, 1.f, eta);
	float sigmaTr = Luminance(bssrdf.GetSigmaTr());
	float3 sigmaS = bssrdf.GetSigmaS();
	float3 sigmaT = bssrdf.GetSigmaT();
	float3 rdir = Reflect(in, nor);
	float3 tdir = Refract(in, nor, 1.f, eta);
	float3 L = { 0, 0, 0 };
	Intersection rIsect;
	if (Intersect(Ray(pos, rdir, nullptr, kernel_Resource.kernel_epsilon), &rIsect)) {
		if (rIsect.lightIdx != -1) {
			L += (1.f - fresnel) * kernel_Resource.kernel_lights[rIsect.lightIdx].Le(rIsect.nor, -rdir);
		}
	}
	Intersection tIsect;
	Intersect(Ray(pos, tdir, nullptr, kernel_Resource.kernel_hdr_height), &tIsect);
	float len = length(tIsect.pos - pos);
	int samples = 1;
	for (int i = 0; i < samples; ++i) {
		float d = Exponential(uniform(rng), sigmaTr);
		if (d > len) continue;
		float3 pSample = pos + tdir * d;
		float pdf = ExponentialPdf(d, sigmaTr);
		float choicePdf;
		float u = uniform(rng);
		int idx = LookUpLightDistribution(u, choicePdf);
		Area light = kernel_Resource.kernel_lights[idx];
		float lightPdf;
		Ray shadowRay;
		float3 radiance, lightNor;
		float2 u1 = make_float2(uniform(rng), uniform(rng));
		light.SampleLight(pSample, u1, radiance, shadowRay, lightNor, lightPdf, kernel_Resource.kernel_epsilon);
		if (IsBlack(radiance))
			continue;

		float tmax = shadowRay.tmax;
		Intersection wiIsect;
		if (Intersect(shadowRay, &wiIsect)) {
			if (wiIsect.bssrdf == isect->bssrdf) {
				float3 wiPos = wiIsect.pos;
				float3 wiNor = wiIsect.nor;
				shadowRay.tmin += shadowRay.tmax;
				shadowRay.tmax = tmax;
				if (!IntersectP(shadowRay)) {
					float p = bssrdf.GetPhase();
					float cosi = fabs(dot(wiNor, shadowRay.destination));
					float sini2 = 1.f - cosi * cosi;
					float coso = sqrtf(1.f - sini2 / (eta * eta));
					float fresnelI = 1.f - DielectricFresnel(cosi, coso, 1.f, eta);
					float G = fabs(dot(wiNor, tdir)) / cosi;
					float3 sigmaTC = sigmaT * (1.f + G);
					float di = length(wiPos - pSample);
					float et = 1.f / eta;
					float diPrime = di * fabs(dot(shadowRay.destination, wiNor)) /
						sqrt(1.f - et * et * (1.f - cosi * cosi));
					L += (fresnel * fresnelI * p * sigmaS / sigmaTC) *
						Exp(-diPrime * sigmaT) *
						Exp(-d * sigmaT) * radiance / (lightPdf * choicePdf * pdf);
				}
			}
		}
	}

	L /= samples;
	return L;
}

__device__ inline float3 MultipleScatter(Intersection* isect, float3 in, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng) {
	float3 pos = isect->pos;
	float3 nor = isect->nor;
	float coso = fabs(dot(in, nor));
	Bssrdf bssrdf = kernel_Resource.kernel_bssrdfs[isect->bssrdf];
	float eta = bssrdf.eta;
	float sino2 = 1.f - coso * coso;
	float cosi = sqrtf(1.f - sino2 / (eta * eta));
	float fresnel = 1.f - DielectricFresnel(coso, cosi, 1.f, eta);
	float sigmaTr = Luminance(bssrdf.GetSigmaTr());
	float skipRatio = 0.01f;
	float rMax = sqrt(log(skipRatio) / -sigmaTr);
	float3 L = { 0, 0, 0 };
	int samples = 1;
	for (int i = 0; i < samples; ++i) {
		Ray probeRay;
		float pdf;
		float2 u = make_float2(uniform(rng), uniform(rng));
		bssrdf.SampleProbeRay(pos, nor, u, sigmaTr, rMax, probeRay, pdf);
		probeRay.tmin = kernel_Resource.kernel_epsilon;

		Intersection probeIsect;
		if (Intersect(probeRay, &probeIsect)) {
			if (isect->bssrdf == probeIsect.bssrdf) {
				float3 probePos = probeIsect.pos;
				float3 probeNor = probeIsect.nor;
				float3 rd = bssrdf.Rd(dot(probePos - pos, probePos - pos));
				float choicePdf;
				float u = uniform(rng);
				int idx = LookUpLightDistribution(u, choicePdf);
				Area light = kernel_Resource.kernel_lights[idx];
				float lightPdf;
				float2 u1 = make_float2(uniform(rng), uniform(rng));
				float3 radiance, lightNor;
				Ray shadowRay;
				light.SampleLight(probePos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_Resource.kernel_epsilon);
				if (!IsBlack(radiance) && !IntersectP(shadowRay)) {
					float cosi = fabs(dot(shadowRay.destination, probeNor));
					float sini2 = 1.f - cosi * cosi;
					float cost = sqrtf(1.f - sini2 / (eta * eta));
					float3 irradiance = radiance * cosi / (lightPdf * choicePdf);
					float fresnelI = 1.f - DielectricFresnel(cosi, cost, 1.f, eta);
					pdf *= fabs(dot(probeRay.destination, probeNor));
					L += (ONE_OVER_PI * fresnel * fresnelI * rd * irradiance) / pdf;
				}
			}
		}

		L /= samples;
		return L;
	}
}
//**************************bssrdf end*************

//**************************BSDF Sampling**************************
__device__ inline void SampleBSDF(Material material, float3 in, float3 nor, float2 uv, float3 dpdu, float3 u, float3& out, float3& fr, float& pdf, TransportMode mode = TransportMode::Radiance) {
	switch (material.type) {
	case MT_LAMBERTIAN: {
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		out = CosineSampleHemiSphere(u.x, u.y, n, pdf);
		float3 uu = dpdu, ww;
		ww = cross(uu, n);
		out = ToWorld(out, uu, n, ww);
		fr = make_float3(GetTexel(material, uv)) * ONE_OVER_PI;
		break;
	}

	case MT_MIRROR:
		out = Reflect(in, nor);
		fr = material.specular / fabs(dot(out, nor));
		pdf = 1.f;
		break;

	case MT_DIELECTRIC: {
		float3 wi = -in;
		float3 normal = nor;

		float ei = material.outsideIOR, et = material.insideIOR;
		float cosi = dot(wi, normal);
		bool enter = cosi < 0;
		if (!enter) {
			float t = ei;
			ei = et;
			et = t;
		}

		float eta = ei / et, cost;
		float sint2 = eta * eta * (1.f - cosi * cosi);
		cost = sqrtf(1.f - sint2 < 0.f ? 0.f : 1.f - sint2);
		float3 rdir = Reflect(-wi, normal);
		float3 tdir = Refract(in, nor, material.outsideIOR, material.insideIOR);
		if (sint2 > 1.f) {//total reflection
			out = rdir;
			fr = material.specular / fabs(dot(out, normal));
			pdf = 1.f;
			return;
		}

		float fresnel = DielectricFresnel(fabs(cost), fabs(cosi), et, ei);
		if (u.x > fresnel) {//refract
			out = tdir;
			fr = material.specular / fabs(dot(out, normal)) * (1.f - fresnel);
			if (mode == TransportMode::Radiance)
				fr *= eta * eta;
			pdf = 1.f - fresnel;
		}
		else {//reflect
			out = rdir;
			fr = material.specular / fabs(dot(out, normal)) * fresnel;
			pdf = fresnel;
		}
		break;
	}

	case MT_ROUGHCONDUCTOR: {
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		float3 wh = SampleGGX(material.alphaU, material.alphaV, u.x, u.y);
		float3 uu = dpdu, ww;
		ww = cross(uu, n);
		wh = ToWorld(wh, uu, n, ww);
		out = Reflect(in, wh);
		if (!SameHemiSphere(in, out, nor)) {
			fr = { 0, 0, 0 };
			pdf = 0.f;
			return;
		}

		float cosi = dot(out, wh);
		float3 F = ConductFresnel(fabs(cosi), material.eta, material.k);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);

		fr = material.specular * F * D * G /
			(4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
		pdf = D * fabs(dot(wh, n)) / (4.f * fabs(dot(in, wh)));
		break;
	}

	case MT_SUBSTRATE: {
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;
		if (u.x < 0.5) {
			float ux = u.x * 2.f;
			out = CosineSampleHemiSphere(ux, u.y, n, pdf);
			float3 uu = dpdu, ww;
			ww = cross(uu, n);
			out = ToWorld(out, uu, n, ww);
		}
		else {
			float ux = (u.x - 0.5f) * 2.f;
			float3 wh = SampleGGX(material.alphaU, material.alphaV, ux, u.y);
			float3 uu = dpdu, ww;
			ww = cross(uu, n);
			wh = ToWorld(wh, uu, n, ww);
			out = Reflect(in, wh);
		}
		if (!SameHemiSphere(in, out, n)) {
			fr = { 0.f, 0.f, 0.f };
			pdf = 0.f;
			return;
		}
		float c0 = fabs(dot(in, n));
		float c1 = fabs(dot(out, n));
		float3 Rd = make_float3(GetTexel(material, uv));
		float3 Rs = material.specular;
		float cons0 = 1 - 0.5f * c0;
		float cons1 = 1 - 0.5f * c1;
		/*if (u.x < 0.5f){
			float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
				(1 - cons0*cons0*cons0*cons0*cons0) *
				(1 - cons1*cons1*cons1*cons1*cons1);
			fr = diffuse;
			pdf = fabs(dot(out, n)) * ONE_OVER_PI*0.5f;
		}
		else{
			float3 wh = normalize(in + out);
			float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
			float3 specular = D /
				(4.f * fabs(dot(out, wh))*Max(c0, c1))*
				SchlickFresnel(Rs, dot(out, wh));

			fr =  specular;
			pdf = 0.5f * (D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));
		}*/
		float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
			(1 - cons0 * cons0 * cons0 * cons0 * cons0) *
			(1 - cons1 * cons1 * cons1 * cons1 * cons1);
		float3 wh = normalize(in + out);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float3 specular = D /
			(4.f * fabs(dot(out, wh)) * Max(c0, c1)) *
			SchlickFresnel(Rs, dot(out, wh));

		fr = diffuse + specular;
		pdf = 0.5f * (fabs(dot(out, n)) * ONE_OVER_PI + D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));

		break;
	}

	case MT_ROUGHDIELECTRIC: {
		float3 wi = -in;
		float3 n = nor;
		float3 wh = SampleGGX(material.alphaU, material.alphaV, u.x, u.y);
		float3 uu = dpdu, ww;
		ww = cross(uu, n);
		wh = ToWorld(wh, uu, n, ww);

		float ei = material.outsideIOR, et = material.insideIOR;
		float cosi = dot(wi, n);
		bool enter = cosi < 0;
		if (!enter) {
			float t = ei;
			ei = et;
			et = t;
		}

		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float eta = ei / et, cost;
		cosi = dot(wi, wh);
		float sint2 = eta * eta * (1.f - cosi * cosi);
		cost = sqrtf(1.f - sint2 < 0.f ? 0.f : 1.f - sint2);
		float3 rdir = Reflect(-wi, wh);
		float3 tdir = normalize((wi - wh * cosi) * eta + (enter ? -cost : cost) * wh);
		if (sint2 > 1.f) {//total reflection
			out = rdir;
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			fr = material.specular * D * G / (4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
			pdf = D * fabs(dot(wh, n)) / (4.f * fabs(dot(wh, in)));
			return;
		}

		float fresnel = DielectricFresnel(fabs(cost), fabs(cosi), et, ei);
		if (u.z > fresnel) {//refract
			out = tdir;
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			float c = et * dot(out, wh) + ei * dot(in, wh);
			fr = material.specular * ei * ei * D * G * (1.f - fresnel) * fabs(dot(in, wh)) * fabs(dot(out, wh)) /
				(fabs(dot(out, n)) * fabs(dot(in, n)) * c * c);
			if (mode == TransportMode::Radiance)
				fr *= (1.f / (eta * eta));

			pdf = (1.f - fresnel) * D * fabs(dot(wh, n)) * et * et * fabs(dot(out, wh)) / (c * c);
		}
		else {//reflect
			out = rdir;
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			fr = material.specular * fresnel * D * G / (4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
			pdf = D * fabs(dot(wh, n)) / (4.f * fabs(dot(wh, in))) * fresnel;
		}
		break;
	}
	}
}

//__device__ void Fr(Material material, float3 in, float3 out, float3 nor, float2 uv, float3 dpdu, float u, float3& fr, float& pdf){
__device__ inline void Fr(Material material, float3 in, float3 out, float3 nor, float2 uv, float3 dpdu, float3& fr, float& pdf, TransportMode mode = TransportMode::Radiance) {
	switch (material.type) {
	case MT_LAMBERTIAN:
		if (!SameHemiSphere(in, out, nor)) {
			fr = make_float3(0.f, 0.f, 0.f);
			pdf = 0.f;
			return;
		}

		fr = make_float3(GetTexel(material, uv)) * ONE_OVER_PI;
		pdf = fabs(dot(out, nor)) * ONE_OVER_PI;
		break;

	case MT_MIRROR:
		fr = make_float3(0.f, 0.f, 0.f);
		pdf = 0.f;
		break;

	case MT_DIELECTRIC:
		fr = make_float3(0.f, 0.f, 0.f);
		pdf = 0.f;
		break;

	case MT_ROUGHCONDUCTOR: {
		if (!SameHemiSphere(in, out, nor)) {
			fr = { 0, 0, 0 };
			pdf = 0;
			return;
		}
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		float3 wh = normalize(in + out);
		float cosi = dot(out, wh);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
		float3 F = ConductFresnel(fabs(cosi), material.eta, material.k);
		fr = material.specular * F * D * G /
			(4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
		pdf = D * fabs(dot(wh, n)) / (4.f * fabs(dot(in, wh)));
		break;
	}

	case MT_SUBSTRATE: {
		if (!SameHemiSphere(in, out, nor)) {
			fr = { 0, 0, 0 };
			pdf = 0;
			return;
		}

		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		float c0 = fabs(dot(in, n));
		float c1 = fabs(dot(out, n));
		float3 Rd = make_float3(GetTexel(material, uv));
		float3 Rs = material.specular;
		float cons0 = 1 - 0.5f * c0;
		float cons1 = 1 - 0.5f * c1;
		float3 wh = normalize(in + out);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		/*if (D < 1e-4 || u < 0.5f){
			float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
				(1 - cons0*cons0*cons0*cons0*cons0) *
				(1 - cons1*cons1*cons1*cons1*cons1);
			fr = diffuse;
			pdf = 0.5f*fabs(dot(out, n)) * ONE_OVER_PI;
		}
		else{
			float3 specular = D /
				(4.f * fabs(dot(out, wh))*Max(c0, c1))*
				SchlickFresnel(Rs, dot(out, wh));

			fr =  specular;
			pdf = 0.5f * (D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));
		}*/
		float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
			(1 - cons0 * cons0 * cons0 * cons0 * cons0) *
			(1 - cons1 * cons1 * cons1 * cons1 * cons1);
		float3 specular = D /
			(4.f * fabs(dot(out, wh)) * Max(c0, c1)) *
			SchlickFresnel(Rs, dot(out, wh));
		fr = diffuse + specular;
		pdf = 0.5f * (fabs(dot(out, n)) * ONE_OVER_PI + D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));
		break;
	}

	case MT_ROUGHDIELECTRIC: {
		float3 wi = -in;
		float3 n = nor;
		bool reflect = dot(in, n) * dot(out, n) > 0;

		float ei = material.outsideIOR, et = material.insideIOR;
		float cosi = dot(wi, n);
		bool enter = cosi < 0;
		if (!enter) {
			float t = ei;
			ei = et;
			et = t;
		}

		float3 wh = normalize(-(ei * in + et * out));
		float eta = ei / et, cost;
		cosi = dot(wi, wh);
		float sint2 = eta * eta * (1.f - cosi * cosi);
		cost = sqrtf(1.f - sint2 < 0.f ? 0.f : 1.f - sint2);
		float fresnel = DielectricFresnel(fabs(cost), fabs(cosi), et, ei);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		if (!reflect) {//refract
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			float c = et * dot(out, wh) + ei * dot(in, wh);
			fr = material.specular * ei * ei * D * G * (1.f - fresnel) * fabs(dot(in, wh)) * fabs(dot(out, wh)) /
				(fabs(dot(out, n)) * fabs(dot(in, n)) * c * c);
			if (mode == TransportMode::Radiance)
				fr *= (1.f / (eta * eta));
			pdf = (1.f - fresnel) * D * fabs(dot(wh, n)) * et * et * fabs(dot(out, wh)) / (c * c);
		}
		else {
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			fr = material.specular * fresnel * D * G / (4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
			pdf = fresnel * D * fabs(dot(wh, n)) / (4.f * fabs(dot(wh, in)));

		}
		break;
	}
	}
}
//**************************BSDF End*******************************
