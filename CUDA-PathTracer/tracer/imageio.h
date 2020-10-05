#pragma once
#include "common.h"
#include <vector>

class ImageIO{
public:
	static bool LoadTexture(const char* filename, int& width, int& height, bool srgb, vector<float4>& output);
	static bool SavePng(const char* filename, int width, int height, float3* input);
	static bool LoadExr(const char* filename, int& width, int& height, vector<float3>& output);
	static bool SaveExr(const char* filename, int width, int height, vector<float3>& input);
};