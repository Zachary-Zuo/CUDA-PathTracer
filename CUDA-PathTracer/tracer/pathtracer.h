#pragma once
#include "common.h"
#include "../ShaderStructs.h"

class Camera;
class Scene;
class BVH;

//void Render(Scene& scene, unsigned width, unsigned height, Camera* camera, unsigned iter, bool reset, DXVertex* output);
void Render(Scene& scene, unsigned width, unsigned height, Camera* camera, unsigned iter, bool reset, DXVertex* output,
	cudaExternalSemaphore_t& extSemaphore, uint64_t& key, unsigned int timeoutMs, cudaStream_t streamToRun);
void BeginRender(Scene& scene, unsigned width, unsigned height, float ep);
void EndRender();