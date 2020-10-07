#pragma once
#include<mutex>
#include "scene.h"
#include "parsescene.h"

class CudaRender
{
public:
	static CudaRender* getInstance();
	~CudaRender();

    void InitCudaScene(std::string file = "E:/Project/CUDA-PathTracer/x64/Debug/scene.json");

    float3* renderImage();
    float3* ReverseImage(float3* image);
    float3* GetRenderReverseImage();
    GlobalConfig GetConfig();

private:
	CudaRender();
    

    GlobalConfig config;
    unsigned iteration = 0;
    bool vision_bvh = false;
    bool reset_acc_image = false;
    clock_t start = 0, last = 0;
    float3* image, * dev_ptr;
    float3* reverseImage;
    Scene scene;
    cudaGraphicsResource* resource = NULL;
    bool m_bInitImage = false;
    bool m_bChangeImage = false;


	static CudaRender* s_pInstance;
	static std::mutex s_mutex;
};

