#include "CudaRender.h"
#include "pathtracer.h"

CudaRender* CudaRender::s_pInstance = nullptr;
std::mutex CudaRender::s_mutex;

CudaRender* CudaRender::getInstance()
{
	if (s_pInstance == nullptr)
	{
		s_mutex.lock();
		if (s_pInstance == nullptr)
		{
			s_pInstance = new CudaRender();
		}
		s_mutex.unlock();
	}
	return s_pInstance;
}

CudaRender::CudaRender()
{
}

CudaRender::~CudaRender()
{
	cudaFree(dev_ptr);
	delete[]image;
    delete[]reverseImage;
}



void CudaRender::InitCudaScene(std::string file)
{
    InitScene(file, config, scene);
    BeginRender(scene, config.width, config.height, config.epsilon);
    image = new float3[config.width * config.height];
    reverseImage = new float3[config.width * config.height];
    cudaMalloc(&dev_ptr, config.width * config.height * sizeof(float3));
    m_bInitImage = true;
}

float3* CudaRender::renderImage()
{
    static uint64_t key = 0;
    // Launch cuda kernel to generate sinewave in vertex buffer
    //RunSineWaveKernel(m_extSemaphore, key, INFINITE, m_nWindowWidth, m_nWindowHeight, m_VertexBufPtr, m_cuda_stream);
    iteration++;

    reset_acc_image = false;
    //Render(scene, 1920, 1080, scene.camera, iteration, false, m_VertexBufPtr,m_extSemaphore, key, INFINITE, m_cuda_stream);

    Render(scene, config.width, config.height, scene.camera, iteration, reset_acc_image, dev_ptr);
    cudaMemcpy(image, dev_ptr, config.width * config.height * sizeof(float3), cudaMemcpyDeviceToHost);
    //char buffer[2048] = { 0 };
    //sprintf(buffer, "E:/Project/CUDA-PathTracer/result/%ds iteration %dpx-%dpx.png", iteration, 1920, 1080);
    //ImageIO::SavePng(buffer, config.width, config.height, &image[0]);
    return image;
}

float3* CudaRender::ReverseImage(float3 * image)
{
    for (int i = 0; i < config.height; i++)
    {
        for (int j = 0; j < config.width; j++)
        {
            reverseImage[(config.height - i - 1) * config.width + j] = image[i * config.width + j];
        }
    }
    return reverseImage;
}

float3* CudaRender::GetRenderReverseImage()
{
    return ReverseImage(renderImage());
}

GlobalConfig CudaRender::GetConfig()
{
    return config;
}