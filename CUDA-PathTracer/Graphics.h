#pragma once
#pragma warning(disable : 4312)
#include <Windows.h>
#include "CasterLabWin.h"
#include "WindowException.h"
#include <d3d11.h>
#include <wrl.h>
#include <vector>
#include "DxgiInfoManager.h"
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <memory>
#include <random>

#include "ShaderStructs.h"
#include "tracer/scene.h"
#include "tracer/parsescene.h"


class Graphics
{
    friend class Bindable;

public:
    class Exception : public WindowException
    {
        using WindowException::WindowException;
    };
    class HrException : public Exception
    {
    public:
        HrException(int line, const char* file, HRESULT hr, std::vector<std::string> infoMsgs = {}) noexcept;
        const char* what() const noexcept override;
        const char* GetType() const noexcept override;
        HRESULT GetErrorCode() const noexcept;
        std::string GetErrorString() const noexcept;
        std::string GetErrorDescription() const noexcept;
        std::string GetErrorInfo() const noexcept;

    private:
        HRESULT hr;
        std::string info;
    };
    class InfoException : public Exception
    {
    public:
        InfoException(int line, const char* file, std::vector<std::string> infoMsgs) noexcept;
        const char* what() const noexcept override;
        const char* GetType() const noexcept override;
        std::string GetErrorInfo() const noexcept;

    private:
        std::string info;
    };
    class DeviceRemovedException : public HrException
    {
        using HrException::HrException;

    public:
        const char* GetType() const noexcept override;

    private:
        std::string reason;
    };

public:
    Graphics(HWND hWnd, int width, int height);
    Graphics(const Graphics&) = delete;
    Graphics& operator=(const Graphics&) = delete;
    ~Graphics();
    void EndFrame();
    void BeginFrame(float red, float green, float blue) noexcept;
    void DrawIndexed(UINT count) noexcept(!IS_DEBUG);
    void SetProjection(DirectX::FXMMATRIX proj) noexcept;
    DirectX::XMMATRIX GetProjection() const noexcept;
    void SetCamera(DirectX::FXMMATRIX cam) noexcept;
    DirectX::XMMATRIX GetCamera() const noexcept;
    void EnableImgui() noexcept;
    void DisableImgui() noexcept;
    bool IsImguiEnabled() const noexcept;
    bool findCUDADevice();
    static bool dynlinkLoadD3D11API();
    bool findDXDevice(char* device_name);
    void Graphics::InitCudaScene();

private:
    DirectX::XMMATRIX projection;
    DirectX::XMMATRIX camera;
    bool m_bImguiEnabled = true;
    int m_nWindowWidth;
    int m_nWindowHeight;

    Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
    Microsoft::WRL::ComPtr<IDXGISwapChain> m_pSwap;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pContext;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_pTarget;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> m_pCudaCapableAdapter;
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_pVertexBuffer;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_pRasterState;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> m_pLayout;
    IDXGIKeyedMutex* m_pKeyedMutex11;
    DXVertex* m_VertexBufPtr = NULL;
    cudaExternalMemory_t m_extMemory;
    cudaExternalSemaphore_t m_extSemaphore;
    cudaStream_t m_cuda_stream;
    char m_szdevice_name[256];

    GlobalConfig config;
    unsigned iteration = 0;
    bool vision_bvh = false;
    bool reset_acc_image = false;
    clock_t start = 0, last = 0;
    //float3* image, * dev_ptr;
    Scene scene;
    cudaGraphicsResource* resource = NULL;

#ifndef NDEBUG
    DxgiInfoManager infoManager;
#endif
};