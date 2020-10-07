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
    void InitCudaScene();
    ID3D11ShaderResourceView* getTex(float3* image);
    float3* renderImage();

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
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> m_pDSV;

    GlobalConfig config;
    unsigned iteration = 0;
    bool vision_bvh = false;
    bool reset_acc_image = false;
    clock_t start = 0, last = 0;
    //float3* image, * dev_ptr;
    Scene scene;
    cudaGraphicsResource* resource = NULL;

    float3* image, * dev_ptr;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> pimgTex;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;

#ifndef NDEBUG
    DxgiInfoManager infoManager;
#endif
};