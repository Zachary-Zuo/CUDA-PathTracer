#pragma warning(disable : 4312)
#include "Graphics.h"
#include "Dxerr/dxerr.h"
#include <sstream>
#include <cmath>
#include <DirectXMath.h>
#include "GraphicsThrowMacros.h"
#include "../imgui/imgui_impl_dx11.h"
#include "../imgui/imgui_impl_win32.h"

// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <dxgi1_2.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>

// includes, project
#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h
#include "../tracer/pathtracer.h"

namespace wrl = Microsoft::WRL;
namespace dx = DirectX;

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3DCompiler.lib")

Graphics::Graphics(HWND hWnd, int width, int height)
{
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 0;
    sd.BufferDesc.RefreshRate.Denominator = 0;
    sd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 1;
    sd.OutputWindow = hWnd;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = 0;

    UINT swapCreateFlags = 0u;
#ifndef NDEBUG
    swapCreateFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // for checking results of d3d functions
    HRESULT hr;

    // create device and front/back buffers, and swap chain and rendering context
    GFX_THROW_INFO(D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        swapCreateFlags,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &sd,
        &m_pSwap,
        &m_pDevice,
        nullptr,
        &m_pContext
    ));

    // gain access to texture subresource in swap chain (back buffer)
    wrl::ComPtr<ID3D11Resource> pBackBuffer;
    GFX_THROW_INFO(m_pSwap->GetBuffer(0, __uuidof(ID3D11Resource), &pBackBuffer));
    GFX_THROW_INFO(m_pDevice->CreateRenderTargetView(pBackBuffer.Get(), nullptr, &m_pTarget));

    // create depth stensil state
    D3D11_DEPTH_STENCIL_DESC dsDesc = {};
    dsDesc.DepthEnable = TRUE;
    dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsDesc.DepthFunc = D3D11_COMPARISON_LESS;
    wrl::ComPtr<ID3D11DepthStencilState> pDSState;
    GFX_THROW_INFO(m_pDevice->CreateDepthStencilState(&dsDesc, &pDSState));

    // bind depth state
    m_pContext->OMSetDepthStencilState(pDSState.Get(), 1u);

    // create depth stensil texture
    wrl::ComPtr<ID3D11Texture2D> pDepthStencil;
    D3D11_TEXTURE2D_DESC descDepth = {};
    descDepth.Width = static_cast<UINT>(width);
    descDepth.Height = static_cast<UINT>(height);
    descDepth.MipLevels = 1u;
    descDepth.ArraySize = 1u;
    descDepth.Format = DXGI_FORMAT_D32_FLOAT;
    descDepth.SampleDesc.Count = 1u;
    descDepth.SampleDesc.Quality = 0u;
    descDepth.Usage = D3D11_USAGE_DEFAULT;
    descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    GFX_THROW_INFO(m_pDevice->CreateTexture2D(&descDepth, nullptr, &pDepthStencil));

    // create view of depth stensil texture
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = {};
    descDSV.Format = DXGI_FORMAT_D32_FLOAT;
    descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    descDSV.Texture2D.MipSlice = 0u;
    GFX_THROW_INFO(m_pDevice->CreateDepthStencilView(
        pDepthStencil.Get(), &descDSV, &m_pDSV
    ));

    // bind depth stensil view to OM
    m_pContext->OMSetRenderTargets(1u, m_pTarget.GetAddressOf(), m_pDSV.Get());

    // configure viewport
    D3D11_VIEWPORT vp;
    vp.Width = static_cast<float>(width);
    vp.Height = static_cast<float>(height);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    m_pContext->RSSetViewports(1u, &vp);

    // init imgui d3d impl
    ImGui_ImplDX11_Init(m_pDevice.Get(), m_pContext.Get());
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

Graphics::~Graphics()
{
    ImGui_ImplDX11_Shutdown();
}

void Graphics::EndFrame()
{
    // imgui frame end
    if (m_bImguiEnabled)
    {
        ImGui::Render();
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    }

    HRESULT hr;
#ifndef NDEBUG
    infoManager.Set();
#endif
    if (FAILED(hr = m_pSwap->Present(1u, 0u)))
    {
        if (hr == DXGI_ERROR_DEVICE_REMOVED)
        {
            throw GFX_DEVICE_REMOVED_EXCEPT(m_pDevice->GetDeviceRemovedReason());
        }
        else
        {
            throw GFX_EXCEPT(hr);
        }
    }
    tex.Reset();
    pimgTex.Reset();
}

void Graphics::BeginFrame(float red, float green, float blue) noexcept
{
    // imgui begin frame
    if (m_bImguiEnabled)
    {
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
    }

    const float color[] = { red, green, blue, 1.0f };
    m_pContext->ClearRenderTargetView(m_pTarget.Get(), color);
    m_pContext->ClearDepthStencilView(m_pDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0u);
}

ID3D11ShaderResourceView* Graphics::GetTex(float3* image, int width, int height)
{
    // 创建纹理
    D3D11_TEXTURE2D_DESC texDesc;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
    texDesc.SampleDesc.Count = 1;		// 不使用多重采样
    texDesc.SampleDesc.Quality = 0;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;	// 指定需要生成mipmap

    D3D11_SUBRESOURCE_DATA sd;
    float3* pData = image;
    sd.pSysMem = pData;
    sd.SysMemPitch = width * sizeof(float3);
    sd.SysMemSlicePitch = width * height * sizeof(float3);

    m_pDevice->CreateTexture2D(&texDesc, &sd, tex.GetAddressOf());

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    srvDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.MostDetailedMip = 0;
    m_pDevice->CreateShaderResourceView(tex.Get(), &srvDesc, pimgTex.GetAddressOf());

    return pimgTex.Get();
}

void Graphics::ClearBuffer(float red, float green, float blue) noexcept
{
    const float color[] = { red,green,blue,1.0f };
    m_pContext->ClearRenderTargetView(m_pTarget.Get(), color);
    m_pContext->ClearDepthStencilView(m_pDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0u);
}

void Graphics::DrawIndexed(UINT count) noexcept(!IS_DEBUG)
{
    GFX_THROW_INFO_ONLY(m_pContext->DrawIndexed(count, 0u, 0u));
}

void Graphics::SetProjection(DirectX::FXMMATRIX proj) noexcept
{
    projection = proj;
}

DirectX::XMMATRIX Graphics::GetProjection() const noexcept
{
    return projection;
}

void Graphics::SetCamera(DirectX::FXMMATRIX cam) noexcept
{
    camera = cam;
}

DirectX::XMMATRIX Graphics::GetCamera() const noexcept
{
    return camera;
}

void Graphics::EnableImgui() noexcept
{
    m_bImguiEnabled = true;
}

void Graphics::DisableImgui() noexcept
{
    m_bImguiEnabled = false;
}

bool Graphics::IsImguiEnabled() const noexcept
{
    return m_bImguiEnabled;
}

// Graphics exception stuff
Graphics::HrException::HrException(int line, const char* file, HRESULT hr, std::vector<std::string> infoMsgs) noexcept
    : Exception(line, file),
    hr(hr)
{
    // join all info messages with newlines into single string
    for (const auto& m : infoMsgs)
    {
        info += m;
        info.push_back('\n');
    }
    // remove final newline if exists
    if (!info.empty())
    {
        info.pop_back();
    }
}

const char* Graphics::HrException::what() const noexcept
{
    std::ostringstream oss;
    oss << GetType() << std::endl
        << "[Error Code] 0x" << std::hex << std::uppercase << GetErrorCode()
        << std::dec << " (" << (unsigned long)GetErrorCode() << ")" << std::endl
        << "[Error String] " << GetErrorString() << std::endl
        << "[Description] " << GetErrorDescription() << std::endl;
    if (!info.empty())
    {
        oss << "\n[Error Info]\n"
            << GetErrorInfo() << std::endl
            << std::endl;
    }
    oss << GetOriginString();
    whatBuffer = oss.str();
    return whatBuffer.c_str();
}

const char* Graphics::HrException::GetType() const noexcept
{
    return "Steve Graphics Exception";
}

HRESULT Graphics::HrException::GetErrorCode() const noexcept
{
    return hr;
}

std::string Graphics::HrException::GetErrorString() const noexcept
{
    return DXGetErrorString(hr);
}

std::string Graphics::HrException::GetErrorDescription() const noexcept
{
    char buf[512];
    DXGetErrorDescription(hr, buf, sizeof(buf));
    return buf;
}

std::string Graphics::HrException::GetErrorInfo() const noexcept
{
    return info;
}

const char* Graphics::DeviceRemovedException::GetType() const noexcept
{
    return "Steve Graphics Exception [Device Removed] (DXGI_ERROR_DEVICE_REMOVED)";
}
Graphics::InfoException::InfoException(int line, const char* file, std::vector<std::string> infoMsgs) noexcept
    : Exception(line, file)
{
    // join all info messages with newlines into single string
    for (const auto& m : infoMsgs)
    {
        info += m;
        info.push_back('\n');
    }
    // remove final newline if exists
    if (!info.empty())
    {
        info.pop_back();
    }
}

const char* Graphics::InfoException::what() const noexcept
{
    std::ostringstream oss;
    oss << GetType() << std::endl
        << "\n[Error Info]\n"
        << GetErrorInfo() << std::endl
        << std::endl;
    oss << GetOriginString();
    whatBuffer = oss.str();
    return whatBuffer.c_str();
}

const char* Graphics::InfoException::GetType() const noexcept
{
    return "Steve Graphics Info Exception";
}

std::string Graphics::InfoException::GetErrorInfo() const noexcept
{
    return info;
}
