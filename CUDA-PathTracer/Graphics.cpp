#pragma warning(disable : 4312)
#include "Graphics.h"
#include "dxerr.h"
#include <sstream>
#include <cmath>
#include <DirectXMath.h>
#include "GraphicsThrowMacros.h"
#include "imgui/imgui_impl_dx11.h"
#include "imgui/imgui_impl_win32.h"
#include "ShaderStructs.h"
#include "sinewave_cuda.h"

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
#include "tracer\pathtracer.h"

namespace wrl = Microsoft::WRL;
namespace dx = DirectX;

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3DCompiler.lib")

Graphics::Graphics(HWND hWnd, int width, int height)
{

    if (!findCUDADevice()) // Search for CUDA GPU
    {
        printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", m_szdevice_name);
        exit(EXIT_SUCCESS);
    }

    if (!dynlinkLoadD3D11API()) // Search for D3D API (locate drivers, does not mean device is found)
    {
        printf("> D3D11 API libraries NOT found on.. Exiting.\n");
        dynlinkUnloadD3D11API();
        exit(EXIT_SUCCESS);
    }

    if (!findDXDevice(m_szdevice_name)) // Search for D3D Hardware Device
    {
        printf("> D3D11 Graphics Device NOT found.. Exiting.\n");
        dynlinkUnloadD3D11API();
        exit(EXIT_SUCCESS);
    }

    m_nWindowWidth = width;
    m_nWindowHeight = height;
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 1;
    sd.BufferDesc.Width = m_nWindowWidth;
    sd.BufferDesc.Height = m_nWindowHeight;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

    UINT swapCreateFlags = 0u;
    D3D_FEATURE_LEVEL tour_fl[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL flRes;
#ifndef NDEBUG
    swapCreateFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // for checking results of d3d functions
    HRESULT hr;

    // create device and front/back buffers, and swap chain and rendering context
    GFX_THROW_INFO(D3D11CreateDeviceAndSwapChain(
        m_pCudaCapableAdapter.Get(),
        D3D_DRIVER_TYPE_UNKNOWN, //D3D_DRIVER_TYPE_HARDWARE
        nullptr,
        swapCreateFlags,
        tour_fl, // D3D_FEATURE_LEVEL* pFeatureLevels
        2,
        D3D11_SDK_VERSION,
        &sd,
        &m_pSwap,
        &m_pDevice,
        &flRes, //D3D_FEATURE_LEVEL* pFeatureLevel
        &m_pContext));

    // Get the immediate DeviceContext
    m_pDevice->GetImmediateContext(&m_pContext);

    // Create a render target view of the swapchain
    ID3D11Texture2D* pBuffer;
    GFX_THROW_INFO(m_pSwap->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBuffer));

    GFX_THROW_INFO(m_pDevice->CreateRenderTargetView(pBuffer, nullptr, m_pTarget.GetAddressOf()));

    pBuffer->Release();

    m_pContext->OMSetRenderTargets(1, m_pTarget.GetAddressOf(), NULL);

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = static_cast<float>(m_nWindowWidth);
    vp.Height = static_cast<float>(m_nWindowHeight);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    m_pContext->RSSetViewports(1u, &vp);

    wrl::ComPtr<ID3DBlob> VS;
    wrl::ComPtr<ID3DBlob> PS;
    wrl::ComPtr<ID3D11VertexShader> pVertexShader;
    wrl::ComPtr<ID3D11PixelShader> pPixelShader;
    // DXVertex shader
    D3DReadFileToBlob(L"VertexShader.cso", VS.GetAddressOf());
    GFX_THROW_INFO(m_pDevice->CreateVertexShader(VS->GetBufferPointer(), VS->GetBufferSize(), NULL, pVertexShader.GetAddressOf()));
    // Let's bind it now : no other vtx shader will replace it...
    m_pContext->VSSetShader(pVertexShader.Get(), NULL, 0);

    // Pixel shader
    D3DReadFileToBlob(L"PixelShader.cso", PS.GetAddressOf());
    GFX_THROW_INFO(m_pDevice->CreatePixelShader(PS->GetBufferPointer(), PS->GetBufferSize(), NULL, pPixelShader.GetAddressOf()));
    // Let's bind it now : no other pix shader will replace it...
    m_pContext->PSSetShader(pPixelShader.Get(), NULL, 0);

    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(DXVertex) * width * height;
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    GFX_THROW_INFO(m_pDevice->CreateBuffer(&bufferDesc, NULL, m_pVertexBuffer.GetAddressOf()));

    GFX_THROW_INFO(m_pVertexBuffer->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&m_pKeyedMutex11));

    D3D11_INPUT_ELEMENT_DESC inputElementDescs[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 12, D3D11_INPUT_PER_VERTEX_DATA, 0} };

    GFX_THROW_INFO(m_pDevice->CreateInputLayout(inputElementDescs, 2, VS->GetBufferPointer(), VS->GetBufferSize(), m_pLayout.GetAddressOf()));

    // Setup  Input Layout
    m_pContext->IASetInputLayout(m_pLayout.Get());
    m_pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

    wrl::ComPtr<IDXGIResource1> pResource;
    HANDLE sharedHandle;

    GFX_THROW_INFO(m_pVertexBuffer.As(&pResource));

    GFX_THROW_INFO(pResource->GetSharedHandle(&sharedHandle));

    // Import the D3D11 DXVertex Buffer into CUDA
    m_VertexBufPtr = cudaImportVertexBuffer(sharedHandle, m_extMemory, width, height);
    pResource->Release();

    GFX_THROW_INFO(m_pKeyedMutex11->QueryInterface(__uuidof(IDXGIResource1), (void**)pResource.GetAddressOf()));

    pResource->GetSharedHandle(&sharedHandle);
    // Import the D3D11 Keyed Mutex into CUDA
    cudaImportKeyedMutex(sharedHandle, m_extSemaphore);
    pResource->Release();

    D3D11_RASTERIZER_DESC rasterizerState;
    rasterizerState.FillMode = D3D11_FILL_SOLID;
    rasterizerState.CullMode = D3D11_CULL_FRONT;
    rasterizerState.FrontCounterClockwise = false;
    rasterizerState.DepthBias = false;
    rasterizerState.DepthBiasClamp = 0;
    rasterizerState.SlopeScaledDepthBias = 0;
    rasterizerState.DepthClipEnable = false;
    rasterizerState.ScissorEnable = false;
    rasterizerState.MultisampleEnable = false;
    rasterizerState.AntialiasedLineEnable = false;
    m_pDevice->CreateRasterizerState(&rasterizerState, m_pRasterState.GetAddressOf());
    m_pContext->RSSetState(m_pRasterState.Get());

    // init imgui d3d impl
    ImGui_ImplDX11_Init(m_pDevice.Get(), m_pContext.Get());
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

bool Graphics::findCUDADevice()
{
    int deviceCount = 0;
    // This function call returns 0 if there are no CUDA capable devices.
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        HRESULT hr = S_OK;
        std::vector<std::string> info;
        info.push_back("There are no device(s) supporting CUDA");
        throw Graphics::HrException(__LINE__, __FILE__, hr, info);
    }

    return true;
}

// Dynamically load the D3D11 DLLs loaded and map the function pointers
bool Graphics::dynlinkLoadD3D11API()
{
    // If both modules are non-NULL, this function has already been called.  Note
    // that this doesn't guarantee that all ProcAddresses were found.
    if (s_hModD3D11 != NULL && s_hModDXGI != NULL)
    {
        return true;
    }

    // This may fail if Direct3D 11 isn't installed
    s_hModD3D11 = LoadLibrary("d3d11.dll");

    if (s_hModD3D11 != NULL)
    {
        sFnPtr_D3D11CreateDevice = (LPD3D11CREATEDEVICE)GetProcAddress(s_hModD3D11, "D3D11CreateDevice");
        sFnPtr_D3D11CreateDeviceAndSwapChain = (LPD3D11CREATEDEVICEANDSWAPCHAIN)GetProcAddress(s_hModD3D11, "D3D11CreateDeviceAndSwapChain");
    }
    else
    {
        HRESULT hr = S_OK;
        std::vector<std::string> info;
        info.push_back("Load d3d11.dll failed");
        throw Graphics::HrException(__LINE__, __FILE__, hr, info);
    }

    if (!sFnPtr_CreateDXGIFactory)
    {
        s_hModDXGI = LoadLibrary("dxgi.dll");

        if (s_hModDXGI)
        {
            sFnPtr_CreateDXGIFactory = (LPCREATEDXGIFACTORY)GetProcAddress(s_hModDXGI, "CreateDXGIFactory1");
        }

        return (s_hModDXGI != NULL) && (s_hModD3D11 != NULL);
    }
    return (s_hModD3D11 != NULL);
}

bool Graphics::findDXDevice(char* dev_name)
{
    HRESULT hr = S_OK;
    cudaError cuStatus;
    int cuda_dev = -1;

    // Iterate through the candidate adapters
    IDXGIFactory1* pFactory;
    hr = sFnPtr_CreateDXGIFactory(__uuidof(IDXGIFactory1), (void**)(&pFactory));

    if (!SUCCEEDED(hr))
    {
        HRESULT hr = S_OK;
        std::vector<std::string> info;
        info.push_back("Load d3d11.dll failed");
        throw Graphics::HrException(__LINE__, __FILE__, hr, info);
    }

    UINT adapter = 0;

    for (; !m_pCudaCapableAdapter; ++adapter)
    {
        // Get a candidate DXGI adapter
        IDXGIAdapter1* pAdapter = NULL;

        hr = pFactory->EnumAdapters1(adapter, &pAdapter);

        if (FAILED(hr))
        {
            break; // no compatible adapters found
        }

        // Query to see if there exists a corresponding compute device
        int cuDevice;
        cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
        printLastCudaError("cudaD3D11GetDevice failed"); //This prints and resets the cudaError to cudaSuccess

        if (cudaSuccess == cuStatus)
        {
            // If so, mark it as the one against which to create our d3d11 device
            m_pCudaCapableAdapter = pAdapter;
            m_pCudaCapableAdapter->AddRef();
            cuda_dev = cuDevice;
            printf("\ncuda device id selected = %d\n", cuda_dev);
        }

        pAdapter->Release();
    }

    printf("> Found %d D3D11 Adapater(s).\n", (int)adapter);

    pFactory->Release();

    if (!m_pCudaCapableAdapter)
    {
        printf("> Found 0 D3D11 Adapater(s) /w Compute capability.\n");
        return false;
    }

    DXGI_ADAPTER_DESC adapterDesc;
    m_pCudaCapableAdapter->GetDesc(&adapterDesc);
    wcstombs(dev_name, adapterDesc.Description, 128);

    checkCudaErrors(cudaSetDevice(cuda_dev));
    checkCudaErrors(cudaStreamCreateWithFlags(&m_cuda_stream, cudaStreamNonBlocking));

    printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
    printf("> %s\n", dev_name);

    return true;
}

Graphics::~Graphics()
{
    ImGui_ImplDX11_Shutdown();
    checkCudaErrors(cudaFree(m_VertexBufPtr));
    checkCudaErrors(cudaDestroyExternalMemory(m_extMemory));
    checkCudaErrors(cudaDestroyExternalSemaphore(m_extSemaphore));
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


    static uint64_t key = 0;
    // Launch cuda kernel to generate sinewave in vertex buffer
    //RunSineWaveKernel(m_extSemaphore, key, INFINITE, m_nWindowWidth, m_nWindowHeight, m_VertexBufPtr, m_cuda_stream);
    iteration++;

    if (reset_acc_image) {
        iteration = 1;
    }
    Render(scene, 1920, 1080, scene.camera, iteration, reset_acc_image, m_VertexBufPtr,m_extSemaphore, key, INFINITE, m_cuda_stream);
    reset_acc_image = false;

    // Draw the scene using them
    HRESULT hr = S_OK;

    GFX_THROW_INFO(m_pKeyedMutex11->AcquireSync(key++, INFINITE));

    UINT stride = sizeof(DXVertex);
    UINT offset = 0;
    m_pContext->IASetVertexBuffers(0, 1, m_pVertexBuffer.GetAddressOf(), &stride, &offset);
    m_pContext->Draw(m_nWindowHeight * m_nWindowWidth, 0);
    GFX_THROW_INFO(m_pKeyedMutex11->ReleaseSync(key));
}

void Graphics::InitCudaScene()
{
    std::string f = "E:/Project/CUDA-PathTracer/x64/Debug/scene.json";
    InitScene(f, config, scene);
    BeginRender(scene, 1920, 1080, 0.00100000005);
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
