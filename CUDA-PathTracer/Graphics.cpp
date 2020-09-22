#pragma warning(disable: 4312)
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
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

namespace wrl = Microsoft::WRL;
namespace dx = DirectX;

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"D3DCompiler.lib")

Graphics::Graphics(HWND hWnd, int width, int height)
{

	if (!findCUDADevice())                   // Search for CUDA GPU
	{
		printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
		exit(EXIT_SUCCESS);
	}

	if (!dynlinkLoadD3D11API())                  // Search for D3D API (locate drivers, does not mean device is found)
	{
		printf("> D3D11 API libraries NOT found on.. Exiting.\n");
		dynlinkUnloadD3D11API();
		exit(EXIT_SUCCESS);
	}

	if (!findDXDevice(device_name))           // Search for D3D Hardware Device
	{
		printf("> D3D11 Graphics Device NOT found.. Exiting.\n");
		dynlinkUnloadD3D11API();
		exit(EXIT_SUCCESS);
	}
	
	windowWidth = width;
	windowHeight = height;
	DXGI_SWAP_CHAIN_DESC sd = {};
	sd.BufferCount = 1;
	sd.BufferDesc.Width = windowWidth;
	sd.BufferDesc.Height = windowHeight;
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
		D3D_FEATURE_LEVEL_11_0
	};
	D3D_FEATURE_LEVEL flRes;
#ifndef NDEBUG
	swapCreateFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	// for checking results of d3d functions
	HRESULT hr;
	cudaError cuStatus;

	// create device and front/back buffers, and swap chain and rendering context
	GFX_THROW_INFO(D3D11CreateDeviceAndSwapChain(
		g_pCudaCapableAdapter,
		D3D_DRIVER_TYPE_UNKNOWN,//D3D_DRIVER_TYPE_HARDWARE
		nullptr,
		swapCreateFlags,
		tour_fl,// D3D_FEATURE_LEVEL* pFeatureLevels
		2,
		D3D11_SDK_VERSION,
		&sd,
		&pSwap,
		&pDevice,
		&flRes,//D3D_FEATURE_LEVEL* pFeatureLevel
		&pContext
	));

	g_pCudaCapableAdapter->Release();

	// Get the immediate DeviceContext
	pDevice->GetImmediateContext(&pContext);

	// Create a render target view of the swapchain
	ID3D11Texture2D* pBuffer;
	GFX_THROW_INFO(pSwap->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBuffer));

	GFX_THROW_INFO(pDevice->CreateRenderTargetView(pBuffer, nullptr, pTarget.GetAddressOf()));

	pBuffer->Release();

	pContext->OMSetRenderTargets(1, pTarget.GetAddressOf(), NULL);

	// Setup the viewport
	D3D11_VIEWPORT vp;
	vp.Width = static_cast <float>(windowWidth);
	vp.Height = static_cast <float>(windowHeight);
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	pContext->RSSetViewports(1u, &vp);

	static const char g_simpleShaders[] =
"struct PSInput\n" \
"{ \n" \
"    float4 position : SV_POSITION;\n" \
"    float4 color : COLOR; \n" \
"};\n" \
"PSInput VSMain(float3 position : POSITION, float4 color : COLOR)\n" \
"{ \n" \
"    PSInput result;\n" \
"    result.position = float4(position, 1.0f); \n" \
"    // Pass the color through without modification. \n" \
"    result.color = color; \n" \
"    return result; \n" \
"} \n" \
"float4 PSMain(PSInput input) : SV_TARGET \n" \
"{ \n" \
"    return input.color; \n" \
"} \n" \
;

	ID3DBlob* VS;
	ID3DBlob* PS;
	ID3DBlob* pErrorMsgs;
	// Vertex shader
	{
		GFX_THROW_INFO(D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL, NULL, "VSMain", "vs_4_0", 0/*Flags1*/, 0/*Flags2*/, &VS, &pErrorMsgs));

		if (FAILED(hr))
		{
			const char* pStr = (const char*)pErrorMsgs->GetBufferPointer();
			printf(pStr);
		}

		GFX_THROW_INFO(pDevice->CreateVertexShader(VS->GetBufferPointer(), VS->GetBufferSize(), NULL, &g_pVertexShader));

		// Let's bind it now : no other vtx shader will replace it...
		pContext->VSSetShader(g_pVertexShader, NULL, 0);
	}
	// Pixel shader
	{
		GFX_THROW_INFO(D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL, NULL, "PSMain", "ps_4_0", 0/*Flags1*/, 0/*Flags2*/, &PS, &pErrorMsgs));

		GFX_THROW_INFO(pDevice->CreatePixelShader(PS->GetBufferPointer(), PS->GetBufferSize(), NULL, &g_pPixelShader));
		// Let's bind it now : no other pix shader will replace it...
		pContext->PSSetShader(g_pPixelShader, NULL, 0);
	}

	D3D11_BUFFER_DESC bufferDesc;
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.ByteWidth = sizeof(Vertex) * width * height;
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = 0;
	bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

	GFX_THROW_INFO(pDevice->CreateBuffer(&bufferDesc, NULL, &g_VertexBuffer));

	GFX_THROW_INFO(g_VertexBuffer->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&g_pKeyedMutex11));

	D3D11_INPUT_ELEMENT_DESC inputElementDescs[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	GFX_THROW_INFO(pDevice->CreateInputLayout(inputElementDescs, 2, VS->GetBufferPointer(), VS->GetBufferSize(), &g_pLayout));

	// Setup  Input Layout
	pContext->IASetInputLayout(g_pLayout);

	pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);


	IDXGIResource1* pResource;
	HANDLE sharedHandle;
	g_VertexBuffer->QueryInterface(__uuidof(IDXGIResource1), (void**)&pResource);
	hr = pResource->GetSharedHandle(&sharedHandle);
	if (!SUCCEEDED(hr))
	{
		std::cout << "Failed GetSharedHandle hr= " << hr << std::endl;
	}
	// Import the D3D11 Vertex Buffer into CUDA
	d_VertexBufPtr = cudaImportVertexBuffer(sharedHandle, extMemory, width, height);
	pResource->Release();

	g_pKeyedMutex11->QueryInterface(__uuidof(IDXGIResource1), (void**)&pResource);
	pResource->GetSharedHandle(&sharedHandle);
	// Import the D3D11 Keyed Mutex into CUDA
	cudaImportKeyedMutex(sharedHandle, extSemaphore);
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
	pDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
	pContext->RSSetState(g_pRasterState);

	// init imgui d3d impl
	ImGui_ImplDX11_Init(pDevice.Get(), pContext.Get());
}

void Graphics::Render()
{
	static uint64_t key = 0;

	// Launch cuda kernel to generate sinewave in vertex buffer
	RunSineWaveKernel(extSemaphore, key, INFINITE, windowWidth, windowHeight, d_VertexBufPtr, cuda_stream);

	// Draw the scene using them
	DrawScene(key);
}

bool Graphics::DrawScene(uint64_t& key)
{

	HRESULT hr = S_OK;
	// Clear the backbuffer
	float ClearColor[4] = { 0.5f, 0.5f, 0.6f, 1.0f };

	pContext->ClearRenderTargetView(pTarget.Get(), ClearColor);

	GFX_THROW_INFO(g_pKeyedMutex11->AcquireSync(key++, INFINITE));

	UINT stride = sizeof(Vertex);
	UINT offset = 0;
	pContext->IASetVertexBuffers(0, 1, &g_VertexBuffer, &stride, &offset);
	pContext->Draw(windowHeight * windowWidth, 0);
	GFX_THROW_INFO(g_pKeyedMutex11->ReleaseSync(key));

	// Present the backbuffer contents to the display
	pSwap->Present(1, 0);

	return true;
}
bool Graphics::findCUDADevice()
{
	int deviceCount = 0;
	// This function call returns 0 if there are no CUDA capable devices.
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
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

#if 1
	// This may fail if Direct3D 11 isn't installed
	s_hModD3D11 = LoadLibrary("d3d11.dll");

	if (s_hModD3D11 != NULL)
	{
		sFnPtr_D3D11CreateDevice = (LPD3D11CREATEDEVICE)GetProcAddress(s_hModD3D11, "D3D11CreateDevice");
		sFnPtr_D3D11CreateDeviceAndSwapChain = (LPD3D11CREATEDEVICEANDSWAPCHAIN)GetProcAddress(s_hModD3D11, "D3D11CreateDeviceAndSwapChain");
	}
	else
	{
		printf("\nLoad d3d11.dll failed\n");
		fflush(0);
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
#else
	sFnPtr_D3D11CreateDevice = (LPD3D11CREATEDEVICE)D3D11CreateDeviceAndSwapChain;
	sFnPtr_D3D11CreateDeviceAndSwapChain = (LPD3D11CREATEDEVICEANDSWAPCHAIN)D3D11CreateDeviceAndSwapChain;
	//sFnPtr_D3DX11CreateEffectFromMemory  = ( LPD3DX11CREATEEFFECTFROMMEMORY )D3DX11CreateEffectFromMemory;
	sFnPtr_D3DX11CompileFromMemory = (LPD3DX11COMPILEFROMMEMORY)D3DX11CompileFromMemory;
	sFnPtr_CreateDXGIFactory = (LPCREATEDXGIFACTORY)CreateDXGIFactory;
	return true;
#endif
	return true;
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
		printf("> No DXGI Factory created.\n");
		return false;
	}

	UINT adapter = 0;

	for (; !g_pCudaCapableAdapter; ++adapter)
	{
		// Get a candidate DXGI adapter
		IDXGIAdapter1* pAdapter = NULL;

		hr = pFactory->EnumAdapters1(adapter, &pAdapter);

		if (FAILED(hr))
		{
			break;    // no compatible adapters found
		}

		// Query to see if there exists a corresponding compute device
		int cuDevice;
		cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
		printLastCudaError("cudaD3D11GetDevice failed"); //This prints and resets the cudaError to cudaSuccess

		if (cudaSuccess == cuStatus)
		{
			// If so, mark it as the one against which to create our d3d11 device
			g_pCudaCapableAdapter = pAdapter;
			g_pCudaCapableAdapter->AddRef();
			cuda_dev = cuDevice;
			printf("\ncuda device id selected = %d\n", cuda_dev);
		}

		pAdapter->Release();
	}

	printf("> Found %d D3D11 Adapater(s).\n", (int)adapter);

	pFactory->Release();

	if (!g_pCudaCapableAdapter)
	{
		printf("> Found 0 D3D11 Adapater(s) /w Compute capability.\n");
		return false;
	}

	DXGI_ADAPTER_DESC adapterDesc;
	g_pCudaCapableAdapter->GetDesc(&adapterDesc);
	wcstombs(dev_name, adapterDesc.Description, 128);

	checkCudaErrors(cudaSetDevice(cuda_dev));
	checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));

	printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
	printf("> %s\n", dev_name);

	return true;
}

void Graphics::Cleanup()
{
	checkCudaErrors(cudaFree(d_VertexBufPtr));
	checkCudaErrors(cudaDestroyExternalMemory(extMemory));
	checkCudaErrors(cudaDestroyExternalSemaphore(extSemaphore));
}


Graphics::~Graphics()
{
	ImGui_ImplDX11_Shutdown();
}

void Graphics::EndFrame()
{
	// imgui frame end
	if (imguiEnabled)
	{
		ImGui::Render();
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
	}

	HRESULT hr;
#ifndef NDEBUG
	infoManager.Set();
#endif
	if (FAILED(hr = pSwap->Present(1u, 0u)))
	{
		if (hr == DXGI_ERROR_DEVICE_REMOVED)
		{
			throw GFX_DEVICE_REMOVED_EXCEPT(pDevice->GetDeviceRemovedReason());
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
	if (imguiEnabled)
	{
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
	}

	const float color[] = { red,green,blue,1.0f };
	pContext->ClearRenderTargetView(pTarget.Get(), color);
	pContext->ClearDepthStencilView(pDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0u);
}

void Graphics::DrawIndexed(UINT count) noexcept(!IS_DEBUG)
{
	GFX_THROW_INFO_ONLY(pContext->DrawIndexed(count, 0u, 0u));
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
	imguiEnabled = true;
}

void Graphics::DisableImgui() noexcept
{
	imguiEnabled = false;
}

bool Graphics::IsImguiEnabled() const noexcept
{
	return imguiEnabled;
}


// Graphics exception stuff
Graphics::HrException::HrException(int line, const char* file, HRESULT hr, std::vector<std::string> infoMsgs) noexcept
	:
	Exception(line, file),
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
		oss << "\n[Error Info]\n" << GetErrorInfo() << std::endl << std::endl;
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
	:
	Exception(line, file)
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
		<< "\n[Error Info]\n" << GetErrorInfo() << std::endl << std::endl;
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
