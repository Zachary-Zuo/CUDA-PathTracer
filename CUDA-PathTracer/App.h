#pragma once

#include "imgui/imgui_impl_win32.h"
#include "Window.h"
#include "WindowTimer.h"
#include "ImguiManager.h"
#include "Graphics.h"
#include "tracer/scene.h"
#include "tracer/parsescene.h"

class App
{
public:
	App();
	// master frame / message loop
	int Go();
	~App();
private:
	void DoFrame();
private:
	ImguiManager imgui;
	Window wnd;
	WindowTimer timer;
	float speed_factor = 1.0f;

	static constexpr size_t nDrawables = 180;

	GlobalConfig config;
	unsigned iteration = 0;
	bool vision_bvh = false;
	bool reset_acc_image = false;
	clock_t start = 0, last = 0;
	float3* image, * dev_ptr;
	Scene scene;
	cudaGraphicsResource* resource = NULL;
};