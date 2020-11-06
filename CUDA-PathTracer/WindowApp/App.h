#pragma once

#include "../imgui/imgui_impl_win32.h"
#include "Window.h"
#include "WindowTimer.h"
#include "ImguiManager.h"
#include "Graphics.h"
#include "../Editor/Camera.h"
#include "../tracer/CudaRender.h"
#include "../Editor/Drawable/Box.h"


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
	bool enableCudaRenderer=false;
	ImguiManager imgui;
	Window wnd;
	WindowTimer timer;
	CudaRender* render;
	EditorCamera cam;
	std::vector<std::unique_ptr<class Drawable>> drawables;
	static constexpr size_t nDrawables = 180;
	float speed_factor = 1.0f;

};