#include "App.h"
#include <memory>
#include <algorithm>
#include "../CasterLabMath.h"
#include "GDIPlusManager.h"
#include "../imgui/imgui.h"
#include <cstring>
#include "../tracer/parsescene.h"

namespace dx = DirectX;

GDIPlusManager gdipm;

App::App()
	:
	wnd(1920, 1080, "CUDA Path Tracer")
{
	class Factory
	{
	public:
		Factory(Graphics& gfx)
			:
			gfx(gfx)
		{}
	private:
		Graphics& gfx;
	};
}

void App::DoFrame()
{
	wnd.Gfx().BeginFrame(0.5f, 0.5f, 0.6f);
	// imgui window to control simulation speed
	//if (ImGui::Begin("Simulation Speed"))
	//{
	//	ImGui::SliderFloat("Speed Factor", &speed_factor, 0.0f, 4.0f);
	//	ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	//	ImGui::Text("Status: %s", wnd.kbd.KeyIsPressed(VK_SPACE) ? "PAUSED" : "RUNNING (hold spacebar to pause)");
	//}
	//ImGui::End();
	// imgui windows to control camera and light

	if (ImGui::Begin(u8"IMGUI", false, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImTextureID my_tex_id = wnd.Gfx().getTex(wnd.Gfx().renderImage());
		float my_tex_w = (float)1920;
		float my_tex_h = (float)1080;
		ImGui::Image(my_tex_id, ImVec2(my_tex_w, my_tex_h), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));

	}

	//bool imshow = true;
	//ImGui::ShowDemoWindow(&imshow);

	// present
	wnd.Gfx().EndFrame();
}

App::~App()
{}


int App::Go()
{
	wnd.Gfx().InitCudaScene();
	while (true)
	{
		// process all messages pending, but to not block for new messages
		if (const auto ecode = Window::ProcessMessages())
		{
			// if return optional has value, means we're quitting so return exit code
			return *ecode;
		}
		DoFrame();
	}
}