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
	if (enableCudaRenderer)
	{
		render = CudaRender::getInstance();
	}
	
}

void App::DoFrame()
{
	wnd.Gfx().BeginFrame(0.5f, 0.5f, 0.6f);

	if (enableCudaRenderer)
	{
		//Dock space
		{
			static bool opt_fullscreen = true;
			static bool opt_padding = false;
			static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

			ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
			ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(viewport->GetWorkPos());
			ImGui::SetNextWindowSize(viewport->GetWorkSize());
			ImGui::SetNextWindowViewport(viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
			window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

			if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
				window_flags |= ImGuiWindowFlags_NoBackground;

			if (!opt_padding)
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
			ImGui::Begin("DockSpace Demo", false, window_flags);
			if (!opt_padding)
				ImGui::PopStyleVar();

			if (opt_fullscreen)
				ImGui::PopStyleVar(2);

			// DockSpace
			ImGuiIO& io = ImGui::GetIO();
			ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
			ImGui::End();
		}

		//ImGui::SetNextWindowSizeConstraints(ImVec2(imageWidth+15, imageHeight+35), ImVec2(FLT_MAX, FLT_MAX));
		if (ImGui::Begin(u8"Render result", false))
		{
			int imageWidth = render->GetConfig().width;
			int imageHeight = render->GetConfig().height;
			ImTextureID my_tex_id = wnd.Gfx().GetTex(render->GetRenderReverseImage(), imageWidth, imageHeight);
			float my_tex_w = (float)imageWidth;
			float my_tex_h = (float)imageHeight;
			ImGui::Image(my_tex_id, ImVec2(my_tex_w, my_tex_h));
		}
		ImGui::End();

		// imgui window to show rendering information
		if (ImGui::Begin("Information"))
		{
			ImGui::Text("Framerate: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::Text("Iteration: %d", render->GetIteration());
			ImGui::Text("Rendered for %.0f s", timer.Peek());
			//ImGui::Text("Status: %s", wnd.kbd.KeyIsPressed(VK_SPACE) ? "PAUSED" : "RUNNING (hold spacebar to pause)");
			if (ImGui::Button("Save the result"))
				render->SaveImage();
		}
		ImGui::End();
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
	if (enableCudaRenderer)
	{
		render->InitCudaScene();
	}
	
	timer.Mark();
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