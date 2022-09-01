#pragma once

#include <SDL.h>
#include <SDL_filesystem.h>
#include <SDL_vulkan.h>
#include <tool.hh>
#include <vulkanrender.hh>

class Application {
public:
  Application();
  int OnExecute();

  void OnInit();

  void OnEvent(SDL_Event *event);

  void OnLoop();

  void OnRender();

  void OnCleanup();

  ~Application();

  static std::string GetBasePath() noexcept {
    const auto *path = SDL_GetBasePath();
    return std::string(path);
  }

private:
  bool m_running;
  SDL_Window *m_window = nullptr;
  VulkanRender m_vulkanRender;
};
