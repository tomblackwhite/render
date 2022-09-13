#pragma once

#include <SDL.h>
#include <SDL_filesystem.h>
#include <SDL_vulkan.h>
#include <tool.hh>
#include <vulkanrender.hh>
#include <cstdint>

class Application {
public:
  Application();
  int onExecute();

  void onInit();

  void onEvent(SDL_Event *event);

  void onLoop();

  void onRender();

  void onCleanup();

  ~Application();

  static std::string GetBasePath() noexcept {
    const auto *path = SDL_GetBasePath();
    return std::string(path);
  }

private:
  bool m_running;

  SDL_Window *m_window = nullptr;
  uint32_t m_windowWitdth = 960;
  uint32_t m_windowHeight = 540;

  VulkanRender m_vulkanRender;
};
