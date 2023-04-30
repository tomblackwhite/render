#pragma once

#include "scene.hh"
#include <SDL.h>
#include <SDL_filesystem.h>
#include <SDL_vulkan.h>
#include <cstdint>
#include <tool.hh>
#include <vulkanrender.hh>

class Application {
public:
  Application();
  int onExecute();

  void onEvent(SDL_Event *event);

  void onLoop();

  void onRender();

  void onCleanup();

  ~Application();

  static std::string GetBasePath() noexcept {
    const auto *path = SDL_GetBasePath();
    return path;
  }

private:
  void onInit();

  bool m_running;

  inline static constexpr App::Extent2D defaultWindowSize = {960, 540};

  App::Extent2D m_windowSize = defaultWindowSize;

  std::string m_appName = "Game";
  uint32_t m_appVersion = VK_MAKE_VERSION(
      PROJECT_VERSION_MAJOR, PROJECT_VERSION_MINOR, PROJECT_VERSION_PATCH);

  std::string m_engineName = "Game";
  uint32_t m_engineVersion = VK_MAKE_VERSION(
      PROJECT_VERSION_MAJOR, PROJECT_VERSION_MINOR, PROJECT_VERSION_PATCH);

  //  uint32_t m_vulkanApiVersion = VK_VERSION_1_3;

  SDL_Window *m_window = nullptr;

  VulkanRender m_vulkanRender;

  App::SceneManager m_manager;

  // 构建SDL windows并解决依赖问题。
  static SDL_Window *createWindow(App::Extent2D size);

  static VulkanRender createRender(const string &path, App::Extent2D size,SDL_Window *window,
                                   const std::string &appName,
                                   uint32_t appVersion,
                                   const std::string &engineName,
                                   uint32_t engineVersion);
};
