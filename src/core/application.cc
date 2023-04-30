#include "application.hh"

Application::Application()
    : m_running(true), m_window(createWindow(m_windowSize)),
      m_vulkanRender(createRender(
          Application::GetBasePath() + "/..", m_windowSize,
          m_window, m_appName, m_appVersion, m_engineName, m_engineVersion)),
      m_manager(m_vulkanRender.getVulkanMemory()) {
  onInit();
}

Application::~Application() { onCleanup(); }

int Application::onExecute() {
  SDL_Event event;

  while (m_running) {
    while (static_cast<bool>(SDL_PollEvent(&event))) {
      onEvent(&event);
    }
    onLoop();
    onRender();
  }
  return 0;
}

void Application::onInit() {

  // App::CheckAudioDriver();

  m_manager.setMainScene("neptune.gltf");
  m_manager.init();
}
void Application::onEvent(SDL_Event *event) {
  if (event->type == SDL_QUIT) {
    m_running = false;
  }
}

void Application::onLoop() { m_manager.update(); }
void Application::onRender() { m_vulkanRender.drawFrame(m_manager.m_scene); }

void Application::onCleanup() {
  m_vulkanRender.waitDrawClean();
  m_vulkanRender.cleanup();
  if (m_window != nullptr) {
    SDL_DestroyWindow(m_window);
  }
  SDL_Quit();
}

VulkanRender Application::createRender(const string &path, App::Extent2D size,
                                       SDL_Window *window,
                                       const std::string &appName,
                                       uint32_t appVersion,
                                       const std::string &engineName,
                                       uint32_t engineVersion) {

  VulkanRender vulkanRender(path, size, appName, appVersion,
                            engineName, engineVersion);

  // Get SDL_window require extensions
  unsigned int count = 0;
  if (SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr) != SDL_TRUE) {
    App::ThrowException("GetVulkanExtensins error", true);
  }
  std::vector<const char *> extensions(count);
  if (SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data()) !=
      SDL_TRUE) {
    App::ThrowException("GetVulkanExtensins error", true);
  }

  std::vector<std::string> strExtensions(count);
  std::transform(extensions.begin(), extensions.end(), strExtensions.begin(),
                 [](const char *str) { return std::string(str); });

  // 初始化vulkanInstance
  vulkanRender.initVulkanInstance(std::move(strExtensions));
  VkSurfaceKHR surface = nullptr;
  if (SDL_Vulkan_CreateSurface(window, vulkanRender.getVulkanInstance(),
                               &surface) != SDL_TRUE) {
    App::ThrowException("createSurface error", true);
  }
  vulkanRender.initOthers(surface);

  return vulkanRender;
}

SDL_Window *Application::createWindow(App::Extent2D size) {

  if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
    App::ThrowException("init sdl failed", true);
  }

  auto *window =
      SDL_CreateWindow("Game", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                       static_cast<int>(size.width),
                       static_cast<int>(size.height), SDL_WINDOW_VULKAN);
  if (window == nullptr) {
    App::ThrowException("create window failed", true);
  }
  return window;
}
