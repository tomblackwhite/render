#include "application.hh"

Application::Application()
    : m_running(true), m_vulkanRender(Application::GetBasePath()+"/..",
                                      m_windowWitdth, m_windowHeight) {
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

  if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
    App::ThrowException("init sdl failed", true);
  }

  m_window =
      SDL_CreateWindow("Game", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                       static_cast<int>(m_windowWitdth),
                       static_cast<int>(m_windowHeight), SDL_WINDOW_VULKAN);
  if (m_window == nullptr) {
    App::ThrowException("create window failed", true);
  }

  // Get SDL_window require extensions
  unsigned int count = 0;
  if (SDL_Vulkan_GetInstanceExtensions(m_window, &count, nullptr) != SDL_TRUE) {
    App::ThrowException("GetVulkanExtensins error", true);
  }
  std::vector<const char *> extensions(count);
  if (SDL_Vulkan_GetInstanceExtensions(m_window, &count, extensions.data()) !=
      SDL_TRUE) {
    App::ThrowException("GetVulkanExtensins error", true);
  }

  std::vector<std::string> strExtensions(count);
  std::transform(extensions.begin(), extensions.end(), strExtensions.begin(),
                 [](const char *str) { return std::string(str); });

  //初始化vulkanInstance
  m_vulkanRender.initVulkanInstance(std::move(strExtensions));
  VkSurfaceKHR surface = nullptr;
  if (SDL_Vulkan_CreateSurface(m_window, m_vulkanRender.getVulkanInstance(),
                               &surface) != SDL_TRUE) {
    App::ThrowException("createSurface error", true);
  }
  m_vulkanRender.initOthers(surface);

  m_manager.setMainScene("neptune.gltf");
  m_manager.init();
}
void Application::onEvent(SDL_Event *event) {
  if (event->type == SDL_QUIT) {
    m_running = false;
  }
}

void Application::onLoop() {}
void Application::onRender() {
  m_vulkanRender.drawFrame();
}

void Application::onCleanup() {
  m_vulkanRender.waitDrawClean();
  m_vulkanRender.cleanup();
  if (m_window != nullptr) {
    SDL_DestroyWindow(m_window);
  }
  SDL_Quit();
}
