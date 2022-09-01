#include "application.hh"

Application::Application() : m_running(true),m_vulkanRender(Application::GetBasePath()){ OnInit(); }

Application::~Application() { OnCleanup(); }

int Application::OnExecute() {
  SDL_Event event;

  while (m_running) {
    while (static_cast<bool>(SDL_PollEvent(&event))) {
      OnEvent(&event);
    }
    OnLoop();
    OnRender();
  }
  return 0;
}

void Application::OnInit() {

  if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
    App::ThrowException("init sdl failed", true);
  }

  m_window =
      SDL_CreateWindow("Game", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                       640, 360, SDL_WINDOW_VULKAN);
  if (m_window == nullptr) {
    App::ThrowException("create window failed", true);
  }

  // Get SDL_window require extensions
  unsigned int count;
  if (!SDL_Vulkan_GetInstanceExtensions(m_window, &count, nullptr)) {
    App::ThrowException("GetVulkanExtensins error", true);
  }
  std::vector<const char *> extensions(count);
  if (!SDL_Vulkan_GetInstanceExtensions(m_window, &count, extensions.data())) {
    App::ThrowException("GetVulkanExtensins error", true);
  }


}
void Application::OnEvent(SDL_Event *event) {
  if (event->type == SDL_QUIT) {
    m_running = false;
  }
}

void Application::OnLoop() {}
void Application::OnRender() {}

void Application::OnCleanup() {
  if (m_window != nullptr) {
    SDL_DestroyWindow(m_window);
  }
  SDL_Quit();
}
