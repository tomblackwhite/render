#pragma once

#include <SDL.h>
#include <tool.hh>


class Application {
public:
  Application();
  int OnExecute();

  void OnInit();

  void OnEvent(SDL_Event* event);

  void OnLoop();

  void OnRender();

  void OnCleanup();

  ~Application();

private:
  bool m_running;
};
