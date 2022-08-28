#include "application.hh"

Application::Application():m_running(true){
  OnInit();
}

Application::~Application(){
  OnCleanup();
  SDL_Quit();
}

int Application::OnExecute(){
  SDL_Event event;

  while (m_running) {
    App::ThrowException("测试");
    while (static_cast<bool>(SDL_PollEvent(&event))) {
      OnEvent(&event);
    }
    OnLoop();
    OnRender();
  }
  return 0;
}


void Application::OnInit(){

}

void Application::OnEvent(SDL_Event *event){

}

void Application::OnLoop(){

}
void Application::OnRender(){

}

void Application::OnCleanup(){

}
