#pragma once
#include <QApplication>
#include <window.hh>

class Application {
public:
  Application(int argc, char *argv[]) : m_app(argc, argv){};

  void run(){

  }
private:
  QApplication m_app;
};
