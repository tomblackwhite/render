#include <application.hh>
#include <iostream>





int main() {

  try {
    Application app;
    return app.onExecute();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    std::terminate();
  }
}
