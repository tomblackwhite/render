#include <application.hh>
#include <tool.hh>
#include <iostream>

int main() {

  try {
    Application app;
    return app.onExecute();
  } catch (const App::RunTimeErrorWithTrace &e) {
    std::cerr << e.what() << "\n" << e.stacktrace() << "\n";
    std::terminate();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    std::terminate();
  }
}
