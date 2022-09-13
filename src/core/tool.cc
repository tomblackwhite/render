#include "tool.hh"

namespace App {

void ThrowException(string const &message, bool hasSDL) {
  auto backtraceStr =
      boost::stacktrace::to_string(boost::stacktrace::stacktrace());

  string whatStr;
  if (hasSDL) {
    auto const *error = SDL_GetError();
    whatStr = fmt::format("{}\n{}\n{}", message, error, backtraceStr);
  } else {
    whatStr = fmt::format("{}\n{}", message, backtraceStr);
  }

  throw std::runtime_error(whatStr);
}

void CheckAudioDriver(){
  int num = SDL_GetNumAudioDrivers();
  for(int i = 0; i < num; ++i){
    const char *name = SDL_GetAudioDriver(i);
    if(SDL_AudioInit(name)!=0){
      const auto *error = SDL_GetError();
      std::cerr << error << "\n";
    }
  }
}


} // namespace App
