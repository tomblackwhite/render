#include "tool.hh"

namespace App {
void ThrowException(string const &message, bool hasSDL) {

  string whatStr;
  if (hasSDL) {
    auto const *error = SDL_GetError();
    whatStr = fmt::format("{}\n{}\n", message, error);
  } else {
    whatStr = fmt::format("{}\n", message);
  }

  throw RunTimeErrorWithTrace(whatStr);
}

void VulkanCheck(VkResult result, std::string_view message) {

  vk::Result re{std::to_underlying(result)};
  string str = fmt::format("message{}\n result{}\n", message, result);
  vk::resultCheck(re, str.c_str());
}

void CheckAudioDriver() {
  int num = SDL_GetNumAudioDrivers();
  for (int i = 0; i < num; ++i) {
    const char *name = SDL_GetAudioDriver(i);
    if (SDL_AudioInit(name) != 0) {
      const auto *error = SDL_GetError();
      std::cerr << error << "\n";
    }
  }
}

} // namespace App
