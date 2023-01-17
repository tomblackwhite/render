#pragma once
#include <SDL_audio.h>
#include <SDL_error.h>
#include <boost/stacktrace.hpp>
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vulkan/vulkan.hpp>

namespace App {
using std::string;
template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

//标识非所有权
template <typename T> using observer = T;
template <typename T> using observer_ptr = T;

// throw sdl exception and backtrace
void ThrowException(string const &message, bool hasSDL = false);

// check audio driver

void CheckAudioDriver();

#define VULKAN_CHECK(expr, message)                                            \
  do {                                                                         \
    if (VkResult result = expr; result != VK_SUCCESS) {                        \
      App::ThrowException(fmt::format("message{}", result));                   \
    }                                                                          \
  } while (false)

} // namespace App
