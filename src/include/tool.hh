#pragma once
#include <SDL_audio.h>
#include <SDL_error.h>
#include <boost/stacktrace.hpp>
#include <exception>
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vulkan/vulkan.hpp>
#include <utility>
#include <type_traits>
#include <concepts>
#include <string_view>
#include <ranges>
#include <algorithm>


namespace App {
using std::string;
namespace ranges = std::ranges;
namespace stacktrace = boost::stacktrace;
template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

// 标识所有权
template <typename T> using own_ptr = T*;

// throw sdl exception and backtrace
void ThrowException(string const &message, bool hasSDL = false);


// check audio driver

void CheckAudioDriver();

void VulkanCheck(VkResult result,std::string_view message);

// #define VULKAN_CHECK(expr, message)                                            \
//   do {                                                                         \
//     if (VkResult result = expr; result != VK_SUCCESS) {                        \
//       App::ThrowException(fmt::format("message{}", result));                   \
//     }                                                                          \
//   } while (false)

// stacktrace in exception interface
class IStackTraceError {
public:
  IStackTraceError() = default;
  explicit IStackTraceError(stacktrace::stacktrace stacktrace)
      : m_stacktrace(std::move(stacktrace)){};
  IStackTraceError(const IStackTraceError &) = default;
  IStackTraceError(IStackTraceError &&) = default;
  IStackTraceError &operator=(IStackTraceError const &) = default;
  IStackTraceError &operator=(IStackTraceError &&) = default;

  [[nodiscard("stacktrace not ignore")]] const stacktrace::stacktrace &
  stacktrace() const {
    return m_stacktrace;
  };

private:
  stacktrace::stacktrace m_stacktrace = stacktrace::stacktrace();

protected:
  // don't cast to IStackTraceError
  ~IStackTraceError() = default;
};

class RunTimeErrorWithTrace : public std::runtime_error,
                              public IStackTraceError {
public:
  using std::runtime_error::runtime_error;

  RunTimeErrorWithTrace(const char *what_arg, stacktrace::stacktrace stack)
      : std::runtime_error(what_arg), IStackTraceError(std::move(stack)) {}
  RunTimeErrorWithTrace(const std::string &what_arg,
                        stacktrace::stacktrace stack)
      : std::runtime_error(what_arg), IStackTraceError(std::move(stack)) {}
};

} // namespace App
