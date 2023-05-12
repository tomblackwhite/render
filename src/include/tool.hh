#pragma once
#include <SDL_audio.h>
#include <SDL_error.h>
#include <algorithm>
#include <boost/mp11.hpp>
#include <boost/stacktrace.hpp>
#include <concepts>
#include <exception>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <memory>

#include <ranges>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <filesystem>
#include <vulkan/vulkan.hpp>

namespace App {
using std::string;
namespace ranges = std::ranges;
namespace fs = std::filesystem;
namespace mp = boost::mp11;
namespace stacktrace = boost::stacktrace;
template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

// 标识所有权
template <typename T> using own_ptr = T *;

// throw sdl exception and backtrace
void ThrowException(string const &message, bool hasSDL = false);

// 给variant赋值，当内部类型匹配时才会触发。
template <typename... L, typename... R>
bool VariantAssign(std::variant<L...> &left, std::variant<R...> const &right) {
  bool canAssign = false;

  std::visit(
      [&canAssign, &left](auto &&var) {
        using RightCurrentElementType = std::decay_t<decltype(var)>;

        using LeftType = std::variant<L...>;

        if constexpr (mp::mp_contains<LeftType,
                                      RightCurrentElementType>::value) {
          canAssign = true;
          left = var;
        }
      },
      right);

  return canAssign;
}

// check audio driver

void CheckAudioDriver();

void VulkanCheck(VkResult result, std::string const &message);

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

inline std::vector<unsigned char> readFile(const fs::path &filename) {

  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    App::ThrowException(fmt::format("can't open file {}",filename.string()));
  }
  auto fileSize = file.tellg();
  std::vector<unsigned char> buffer(fileSize);
  file.seekg(0);
  file.read(std::launder(reinterpret_cast<char*>(buffer.data())), fileSize);
  return buffer;
}

} // namespace App
