#include "tool.hh"

namespace App {

void ThrowException(string const &message, bool hasSDL) {
  auto backtraceStr =
      boost::stacktrace::to_string(boost::stacktrace::stacktrace());

  std::cerr << backtraceStr << "\n backTrace\n";

  string whatStr;
  if (hasSDL) {
    auto const *error = SDL_GetError();
    whatStr = fmt::format("{}\n{}\n{}", message, error, backtraceStr);
  } else {
    whatStr = fmt::format("{}\n{}", message, backtraceStr);
  }

  throw std::runtime_error(whatStr);
}

} // namespace App
