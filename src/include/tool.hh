#pragma once
#include <SDL_error.h>
#include <boost/stacktrace.hpp>
#include <string>
#include <system_error>
#include <fmt/core.h>
#include <iostream>

namespace App {
using std::string;


//throw sdl exception and backtrace
void ThrowException(string const & message, bool hasSDL=false);

} // namespace App
