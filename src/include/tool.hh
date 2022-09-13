#pragma once
#include <SDL_error.h>
#include <SDL_audio.h>
#include <boost/stacktrace.hpp>
#include <string>
#include <system_error>
#include <fmt/core.h>
#include <iostream>
#include <memory>

namespace App {
using std::string;
using std::unique_ptr;

//标识非所有权
template<typename T> using observer = T;
template<typename T> using observer_ptr = T;

//throw sdl exception and backtrace
void ThrowException(string const & message, bool hasSDL=false);


//check audio driver

void CheckAudioDriver();


} // namespace App
