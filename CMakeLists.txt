#set minimal cmake version

cmake_minimum_required(VERSION "3.20" )

project(render LANGUAGES CXX)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")


#set c++ standard
set(CMAKE_CXX_STANDARD 20)
# enable_testing()
#add_compile_options(-g)

add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
add_compile_definitions(
  "GLM_FORCE_DEPTH_ZERO_TO_ONE"
  "VULKAN_HPP_NO_CONSTRUCTORS")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)

#DEFINITIONS boost
set(CONAN_DEFINES_BOOST "-DBOOST_STACKTRACE_USE_BACKTRACE")
set(CONAN_COMPILE_DEFINITIONS_BOOST "BOOST_STACKTRACE_USE_BACKTRACE")

conan_basic_setup(TARGETS)

add_subdirectory("src")

#export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

#copy assets to obj
file(
  GLOB asset_file
  "${CMAKE_SOURCE_DIR}/asset/*"
)
add_custom_target(copyasset ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${asset_file} "${CMAKE_BINARY_DIR}/asset")
