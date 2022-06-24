set(CMAKE_AUTOMOC ON) # Meta-Object Compiler
set(CMAKE_AUTORCC ON) # Resource Compiler
set(CMAKE_AUTOUIC ON) # User Interface Compiler


find_package(fmt 8.0.1 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(Microsoft.GSL REQUIRED)

file(
  GLOB core_src
  "./core/*.cc"
)
add_library(
  core
  ${core_src}
)
target_compile_options(core PRIVATE)
target_include_directories(core PUBLIC "./include")
target_link_libraries(
  core PUBLIC
  fmt::fmt
  glfw
  Qt5::Widgets
  spdlog::spdlog
  Vulkan::Vulkan
  Microsoft.GSL::GSL
)

file(
  GLOB ui_src
  "./ui/*.cc"
  "./ui/*.ui"
)
add_library(ui ${ui_src})
target_include_directories(ui PUBLIC "./include")
target_link_libraries(ui PUBLIC Qt5::Widgets)
add_executable(render main.cc)
message("${CMAKE_BINARY_DIR}${Vulkan_GLSLC_EXECUTABLE}")

target_include_directories(render PUBLIC "./ui")
target_link_libraries(render PUBLIC core ui glfw Vulkan::Vulkan)


#compile shader
add_custom_command(
  OUTPUT spirv
  COMMAND ${Vulkan_GLSLC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.frag" -o "frag.spv"
  COMMAND ${Vulkan_GLSLC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.vert" -o "vert.spv"
  VERBATIM
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.frag" "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.vert"
)

add_custom_target(
  shaderbin
  DEPENDS spirv
)
add_dependencies(render shaderbin)
