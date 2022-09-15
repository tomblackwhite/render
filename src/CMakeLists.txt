


file(
  GLOB core_src
  "./core/*.cc"
)

add_library(
  tool
  "./core/tool.cc"
  )
target_include_directories(tool PUBLIC "./include")
target_link_libraries(tool PUBLIC CONAN_PKG::boost CONAN_PKG::sdl CONAN_PKG::fmt)
#conan_target_link_libraries(tool PUBLIC)


add_library(
  asset_manager
  "./core/asset.cc"
  )

set(VULKAN_PKG
  CONAN_PKG::vulkan-headers
  CONAN_PKG::vulkan-loader
  CONAN_PKG::vulkan-validationlayers
  CONAN_PKG::vulkan-memory-allocator)

target_include_directories(asset_manager PUBLIC "./include")
target_link_libraries(asset_manager PUBLIC
  ${VULKAN_PKG}
  CONAN_PKG::stb
  CONAN_PKG::glm
  tool)

add_library(
  core
  "./core/application.cc"
  "./core/vulkanrender.cc"
)
target_include_directories(core PUBLIC "./include")
target_link_libraries(core PUBLIC tool asset_manager)

add_executable(render main.cc)
target_link_libraries(render PUBLIC core)




#compile shader
set(Vulkan_GLSLC_EXECUTABLE "${CONAN_BIN_DIRS_SHADERC}/glslc")
message("${CMAKE_BINARY_DIR}${Vulkan_GLSLC_EXECUTABLE}")
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
