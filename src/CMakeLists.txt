


file(
  GLOB core_src
  "./core/*.cc"
)

add_library(
  tool
  "./core/tool.cc"
  )
target_include_directories(tool PUBLIC "./include")
target_link_libraries(tool PUBLIC CONAN_PKG::boost CONAN_PKG::sdl CONAN_PKG::fmt
  )
#conan_target_link_libraries(tool PUBLIC)


add_library(
  asset_manager
  "./core/asset.cc"
  )
add_library(core_other
  "./core/render_target.cc"
  "./core/pipeline.cc"
  "./core/node.cc")

target_link_libraries(core_other PUBLIC
  asset_manager tool)

target_include_directories(core_other PUBLIC "./include")

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
  CONAN_PKG::tinyobjloader
  CONAN_PKG::tinygltf
  tool)

add_library(
  core
  "./core/application.cc"
  "./core/vulkanrender.cc"
)
target_include_directories(core PUBLIC "./include")
target_link_libraries(core PUBLIC tool
  asset_manager
  core_other)

add_executable(render main.cc)
target_link_libraries(render PUBLIC core)



message("in  src ${CMAKE_CURRENT_BINARY_DIR}")

#compile shader
set(Vulkan_GLSLC_EXECUTABLE "${CONAN_BIN_DIRS_SHADERC}/glslc")
set(SHADER_BIN_PATH "${CMAKE_BINARY_DIR}/shader")
message("${CMAKE_BINARY_DIR}${Vulkan_GLSLC_EXECUTABLE}${SHADER_BIN_PATH}")
add_custom_command(
  OUTPUT spirv
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BIN_PATH}
  COMMAND ${Vulkan_GLSLC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.frag" -o "${SHADER_BIN_PATH}/frag.spv"
  COMMAND ${Vulkan_GLSLC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.vert" -o "${SHADER_BIN_PATH}/vert.spv"
  VERBATIM
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.frag"
  "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.vert"
)

add_custom_target(
  shaderbin
  DEPENDS spirv
)

add_dependencies(render shaderbin)
