file(
  GLOB core_src
  "./core/*.cc"
)

add_library(
  tool
  "./core/tool.cc"
)
target_include_directories(tool PUBLIC "./include")
target_link_libraries(
  tool PUBLIC CONAN_PKG::boost CONAN_PKG::sdl CONAN_PKG::fmt
  CONAN_PKG::spdlog
)
#conan_target_link_libraries(tool PUBLIC)


add_library(
  asset_manager
  "./core/asset.cc"
)
add_library(
  core_other
  "./core/render_target.cc"
  "./core/pipeline.cc"
  "./core/scene.cc"
)

add_library(
  node
  "./core/node.cc"
)

add_library(
  script_interface
  "./script/script.cc"
)

add_library(
  script
  "./script/script_implement.cc"
)

target_link_libraries(script PRIVATE node PUBLIC script_interface)

target_include_directories(script_interface PUBLIC "./include")
target_link_libraries(
  node PRIVATE script PUBLIC
  asset_manager tool
)
target_link_libraries(
  core_other PUBLIC
  node script script_interface
)

target_include_directories(core_other PUBLIC "./include")

set(
  VULKAN_PKG
  CONAN_PKG::vulkan-headers
  CONAN_PKG::vulkan-loader
  CONAN_PKG::vulkan-validationlayers
  CONAN_PKG::vulkan-memory-allocator
)

target_include_directories(asset_manager PUBLIC "./include")
target_link_libraries(
  asset_manager PUBLIC
  ${VULKAN_PKG}
  CONAN_PKG::stb
  CONAN_PKG::glm
  CONAN_PKG::tinyobjloader
  CONAN_PKG::tinygltf
  tool
)

add_library(
  core
  "./core/application.cc"
  "./core/vulkanrender.cc"
)
target_include_directories(core PUBLIC "./include")
target_link_libraries(
  core PUBLIC tool
  asset_manager
  core_other
)

add_executable(render main.cc)
target_link_libraries(render PUBLIC core)



message("in  src ${CMAKE_CURRENT_BINARY_DIR}")

#compile shader

#shader files
file(
  GLOB shader_files
  "${CMAKE_CURRENT_SOURCE_DIR}/shader/*.frag"
  "${CMAKE_CURRENT_SOURCE_DIR}/shader/*.vert"
)

set(Vulkan_GLSLC_EXECUTABLE "${CONAN_BIN_DIRS_SHADERC}/glslc")
set(SHADER_BIN_PATH "${CMAKE_BINARY_DIR}/shader")
message("${CMAKE_BINARY_DIR}${Vulkan_GLSLC_EXECUTABLE}${SHADER_BIN_PATH}")
message("${shader_files}")
#创建文件夹
add_custom_target(
  create_shader_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BIN_PATH}
)
add_custom_command(
  OUTPUT spirv
  COMMAND ${Vulkan_GLSLC_EXECUTABLE} "-O" "-c" "${shader_files}"
  DEPENDS create_shader_dir "${shader_files}"
  WORKING_DIRECTORY "${SHADER_BIN_PATH}"
  VERBATIM
  COMMAND_EXPAND_LISTS
)


add_custom_target(
  shaderbin
  DEPENDS spirv
)




add_dependencies(render shaderbin)
