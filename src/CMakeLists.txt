


file(
  GLOB core_src
  "./core/*.cc"
)

add_library(
  tool
  "./core/tool.cc"
  )
target_include_directories(tool PUBLIC "./include")
target_link_libraries(tool PUBLIC ${CONAN_TARGETS})
#conan_target_link_libraries(tool PUBLIC)

add_library(
  core
  "./core/application.cc"
)
target_include_directories(core PUBLIC "./include")
target_link_libraries(core PUBLIC tool)

add_executable(render main.cc)
target_link_libraries(render PUBLIC core)

message("binary dir${CMAKE_BINARY_DIR}")


#message("${CMAKE_BINARY_DIR}${Vulkan_GLSLC_EXECUTABLE}")
#compile shader
# add_custom_command(
#   OUTPUT spirv
#   COMMAND ${Vulkan_GLSLC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.frag" -o "frag.spv"
#   COMMAND ${Vulkan_GLSLC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.vert" -o "vert.spv"
#   VERBATIM
#   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.frag" "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.vert"
# )

# add_custom_target(
#   shaderbin
#   DEPENDS spirv
# )
# add_dependencies(render shaderbin)
