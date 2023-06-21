#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 texCoordinate;

// out
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outView;
layout(location = 2) out vec2 texCoord;
layout(location = 3 ) out vec4 testVec;

layout(set = 0, binding = 0) uniform CameraBuffer {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
}
cameraData;

layout(set = 0, binding = 1) uniform SceneData {
  vec4 fogColor;
  vec4 fogDistances;
  vec4 ambientColor;
  vec4 sunlightDirection;
  vec4 sunlightColor;
}
sceneData;

layout(std140, set = 1, binding = 0) readonly buffer ObjectData {
  mat4 modelMatrix[];
}
objectData;

// layout(set = 2,binding = 0) uniform sampler2D texSampler;

layout(std140, set = 2, binding = 1) uniform Material {
  vec4 baseColorFactor;

  float metallicFactor;
  float roughnessFactor;
  int baseColorFactorIndex;
  int metallicRoughnessCoordIndex;
}
material;

layout(push_constant) uniform constants {
  vec4 data;
  mat4 renderMatrix;
}
pushConstants;

// const vec3 positions[3] =
//     vec3[3](vec3(1.f, 1.f, 0.0f), vec3(-1.f, 1.f, 0.0f), vec3(1.f, -1.f, 0.0f));
// const vec3 colors[3] = vec3[3](vec3(1.0f, 0.0f, 0.0f), // red
//                                vec3(0.0f, 1.0f, 0.0f), // green
//                                vec3(00.f, 0.0f, 1.0f)  // blue
// );

void main() {

  // output the position of each vertex
  vec4 color = sceneData.ambientColor;

  mat4 transformMatrix =
      cameraData.viewProj * objectData.modelMatrix[gl_InstanceIndex];
  // mat4 transformMatrix = cameraData.viewProj;
  gl_Position = transformMatrix * vec4(inPosition, 1.0f);
  texCoord = texCoordinate;
  outNormal= inNormal;
  outView = (-cameraData.view[3].xyz - inPosition) ;

  // gl_Position = vec4(inPosition, 1.0f);
  // gl_Position= transformMatrix* vec4(positions[gl_VertexIndex%3],1);

 // vertexColor= texture(texSampler, texCoordinate);
  vec4 testVec = material.baseColorFactor;
}
