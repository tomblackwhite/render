#version 460

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inView;
layout(location = 2) in vec2 texCoord;

layout(location = 3) in vec4 testVec;
layout(location = 0) out vec4 outColor;
// layout(location = 1) in vec2 fragTexCoord;
layout(set = 0, binding = 1) uniform SceneData {
  vec4 fogColor;
  vec4 fogDistances;
  vec4 ambientColor;
  vec4 sunlightDirection;
  vec4 sunlightColor;
}
sceneData;

layout(set = 2, binding = 0) uniform sampler2D texSampler;
layout(std140, set = 2, binding = 1) uniform Material {
  vec4 baseColorFactor;
  float metallicFactor;
  float roughnessFactor;
  int baseColorFactorIndex;
  int metallicRoughnessCoordIndex;
}
material;

// layout(set = 2,binding = 0) uniform sampler2D texSampler;

// layout(location = 0) out vec4 outColor;

const float PI = radians(180);

float dCookTorrance(in float alpha, in vec3 halfNormal, in vec3 normal) {
  float alpha2 = alpha * alpha;

  float hDotN = dot(halfNormal, normal);
  hDotN = max(hDotN, 0.0);
  float hDotN2 = hDotN * hDotN;

  float low = (hDotN2 * (alpha2 - 1)) + 1;

  low = PI * low * low;

  return hDotN2 / low;
}

float visibleFunction1(in vec3 halfNormal, in vec3 direction, in float alpha) {
  float hDotD = dot(halfNormal, direction);
  float alpha2 = alpha * alpha;
  float hDotD2 = hDotD * hDotD;

  float low = abs(hDotD) + sqrt(alpha2 + (1 - alpha2) * hDotD2);

  float result = (hDotD > 0 ? 1 : 0) / low;
  return result;
}

float test() { return 1.0; }

float specularBrdf(in float alpha, in vec3 halfNormal, in vec3 normal,
                   in vec3 lightDirection, in vec3 view) {
  return dCookTorrance(alpha, halfNormal, normal) *
         visibleFunction1(halfNormal, lightDirection, alpha) *
         visibleFunction1(halfNormal, view, alpha);
}

vec3 getLight(in vec4 baseColor, in float metallic, in float roughness,
              in vec3 lightDirection, in vec3 normal, in vec3 view) {

  const vec3 black = vec3(0);

  vec3 cDiff = mix(baseColor.rgb, black, metallic);
  // 菲涅尔垂直表面时反射率
  vec3 f0 = mix(vec3(0.04), baseColor.rgb, metallic);
  float alpha = roughness * roughness;
  vec3 halfNormal = normalize(lightDirection + view);

  // 菲涅尔反射率近似
  vec3 rF = f0 + (1 - f0) * pow((1 - abs(dot(normal, lightDirection))), 5);

  vec3 diffuse = (1 - rF) * (1 / PI) * cDiff;
  vec3 specular =
      rF * specularBrdf(alpha, halfNormal, normal, lightDirection, view) *
      baseColor.rgb;

  return diffuse + specular;
}

void main() {
  // outColor = vec4(texSampler[],1.0f);
  // outColor = vertexInColor;
  // global light
  vec3 resultColor = vec3(0);

  vec4 baseColor = vec4(0, 0, 0, 0);
  if (material.baseColorFactorIndex != -1) {

    baseColor = texture(texSampler, texCoord);
  } else {

    baseColor = vec4(1, 1, 1, 1);
  }

  baseColor *= material.baseColorFactor;

  {
    float lDotN = dot(sceneData.sunlightDirection.xyz, inNormal);
    if (lDotN > 0) {
      resultColor +=
          getLight(baseColor, material.metallicFactor, material.roughnessFactor,
                   sceneData.sunlightDirection.xyz, inNormal, inView) *
          sceneData.sunlightColor.xyz;
    }
  }

  outColor = vec4(resultColor, 1);
}
