#version 460

layout(location = 0) in vec3 vertexInColor;
layout(location = 1) in vec2 texCoord;

layout(location = 0) out vec4 outColor;
// layout(location = 1) in vec2 fragTexCoord;

layout(set = 2,binding = 0) uniform sampler2D texSampler;

// layout(location = 0) out vec4 outColor;

void main() {
    // outColor = vec4(texSampler[],1.0f);
    outColor = texture(texSampler, texCoord);
}
