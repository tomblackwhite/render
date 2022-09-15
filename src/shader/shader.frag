#version 450

layout(location = 0) in vec3 vertexInColor;

layout(location = 0) out vec4 outColor;
// layout(location = 1) in vec2 fragTexCoord;

// layout(binding = 1) uniform sampler2D texSampler;

// layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(vertexInColor,1.0f);
    //outColor = texture(texSampler, fragTexCoord);
}
