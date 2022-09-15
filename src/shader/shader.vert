#version 450

// layout(binding = 0) uniform UniformBufferObject {
//     mat4 model;
//     mat4 view;
//     mat4 proj;
// } ubo;


// layout(location = 0) in vec2 inPosition;
// layout(location = 1) in vec3 inColor;
// layout(location = 2) in vec2 inTexCoord;

// layout(location = 0) out vec3 fragColor;
// layout(location = 1) out vec2 fragTexCoord;
layout(location = 0) out vec3 vertexColor;

const vec3 positions[3] = vec3[3](
    vec3(1.f,1.f, 0.0f),
    vec3(-1.f,1.f, 0.0f),
    vec3(1.f,-1.f, 0.0f)
    );
const vec3 colors[3] = vec3[3](
    vec3(1.0f, 0.0f, 0.0f), //red
    vec3(0.0f, 1.0f, 0.0f), //green
    vec3(00.f, 0.0f, 1.0f)  //blue
);

void main() {

    //output the position of each vertex
    gl_Position = vec4(positions[gl_VertexIndex], 1.0f);
    vertexColor = colors[gl_VertexIndex];
   // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    // gl_Position = vec4(inPosition, 0.0, 1.0);
    // fragColor = inColor;
    // fragTexCoord = inTexCoord;
}

