#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform RenderDetails {
    float time;
} details;

layout(triangles, equal_spacing, cw) in;
layout(location = 0) in vec3 inColors[];

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 normal;

void main() {
    outColor = (inColors[0] * gl_TessCoord.x) + 
                     (inColors[1] * gl_TessCoord.y) + 
                     (inColors[2] * gl_TessCoord.z);
    gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x) + 
                     (gl_in[1].gl_Position * gl_TessCoord.y) + 
                     (gl_in[2].gl_Position * gl_TessCoord.z);
    vec4 waveDirection = normalize(ubo.proj * ubo.view * ubo.model * vec4(1.0f, 1.0f, 0.0f, 0.0f));
    const float magnitude = (gl_Position.x * waveDirection.x) + (gl_Position.y * waveDirection.y);
    const float frequency = 16.0f;
    const float theta = magnitude * frequency + details.time * 5.0f;
    const float height = sin(theta);
    gl_Position.y += height/ 5.0f;
    normal = normalize(ubo.view * ubo.model * vec4(frequency * cos(theta), 1.0f, 0.0f, 1.0f)).xyz;
}
 