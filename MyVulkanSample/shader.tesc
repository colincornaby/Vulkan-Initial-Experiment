#version 450

layout (vertices = 3) out;

layout(location = 0) in vec3 inColors[];
layout(location = 0) out vec3 outColors[];

layout(push_constant) uniform RenderDetails {
    layout(offset = 4) int tessCount;
} details;

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    float tessCount = details.tessCount;

    gl_TessLevelInner[0] = tessCount;

    gl_TessLevelOuter[0] = tessCount;
    gl_TessLevelOuter[1] = tessCount;
    gl_TessLevelOuter[2] = tessCount;

    outColors[gl_InvocationID] = inColors[gl_InvocationID];
}
