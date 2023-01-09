#version 450

layout (vertices = 3) out;

layout(location = 0) in vec3 inColors[];
layout(location = 0) out vec3 outColors[];

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    gl_TessLevelInner[0] = 64.0f;

    gl_TessLevelOuter[0] = 64.0f;
    gl_TessLevelOuter[1] = 64.0f;
    gl_TessLevelOuter[2] = 64.0f;

    outColors[gl_InvocationID] = inColors[gl_InvocationID];
}
