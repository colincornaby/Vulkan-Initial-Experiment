#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec4 outColor;

void main() {
    const vec3 lightPosition = vec3(-2.0f, 1.0f, -1.0f);
    float lightMagnitude = clamp(dot(lightPosition, normal), 0.0f, 1.0f);
    lightMagnitude += 0.3f;
    lightMagnitude = clamp(lightMagnitude, 0.0f, 1.0f);
    outColor = vec4(fragColor *  lightMagnitude, 1.0);
}