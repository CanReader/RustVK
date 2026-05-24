#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 viewPos;
    vec4 albedoMetallic;
    vec4 roughnessAO;
    vec4 pointLightPos[4];
    vec4 pointLightColor[4];
    vec4 lightCounts;
    vec4 dirLightDir;
    vec4 dirLightColor;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos    = vec3(worldPos);
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor  = inColor;
    gl_Position = ubo.proj * ubo.view * worldPos;
}
