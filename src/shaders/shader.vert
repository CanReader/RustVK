#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in float inMetallic;
layout(location = 4) in float inRoughness;
layout(location = 5) in float inTransmission;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 viewPos;
    vec4 pointLightPos[4];
    vec4 pointLightColor[4];
    vec4 lightCounts;
    vec4 dirLightDir;
    vec4 dirLightColor;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out float fragMetallic;
layout(location = 4) out float fragRoughness;
layout(location = 5) out float fragTransmission;

void main() {
    vec4 worldPos  = ubo.model * vec4(inPosition, 1.0);
    fragPos        = vec3(worldPos);
    fragNormal     = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor      = inColor;
    fragMetallic   = inMetallic;
    fragRoughness  = inRoughness;
    fragTransmission = inTransmission;
    gl_Position = ubo.proj * ubo.view * worldPos;
}
