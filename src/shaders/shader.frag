#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 lightPos;
    float _pad0;
    vec3 lightColor;
    float _pad1;
    vec3 viewPos;
    float _pad2;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 norm     = normalize(fragNormal);
    vec3 toLight  = ubo.lightPos - fragPos;
    float dist    = length(toLight);
    vec3 lightDir = toLight / dist;
    vec3 viewDir  = normalize(ubo.viewPos - fragPos);
    vec3 halfDir  = normalize(lightDir + viewDir);

    // Quadratic attenuation for realistic point light falloff
    float atten = 1.0 / (1.0 + 0.045 * dist + 0.0075 * dist * dist);

    float diff = max(dot(norm, lightDir), 0.0);
    float spec = pow(max(dot(norm, halfDir), 0.0), 128.0);

    vec3 ambient  = 0.06 * ubo.lightColor;
    vec3 diffuse  = diff * ubo.lightColor * atten;
    vec3 specular = 0.85 * spec * ubo.lightColor * atten;

    vec3 result = (ambient + diffuse + specular) * fragColor;
    outColor = vec4(result, 1.0);
}
