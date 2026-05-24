#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in float fragMetallic;
layout(location = 4) in float fragRoughness;
layout(location = 5) in float fragTransmission;

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

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float G_Schlick(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_Schlick(NdotV, roughness) * G_Schlick(NdotL, roughness);
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Cook-Torrance BRDF. transmission reduces diffuse so glass passes light through.
vec3 cookTorrance(
    vec3  N, vec3 V, vec3 L,
    vec3  albedo, float metallic, float roughness, float transmission,
    vec3  radiance
) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    vec3  H     = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0001);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // Glass: F0 = 0.04 (IOR ≈ 1.5), metals use albedo as F0
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(HdotV, F0);

    // Transmission absorbs the diffuse lobe — glass lets light through
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic) * (1.0 - transmission);

    vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

void main() {
    vec3  albedo       = fragColor;
    float metallic     = fragMetallic;
    float roughness    = max(fragRoughness, 0.04); // roughness=0 makes a2=0 → NDF=0/0
    float transmission = fragTransmission;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.viewPos.xyz - fragPos);

    int  numPointLights = int(ubo.lightCounts.x);
    bool hasDirLight    = ubo.lightCounts.y > 0.5;

    vec3 Lo = vec3(0.0);

    for (int i = 0; i < numPointLights; i++) {
        vec3  lpos      = ubo.pointLightPos[i].xyz;
        float intensity = ubo.pointLightPos[i].w;
        vec3  lcolor    = ubo.pointLightColor[i].xyz;

        vec3  toLight = lpos - fragPos;
        float dist2   = dot(toLight, toLight);
        vec3  L       = normalize(toLight);

        vec3 radiance = lcolor * intensity / dist2;
        Lo += cookTorrance(N, V, L, albedo, metallic, roughness, transmission, radiance);
    }

    if (hasDirLight) {
        vec3  L        = normalize(-ubo.dirLightDir.xyz);
        float intensity = ubo.dirLightDir.w;
        vec3  radiance  = ubo.dirLightColor.xyz * intensity;
        Lo += cookTorrance(N, V, L, albedo, metallic, roughness, transmission, radiance);
    }

    // Glass spheres gather ambient through their volume — boost it by transmission
    // so they glow softly with their tint color in unlit areas.
    float ambientStr = 0.03 + transmission * 0.18;
    vec3  ambient    = ambientStr * albedo;

    vec3 color = ambient + Lo;

    // Reinhard tone mapping
    color = color / (color + vec3(1.0));

    outColor = vec4(color, 1.0);
}
