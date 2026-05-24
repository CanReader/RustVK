#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

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

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// GGX normal distribution function
float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Schlick-GGX geometry term (single direction)
float G_Schlick(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's combined geometry
float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_Schlick(NdotV, roughness) * G_Schlick(NdotL, roughness);
}

// Fresnel-Schlick
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Cook-Torrance BRDF contribution for one light direction L and incoming radiance.
// Returns outgoing radiance toward V.
vec3 cookTorrance(
    vec3 N, vec3 V, vec3 L,
    vec3 albedo, float metallic, float roughness,
    vec3 radiance
) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    vec3  H     = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0001);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(HdotV, F0);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

void main() {
    vec3  albedo    = ubo.albedoMetallic.rgb * fragColor;
    float metallic  = ubo.albedoMetallic.w;
    float roughness = ubo.roughnessAO.x;
    float ao        = ubo.roughnessAO.y;

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

        // Physically correct inverse-square falloff
        vec3 radiance = lcolor * intensity / dist2;

        Lo += cookTorrance(N, V, L, albedo, metallic, roughness, radiance);
    }

    if (hasDirLight) {
        vec3  L         = normalize(-ubo.dirLightDir.xyz);
        float intensity = ubo.dirLightDir.w;
        vec3  radiance  = ubo.dirLightColor.xyz * intensity;

        Lo += cookTorrance(N, V, L, albedo, metallic, roughness, radiance);
    }

    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color   = ambient + Lo;

    // Reinhard tone mapping
    color = color / (color + vec3(1.0));

    outColor = vec4(color, 1.0);
}
