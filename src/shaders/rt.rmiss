#version 460
#extension GL_EXT_ray_tracing : require

struct HitPayload {
    vec3  hitPos;
    vec3  normal;
    vec3  albedo;
    float metallic;
    float roughness;
    float transmission;
    bool  missed;
    vec3  missColor;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main() {
    // Sky gradient: horizon to zenith
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float t  = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    vec3 horizon = vec3(0.15, 0.25, 0.45);
    vec3 zenith  = vec3(0.05, 0.10, 0.30) * 2.5;

    payload.missColor = mix(horizon, zenith, t);
    payload.missed    = true;
}
