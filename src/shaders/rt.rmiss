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
    payload.missed = true;
}
