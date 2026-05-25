#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

// ── Payload ───────────────────────────────────────────────────────────────────
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

// ── Vertex / Index SSBOs ──────────────────────────────────────────────────────
// Vertex layout (12 floats = 48 bytes):
//   float[0..2]  position
//   float[3..5]  normal
//   float[6..8]  color
//   float[9]     metallic
//   float[10]    roughness
//   float[11]    transmission

layout(set = 0, binding = 4) readonly buffer VertexBuffer {
    float data[];
} vertexBuf;

layout(set = 0, binding = 5) readonly buffer IndexBuffer {
    uint data[];
} indexBuf;

hitAttributeEXT vec2 barycentrics;

void main() {
    const int FLOATS_PER_VERTEX = 12;

    // Fetch triangle indices
    uint primIdx = gl_PrimitiveID;
    uint i0 = indexBuf.data[primIdx * 3 + 0];
    uint i1 = indexBuf.data[primIdx * 3 + 1];
    uint i2 = indexBuf.data[primIdx * 3 + 2];

    uint base0 = i0 * FLOATS_PER_VERTEX;
    uint base1 = i1 * FLOATS_PER_VERTEX;
    uint base2 = i2 * FLOATS_PER_VERTEX;

    // Barycentric coords
    vec3 bary = vec3(1.0 - barycentrics.x - barycentrics.y,
                     barycentrics.x, barycentrics.y);

    // Positions (object space)
    vec3 p0 = vec3(vertexBuf.data[base0 + 0], vertexBuf.data[base0 + 1], vertexBuf.data[base0 + 2]);
    vec3 p1 = vec3(vertexBuf.data[base1 + 0], vertexBuf.data[base1 + 1], vertexBuf.data[base1 + 2]);
    vec3 p2 = vec3(vertexBuf.data[base2 + 0], vertexBuf.data[base2 + 1], vertexBuf.data[base2 + 2]);
    vec3 posObj = bary.x * p0 + bary.y * p1 + bary.z * p2;

    // Normals (object space)
    vec3 n0 = vec3(vertexBuf.data[base0 + 3], vertexBuf.data[base0 + 4], vertexBuf.data[base0 + 5]);
    vec3 n1 = vec3(vertexBuf.data[base1 + 3], vertexBuf.data[base1 + 4], vertexBuf.data[base1 + 5]);
    vec3 n2 = vec3(vertexBuf.data[base2 + 3], vertexBuf.data[base2 + 4], vertexBuf.data[base2 + 5]);
    vec3 normObj = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    // Transform to world space
    // gl_ObjectToWorldEXT: 3x4 matrix, row-major in Vulkan RT
    payload.hitPos = (gl_ObjectToWorldEXT * vec4(posObj, 1.0)).xyz;

    // Normal transform: use inverse-transpose = WorldToObject transposed
    // gl_WorldToObjectEXT is 3x4; normal is transformed by its 3x3 transpose
    mat3 normalMat = transpose(mat3(gl_WorldToObjectEXT));
    vec3 worldNorm = normalize(normalMat * normObj);

    // Always store outward geometric normal (rgen determines face orientation)
    payload.normal = worldNorm;

    // Albedo, material
    payload.albedo = vec3(
        vertexBuf.data[base0 + 6] * bary.x + vertexBuf.data[base1 + 6] * bary.y + vertexBuf.data[base2 + 6] * bary.z,
        vertexBuf.data[base0 + 7] * bary.x + vertexBuf.data[base1 + 7] * bary.y + vertexBuf.data[base2 + 7] * bary.z,
        vertexBuf.data[base0 + 8] * bary.x + vertexBuf.data[base1 + 8] * bary.y + vertexBuf.data[base2 + 8] * bary.z
    );
    payload.metallic = vertexBuf.data[base0 + 9]  * bary.x + vertexBuf.data[base1 + 9]  * bary.y + vertexBuf.data[base2 + 9]  * bary.z;
    payload.roughness= vertexBuf.data[base0 + 10] * bary.x + vertexBuf.data[base1 + 10] * bary.y + vertexBuf.data[base2 + 10] * bary.z;
    payload.transmission = vertexBuf.data[base0 + 11] * bary.x + vertexBuf.data[base1 + 11] * bary.y + vertexBuf.data[base2 + 11] * bary.z;

    payload.missed    = false;
    payload.missColor = vec3(0.0);
}
