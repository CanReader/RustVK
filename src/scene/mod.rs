use cgmath::{Point3, Vector3};
use std::f32::consts::PI;

// Each vertex carries its own material so both the rasterizer and a future
// ray tracing pass can read material data directly from the vertex buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position:     [f32; 3],  // offset 0
    pub normal:       [f32; 3],  // offset 12
    pub color:        [f32; 3],  // offset 24  (albedo tint)
    pub metallic:     f32,       // offset 36
    pub roughness:    f32,       // offset 40
    pub transmission: f32,       // offset 44  (1.0 = perfect glass)
}

pub struct PointLight {
    pub position:  [f32; 3],
    pub color:     [f32; 3],
    pub intensity: f32,
}

pub struct DirectionalLight {
    pub direction: [f32; 3],
    pub color:     [f32; 3],
    pub intensity: f32,
}

pub struct Camera {
    pub position: Point3<f32>,
    pub target:   Point3<f32>,
    pub up:       Vector3<f32>,
    pub fov_deg:  f32,
    pub aspect:   f32,
    pub near:     f32,
    pub far:      f32,
}

pub const MAX_POINT_LIGHTS: usize = 4;

// std140-compatible UBO. Material is now per-vertex; UBO carries only
// transforms, view info, and lights.
// point_light_pos[i]   = (x, y, z, intensity)
// point_light_color[i] = (r, g, b, unused)
// light_counts         = (numPointLights, hasDirLight, unused, unused)
// dir_light_dir        = (x, y, z, intensity)
// dir_light_color      = (r, g, b, unused)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferObject {
    pub model:             [[f32; 4]; 4],                   // offset 0
    pub view:              [[f32; 4]; 4],                   // offset 64
    pub proj:              [[f32; 4]; 4],                   // offset 128
    pub view_pos:          [f32; 4],                        // offset 192
    pub point_light_pos:   [[f32; 4]; MAX_POINT_LIGHTS],    // offset 208
    pub point_light_color: [[f32; 4]; MAX_POINT_LIGHTS],    // offset 272
    pub light_counts:      [f32; 4],                        // offset 336
    pub dir_light_dir:     [f32; 4],                        // offset 352
    pub dir_light_color:   [f32; 4],                        // offset 368
}

/// Uniform buffer for the ray-tracing path tracer.
/// Layout is std140-compatible (all fields 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtUBO {
    pub inv_view:          [[f32; 4]; 4],                   // offset 0
    pub inv_proj:          [[f32; 4]; 4],                   // offset 64
    pub cam_pos:           [f32; 4],                        // offset 128
    pub light_counts:      [f32; 4],                        // offset 144  (numPoint, hasDir, 0, 0)
    pub point_light_pos:   [[f32; 4]; MAX_POINT_LIGHTS],    // offset 160
    pub point_light_color: [[f32; 4]; MAX_POINT_LIGHTS],    // offset 224
    pub dir_light_dir:     [f32; 4],                        // offset 288  (x,y,z, intensity)
    pub dir_light_color:   [f32; 4],                        // offset 304
    pub frame_index:       u32,                             // offset 320
    pub max_bounces:       u32,                             // offset 324
    pub _pad:              [f32; 2],                        // offset 328  (pad to 336 = 16-byte aligned end)
}

pub struct Scene {
    pub vertices:       Vec<Vertex>,
    pub indices:        Vec<u32>,
    pub camera:         Camera,
    pub point_lights:   Vec<PointLight>,
    pub dir_light:      Option<DirectionalLight>,
    pub model_rotation: f32,
}

fn add_sphere(
    center:       [f32; 3],
    radius:       f32,
    color:        [f32; 3],
    metallic:     f32,
    roughness:    f32,
    transmission: f32,
    stacks:       u32,
    slices:       u32,
    verts:        &mut Vec<Vertex>,
    idxs:         &mut Vec<u32>,
) {
    let base = verts.len() as u32;

    for i in 0..=stacks {
        let phi = PI * (i as f32) / (stacks as f32);
        let y   = radius * phi.cos();
        let r   = radius * phi.sin();

        for j in 0..=slices {
            let theta = 2.0 * PI * (j as f32) / (slices as f32);
            let x = r * theta.cos();
            let z = r * theta.sin();

            verts.push(Vertex {
                position:     [center[0] + x, center[1] + y, center[2] + z],
                normal:       [x / radius, y / radius, z / radius],
                color,
                metallic,
                roughness,
                transmission,
            });
        }
    }

    for i in 0..stacks {
        for j in 0..slices {
            let a = base + i * (slices + 1) + j;
            let b = a + slices + 1;
            idxs.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
}

fn add_floor(
    half_size: f32,
    y:         f32,
    color:     [f32; 3],
    roughness: f32,
    verts:     &mut Vec<Vertex>,
    idxs:      &mut Vec<u32>,
) {
    let base = verts.len() as u32;
    let n    = [0.0_f32, 1.0, 0.0];
    let s    = half_size;

    verts.extend_from_slice(&[
        Vertex { position: [-s, y, -s], normal: n, color, metallic: 0.0, roughness, transmission: 0.0 },
        Vertex { position: [ s, y, -s], normal: n, color, metallic: 0.0, roughness, transmission: 0.0 },
        Vertex { position: [ s, y,  s], normal: n, color, metallic: 0.0, roughness, transmission: 0.0 },
        Vertex { position: [-s, y,  s], normal: n, color, metallic: 0.0, roughness, transmission: 0.0 },
    ]);
    // Vertices are laid out far-left, far-right, near-right, near-left.
    // Camera is above looking down, so "far" projects to screen-top and "near" to screen-bottom.
    // The natural 0-1-2 order is CW in screen space (culled). Reverse each triangle.
    idxs.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
}

impl Scene {
    pub fn spheres() -> Self {
        let mut vertices = Vec::new();
        let mut indices  = Vec::new();

        // Perfect white floor
        add_floor(6.0, 0.0, [1.0, 1.0, 1.0], 0.55, &mut vertices, &mut indices);

        // Glass spheres: (center, radius, albedo tint)
        // All are dielectric (metallic=0), perfect glass surface (roughness=0),
        // near-fully transmissive (transmission=0.95).
        // The tint color is subtle — glass passes most light through.
        let spheres: &[([f32; 3], f32, [f32; 3])] = &[
            ([  0.00, 0.80,  0.00], 0.80, [0.85, 1.00, 0.98]),  // center — largest, pale cyan
            ([ -2.00, 0.50,  0.30], 0.50, [1.00, 0.70, 0.70]),  // left, coral
            ([  2.00, 0.50,  0.30], 0.50, [0.70, 0.85, 1.00]),  // right, sky blue
            ([ -1.20, 0.35, -1.60], 0.35, [0.80, 0.70, 1.00]),  // back-left, lavender
            ([  1.20, 0.35, -1.60], 0.35, [1.00, 0.88, 0.60]),  // back-right, amber
            ([ -0.55, 0.22,  1.20], 0.22, [0.70, 1.00, 0.80]),  // front-left, mint
            ([  0.55, 0.22,  1.20], 0.22, [1.00, 0.75, 0.90]),  // front-right, rose
        ];

        for &(center, radius, color) in spheres {
            add_sphere(
                center, radius, color,
                0.0,  // metallic  — glass is a dielectric
                0.04, // roughness — very smooth glass
                0.95, // transmission
                32, 32,
                &mut vertices, &mut indices,
            );
        }

        Scene {
            vertices,
            indices,
            camera: Camera {
                position: Point3::new(0.0, 2.5, 6.5),
                target:   Point3::new(0.0, 0.5, 0.0),
                up:       Vector3::new(0.0, 1.0, 0.0),
                fov_deg:  45.0,
                aspect:   1280.0 / 720.0,
                near:     0.1,
                far:      100.0,
            },
            point_lights: vec![
                PointLight {
                    position:  [6.0, 8.0, 5.0],
                    color:     [1.0, 0.95, 0.85],
                    intensity: 600.0,
                },
                PointLight {
                    position:  [-5.0, 3.0, 2.0],
                    color:     [0.5, 0.7, 1.0],
                    intensity: 250.0,
                },
                PointLight {
                    position:  [2.0, -2.0, -6.0],
                    color:     [0.9, 0.3, 1.0],
                    intensity: 120.0,
                },
            ],
            dir_light: Some(DirectionalLight {
                direction: [-0.3, -0.7, -0.5],
                color:     [1.0, 0.98, 0.90],
                intensity: 1.5,
            }),
            model_rotation: 0.0,
        }
    }
}
