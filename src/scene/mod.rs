use cgmath::{Point3, Vector3};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub color:    [f32; 3],
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

pub struct Material {
    pub albedo:    [f32; 3],
    pub metallic:  f32,
    pub roughness: f32,
    pub ao:        f32,
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

// std140-compatible UBO. Each row is a vec4 (16 bytes).
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
    pub albedo_metallic:   [f32; 4],                        // offset 208
    pub roughness_ao:      [f32; 4],                        // offset 224
    pub point_light_pos:   [[f32; 4]; MAX_POINT_LIGHTS],    // offset 240
    pub point_light_color: [[f32; 4]; MAX_POINT_LIGHTS],    // offset 304
    pub light_counts:      [f32; 4],                        // offset 368
    pub dir_light_dir:     [f32; 4],                        // offset 384
    pub dir_light_color:   [f32; 4],                        // offset 400
}

pub struct Scene {
    pub vertices:       Vec<Vertex>,
    pub indices:        Vec<u32>,
    pub camera:         Camera,
    pub material:       Material,
    pub point_lights:   Vec<PointLight>,
    pub dir_light:      Option<DirectionalLight>,
    pub model_rotation: f32,
}

impl Scene {
    pub fn cube() -> Self {
        let faces: &[([f32; 3], [[f32; 3]; 4])] = &[
            (
                [0.0,  0.0,  1.0],
                [[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]],
            ),
            (
                [0.0,  0.0, -1.0],
                [[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5]],
            ),
            (
                [-1.0, 0.0,  0.0],
                [[-0.5,-0.5,-0.5],[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5]],
            ),
            (
                [ 1.0, 0.0,  0.0],
                [[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5]],
            ),
            (
                [0.0,  1.0,  0.0],
                [[-0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5]],
            ),
            (
                [0.0, -1.0,  0.0],
                [[-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5],[-0.5,-0.5, 0.5]],
            ),
        ];

        let mut vertices = Vec::with_capacity(24);
        let mut indices  = Vec::with_capacity(36);

        for (normal, corners) in faces {
            let base = vertices.len() as u32;
            for &pos in corners {
                vertices.push(Vertex {
                    position: pos,
                    normal:   *normal,
                    color:    [1.0, 1.0, 1.0],
                });
            }
            indices.extend_from_slice(&[
                base, base + 1, base + 2,
                base, base + 2, base + 3,
            ]);
        }

        Scene {
            vertices,
            indices,
            camera: Camera {
                position: Point3::new(0.0, 2.0, 5.0),
                target:   Point3::new(0.0, 0.0, 0.0),
                up:       Vector3::new(0.0, 1.0, 0.0),
                fov_deg:  45.0,
                aspect:   1280.0 / 720.0,
                near:     0.1,
                far:      100.0,
            },
            material: Material {
                albedo:    [0.8, 0.8, 0.82],
                metallic:  0.95,
                roughness: 0.2,
                ao:        1.0,
            },
            point_lights: vec![
                PointLight {
                    position:  [5.0,  8.0,  4.0],
                    color:     [1.0,  0.9,  0.7],
                    intensity: 200.0,
                },
                PointLight {
                    position:  [-4.0, 2.0, -3.0],
                    color:     [0.5,  0.7,  1.0],
                    intensity: 80.0,
                },
                PointLight {
                    position:  [0.0, -4.0, -5.0],
                    color:     [1.0,  0.4,  0.8],
                    intensity: 50.0,
                },
            ],
            dir_light: Some(DirectionalLight {
                direction: [-0.4, -0.8, -0.4],
                color:     [1.0,  0.95, 0.85],
                intensity: 1.2,
            }),
            model_rotation: 0.0,
        }
    }
}
