use cgmath::{Point3, Vector3};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub color:    [f32; 3],
}

pub struct Light {
    pub position: [f32; 3],
    pub color:    [f32; 3],
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferObject {
    pub model:       [[f32; 4]; 4],
    pub view:        [[f32; 4]; 4],
    pub proj:        [[f32; 4]; 4],
    pub light_pos:   [f32; 3],
    pub _pad0:       f32,
    pub light_color: [f32; 3],
    pub _pad1:       f32,
    pub view_pos:    [f32; 3],
    pub _pad2:       f32,
}

pub struct Scene {
    pub vertices:       Vec<Vertex>,
    pub indices:        Vec<u32>,
    pub camera:         Camera,
    pub light:          Light,
    pub model_rotation: f32, // radians, updated each frame
}

impl Scene {
    pub fn cube() -> Self {
        // All faces use the same solid white albedo; lighting provides all the variation.
        let faces: &[([f32; 3], [[f32; 3]; 4])] = &[
            (
                [0.0,  0.0,  1.0],  // +Z front
                [[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]],
            ),
            (
                [0.0,  0.0, -1.0],  // -Z back
                [[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5]],
            ),
            (
                [-1.0, 0.0,  0.0],  // -X left
                [[-0.5,-0.5,-0.5],[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5]],
            ),
            (
                [ 1.0, 0.0,  0.0],  // +X right
                [[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5]],
            ),
            (
                [0.0,  1.0,  0.0],  // +Y top
                [[-0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5]],
            ),
            (
                [0.0, -1.0,  0.0],  // -Y bottom
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
            light: Light {
                position: [4.0, 5.0, 3.0],
                color:    [1.0, 0.95, 0.85], // warm white
            },
            model_rotation: 0.0,
        }
    }
}
