use std::collections::HashSet;
use cgmath::{Matrix4, Point3, Vector3};
use winit::keyboard::KeyCode;

/// First-person free-look flying camera.
///
/// Convention: yaw=0, pitch=0 looks toward -Z.
/// Positive yaw rotates right; positive pitch looks up.
pub struct FreeCamera {
    pub position:    Point3<f32>,
    pub yaw:         f32,   // radians
    pub pitch:       f32,   // radians, clamped to ±89°
    pub fov_deg:     f32,
    pub aspect:      f32,
    pub near:        f32,
    pub far:         f32,
    pub move_speed:  f32,
    pub sensitivity: f32,
}

impl FreeCamera {
    pub fn new(
        position: Point3<f32>,
        yaw:      f32,
        pitch:    f32,
        fov_deg:  f32,
        aspect:   f32,
    ) -> Self {
        Self {
            position, yaw, pitch, fov_deg, aspect,
            near:        0.1,
            far:         200.0,
            move_speed:  6.0,
            sensitivity: 0.0018,
        }
    }

    /// Unit forward vector derived from yaw and pitch.
    pub fn forward(&self) -> Vector3<f32> {
        let (sy, cy) = (self.yaw.sin(), self.yaw.cos());
        let (sp, cp) = (self.pitch.sin(), self.pitch.cos());
        Vector3::new(cp * sy, sp, -cp * cy)
    }

    /// Horizontal right vector (always in the XZ plane).
    pub fn right(&self) -> Vector3<f32> {
        let (sy, cy) = (self.yaw.sin(), self.yaw.cos());
        Vector3::new(cy, 0.0, sy)
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let target = self.position + self.forward();
        Matrix4::look_at_rh(self.position, target, Vector3::new(0.0, 1.0, 0.0))
    }

    /// Apply raw mouse delta (pixels). dx > 0 = look right, dy > 0 = look down.
    pub fn update_mouse(&mut self, dx: f32, dy: f32) {
        self.yaw   += dx * self.sensitivity;
        self.pitch -= dy * self.sensitivity;
        let limit = 89.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    /// Move the camera according to currently-held keys and elapsed seconds.
    pub fn update_movement(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        let speed = self.move_speed * dt;
        let fwd   = self.forward();
        let right = self.right();
        let up    = Vector3::new(0.0_f32, 1.0, 0.0);

        if keys.contains(&KeyCode::KeyW)     { self.position += fwd   * speed; }
        if keys.contains(&KeyCode::KeyS)     { self.position -= fwd   * speed; }
        if keys.contains(&KeyCode::KeyD)     { self.position += right * speed; }
        if keys.contains(&KeyCode::KeyA)     { self.position -= right * speed; }
        if keys.contains(&KeyCode::Space)    { self.position += up    * speed; }
        if keys.contains(&KeyCode::ShiftLeft){ self.position -= up    * speed; }
    }

    pub fn is_moving(&self, keys: &HashSet<KeyCode>) -> bool {
        [KeyCode::KeyW, KeyCode::KeyS, KeyCode::KeyA, KeyCode::KeyD,
         KeyCode::Space, KeyCode::ShiftLeft]
            .iter()
            .any(|k| keys.contains(k))
    }
}
