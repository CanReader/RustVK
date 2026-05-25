mod camera;
mod renderer;
mod scene;

use renderer::VulkanRenderer;
use scene::Scene;

use std::collections::HashSet;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

struct App {
    window:         Option<Window>,
    renderer:       Option<VulkanRenderer>,
    scene:          Scene,
    held_keys:      HashSet<KeyCode>,
    mouse_captured: bool,
    mouse_delta:    (f64, f64),
    last_frame:     std::time::Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window:         None,
            renderer:       None,
            scene:          Scene::spheres(),
            held_keys:      HashSet::new(),
            mouse_captured: false,
            mouse_delta:    (0.0, 0.0),
            last_frame:     std::time::Instant::now(),
        }
    }

    fn grab_cursor(&mut self) {
        let Some(w) = &self.window else { return };
        let _ = w.set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined));
        w.set_cursor_visible(false);
        self.mouse_captured = true;
    }

    fn release_cursor(&mut self) {
        let Some(w) = &self.window else { return };
        let _ = w.set_cursor_grab(CursorGrabMode::None);
        w.set_cursor_visible(true);
        self.mouse_captured = false;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("RustVK  |  Click to capture mouse  |  Esc to release")
            .with_inner_size(winit::dpi::LogicalSize::new(1920u32, 1080u32));

        let window = event_loop.create_window(attrs).expect("Failed to create window");

        match VulkanRenderer::new(&window) {
            Ok(r) => self.renderer = Some(r),
            Err(e) => {
                log::error!("Failed to create Vulkan renderer: {}", e);
                event_loop.exit();
                return;
            }
        }

        self.window = Some(window);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id:  DeviceId,
        event:       DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.mouse_captured {
                self.mouse_delta.0 += dx;
                self.mouse_delta.1 += dy;
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id:        WindowId,
        event:      WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Focused(false) => {
                self.release_cursor();
                self.held_keys.clear();
            }

            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if let PhysicalKey::Code(key) = key_event.physical_key {
                    match key_event.state {
                        ElementState::Pressed  => { self.held_keys.insert(key); }
                        ElementState::Released => { self.held_keys.remove(&key); }
                    }
                    if key == KeyCode::Escape && key_event.state == ElementState::Pressed {
                        self.release_cursor();
                    }
                }
            }

            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !self.mouse_captured {
                    self.grab_cursor();
                }
            }

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.scene.camera.aspect = size.width as f32 / size.height as f32;
                    if let Some(r) = self.renderer.as_mut() {
                        r.resize(size.width, size.height);
                        r.reset_accumulation();
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt  = now.duration_since(self.last_frame)
                              .as_secs_f32()
                              .min(0.05);   // cap at 50 ms so a lag spike doesn't teleport
                self.last_frame = now;

                let moved_mouse = self.mouse_delta != (0.0, 0.0);
                let moved_keys  = self.scene.camera.is_moving(&self.held_keys);

                if moved_mouse {
                    let (dx, dy) = self.mouse_delta;
                    self.scene.camera.update_mouse(dx as f32, dy as f32);
                    self.mouse_delta = (0.0, 0.0);
                }
                self.scene.camera.update_movement(&self.held_keys, dt);

                if let Some(r) = self.renderer.as_mut() {
                    if moved_mouse || moved_keys {
                        r.reset_accumulation();
                    }
                    if let Err(e) = r.render(&self.scene) {
                        log::error!("Render error: {}", e);
                        event_loop.exit();
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
