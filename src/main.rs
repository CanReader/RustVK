mod renderer;
mod scene;

use renderer::VulkanRenderer;
use scene::Scene;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
struct App {
    window:   Option<Window>,
    renderer: Option<VulkanRenderer>,
    scene:    Scene,
}

impl App {
    fn new() -> Self {
        Self {
            window:   None,
            renderer: None,
            scene:    Scene::spheres(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes() 
            .with_title("RustVK")
            .with_inner_size(winit::dpi::LogicalSize::new(1920u32, 1080u32)).with_blur(true).with_active(true);

        let window = event_loop.create_window(attrs).expect("Failed to create window");

        log::info!("The window has been created successfully!");

        match VulkanRenderer::new(&window) {
            Ok(r) => self.renderer = Some(r),
            Err(e) => {
                log::error!("Failed to create Vulkan renderer: {}", e);
                event_loop.exit();
                return;
            }
        }

        log::info!("The vulkan renderer has been created successfully!");

        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id:        WindowId,
        event:      WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.scene.camera.aspect = size.width as f32 / size.height as f32;
                    if let Some(r) = self.renderer.as_mut() {
                        r.resize(size.width, size.height);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // Static scene — no model rotation applied.

                if let Some(r) = self.renderer.as_mut() {
                    if let Err(e) = r.render(&self.scene) {
                        log::error!("Render error: {}", e);
                        event_loop.exit();
                    }
                }
            }

            _ => {}
        }
    }

    // Request a redraw every time there's nothing else to process → continuous render loop
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
