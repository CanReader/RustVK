pub mod instance;
pub mod device;
pub mod swapchain;
pub mod render_pass;
pub mod pipeline;
pub mod framebuffer;
pub mod buffer;
pub mod descriptor;
pub mod command;
pub mod sync;
pub mod depth;
pub mod msaa;

use ash::vk;
use cgmath::{Matrix4, Rad, Deg, perspective};

use instance::VulkanInstance;
use device::VulkanDevice;
use swapchain::VulkanSwapchain;
use render_pass::VulkanRenderPass;
use pipeline::VulkanPipeline;
use framebuffer::VulkanFramebuffers;
use buffer::VulkanBuffer;
use descriptor::VulkanDescriptorSets;
use command::VulkanCommandPool;
use sync::{VulkanSync, MAX_FRAMES_IN_FLIGHT};
use depth::VulkanDepthBuffer;
use msaa::{VulkanMsaaBuffer, MSAA_SAMPLES};

use crate::scene::{Scene, UniformBufferObject, MAX_POINT_LIGHTS};
use std::mem::size_of;

// Wraps the instance-level surface so it implements Drop correctly.
// Declared AFTER device so it's destroyed after the device but before the instance.
struct VulkanSurface {
    surface: vk::SurfaceKHR,
    loader:  ash::khr::surface::Instance,
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

// ── Field declaration order == drop order ─────────────────────────────────────
// Rust drops struct fields in declaration order. Innermost resources (those that
// depend on others) must be declared FIRST so they are destroyed first.
pub struct VulkanRenderer {
    // ── device-level children (dropped first) ─────────────────────────────
    sync:                VulkanSync,
    uniform_buffers:     Vec<VulkanBuffer>,
    index_buffer:        VulkanBuffer,
    vertex_buffer:       VulkanBuffer,
    framebuffers:        VulkanFramebuffers,  // references msaa + depth views
    pipeline:            VulkanPipeline,
    descriptor_sets:     VulkanDescriptorSets,
    msaa_buffer:         VulkanMsaaBuffer,    // must drop after framebuffers
    depth_buffer:        VulkanDepthBuffer,   // must drop after framebuffers
    render_pass:         VulkanRenderPass,
    swapchain:           VulkanSwapchain,
    command_pool:        VulkanCommandPool,

    // ── plain data ────────────────────────────────────────────────────────
    acquire_sem_index:   usize,
    index_count:         u32,
    current_frame:       usize,
    framebuffer_resized: bool,
    window_width:        u32,
    window_height:       u32,

    // ── device (dropped after all device-level children) ──────────────────
    device:              VulkanDevice,

    // ── surface (dropped after device, before instance) ───────────────────
    surface:             VulkanSurface,

    // ── instance (dropped last) ───────────────────────────────────────────
    instance:            VulkanInstance,
}

impl VulkanRenderer {
    pub fn new(window: &winit::window::Window) -> Result<Self, Box<dyn std::error::Error>> {
        let enable_validation = cfg!(debug_assertions);

        let instance = VulkanInstance::new(window, enable_validation)?;

        let surface_loader = ash::khr::surface::Instance::new(&instance.entry, &instance.instance);
        let raw_surface = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.instance,
                raw_window_handle::HasRawDisplayHandle::raw_display_handle(window)?,
                raw_window_handle::HasRawWindowHandle::raw_window_handle(window)?,
                None,
            )?
        };

        let device = VulkanDevice::new(&instance.instance, &surface_loader, raw_surface)?;

        let size = window.inner_size();
        let swapchain = VulkanSwapchain::new(
            &instance.instance,
            &device,
            &surface_loader,
            raw_surface,
            size.width,
            size.height,
            vk::SwapchainKHR::null(),
        )?;

        let msaa_buffer  = VulkanMsaaBuffer::new(&device, swapchain.extent, swapchain.format)?;
        let depth_buffer = VulkanDepthBuffer::new(&instance.instance, &device, swapchain.extent, MSAA_SAMPLES)?;
        let render_pass  = VulkanRenderPass::new(&device, swapchain.format, depth_buffer.format)?;
        let command_pool = VulkanCommandPool::new(&device, MAX_FRAMES_IN_FLIGHT)?;

        let cube        = Scene::cube();
        let index_count = cube.indices.len() as u32;

        let vertex_buffer = VulkanBuffer::new_vertex(&device, &command_pool, &cube.vertices)?;
        let index_buffer  = VulkanBuffer::new_index(&device, &command_pool, &cube.indices)?;

        let ubo_size = size_of::<UniformBufferObject>();
        let uniform_buffers: Vec<VulkanBuffer> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| VulkanBuffer::new_uniform(&device, ubo_size))
            .collect::<Result<_, _>>()?;

        let descriptor_sets = VulkanDescriptorSets::new(
            &device,
            &uniform_buffers,
            ubo_size as vk::DeviceSize,
        )?;

        let pipeline     = VulkanPipeline::new(&device, &render_pass, &descriptor_sets, swapchain.extent)?;
        let framebuffers = VulkanFramebuffers::new(&device, &swapchain, &render_pass, &depth_buffer, &msaa_buffer)?;
        let sync         = VulkanSync::new(&device, swapchain.images.len())?;

        Ok(Self {
            sync,
            uniform_buffers,
            index_buffer,
            vertex_buffer,
            framebuffers,
            pipeline,
            descriptor_sets,
            msaa_buffer,
            depth_buffer,
            render_pass,
            swapchain,
            command_pool,
            acquire_sem_index: 0,
            index_count,
            current_frame: 0,
            framebuffer_resized: false,
            window_width:  size.width,
            window_height: size.height,
            device,
            surface: VulkanSurface { surface: raw_surface, loader: surface_loader },
            instance,
        })
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    pub fn render(&mut self, scene: &Scene) -> Result<(), Box<dyn std::error::Error>> {
        let frame = self.current_frame;

        unsafe {
            self.device.device.wait_for_fences(
                &[self.sync.in_flight_fences[frame]],
                true,
                u64::MAX,
            )?;
        }

        // Rotate through acquire semaphores independently from the frame index.
        // A binary semaphore passed to acquire_next_image is "owned" by the
        // presentation engine until that image is displayed and retired. Reusing
        // the same semaphore too quickly (before the engine re-signals it for
        // the same image slot) is a validation error and causes GPU faults.
        // With NUM_ACQUIRE_SEMS = MAX_FRAMES_IN_FLIGHT + 1, there is always at
        // least one semaphore that is not pending a presentation signal.
        let acquire_sem = self.sync.image_available_semaphores[self.acquire_sem_index];
        self.acquire_sem_index =
            (self.acquire_sem_index + 1) % self.sync.image_available_semaphores.len();

        let (image_index, _suboptimal) = unsafe {
            match self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                acquire_sem,
                vk::Fence::null(),
            ) {
                Ok(r) => r,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                    return Err(vk::Result::ERROR_SURFACE_LOST_KHR.into());
                }
                Err(e) => return Err(e.into()),
            }
        };

        unsafe {
            self.device.device.reset_fences(&[self.sync.in_flight_fences[frame]])?;
        }

        self.update_uniform_buffer(frame, scene)?;

        let cb = self.command_pool.command_buffers[frame];
        unsafe {
            self.device.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
        }
        self.record_command_buffer(cb, image_index as usize, frame)?;

        let wait_semaphores   = [acquire_sem];
        // Index by image_index (not frame): a render_finished semaphore is only safe to
        // re-signal after the presentation engine has released that image slot, which
        // happens exactly when the image is re-acquired. Using frame-index here would
        // reuse the semaphore for a different image while the engine may still hold it.
        let signal_semaphores = [self.sync.render_finished_semaphores[image_index as usize]];
        let wait_stages       = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count:   wait_semaphores.len() as u32,
            p_wait_semaphores:      wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask:  wait_stages.as_ptr(),
            command_buffer_count:   1,
            p_command_buffers:      &cb,
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores:    signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.device.queue_submit(
                self.device.graphics_queue,
                &[submit_info],
                self.sync.in_flight_fences[frame],
            )?;
        }

        let swapchains    = [self.swapchain.swapchain];
        let image_indices = [image_index];
        let present_info  = vk::PresentInfoKHR {
            wait_semaphore_count: signal_semaphores.len() as u32,
            p_wait_semaphores:    signal_semaphores.as_ptr(),
            swapchain_count:      swapchains.len() as u32,
            p_swapchains:         swapchains.as_ptr(),
            p_image_indices:      image_indices.as_ptr(),
            ..Default::default()
        };

        let present_result = unsafe {
            self.swapchain.swapchain_loader.queue_present(self.device.present_queue, &present_info)
        };

        let needs_recreate = match present_result {
            Ok(true)  => true,
            Ok(false) => self.framebuffer_resized,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                return Err(vk::Result::ERROR_SURFACE_LOST_KHR.into());
            }
            Err(e) => return Err(e.into()),
        };

        if needs_recreate {
            self.framebuffer_resized = false;
            self.recreate_swapchain()?;
        }

        self.current_frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.window_width  = width;
        self.window_height = height;
        self.framebuffer_resized = true;
    }

    // ── Internals ──────────────────────────────────────────────────────────────

    fn update_uniform_buffer(
        &self,
        frame: usize,
        scene: &Scene,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rot = scene.model_rotation;
        let model = Matrix4::from_angle_y(Rad(rot))
            * Matrix4::from_angle_x(Rad(rot * 0.35));

        let view = Matrix4::look_at_rh(
            scene.camera.position,
            scene.camera.target,
            scene.camera.up,
        );

        let mut proj = perspective(
            Deg(scene.camera.fov_deg),
            scene.camera.aspect,
            scene.camera.near,
            scene.camera.far,
        );
        proj[1][1] *= -1.0;

        let num_point = scene.point_lights.len().min(MAX_POINT_LIGHTS);
        let mut point_light_pos   = [[0f32; 4]; MAX_POINT_LIGHTS];
        let mut point_light_color = [[0f32; 4]; MAX_POINT_LIGHTS];
        for (i, pl) in scene.point_lights.iter().take(num_point).enumerate() {
            point_light_pos[i]   = [pl.position[0], pl.position[1], pl.position[2], pl.intensity];
            point_light_color[i] = [pl.color[0],    pl.color[1],    pl.color[2],    0.0];
        }

        let (has_dir, dir_dir, dir_color) = match &scene.dir_light {
            Some(dl) => (
                1.0f32,
                [dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity],
                [dl.color[0],     dl.color[1],     dl.color[2],     0.0f32],
            ),
            None => (0.0, [0f32; 4], [0f32; 4]),
        };

        let m = &scene.material;
        let cp = scene.camera.position;
        let ubo = UniformBufferObject {
            model:             matrix4_to_array(model),
            view:              matrix4_to_array(view),
            proj:              matrix4_to_array(proj),
            view_pos:          [cp.x, cp.y, cp.z, 0.0],
            albedo_metallic:   [m.albedo[0], m.albedo[1], m.albedo[2], m.metallic],
            roughness_ao:      [m.roughness, m.ao, 0.0, 0.0],
            point_light_pos,
            point_light_color,
            light_counts:      [num_point as f32, has_dir, 0.0, 0.0],
            dir_light_dir:     dir_dir,
            dir_light_color:   dir_color,
        };

        self.uniform_buffers[frame].update_uniform(&self.device, &ubo)
    }

    fn record_command_buffer(
        &self,
        cb:          vk::CommandBuffer,
        image_index: usize,
        frame:       usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let begin_info = vk::CommandBufferBeginInfo { ..Default::default() };
        unsafe { self.device.device.begin_command_buffer(cb, &begin_info)? };

        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        let render_pass_begin = vk::RenderPassBeginInfo {
            render_pass:       self.render_pass.render_pass,
            framebuffer:       self.framebuffers.framebuffers[image_index],
            render_area:       vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values:    clear_values.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.device.cmd_begin_render_pass(cb, &render_pass_begin, vk::SubpassContents::INLINE);
            self.device.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            self.device.device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer.buffer], &[0]);
            self.device.device.cmd_bind_index_buffer(cb, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
            self.device.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout,
                0,
                &[self.descriptor_sets.sets[frame]],
                &[],
            );
            self.device.device.cmd_draw_indexed(cb, self.index_count, 1, 0, 0, 0);
            self.device.device.cmd_end_render_pass(cb);
            self.device.device.end_command_buffer(cb)?;
        }

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { self.device.device.device_wait_idle()? };

        let old_handle = self.swapchain.swapchain;
        let new_swapchain = VulkanSwapchain::new(
            &self.instance.instance,
            &self.device,
            &self.surface.loader,
            self.surface.surface,
            self.window_width,
            self.window_height,
            old_handle,
        )?;

        let new_msaa     = VulkanMsaaBuffer::new(&self.device, new_swapchain.extent, new_swapchain.format)?;
        let new_depth    = VulkanDepthBuffer::new(&self.instance.instance, &self.device, new_swapchain.extent, MSAA_SAMPLES)?;
        let new_rp       = VulkanRenderPass::new(&self.device, new_swapchain.format, new_depth.format)?;
        let new_pipeline = VulkanPipeline::new(&self.device, &new_rp, &self.descriptor_sets, new_swapchain.extent)?;
        let new_fb       = VulkanFramebuffers::new(&self.device, &new_swapchain, &new_rp, &new_depth, &new_msaa)?;

        let new_sync = VulkanSync::new(&self.device, new_swapchain.images.len())?;

        // Each assignment drops the old object while the device is still alive.
        self.sync         = new_sync;
        self.pipeline     = new_pipeline;
        self.framebuffers = new_fb;
        self.msaa_buffer  = new_msaa;
        self.depth_buffer = new_depth;
        self.render_pass  = new_rp;
        self.swapchain    = new_swapchain;
        self.acquire_sem_index = 0;

        Ok(())
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device.device_wait_idle();
        }
    }
}

fn matrix4_to_array(m: Matrix4<f32>) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [m[3][0], m[3][1], m[3][2], m[3][3]],
    ]
}
