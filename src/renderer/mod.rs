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
pub mod acceleration_structure;
pub mod rt_pipeline;
pub mod rt_resources;

use ash::vk;
use cgmath::{Matrix4, Rad, Deg, perspective, SquareMatrix};

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
use acceleration_structure::AccelerationStructure;
use rt_pipeline::RtPipeline;
use rt_resources::RtResources;

use crate::scene::{Scene, UniformBufferObject, RtUBO, MAX_POINT_LIGHTS};
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
    // ── RT resources (device children, dropped before device) ─────────────
    // Many of these fields exist solely for ownership / Drop order. The
    // `#[allow(dead_code)]` suppresses false "never read" warnings.
    rt_resources:    Option<RtResources>,
    rt_pipeline:     Option<RtPipeline>,
    tlas:            Option<AccelerationStructure>,
    #[allow(dead_code)]
    blas:            Option<AccelerationStructure>,
    rt_ubo_buffers:  Vec<VulkanBuffer>,
    #[allow(dead_code)]
    rt_vertex_buf:   Option<VulkanBuffer>,
    #[allow(dead_code)]
    rt_index_buf:    Option<VulkanBuffer>,

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
    sample_count:        u32,   // RT progressive accumulation frame counter

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

        let init_scene  = Scene::spheres();
        let index_count = init_scene.indices.len() as u32;

        let vertex_buffer = VulkanBuffer::new_vertex(&device, &command_pool, &init_scene.vertices)?;
        let index_buffer  = VulkanBuffer::new_index(&device, &command_pool, &init_scene.indices)?;

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

        // ── RT initialisation (optional) ──────────────────────────────────────
        let (blas, tlas, rt_pipeline, rt_resources, rt_vertex_buf, rt_index_buf, rt_ubo_buffers) =
            if device.rt_supported {
                match Self::init_rt(
                    &device,
                    &command_pool,
                    &init_scene,
                    swapchain.extent,
                ) {
                    Ok(rt) => {
                        log::info!("Ray tracing initialised successfully.");
                        let (b, t, rtp, rtr, rvb, rib, rub) = rt;
                        (Some(b), Some(t), Some(rtp), Some(rtr), Some(rvb), Some(rib), rub)
                    }
                    Err(e) => {
                        log::warn!("RT init failed ({}), falling back to rasterizer.", e);
                        (None, None, None, None, None, None, Vec::new())
                    }
                }
            } else {
                log::info!("RT extensions not supported; using rasterizer.");
                (None, None, None, None, None, None, Vec::new())
            };

        Ok(Self {
            rt_resources,
            rt_pipeline,
            tlas,
            blas,
            rt_ubo_buffers,
            rt_vertex_buf,
            rt_index_buf,
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
            sample_count:  0,
            device,
            surface: VulkanSurface { surface: raw_surface, loader: surface_loader },
            instance,
        })
    }

    /// Initialise all RT-specific resources in one shot.
    /// Returns (blas, tlas, rt_pipeline, rt_resources, rt_vertex_buf, rt_index_buf, rt_ubo_buffers).
    fn init_rt(
        device:   &VulkanDevice,
        cmd_pool: &VulkanCommandPool,
        scene:    &Scene,
        extent:   vk::Extent2D,
    ) -> Result<(
        AccelerationStructure,
        AccelerationStructure,
        RtPipeline,
        RtResources,
        VulkanBuffer,
        VulkanBuffer,
        Vec<VulkanBuffer>,
    ), Box<dyn std::error::Error>> {

        // Vertex/index buffers with RT-compatible usage + device address
        let rt_vertex_buf = VulkanBuffer::new_rt_vertex(device, cmd_pool, &scene.vertices)?;
        let rt_index_buf  = VulkanBuffer::new_rt_index(device, cmd_pool, &scene.indices)?;

        let vertex_address = rt_vertex_buf.device_address(&device.device);
        let index_address  = rt_index_buf.device_address(&device.device);
        let vertex_count   = scene.vertices.len() as u32;
        let index_count    = scene.indices.len() as u32;
        // Vertex struct is 12 floats = 48 bytes
        let vertex_stride  = size_of::<crate::scene::Vertex>() as vk::DeviceSize;

        // BLAS
        let blas = AccelerationStructure::build_blas(
            device, cmd_pool,
            vertex_address, vertex_count, vertex_stride,
            index_address,  index_count,
        )?;

        // TLAS
        let tlas = AccelerationStructure::build_tlas(device, cmd_pool, blas.device_address)?;

        // RT UBO buffers (one per frame in flight)
        let rt_ubo_size = size_of::<RtUBO>();
        let rt_ubo_buffers: Vec<VulkanBuffer> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| VulkanBuffer::new_uniform(device, rt_ubo_size))
            .collect::<Result<_, _>>()?;

        // RT resources (descriptor set, storage images)
        let rt_resources = RtResources::new(
            device, cmd_pool, extent,
            &tlas,
            &rt_ubo_buffers,
            &rt_vertex_buf,
            &rt_index_buf,
        )?;

        // RT pipeline — shader dir is next to the binary; use the same approach as the
        // rasterizer pipeline: embed the path at compile time relative to CARGO_MANIFEST_DIR.
        let shader_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders");

        let rt_pipeline = RtPipeline::new(
            device,
            rt_resources.descriptor_layout,
            shader_dir,
        )?;

        Ok((blas, tlas, rt_pipeline, rt_resources, rt_vertex_buf, rt_index_buf, rt_ubo_buffers))
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

        let cb = self.command_pool.command_buffers[frame];
        unsafe {
            self.device.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
        }

        if self.tlas.is_some() {
            // ── Ray tracing path ───────────────────────────────────────────────
            self.update_rt_uniform_buffer(frame, scene)?;
            self.record_rt_command_buffer(cb, image_index as usize, frame)?;
        } else {
            // ── Rasterizer path ────────────────────────────────────────────────
            self.update_uniform_buffer(frame, scene)?;
            self.record_command_buffer(cb, image_index as usize, frame)?;
        }

        let wait_semaphores   = [acquire_sem];
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
        if self.tlas.is_some() {
            self.sample_count = self.sample_count.wrapping_add(1);
        }
        Ok(())
    }

    pub fn reset_accumulation(&mut self) {
        self.sample_count = 0;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.window_width  = width;
        self.window_height = height;
        self.framebuffer_resized = true;
    }

    // ── Internals ──────────────────────────────────────────────────────────────

    fn update_rt_uniform_buffer(
        &self,
        frame: usize,
        scene: &Scene,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let view = scene.camera.view_matrix();
        let mut proj = perspective(
            Deg(scene.camera.fov_deg),
            scene.camera.aspect,
            scene.camera.near,
            scene.camera.far,
        );
        proj[1][1] *= -1.0;

        let inv_view = view.invert().unwrap_or(Matrix4::identity());
        let inv_proj = proj.invert().unwrap_or(Matrix4::identity());

        let num_point = scene.point_lights.len().min(MAX_POINT_LIGHTS);
        let mut point_light_pos   = [[0f32; 4]; MAX_POINT_LIGHTS];
        let mut point_light_color = [[0f32; 4]; MAX_POINT_LIGHTS];
        for (i, pl) in scene.point_lights.iter().take(num_point).enumerate() {
            point_light_pos[i]   = [pl.position[0], pl.position[1], pl.position[2], pl.intensity];
            point_light_color[i] = [pl.color[0], pl.color[1], pl.color[2], 0.0];
        }

        let (has_dir, dir_dir, dir_color) = match &scene.dir_light {
            Some(dl) => (
                1.0f32,
                [dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity],
                [dl.color[0],     dl.color[1],     dl.color[2],     0.0f32],
            ),
            None => (0.0, [0f32; 4], [0f32; 4]),
        };

        let cp = scene.camera.position;
        let ubo = RtUBO {
            inv_view:          matrix4_to_array(inv_view),
            inv_proj:          matrix4_to_array(inv_proj),
            cam_pos:           [cp.x, cp.y, cp.z, 0.0],
            light_counts:      [num_point as f32, has_dir, 0.0, 0.0],
            point_light_pos,
            point_light_color,
            dir_light_dir:     dir_dir,
            dir_light_color:   dir_color,
            frame_index:       self.sample_count,
            max_bounces:       16,
            _pad:              [0.0; 2],
        };

        // Update this frame's UBO buffer contents only — no descriptor update needed
        // because each frame already has its own descriptor set pointing to its own buffer.
        self.rt_ubo_buffers[frame].update_uniform(&self.device, &ubo)?;
        Ok(())
    }

    fn record_rt_command_buffer(
        &self,
        cb:          vk::CommandBuffer,
        image_index: usize,
        frame:       usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rt_loader  = self.device.rt_pipeline_loader.as_ref().ok_or("no RT loader")?;
        let rt_pipe    = self.rt_pipeline.as_ref().ok_or("no rt_pipeline")?;
        let rt_res     = self.rt_resources.as_ref().ok_or("no rt_resources")?;
        let swapchain_image = self.swapchain.images[image_index];
        let extent     = self.swapchain.extent;

        let begin_info = vk::CommandBufferBeginInfo { ..Default::default() };
        unsafe { self.device.device.begin_command_buffer(cb, &begin_info)?; }

        // Bind RT pipeline + this frame's descriptor set
        unsafe {
            self.device.device.cmd_bind_pipeline(
                cb, vk::PipelineBindPoint::RAY_TRACING_KHR, rt_pipe.pipeline,
            );
            self.device.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                rt_pipe.pipeline_layout,
                0,
                &[rt_res.descriptor_sets[frame]],
                &[],
            );
        }

        // Dispatch rays
        unsafe {
            rt_loader.cmd_trace_rays(
                cb,
                &rt_pipe.rgen_region,
                &rt_pipe.miss_region,
                &rt_pipe.hit_region,
                &rt_pipe.callable_region,
                extent.width,
                extent.height,
                1,
            );
        }

        // ── Barrier: out_image GENERAL -> TRANSFER_SRC_OPTIMAL ───────────────
        let out_to_src = vk::ImageMemoryBarrier {
            old_layout:    vk::ImageLayout::GENERAL,
            new_layout:    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: rt_res.out_image,
            subresource_range: full_color_range(),
            ..Default::default()
        };
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[out_to_src],
            );
        }

        // ── Barrier: swapchain image UNDEFINED -> TRANSFER_DST_OPTIMAL ───────
        let sc_to_dst = vk::ImageMemoryBarrier {
            old_layout:    vk::ImageLayout::UNDEFINED,
            new_layout:    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: swapchain_image,
            subresource_range: full_color_range(),
            ..Default::default()
        };
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[sc_to_dst],
            );
        }

        // ── Blit out_image -> swapchain image ─────────────────────────────────
        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                mip_level:        0,
                base_array_layer: 0,
                layer_count:      1,
            },
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D { x: extent.width as i32, y: extent.height as i32, z: 1 },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                mip_level:        0,
                base_array_layer: 0,
                layer_count:      1,
            },
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D { x: extent.width as i32, y: extent.height as i32, z: 1 },
            ],
        };
        unsafe {
            self.device.device.cmd_blit_image(
                cb,
                rt_res.out_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );
        }

        // ── Barrier: swapchain image TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR ─
        let sc_to_present = vk::ImageMemoryBarrier {
            old_layout:    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout:    vk::ImageLayout::PRESENT_SRC_KHR,
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::empty(),
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: swapchain_image,
            subresource_range: full_color_range(),
            ..Default::default()
        };
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[], &[], &[sc_to_present],
            );
        }

        // ── Transition out_image back to GENERAL for next frame ───────────────
        let src_to_general = vk::ImageMemoryBarrier {
            old_layout:    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout:    vk::ImageLayout::GENERAL,
            src_access_mask: vk::AccessFlags::TRANSFER_READ,
            dst_access_mask: vk::AccessFlags::SHADER_WRITE,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: rt_res.out_image,
            subresource_range: full_color_range(),
            ..Default::default()
        };
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[], &[], &[src_to_general],
            );
            self.device.device.end_command_buffer(cb)?;
        }
        Ok(())
    }

    fn update_uniform_buffer(
        &self,
        frame: usize,
        scene: &Scene,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rot = scene.model_rotation;
        let model = Matrix4::from_angle_y(Rad(rot))
            * Matrix4::from_angle_x(Rad(rot * 0.35));

        let view = scene.camera.view_matrix();
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

        let cp = scene.camera.position;
        let ubo = UniformBufferObject {
            model:             matrix4_to_array(model),
            view:              matrix4_to_array(view),
            proj:              matrix4_to_array(proj),
            view_pos:          [cp.x, cp.y, cp.z, 0.0],
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
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.03, 0.05, 0.12, 1.0] } },
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

fn full_color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask:      vk::ImageAspectFlags::COLOR,
        base_mip_level:   0,
        level_count:      1,
        base_array_layer: 0,
        layer_count:      1,
    }
}
