use ash::vk;
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use super::device::VulkanDevice;
use super::swapchain::VulkanSwapchain;
use super::command::VulkanCommandPool;

/// Owns the imgui context, winit platform glue, and the Vulkan renderer.
/// Also owns the overlay render pass and per-swapchain-image framebuffers.
///
/// Drop order: this struct must be dropped before the VulkanDevice that
/// created it. Declare it first in VulkanRenderer so Rust drops it first.
pub struct ImGuiLayer {
    pub context:    imgui::Context,
    pub platform:   WinitPlatform,
    pub renderer:   imgui_rs_vulkan_renderer::Renderer,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    // Kept for Drop so we can clean up without a device reference.
    device_ref:     ash::Device,
}

impl ImGuiLayer {
    /// Create the layer. The render pass uses LOAD_OP_LOAD so the ray-traced
    /// blit content underneath is preserved. finalLayout = PRESENT_SRC_KHR
    /// means we no longer need a manual barrier after the render pass ends.
    pub fn new(
        instance:     &ash::Instance,
        device:       &VulkanDevice,
        swapchain:    &VulkanSwapchain,
        command_pool: &VulkanCommandPool,
        window:       &winit::window::Window,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut context = imgui::Context::create();
        context.set_ini_filename(None);
        context.io_mut().config_flags |= imgui::ConfigFlags::NO_MOUSE_CURSOR_CHANGE;

        let mut platform = WinitPlatform::new(&mut context);
        platform.attach_window(context.io_mut(), window, HiDpiMode::Default);

        let render_pass = Self::create_render_pass(&device.device, swapchain.format)?;

        let framebuffers = Self::create_framebuffers(
            &device.device,
            swapchain,
            render_pass,
        )?;

        use imgui_rs_vulkan_renderer::{Options, Renderer};
        use super::sync::MAX_FRAMES_IN_FLIGHT;

        let renderer = Renderer::with_default_allocator(
            instance,
            device.physical_device,
            device.device.clone(),
            device.graphics_queue,
            command_pool.pool,
            render_pass,
            &mut context,
            Some(Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                ..Default::default()
            }),
        )?;

        Ok(Self {
            context,
            platform,
            renderer,
            render_pass,
            framebuffers,
            device_ref: device.device.clone(),
        })
    }

    /// Recreate framebuffers after a swapchain resize. The render pass itself
    /// does not depend on the swapchain extent, so it is reused.
    pub fn rebuild_framebuffers(
        &mut self,
        device:    &VulkanDevice,
        swapchain: &VulkanSwapchain,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            for fb in self.framebuffers.drain(..) {
                device.device.destroy_framebuffer(fb, None);
            }
        }
        self.framebuffers = Self::create_framebuffers(&device.device, swapchain, self.render_pass)?;
        Ok(())
    }

    /// Pump the winit->imgui event translation and start a new frame.
    /// Returns a reference to the frame UI builder.
    pub fn begin_frame<'ui>(
        &'ui mut self,
        window: &winit::window::Window,
        dt_ms:  f32,
    ) -> &'ui imgui::Ui {
        self.context.io_mut().delta_time = (dt_ms / 1000.0).max(1e-6);
        self.platform.prepare_frame(self.context.io_mut(), window)
            .expect("WinitPlatform::prepare_frame failed");
        self.context.new_frame()
    }

    // ---- private helpers ---------------------------------------------------

    fn create_render_pass(
        device: &ash::Device,
        format: vk::Format,
    ) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        // Single color attachment wrapping the swapchain image.
        // loadOp = LOAD so the ray-traced blit is not cleared.
        // finalLayout = PRESENT_SRC_KHR handles the layout transition for presentation.
        let attachment = vk::AttachmentDescription {
            format,
            samples:        vk::SampleCountFlags::TYPE_1,
            load_op:        vk::AttachmentLoadOp::LOAD,
            store_op:       vk::AttachmentStoreOp::STORE,
            stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout:   vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point:      vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count:   1,
            p_color_attachments:      &color_ref,
            ..Default::default()
        };

        // Dependency: wait for the transfer blit to finish writing before the
        // color attachment output stage reads/writes the same image.
        let dependency = vk::SubpassDependency {
            src_subpass:    vk::SUBPASS_EXTERNAL,
            dst_subpass:    0,
            src_stage_mask:  vk::PipelineStageFlags::TRANSFER,
            dst_stage_mask:  vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        };

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments:    &attachment,
            subpass_count:    1,
            p_subpasses:      &subpass,
            dependency_count: 1,
            p_dependencies:   &dependency,
            ..Default::default()
        };

        let rp = unsafe { device.create_render_pass(&create_info, None)? };
        Ok(rp)
    }

    fn create_framebuffers(
        device:      &ash::Device,
        swapchain:   &VulkanSwapchain,
        render_pass: vk::RenderPass,
    ) -> Result<Vec<vk::Framebuffer>, Box<dyn std::error::Error>> {
        swapchain.image_views.iter().map(|&view| {
            let attachments = [view];
            let info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments:    attachments.as_ptr(),
                width:            swapchain.extent.width,
                height:           swapchain.extent.height,
                layers:           1,
                ..Default::default()
            };
            unsafe { device.create_framebuffer(&info, None).map_err(|e| e.into()) }
        }).collect()
    }
}

impl Drop for ImGuiLayer {
    fn drop(&mut self) {
        unsafe {
            // renderer must be dropped first: it frees descriptor sets / pipelines
            // that depend on the render pass. Rust drops struct fields in declaration
            // order, so renderer is dropped before render_pass and framebuffers here.
            // We do it manually to be explicit about ordering.
            for fb in self.framebuffers.drain(..) {
                self.device_ref.destroy_framebuffer(fb, None);
            }
            self.device_ref.destroy_render_pass(self.render_pass, None);
        }
    }
}
