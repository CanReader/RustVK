use ash::vk;
use super::device::VulkanDevice;
use super::swapchain::VulkanSwapchain;
use super::render_pass::VulkanRenderPass;
use super::depth::VulkanDepthBuffer;
use super::msaa::VulkanMsaaBuffer;

pub struct VulkanFramebuffers {
    pub framebuffers: Vec<vk::Framebuffer>,
    device_ref:       ash::Device,
}

impl VulkanFramebuffers {
    pub fn new(
        device:       &VulkanDevice,
        swapchain:    &VulkanSwapchain,
        render_pass:  &VulkanRenderPass,
        depth_buffer: &VulkanDepthBuffer,
        msaa_buffer:  &VulkanMsaaBuffer,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let framebuffers = swapchain.image_views.iter().map(|&resolve_view| {
            // Attachment order must match the render pass declaration:
            //   0 = MSAA color, 1 = MSAA depth, 2 = resolve (swapchain image)
            let attachments = [
                msaa_buffer.image_view,
                depth_buffer.image_view,
                resolve_view,
            ];

            let fb_info = vk::FramebufferCreateInfo {
                render_pass:      render_pass.render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments:    attachments.as_ptr(),
                width:            swapchain.extent.width,
                height:           swapchain.extent.height,
                layers:           1,
                ..Default::default()
            };

            unsafe { device.device.create_framebuffer(&fb_info, None) }
        }).collect::<Result<Vec<_>, _>>()?;

        Ok(Self { framebuffers, device_ref: device.device.clone() })
    }
}

impl Drop for VulkanFramebuffers {
    fn drop(&mut self) {
        unsafe {
            for &fb in &self.framebuffers {
                self.device_ref.destroy_framebuffer(fb, None);
            }
        }
    }
}
