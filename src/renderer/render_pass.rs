use ash::vk;
use super::device::VulkanDevice;
use super::msaa::MSAA_SAMPLES;

pub struct VulkanRenderPass {
    pub render_pass: vk::RenderPass,
    device_ref:      ash::Device,
}

impl VulkanRenderPass {
    pub fn new(
        device:       &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Attachment 0 — MSAA color (rendered into; not stored, just resolved)
        let msaa_color = vk::AttachmentDescription {
            format:           color_format,
            samples:          MSAA_SAMPLES,
            load_op:          vk::AttachmentLoadOp::CLEAR,
            store_op:         vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout:   vk::ImageLayout::UNDEFINED,
            final_layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        // Attachment 1 — MSAA depth (same sample count as color)
        let depth_att = vk::AttachmentDescription {
            format:           depth_format,
            samples:          MSAA_SAMPLES,
            load_op:          vk::AttachmentLoadOp::CLEAR,
            store_op:         vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout:   vk::ImageLayout::UNDEFINED,
            final_layout:     vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        // Attachment 2 — Resolve target (1× swapchain image; this is what gets presented)
        let resolve_att = vk::AttachmentDescription {
            format:           color_format,
            samples:          vk::SampleCountFlags::TYPE_1,
            load_op:          vk::AttachmentLoadOp::DONT_CARE,
            store_op:         vk::AttachmentStoreOp::STORE,
            stencil_load_op:  vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout:   vk::ImageLayout::UNDEFINED,
            final_layout:     vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let attachments = [msaa_color, depth_att, resolve_att];

        let color_ref = [vk::AttachmentReference {
            attachment: 0,
            layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_ref = vk::AttachmentReference {
            attachment: 1,
            layout:     vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let resolve_ref = [vk::AttachmentReference {
            attachment: 2,
            layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpass = vk::SubpassDescription {
            pipeline_bind_point:        vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count:     color_ref.len() as u32,
            p_color_attachments:        color_ref.as_ptr(),
            p_resolve_attachments:      resolve_ref.as_ptr(),
            p_depth_stencil_attachment: &depth_ref,
            ..Default::default()
        };

        let dependency = vk::SubpassDependency {
            src_subpass:     vk::SUBPASS_EXTERNAL,
            dst_subpass:     0,
            src_stage_mask:  vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                           | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask:  vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                           | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                           | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments:    attachments.as_ptr(),
            subpass_count:    1,
            p_subpasses:      &subpass,
            dependency_count: 1,
            p_dependencies:   &dependency,
            ..Default::default()
        };

        let render_pass = unsafe { device.device.create_render_pass(&render_pass_info, None)? };
        Ok(Self { render_pass, device_ref: device.device.clone() })
    }
}

impl Drop for VulkanRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_render_pass(self.render_pass, None);
        }
    }
}
