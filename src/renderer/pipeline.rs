use ash::vk;
use super::device::VulkanDevice;
use super::render_pass::VulkanRenderPass;
use super::descriptor::VulkanDescriptorSets;
use super::msaa::MSAA_SAMPLES;
use crate::scene::Vertex;
use std::mem;

pub struct VulkanPipeline {
    pub pipeline:        vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    device_ref:          ash::Device,
}

impl VulkanPipeline {
    pub fn new(
        device:      &VulkanDevice,
        render_pass: &VulkanRenderPass,
        descriptors: &VulkanDescriptorSets,
        extent:      vk::Extent2D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let vert_spv = include_bytes!("../shaders/vert.spv");
        let frag_spv = include_bytes!("../shaders/frag.spv");

        let vert_module = Self::create_shader_module(&device.device, vert_spv)?;
        let frag_module = Self::create_shader_module(&device.device, frag_spv)?;

        let entry = std::ffi::CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage:  vk::ShaderStageFlags::VERTEX,
                module: vert_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage:  vk::ShaderStageFlags::FRAGMENT,
                module: frag_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
        ];

        // Vertex input
        let binding = vk::VertexInputBindingDescription {
            binding:    0,
            stride:     mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let attributes = [
            vk::VertexInputAttributeDescription {
                location: 0, binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1, binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2, binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 24,
            },
            vk::VertexInputAttributeDescription {
                location: 3, binding: 0,
                format: vk::Format::R32_SFLOAT,
                offset: 36,
            },
            vk::VertexInputAttributeDescription {
                location: 4, binding: 0,
                format: vk::Format::R32_SFLOAT,
                offset: 40,
            },
            vk::VertexInputAttributeDescription {
                location: 5, binding: 0,
                format: vk::Format::R32_SFLOAT,
                offset: 44,
            },
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count:   1,
            p_vertex_binding_descriptions:      &binding,
            vertex_attribute_description_count: attributes.len() as u32,
            p_vertex_attribute_descriptions:    attributes.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology:                 vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0, y: 0.0,
            width:     extent.width  as f32,
            height:    extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports:    &viewport,
            scissor_count:  1,
            p_scissors:     &scissor,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable:        vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode:              vk::PolygonMode::FILL,
            line_width:                1.0,
            cull_mode:                 vk::CullModeFlags::BACK,
            front_face:                vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable:         vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: MSAA_SAMPLES,
            sample_shading_enable: vk::FALSE,
            ..Default::default()
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable:        vk::TRUE,
            depth_write_enable:       vk::TRUE,
            depth_compare_op:         vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable:      vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable:     vk::FALSE,
            ..Default::default()
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable:  vk::FALSE,
            attachment_count: 1,
            p_attachments:    &color_blend_attachment,
            ..Default::default()
        };

        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts:    &descriptors.layout,
            ..Default::default()
        };

        let pipeline_layout = unsafe { device.device.create_pipeline_layout(&layout_info, None)? };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count:            shader_stages.len() as u32,
            p_stages:               shader_stages.as_ptr(),
            p_vertex_input_state:   &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_viewport_state:       &viewport_state,
            p_rasterization_state:  &rasterizer,
            p_multisample_state:    &multisampling,
            p_depth_stencil_state:  &depth_stencil,
            p_color_blend_state:    &color_blending,
            layout:                 pipeline_layout,
            render_pass:            render_pass.render_pass,
            subpass:                0,
            base_pipeline_handle:   vk::Pipeline::null(),
            base_pipeline_index:    -1,
            ..Default::default()
        };

        let pipeline = unsafe {
            device.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            ).map_err(|(_, e)| e)?[0]
        };

        unsafe {
            device.device.destroy_shader_module(vert_module, None);
            device.device.destroy_shader_module(frag_module, None);
        }

        Ok(Self { pipeline, pipeline_layout, device_ref: device.device.clone() })
    }

    fn create_shader_module(
        device: &ash::Device,
        spv:    &[u8],
    ) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
        assert!(spv.len() % 4 == 0, "SPIR-V byte length must be a multiple of 4");
        let code: Vec<u32> = spv.chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let info = vk::ShaderModuleCreateInfo {
            code_size: spv.len(),
            p_code:    code.as_ptr(),
            ..Default::default()
        };
        Ok(unsafe { device.create_shader_module(&info, None)? })
    }
}

impl Drop for VulkanPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_pipeline(self.pipeline, None);
            self.device_ref.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
