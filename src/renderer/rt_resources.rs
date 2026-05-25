use ash::vk;
use super::device::VulkanDevice;
use super::buffer::VulkanBuffer;
use super::acceleration_structure::AccelerationStructure;
use super::command::VulkanCommandPool;

/// RGBA32F storage images and the RT descriptor set.
pub struct RtResources {
    pub accum_image:      vk::Image,
    pub accum_image_view: vk::ImageView,
    accum_memory:         vk::DeviceMemory,

    pub out_image:      vk::Image,
    pub out_image_view: vk::ImageView,
    out_memory:         vk::DeviceMemory,

    pub descriptor_pool:   vk::DescriptorPool,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub descriptor_set:    vk::DescriptorSet,

    #[allow(dead_code)]
    pub extent: vk::Extent2D,

    device_ref: ash::Device,
}

impl RtResources {
    pub fn new(
        device:      &VulkanDevice,
        cmd_pool:    &VulkanCommandPool,
        extent:      vk::Extent2D,
        tlas:        &AccelerationStructure,
        ubo_buffers: &[VulkanBuffer],
        vertex_buf:  &VulkanBuffer,
        index_buf:   &VulkanBuffer,
    ) -> Result<Self, Box<dyn std::error::Error>> {

        // ── Storage images ────────────────────────────────────────────────────
        let (accum_image, accum_memory) = Self::create_storage_image(device, extent)?;
        let (out_image,   out_memory)   = Self::create_storage_image(device, extent)?;

        let accum_image_view = Self::create_image_view(device, accum_image)?;
        let out_image_view   = Self::create_image_view(device, out_image)?;

        // Transition both to GENERAL
        let cb = cmd_pool.begin_single_time_commands()?;
        for &img in &[accum_image, out_image] {
            let barrier = vk::ImageMemoryBarrier {
                old_layout:          vk::ImageLayout::UNDEFINED,
                new_layout:          vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image:               img,
                subresource_range:   vk::ImageSubresourceRange {
                    aspect_mask:      vk::ImageAspectFlags::COLOR,
                    base_mip_level:   0,
                    level_count:      1,
                    base_array_layer: 0,
                    layer_count:      1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                ..Default::default()
            };
            unsafe {
                device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[], &[], &[barrier],
                );
            }
        }
        cmd_pool.end_single_time_commands(cb, device.graphics_queue)?;

        // ── Descriptor set layout ─────────────────────────────────────────────
        // Binding 0: TLAS
        // Binding 1: storage image (accum)
        // Binding 2: storage image (out)
        // Binding 3: uniform buffer (RT UBO)
        // Binding 4: storage buffer (vertex data)
        // Binding 5: storage buffer (index data)
        let rt_stages = vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR;

        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding:          0,
                descriptor_type:  vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          1,
                descriptor_type:  vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          2,
                descriptor_type:  vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          3,
                descriptor_type:  vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          4,
                descriptor_type:  vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          5,
                descriptor_type:  vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings:    bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_layout = unsafe {
            device.device.create_descriptor_set_layout(&layout_info, None)?
        };

        // ── Descriptor pool ───────────────────────────────────────────────────
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE,              descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER,             descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER,             descriptor_count: 2 },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets:        1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes:    pool_sizes.as_ptr(),
            ..Default::default()
        };
        let descriptor_pool = unsafe {
            device.device.create_descriptor_pool(&pool_info, None)?
        };

        // ── Allocate descriptor set ───────────────────────────────────────────
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts:        &descriptor_layout,
            ..Default::default()
        };
        let descriptor_set = unsafe {
            device.device.allocate_descriptor_sets(&alloc_info)?[0]
        };

        // ── Write descriptors (frame 0's UBO; we update binding 3 each frame) ─
        // Binding 0: TLAS
        let tlas_handle = tlas.handle;
        let mut write_tlas = vk::WriteDescriptorSetAccelerationStructureKHR {
            acceleration_structure_count: 1,
            p_acceleration_structures:    &tlas_handle,
            ..Default::default()
        };
        let write0 = vk::WriteDescriptorSet {
            p_next:            &mut write_tlas as *mut _ as *const std::ffi::c_void,
            dst_set:           descriptor_set,
            dst_binding:       0,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            ..Default::default()
        };

        // Binding 1: accum image
        let accum_info = vk::DescriptorImageInfo {
            image_view:   accum_image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler:      vk::Sampler::null(),
        };
        let write1 = vk::WriteDescriptorSet {
            dst_set:           descriptor_set,
            dst_binding:       1,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::STORAGE_IMAGE,
            p_image_info:      &accum_info,
            ..Default::default()
        };

        // Binding 2: out image
        let out_info = vk::DescriptorImageInfo {
            image_view:   out_image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler:      vk::Sampler::null(),
        };
        let write2 = vk::WriteDescriptorSet {
            dst_set:           descriptor_set,
            dst_binding:       2,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::STORAGE_IMAGE,
            p_image_info:      &out_info,
            ..Default::default()
        };

        // Binding 3: UBO — use first frame's buffer; update each frame via update_rt_ubo_descriptor
        let ubo_info = vk::DescriptorBufferInfo {
            buffer: ubo_buffers[0].buffer,
            offset: 0,
            range:  ubo_buffers[0].size,
        };
        let write3 = vk::WriteDescriptorSet {
            dst_set:           descriptor_set,
            dst_binding:       3,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info:     &ubo_info,
            ..Default::default()
        };

        // Binding 4: vertex SSBO
        let vb_info = vk::DescriptorBufferInfo {
            buffer: vertex_buf.buffer,
            offset: 0,
            range:  vertex_buf.size,
        };
        let write4 = vk::WriteDescriptorSet {
            dst_set:           descriptor_set,
            dst_binding:       4,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info:     &vb_info,
            ..Default::default()
        };

        // Binding 5: index SSBO
        let ib_info = vk::DescriptorBufferInfo {
            buffer: index_buf.buffer,
            offset: 0,
            range:  index_buf.size,
        };
        let write5 = vk::WriteDescriptorSet {
            dst_set:           descriptor_set,
            dst_binding:       5,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info:     &ib_info,
            ..Default::default()
        };

        unsafe {
            device.device.update_descriptor_sets(
                &[write0, write1, write2, write3, write4, write5],
                &[],
            );
        }

        Ok(Self {
            accum_image,
            accum_image_view,
            accum_memory,
            out_image,
            out_image_view,
            out_memory,
            descriptor_pool,
            descriptor_layout,
            descriptor_set,
            extent,
            device_ref: device.device.clone(),
        })
    }

    /// Re-point binding 3 (UBO) to the buffer for the current frame.
    pub fn update_ubo_descriptor(&self, device: &ash::Device, ubo_buf: &VulkanBuffer) {
        let buf_info = vk::DescriptorBufferInfo {
            buffer: ubo_buf.buffer,
            offset: 0,
            range:  ubo_buf.size,
        };
        let write = vk::WriteDescriptorSet {
            dst_set:           self.descriptor_set,
            dst_binding:       3,
            dst_array_element: 0,
            descriptor_count:  1,
            descriptor_type:   vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info:     &buf_info,
            ..Default::default()
        };
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    fn create_storage_image(
        device: &VulkanDevice,
        extent: vk::Extent2D,
    ) -> Result<(vk::Image, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format:     vk::Format::R32G32B32A32_SFLOAT,
            extent: vk::Extent3D {
                width:  extent.width,
                height: extent.height,
                depth:  1,
            },
            mip_levels:     1,
            array_layers:   1,
            samples:        vk::SampleCountFlags::TYPE_1,
            tiling:         vk::ImageTiling::OPTIMAL,
            usage:          vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            sharing_mode:   vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image = unsafe { device.device.create_image(&image_info, None)? };
        let mem_req  = unsafe { device.device.get_image_memory_requirements(image) };
        let mem_type = device.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size:   mem_req.size,
            memory_type_index: mem_type,
            ..Default::default()
        };
        let memory = unsafe {
            let mem = device.device.allocate_memory(&alloc_info, None)?;
            device.device.bind_image_memory(image, mem, 0)?;
            mem
        };
        Ok((image, memory))
    }

    fn create_image_view(
        device: &VulkanDevice,
        image:  vk::Image,
    ) -> Result<vk::ImageView, Box<dyn std::error::Error>> {
        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format:    vk::Format::R32G32B32A32_SFLOAT,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                base_mip_level:   0,
                level_count:      1,
                base_array_layer: 0,
                layer_count:      1,
            },
            ..Default::default()
        };
        Ok(unsafe { device.device.create_image_view(&view_info, None)? })
    }
}

impl Drop for RtResources {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_image_view(self.accum_image_view, None);
            self.device_ref.destroy_image_view(self.out_image_view, None);
            self.device_ref.destroy_image(self.accum_image, None);
            self.device_ref.destroy_image(self.out_image, None);
            self.device_ref.free_memory(self.accum_memory, None);
            self.device_ref.free_memory(self.out_memory, None);
            self.device_ref.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device_ref.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}
